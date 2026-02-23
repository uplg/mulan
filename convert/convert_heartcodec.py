#!/usr/bin/env python3
"""Convert HeartCodec PyTorch checkpoint to MLX format.

Handles:
  - Weight-norm fusion: parametrizations.weight.original0/1 → effective weight
  - Conv1d weight transpose: PyTorch (out, in, k) → MLX (out, k, in)
  - ConvTranspose1d weight transpose: PyTorch (in, out, k) → MLX (out, k, in)
  - Key remapping: legacy paths → MLX module paths
  - Codebook rename: _codebook → codebook
  - Estimator Conv1d (proj_in/proj_out/connection_proj ffn_1) transpose

Usage:
    python ckpt-mlx/convert_heartcodec.py
"""

from __future__ import annotations

import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CKPT_DIR = Path(__file__).parent.parent / "ckpt" / "HeartCodec-oss"
OUT_DIR = Path(__file__).parent / "HeartCodec-oss"

SHARD_FILES = [
    CKPT_DIR / "model-00001-of-00002.safetensors",
    CKPT_DIR / "model-00002-of-00002.safetensors",
]


# ---------------------------------------------------------------------------
# Weight-norm fusion
# ---------------------------------------------------------------------------


def fuse_weight_norm(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fuse weight-norm parameters into an effective weight.

    g: (out_channels, 1, 1) for Conv1d, or (in_channels, 1, 1) for ConvTranspose1d
    v: the full weight matrix
    Returns: g * v / ||v||  (norm over all dims except dim 0)
    """
    # Compute L2 norm of v over all dims except 0
    norm_dims = tuple(range(1, v.ndim))
    v_norm = np.sqrt((v * v).sum(axis=norm_dims, keepdims=True))
    v_norm = np.maximum(v_norm, 1e-12)  # avoid division by zero
    return g * v / v_norm


# ---------------------------------------------------------------------------
# Conv weight transpositions
# ---------------------------------------------------------------------------


def transpose_conv1d_weight(w: np.ndarray) -> np.ndarray:
    """PyTorch Conv1d: (out_ch, in_ch, kernel) → MLX: (out_ch, kernel, in_ch)."""
    return np.swapaxes(w, 1, 2)


def transpose_convtranspose1d_weight(w: np.ndarray) -> np.ndarray:
    """PyTorch ConvTranspose1d: (in_ch, out_ch, kernel) → MLX: (out_ch, kernel, in_ch).

    PyTorch stores ConvTranspose1d weight as (in_channels, out_channels/groups, kernel_size).
    MLX ConvTranspose1d expects (out_channels, kernel_size, in_channels).
    """
    # (in_ch, out_ch, k) → transpose(1,0,2) → (out_ch, in_ch, k) → swap(1,2) → (out_ch, k, in_ch)
    return np.swapaxes(w.transpose(1, 0, 2), 1, 2)


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------

# Pre-compiled patterns for matching

# ScalarModel: top-level Conv1d with weight_norm (decoder.0, decoder.7, encoder.0, encoder.7)
# Legacy: scalar_model.{enc_or_dec}.{idx}.parametrizations.weight.original{0,1}
# Legacy: scalar_model.{enc_or_dec}.{idx}.bias
RE_SM_TOPLEVEL_WN = re.compile(
    r"^scalar_model\.(encoder|decoder)\.(\d+)\."
    r"parametrizations\.weight\.original([01])$"
)
RE_SM_TOPLEVEL_BIAS = re.compile(r"^scalar_model\.(encoder|decoder)\.(\d+)\.bias$")

# ScalarModel: ResBlock conv1/conv2 with weight_norm
# Legacy: scalar_model.decoder.{blk}.convs.{j}.conv{k}.parametrizations.weight.original{0,1}
RE_SM_RESCONV_WN = re.compile(
    r"^scalar_model\.(encoder|decoder)\.(\d+)\.convs\.(\d+)\.(conv[12])\."
    r"parametrizations\.weight\.original([01])$"
)
RE_SM_RESCONV_BIAS = re.compile(
    r"^scalar_model\.(encoder|decoder)\.(\d+)\.convs\.(\d+)\.(conv[12])\.bias$"
)

# ScalarModel: UpsampleLayer ConvTranspose1d with weight_norm
# Legacy: scalar_model.decoder.{blk}.up_conv.layer.parametrizations.weight.original{0,1}
RE_SM_UPCONV_WN = re.compile(
    r"^scalar_model\.decoder\.(\d+)\.up_conv\.layer\."
    r"parametrizations\.weight\.original([01])$"
)
RE_SM_UPCONV_BIAS = re.compile(r"^scalar_model\.decoder\.(\d+)\.up_conv\.layer\.bias$")

# ScalarModel: DownsampleLayer Conv1d with weight_norm
# Legacy: scalar_model.encoder.{blk}.down_conv.layer.parametrizations.weight.original{0,1}
RE_SM_DOWNCONV_WN = re.compile(
    r"^scalar_model\.encoder\.(\d+)\.down_conv\.layer\."
    r"parametrizations\.weight\.original([01])$"
)
RE_SM_DOWNCONV_BIAS = re.compile(r"^scalar_model\.encoder\.(\d+)\.down_conv\.layer\.bias$")

# ScalarModel: PostProcessor conv (no weight_norm)
# Legacy: scalar_model.decoder.6.conv.weight / bias
# Also: scalar_model.encoder.1.conv.weight / bias  (PreProcessor)
RE_SM_PLAIN_CONV = re.compile(r"^scalar_model\.(encoder|decoder)\.(\d+)\.conv\.(weight|bias)$")

# VQ codebook rename: _codebook → codebook
RE_VQ_CODEBOOK = re.compile(r"^(flow_matching\.vq_embed\.layers\.\d+)\._codebook\.(.+)$")

# Estimator Conv1d layers (ffn_1) — need transposition
RE_EST_FFN1 = re.compile(
    r"^(flow_matching\.estimator\.(?:proj_in|proj_out|connection_proj)\.ffn_1)\.(weight|bias)$"
)

# PReLU activation weights — direct copy (all shapes match)
# Down conv activation
RE_SM_DOWNCONV_ACT = re.compile(r"^scalar_model\.encoder\.(\d+)\.down_conv\.activation\.weight$")


def is_decoder_non_causal_toplevel(enc_or_dec: str, idx: int) -> bool:
    """decoder.0 is non-causal (plain nn.Conv1d), others are causal (CausalConv1d)."""
    return enc_or_dec == "decoder" and idx == 0


def build_conv1d_mlx_key(enc_or_dec: str, idx: int, suffix: str) -> str:
    """Build MLX key for a top-level scalar_model Conv1d.

    If non-causal: impl is nn.Conv1d directly → key is .impl.{suffix}
    If causal: impl is CausalConv1d → inner conv → key is .impl.conv.{suffix}
    """
    if is_decoder_non_causal_toplevel(enc_or_dec, idx):
        # decoder.0: Conv1d wrapper → impl = nn.Conv1d (non-causal, direct)
        return f"scalar_model.{enc_or_dec}.{idx}.impl.{suffix}"
    else:
        # causal: Conv1d wrapper → impl = CausalConv1d → conv = nn.Conv1d
        return f"scalar_model.{enc_or_dec}.{idx}.impl.conv.{suffix}"


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def load_all_tensors() -> dict[str, np.ndarray]:
    """Load all tensors from all shards into a single dict."""
    tensors: dict[str, np.ndarray] = {}
    for shard_path in SHARD_FILES:
        with safe_open(str(shard_path), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


def convert() -> dict[str, np.ndarray]:
    """Convert PyTorch HeartCodec checkpoint to MLX format."""
    src = load_all_tensors()
    dst: dict[str, np.ndarray] = {}

    # Collect weight-norm pairs: group by prefix to fuse original0 + original1
    wn_pairs: dict[str, dict[str, np.ndarray]] = {}  # prefix → {"g": ..., "v": ...}

    processed_keys: set[str] = set()

    # First pass: identify weight-norm pairs and collect them
    for key in src:
        # Top-level Conv1d weight-norm
        m = RE_SM_TOPLEVEL_WN.match(key)
        if m:
            enc_or_dec, idx_str, orig_idx = m.groups()
            prefix = f"sm_toplevel_{enc_or_dec}_{idx_str}"
            if prefix not in wn_pairs:
                wn_pairs[prefix] = {
                    "enc_or_dec": enc_or_dec,
                    "idx": int(idx_str),
                    "type": "toplevel",
                }
            if orig_idx == "0":
                wn_pairs[prefix]["g"] = src[key]
            else:
                wn_pairs[prefix]["v"] = src[key]
            processed_keys.add(key)
            continue

        # ResBlock conv weight-norm
        m = RE_SM_RESCONV_WN.match(key)
        if m:
            enc_or_dec, blk, j, conv_name, orig_idx = m.groups()
            prefix = f"sm_resconv_{enc_or_dec}_{blk}_{j}_{conv_name}"
            if prefix not in wn_pairs:
                wn_pairs[prefix] = {
                    "enc_or_dec": enc_or_dec,
                    "blk": int(blk),
                    "j": int(j),
                    "conv_name": conv_name,
                    "type": "resconv",
                }
            if orig_idx == "0":
                wn_pairs[prefix]["g"] = src[key]
            else:
                wn_pairs[prefix]["v"] = src[key]
            processed_keys.add(key)
            continue

        # UpsampleLayer ConvTranspose1d weight-norm
        m = RE_SM_UPCONV_WN.match(key)
        if m:
            blk, orig_idx = m.groups()
            prefix = f"sm_upconv_{blk}"
            if prefix not in wn_pairs:
                wn_pairs[prefix] = {"blk": int(blk), "type": "upconv"}
            if orig_idx == "0":
                wn_pairs[prefix]["g"] = src[key]
            else:
                wn_pairs[prefix]["v"] = src[key]
            processed_keys.add(key)
            continue

        # DownsampleLayer Conv1d weight-norm
        m = RE_SM_DOWNCONV_WN.match(key)
        if m:
            blk, orig_idx = m.groups()
            prefix = f"sm_downconv_{blk}"
            if prefix not in wn_pairs:
                wn_pairs[prefix] = {"blk": int(blk), "type": "downconv"}
            if orig_idx == "0":
                wn_pairs[prefix]["g"] = src[key]
            else:
                wn_pairs[prefix]["v"] = src[key]
            processed_keys.add(key)
            continue

    # Second pass: process weight-norm pairs
    for prefix, info in wn_pairs.items():
        g, v = info["g"], info["v"]
        w = fuse_weight_norm(g, v)

        if info["type"] == "toplevel":
            enc_or_dec = info["enc_or_dec"]
            idx = info["idx"]
            w_mlx = transpose_conv1d_weight(w)
            mlx_key = build_conv1d_mlx_key(enc_or_dec, idx, "weight")
            dst[mlx_key] = w_mlx

        elif info["type"] == "resconv":
            enc_or_dec = info["enc_or_dec"]
            blk = info["blk"]
            j = info["j"]
            conv_name = info["conv_name"]
            w_mlx = transpose_conv1d_weight(w)
            # All resblock convs are causal → .impl.conv.weight
            mlx_key = f"scalar_model.{enc_or_dec}.{blk}.convs.{j}.{conv_name}.impl.conv.weight"
            dst[mlx_key] = w_mlx

        elif info["type"] == "upconv":
            blk = info["blk"]
            w_mlx = transpose_convtranspose1d_weight(w)
            # Causal ConvTranspose1d: .impl.conv_t.weight
            mlx_key = f"scalar_model.decoder.{blk}.up_conv.conv_t.impl.conv_t.weight"
            dst[mlx_key] = w_mlx

        elif info["type"] == "downconv":
            blk = info["blk"]
            w_mlx = transpose_conv1d_weight(w)
            # Causal Conv1d: .impl.conv.weight
            mlx_key = f"scalar_model.encoder.{blk}.down_conv.conv.impl.conv.weight"
            dst[mlx_key] = w_mlx

    # Third pass: process remaining keys
    for key in src:
        if key in processed_keys:
            continue

        val = src[key]

        # Top-level bias
        m = RE_SM_TOPLEVEL_BIAS.match(key)
        if m:
            enc_or_dec, idx_str = m.groups()
            idx = int(idx_str)
            mlx_key = build_conv1d_mlx_key(enc_or_dec, idx, "bias")
            dst[mlx_key] = val
            processed_keys.add(key)
            continue

        # ResBlock conv bias
        m = RE_SM_RESCONV_BIAS.match(key)
        if m:
            enc_or_dec, blk, j, conv_name = m.groups()
            mlx_key = f"scalar_model.{enc_or_dec}.{blk}.convs.{j}.{conv_name}.impl.conv.bias"
            dst[mlx_key] = val
            processed_keys.add(key)
            continue

        # UpsampleLayer bias
        m = RE_SM_UPCONV_BIAS.match(key)
        if m:
            blk = m.group(1)
            mlx_key = f"scalar_model.decoder.{blk}.up_conv.conv_t.impl.conv_t.bias"
            dst[mlx_key] = val
            processed_keys.add(key)
            continue

        # DownsampleLayer bias
        m = RE_SM_DOWNCONV_BIAS.match(key)
        if m:
            blk = m.group(1)
            mlx_key = f"scalar_model.encoder.{blk}.down_conv.conv.impl.conv.bias"
            dst[mlx_key] = val
            processed_keys.add(key)
            continue

        # DownsampleLayer activation (PReLU)
        m = RE_SM_DOWNCONV_ACT.match(key)
        if m:
            blk = m.group(1)
            mlx_key = f"scalar_model.encoder.{blk}.down_conv.activation.weight"
            dst[mlx_key] = val
            processed_keys.add(key)
            continue

        # PostProcessor / PreProcessor plain conv (no weight_norm)
        m = RE_SM_PLAIN_CONV.match(key)
        if m:
            enc_or_dec, idx_str, param_name = m.groups()
            idx = int(idx_str)
            if param_name == "weight":
                val = transpose_conv1d_weight(val)
            # Plain conv in PostProcessor/PreProcessor: .conv.impl.conv.{weight|bias}
            # These are causal Conv1d wrapping CausalConv1d wrapping nn.Conv1d
            mlx_key = f"scalar_model.{enc_or_dec}.{idx}.conv.impl.conv.{param_name}"
            dst[mlx_key] = val
            processed_keys.add(key)
            continue

        # VQ codebook rename: _codebook → codebook
        m = RE_VQ_CODEBOOK.match(key)
        if m:
            prefix_part, remainder = m.groups()
            mlx_key = f"{prefix_part}.codebook.{remainder}"
            dst[mlx_key] = val
            processed_keys.add(key)
            continue

        # Estimator Conv1d ffn_1 weight (needs transpose)
        m = RE_EST_FFN1.match(key)
        if m:
            prefix_part, param_name = m.groups()
            if param_name == "weight":
                val = transpose_conv1d_weight(val)
            dst[f"{prefix_part}.{param_name}"] = val
            processed_keys.add(key)
            continue

        # Everything else: direct copy (activation weights, linear layers,
        # transformer params, etc.)
        dst[key] = val
        processed_keys.add(key)

    return dst


def main() -> None:
    print("Loading PyTorch checkpoint...")
    mlx_weights = convert()

    # Verify: weight-norm fusion merges 2 keys → 1, so dst should have fewer keys
    # Count weight-norm pairs in source
    src = load_all_tensors()
    wn_count = sum(1 for k in src if "parametrizations.weight.original" in k)
    wn_pairs_count = wn_count // 2
    expected_dst = len(src) - wn_pairs_count
    assert len(mlx_weights) == expected_dst, (
        f"Key count mismatch: expected {expected_dst} (src={len(src)} - "
        f"{wn_pairs_count} fused pairs), got {len(mlx_weights)}"
    )

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "weights.safetensors"

    # Convert all to float32 numpy arrays for safetensors
    save_dict = {}
    for k, v in mlx_weights.items():
        if isinstance(v, np.ndarray):
            save_dict[k] = v.astype(np.float32)
        else:
            save_dict[k] = np.array(v, dtype=np.float32)

    print(f"Saving {len(save_dict)} tensors to {out_path}...")
    save_file(save_dict, str(out_path))
    print("Done!")

    # Print summary stats
    total_params = sum(v.size for v in save_dict.values())
    total_bytes = sum(v.nbytes for v in save_dict.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
