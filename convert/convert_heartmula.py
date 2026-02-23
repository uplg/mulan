#!/usr/bin/env python3
"""Convert HeartMuLa PyTorch checkpoint to MLX format.

The HeartMuLa checkpoint is already in a very clean format (torchtune-style keys).
The only transformation needed is renaming RMSNorm ``.scale`` → ``.weight`` to
match the MLX ``nn.RMSNorm`` convention.

Usage:
    python ckpt-mlx/convert_heartmula.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CKPT_DIR = Path(__file__).parent.parent / "ckpt" / "HeartMuLa-oss-3B"
OUT_DIR = Path(__file__).parent / "HeartMuLa-oss-3B"

SHARD_FILES = sorted(CKPT_DIR.glob("model-*.safetensors"))


def convert_key(key: str) -> str:
    """Rename checkpoint key to MLX convention.

    Only change: ``*.scale`` → ``*.weight`` for RMSNorm layers (sa_norm, mlp_norm,
    backbone.norm, decoder.norm).
    """
    if key.endswith(".scale"):
        return key[: -len(".scale")] + ".weight"
    return key


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all shards
    tensors: dict[str, np.ndarray] = {}
    for shard in SHARD_FILES:
        print(f"Loading {shard.name} ...")
        with safe_open(str(shard), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    print(f"Loaded {len(tensors)} tensors from {len(SHARD_FILES)} shards.")

    # Convert
    out_tensors: dict[str, np.ndarray] = {}
    for key, arr in sorted(tensors.items()):
        new_key = convert_key(key)
        if new_key != key:
            print(f"  rename: {key} -> {new_key}")
        out_tensors[new_key] = arr

    print(f"\nWriting {len(out_tensors)} tensors to {OUT_DIR / 'weights.safetensors'} ...")
    save_file(out_tensors, str(OUT_DIR / "weights.safetensors"))

    # Copy config.json
    src_config = CKPT_DIR / "config.json"
    dst_config = OUT_DIR / "config.json"
    shutil.copy2(src_config, dst_config)
    print(f"Copied {src_config.name} -> {dst_config}")

    # Verify
    print("\n--- Verification ---")
    with safe_open(str(OUT_DIR / "weights.safetensors"), framework="numpy") as f:
        out_keys = sorted(f.keys())
    print(f"Output keys: {len(out_keys)}")

    # Quick size report
    total_bytes = sum(arr.nbytes for arr in out_tensors.values())
    total_params = sum(arr.size for arr in out_tensors.values())
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    print("\nDone!")


if __name__ == "__main__":
    main()
