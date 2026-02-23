"""HeartMuLa model ported to MLX.

This re-implements the subset of torchtune (TransformerDecoder, MultiHeadAttention,
FeedForward, KVCache, Llama3ScaledRoPE) needed by HeartMuLa, plus the HeartMuLa
wrapper itself.  Numerics must match the PyTorch original to ~1e-5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .configuration_heartmula import FLAVORS, HeartMuLaConfig, LlamaFlavorSpec


# RoPE  (matches torchtune Llama3ScaledRoPE exactly)


def _apply_scaling(
    freqs: mx.array,
    scale_factor: int,
    low_freq_factor: int,
    high_freq_factor: int,
    old_context_len: int,
) -> mx.array:
    """Llama-3.1 scaled RoPE frequency adjustment."""
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for i in range(freqs.shape[0]):
        freq = freqs[i].item()
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return mx.array(new_freqs, dtype=freqs.dtype)


def _build_rope_cache(
    dim: int,
    max_seq_len: int,
    base: int = 500_000,
    scale_factor: int = 32,
    low_freq_factor: int = 1,
    high_freq_factor: int = 4,
    old_context_len: int = 8192,
) -> mx.array:
    """Build a RoPE cache of shape ``[max_seq_len, dim//2, 2]`` (cos, sin)."""
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32)[: dim // 2] / dim))
    theta = _apply_scaling(freqs, scale_factor, low_freq_factor, high_freq_factor, old_context_len)
    seq_idx = mx.arange(max_seq_len, dtype=mx.float32)
    idx_theta = mx.outer(seq_idx, theta)  # [max_seq_len, dim//2]
    cache = mx.stack([mx.cos(idx_theta), mx.sin(idx_theta)], axis=-1)  # [max_seq_len, dim//2, 2]
    return cache


def _apply_rope(x: mx.array, cache: mx.array) -> mx.array:
    """Apply rotary embeddings to *x* ``[B, S, H, D]`` using *cache* ``[S, D//2, 2]``.

    torchtune reshapes into ``[..., D//2, 2]``, multiplies cos/sin, and flattens back.
    We do the same.  The cache is kept in float32 for precision; the output is cast
    back to *x*'s dtype so downstream attention stays in the model's compute dtype.
    """
    input_dtype = x.dtype
    *shape, D = x.shape  # [B, S, H, D]
    xr = x.reshape(*shape, D // 2, 2)  # [B, S, H, D//2, 2]

    # cache: [S, D//2, 2] -> broadcast-compatible [1, S, 1, D//2, 2]
    c = cache.reshape(1, cache.shape[0], 1, cache.shape[1], 2)

    x0 = xr[..., 0]
    x1 = xr[..., 1]
    c_cos = c[..., 0]
    c_sin = c[..., 1]

    out0 = x0 * c_cos - x1 * c_sin
    out1 = x1 * c_cos + x0 * c_sin
    out = mx.stack([out0, out1], axis=-1)  # [B, S, H, D//2, 2]
    return out.reshape(*shape, D).astype(input_dtype)


# KV-Cache  (plain dataclass — NOT an nn.Module, so not in parameter tree)


@dataclass
class KVCache:
    """Simple KV-cache that mirrors torchtune's KVCache behaviour.

    Stores pre-allocated k/v arrays of shape ``[B, H, max_seq_len, D]`` and
    tracks the current write position.
    """

    k_cache: mx.array  # [B, H, max_seq_len, D]
    v_cache: mx.array
    pos: int = 0

    @staticmethod
    def create(
        batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ) -> KVCache:
        shape = (batch_size, num_heads, max_seq_len, head_dim)
        return KVCache(
            k_cache=mx.zeros(shape, dtype=dtype),
            v_cache=mx.zeros(shape, dtype=dtype),
            pos=0,
        )

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """Write *k*, *v* of shape ``[B, H, S, D]`` and return full cache."""
        seq_len = k.shape[2]
        self.k_cache[:, :, self.pos : self.pos + seq_len, :] = k
        self.v_cache[:, :, self.pos : self.pos + seq_len, :] = v
        self.pos += seq_len
        return self.k_cache, self.v_cache

    def reset(self) -> None:
        # Just reset position — old data beyond pos is never read by attention
        # (attention only reads up to self.pos), and gets overwritten by update().
        # Avoids allocating new mx.zeros_like tensors (7x per frame in decoder).
        self.pos = 0


# Multi-Head Attention  (with GQA + RoPE + KV-cache)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with grouped-query support, matching torchtune exactly."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_per_kv = num_heads // num_kv_heads
        self._scale = math.sqrt(head_dim)

        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

        # KV-cache — set at runtime via setup_cache(), not an nn.Module attribute
        self._kv_cache: KVCache | None = None

    def setup_cache(self, batch_size: int, max_seq_len: int, dtype: mx.Dtype = mx.float32) -> None:
        self._kv_cache = KVCache.create(
            batch_size, self.num_heads, max_seq_len, self.head_dim, dtype=dtype
        )

    def reset_cache(self) -> None:
        if self._kv_cache is not None:
            self._kv_cache.reset()

    @property
    def cache_enabled(self) -> bool:
        return self._kv_cache is not None

    def __call__(
        self,
        x: mx.array,
        y: mx.array | None = None,
        *,
        mask: mx.array | None = None,
        input_pos: mx.array | None = None,
        rope_cache: mx.array | None = None,
    ) -> mx.array:
        B, S_x, _ = x.shape

        # Query
        q = self.q_proj(x)  # [B, S_x, num_heads*head_dim]
        q = q.reshape(B, S_x, self.num_heads, self.head_dim)  # [B, S_x, H, D]

        # Apply RoPE to q
        if rope_cache is not None and input_pos is not None:
            # input_pos: [B, S_x] — select the right cache rows (same across batch)
            rope_sub = rope_cache[input_pos[0]]  # [S_x, D//2, 2]
            q = _apply_rope(q, rope_sub)

        # q: [B, H, S_x, D]
        q = q.transpose(0, 2, 1, 3)

        if y is None:
            # Streaming mode: k/v come entirely from cache
            assert self._kv_cache is not None
            k = self._kv_cache.k_cache
            v = self._kv_cache.v_cache
        else:
            S_y = y.shape[1]
            k = self.k_proj(y).reshape(B, S_y, self.num_kv_heads, self.head_dim)
            v = self.v_proj(y).reshape(B, S_y, self.num_kv_heads, self.head_dim)

            # Apply RoPE to k
            if rope_cache is not None and input_pos is not None:
                rope_sub = rope_cache[input_pos[0]]
                k = _apply_rope(k, rope_sub)

            # GQA expansion: [B, S, n_kv, 1, D] -> [B, S, n_kv, q_per_kv, D] -> [B, S, H, D]
            if self.num_heads != self.num_kv_heads:
                k = mx.repeat(
                    k.reshape(B, S_y, self.num_kv_heads, 1, self.head_dim),
                    repeats=self.q_per_kv,
                    axis=3,
                )
                k = k.reshape(B, S_y, self.num_heads, self.head_dim)
                v = mx.repeat(
                    v.reshape(B, S_y, self.num_kv_heads, 1, self.head_dim),
                    repeats=self.q_per_kv,
                    axis=3,
                )
                v = v.reshape(B, S_y, self.num_heads, self.head_dim)

            # [B, H, S, D]
            k = k.transpose(0, 2, 1, 3)
            v = v.transpose(0, 2, 1, 3)

            # Update KV-cache
            if self._kv_cache is not None:
                k, v = self._kv_cache.update(k, v)

        # Scaled dot-product attention.
        scores = (q @ k.transpose(0, 1, 3, 2)) / self._scale  # [B, H, S_x, S_kv]

        if mask is not None:
            # mask: [B, S_x, S_kv] bool — True means attend
            # expand for heads: [B, 1, S_x, S_kv]
            mask_4d = mask[:, None, :, :]
            large_neg = mx.array(-1e9, dtype=scores.dtype)
            scores = mx.where(mask_4d, scores, large_neg)

        weights = mx.softmax(scores, axis=-1)
        out = weights @ v  # [B, H, S_x, D]

        # Merge heads
        out = out.transpose(0, 2, 1, 3).reshape(B, S_x, -1)  # [B, S_x, H*D]

        # Project back. Keep output dtype consistent with input stream.
        proj = self.output_proj(out)
        if proj.dtype != x.dtype:
            proj = proj.astype(x.dtype)
        return proj


# Feed-forward  (SwiGLU)


class FeedForward(nn.Module):
    """SwiGLU feed-forward matching torchtune's ``FeedForward``."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up_proj

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# Transformer layer


class TransformerSelfAttentionLayer(nn.Module):
    """Pre-norm transformer layer matching torchtune's layout."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_dim: int,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, num_kv_heads, head_dim)
        self.mlp = FeedForward(embed_dim, intermediate_dim)
        self.sa_norm = nn.RMSNorm(embed_dim, eps=norm_eps)
        self.mlp_norm = nn.RMSNorm(embed_dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        input_pos: mx.array | None = None,
        rope_cache: mx.array | None = None,
    ) -> mx.array:
        h = self.sa_norm(x)
        h = self.attn(h, h, mask=mask, input_pos=input_pos, rope_cache=rope_cache) + x
        out = self.mlp(self.mlp_norm(h)) + h
        return out


# Transformer Decoder  (mirrors torchtune TransformerDecoder)


class TransformerDecoder(nn.Module):
    """Stack of transformer layers with norm + output projection.

    In HeartMuLa, ``tok_embeddings`` and ``output`` are replaced with Identity,
    so this class does not include them.  The RoPE cache is stored with an
    underscore prefix so MLX does not treat it as a model parameter.
    """

    def __init__(
        self,
        spec: LlamaFlavorSpec,
    ) -> None:
        super().__init__()
        self.max_seq_len = spec.max_seq_len
        self.num_heads = spec.num_heads
        self.head_dim = spec.head_dim
        self.embed_dim = spec.embed_dim

        # Build RoPE cache (shared across all layers).
        # Underscore prefix: MLX nn.Module skips attributes starting with '_'
        # when traversing parameters — this is a precomputed buffer, not a weight.
        self._rope_cache: mx.array = _build_rope_cache(
            dim=spec.head_dim,
            max_seq_len=spec.max_seq_len,
            base=spec.rope_base,
            scale_factor=spec.scale_factor,
        )

        self.layers = [
            TransformerSelfAttentionLayer(
                embed_dim=spec.embed_dim,
                num_heads=spec.num_heads,
                num_kv_heads=spec.num_kv_heads,
                head_dim=spec.head_dim,
                intermediate_dim=spec.intermediate_dim,
                norm_eps=spec.norm_eps,
            )
            for _ in range(spec.num_layers)
        ]
        self.norm = nn.RMSNorm(spec.embed_dim, eps=spec.norm_eps)

        # Cache state
        self._caches_enabled = False

    def setup_caches(
        self,
        batch_size: int,
        *,
        decoder_max_seq_len: int | None = None,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        max_len = decoder_max_seq_len if decoder_max_seq_len is not None else self.max_seq_len
        for layer in self.layers:
            layer.attn.setup_cache(batch_size, max_len, dtype=dtype)
        self._caches_enabled = True

    def caches_are_enabled(self) -> bool:
        return self._caches_enabled

    def reset_caches(self) -> None:
        for layer in self.layers:
            layer.attn.reset_cache()

    def __call__(
        self,
        h: mx.array,
        *,
        input_pos: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Forward pass.  *h* is already embedded (Identity tok_embeddings)."""
        for layer in self.layers:
            h = layer(h, mask=mask, input_pos=input_pos, rope_cache=self._rope_cache)
        h = self.norm(h)
        # Match torchtune: `output = self.output(h).float()`
        # Since output = Identity(), this is just an upcast to float32.
        # Critical for decoder path — bfloat16 matmul corrupts codebook 1-7 logits.
        return h.astype(mx.float32)


# Sampling helpers


def _sample_topk(logits: mx.array, topk: int, temperature: float) -> mx.array:
    """Top-k sampling matching the legacy ``sample_topk`` exactly.

    Args:
        logits: ``[B, V]``
    Returns:
        ``[B, 1]`` int32 token ids.
    """
    logits = logits / temperature

    # Top-k mask
    # mx.topk returns the top-k values in ASCENDING order (smallest first),
    # unlike PyTorch which returns DESCENDING. So index [0] is the k-th largest.
    topk_vals = mx.topk(logits, topk, axis=-1)  # [B, topk] ascending
    threshold = topk_vals[:, 0:1]  # [B, 1] — the k-th largest value
    logits = mx.where(logits < threshold, -1e9, logits)

    # Log-softmax + softmax (matches legacy exactly)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    probs = mx.exp(log_probs)

    # Gumbel trick for multinomial sampling (no CUDA sync needed)
    q = -mx.log(mx.random.uniform(shape=probs.shape, dtype=probs.dtype))
    return mx.argmax(probs / q, axis=-1, keepdims=True).astype(mx.int32)


# Causal mask helpers


def _create_causal_mask(seq_len: int) -> mx.array:
    """Lower-triangular boolean mask ``[seq_len, seq_len]``."""
    return mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_), k=0)


def _index_causal_mask(mask: mx.array, input_pos: mx.array) -> mx.array:
    """Select rows from a causal mask.

    *input_pos*: ``[B, S]`` — use first batch element (all identical).
    Returns ``[B, S, full_seq_len]``.
    """
    rows = mask[input_pos]  # [B, S, full_seq_len]
    return rows


# HeartMuLa


class HeartMuLa(nn.Module):
    """HeartMuLa music generation model — MLX port."""

    def __init__(self, config: HeartMuLaConfig) -> None:
        super().__init__()
        self.config = config
        self._dtype: mx.Dtype = mx.float32

        backbone_spec = FLAVORS[config.backbone_flavor]
        decoder_spec = FLAVORS[config.decoder_flavor]
        backbone_dim = backbone_spec.embed_dim
        decoder_dim = decoder_spec.embed_dim

        self.backbone = TransformerDecoder(backbone_spec)
        self.decoder = TransformerDecoder(decoder_spec)

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks, backbone_dim
        )
        self.unconditional_text_embedding = nn.Embedding(1, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = mx.zeros(
            (config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size)
        )
        self.muq_linear = nn.Linear(config.muq_dim, backbone_dim)

        # Causal masks — stored with underscore to keep out of parameter tree
        self._backbone_causal_mask: mx.array | None = None
        self._decoder_causal_mask: mx.array | None = None

        # Precomputed decoder curr_pos tensors (set by setup_caches)
        self._decoder_pos_init: mx.array | None = None  # [1, 2] for initial 2-token input
        self._decoder_pos_step: list[mx.array] = []  # [1, 1] for each codebook step

    def setup_caches(self, max_batch_size: int) -> None:
        self.backbone.setup_caches(max_batch_size, dtype=self._dtype)
        self.decoder.setup_caches(
            max_batch_size,
            decoder_max_seq_len=self.config.audio_num_codebooks,
            dtype=self._dtype,
        )
        self._backbone_causal_mask = _create_causal_mask(self.backbone.max_seq_len)
        self._decoder_causal_mask = _create_causal_mask(self.config.audio_num_codebooks)

        # Precompute decoder position tensors (batch-broadcast at generate_frame time)
        self._decoder_pos_init = mx.arange(2)[None, :]  # [1, 2]
        self._decoder_pos_step = [
            mx.array([[i + 2]]) for i in range(self.config.audio_num_codebooks - 1)
        ]  # [1, 1] each

    def reset_caches(self) -> None:
        self.backbone.reset_caches()
        self.decoder.reset_caches()

    def _embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_tokens(self, tokens: mx.array, uncond_mask: mx.array | None) -> mx.array:
        """Embed a combined token tensor ``[B, S, audio_num_codebooks + 1]``.

        Last channel is text, first ``audio_num_codebooks`` channels are audio codebooks.
        Returns ``[B, S, audio_num_codebooks + 1, D]``.
        """
        B, S, _ = tokens.shape

        # Text embeddings from last channel
        text_embeds = self.text_embeddings(tokens[:, :, -1])  # [B, S, D]

        if uncond_mask is not None:
            uncond_text_embed = self.unconditional_text_embedding(
                mx.zeros((1,), dtype=mx.int32)
            )  # [1, D]
            mask_expanded = uncond_mask.reshape(B, 1, 1)  # broadcast over S, D
            text_embeds = mx.where(mask_expanded, uncond_text_embed, text_embeds)

        text_embeds = text_embeds[:, :, None, :]  # [B, S, 1, D]

        # Audio embeddings — offset each codebook
        offsets = self.config.audio_vocab_size * mx.arange(self.config.audio_num_codebooks)
        audio_tokens = tokens[:, :, :-1] + offsets  # [B, S, num_codebooks]
        audio_embeds = self.audio_embeddings(audio_tokens.reshape(-1)).reshape(
            B, S, self.config.audio_num_codebooks, -1
        )  # [B, S, num_codebooks, D]

        return mx.concatenate([audio_embeds, text_embeds], axis=2)  # [B, S, num_codebooks+1, D]

    def generate_frame(
        self,
        tokens: mx.array,
        tokens_mask: mx.array,
        input_pos: mx.array,
        temperature: float,
        topk: int,
        cfg_scale: float,
        continuous_segments: mx.array | None = None,
        starts: mx.array | None = None,
    ) -> mx.array:
        """Generate one frame of audio tokens (all codebooks).

        Args:
            tokens: ``[B, S, audio_num_codebooks + 1]``
            tokens_mask: ``[B, S, audio_num_codebooks + 1]``
            input_pos: ``[B, S]``
            temperature: sampling temperature
            topk: top-k sampling
            cfg_scale: classifier-free guidance scale
            continuous_segments: optional ``[B, muq_dim]``
            starts: optional positions for continuous segments

        Returns:
            ``[B, audio_num_codebooks]`` sampled tokens.
        """
        b, s, _ = tokens.shape

        assert self.backbone.caches_are_enabled(), "backbone caches are not enabled"
        assert self._backbone_causal_mask is not None, "call setup_caches() first"
        assert self._decoder_causal_mask is not None, "call setup_caches() first"
        curr_backbone_mask = _index_causal_mask(self._backbone_causal_mask, input_pos)

        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            uncond_mask = mx.concatenate(
                [
                    mx.zeros((actual_B,), dtype=mx.bool_),
                    mx.ones((actual_B,), dtype=mx.bool_),
                ]
            )

        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask)
        masked_embeds = embeds * tokens_mask[:, :, :, None]
        h = masked_embeds.sum(axis=2)  # merge: [B, S, D]

        if continuous_segments is not None:
            continuous_segments = self.muq_linear(continuous_segments)
            if uncond_mask is not None:
                uncond_embed = self.unconditional_text_embedding(mx.zeros((1,), dtype=mx.int32))
                mask_expanded = uncond_mask.reshape(b, 1)  # [B, 1] -> broadcast over D
                continuous_segments = mx.where(
                    mask_expanded[:, :, None] if continuous_segments.ndim == 3 else mask_expanded,
                    uncond_embed,
                    continuous_segments,
                )
            batch_indices = mx.arange(h.shape[0])
            assert starts is not None, (
                "starts must be provided when continuous_segments is not None"
            )
            starts_arr = cast(mx.array, starts)
            h = _scatter_1d(h, batch_indices, starts_arr, continuous_segments)

        h = self.backbone(h, input_pos=input_pos, mask=curr_backbone_mask)
        last_h = h[:, -1, :]  # [B, D]
        c0_logits = self.codebook0_head(last_h)  # [B, V]

        # CFG for codebook 0
        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            cond_logits = c0_logits[:actual_B]
            uncond_logits = c0_logits[actual_B:]
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            c0_sample = _sample_topk(guided_logits, topk, temperature)
            c0_sample = mx.concatenate([c0_sample, c0_sample], axis=0)
        else:
            c0_sample = _sample_topk(c0_logits, topk, temperature)

        c0_embed = self._embed_audio(0, c0_sample)  # [B, 1, D]

        # Decoder: autoregressively predict codebooks 1..7
        self.decoder.reset_caches()
        curr_h = mx.concatenate([last_h[:, None, :], c0_embed], axis=1)  # [B, 2, D]
        # Match legacy line 248: curr_h = curr_h.to(embeds.dtype)
        # Under torch.autocast, embeds.dtype == bfloat16, so decoder input matches model dtype.
        curr_h = curr_h.astype(embeds.dtype)
        curr_sample = c0_sample  # [B, 1]

        curr_pos = mx.broadcast_to(
            cast(mx.array, self._decoder_pos_init),
            (curr_h.shape[0], 2),
        )  # [B, 2]

        for i in range(1, self.config.audio_num_codebooks):
            curr_decoder_mask = _index_causal_mask(self._decoder_causal_mask, curr_pos)
            decoder_h = self.decoder(
                self.projection(curr_h), input_pos=curr_pos, mask=curr_decoder_mask
            )
            ci_logits = decoder_h[:, -1, :] @ self.audio_head[i - 1]  # [B, V]

            if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
                actual_B = b // 2
                cond_ci = ci_logits[:actual_B]
                uncond_ci = ci_logits[actual_B:]
                guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale
                ci_sample = _sample_topk(guided_ci, topk, temperature)
                ci_sample = mx.concatenate([ci_sample, ci_sample], axis=0)
            else:
                ci_sample = _sample_topk(ci_logits, topk, temperature)

            ci_embed = self._embed_audio(i, ci_sample)
            curr_h = ci_embed
            curr_sample = mx.concatenate([curr_sample, ci_sample], axis=1)
            curr_pos = mx.broadcast_to(
                self._decoder_pos_step[i - 1],
                (curr_h.shape[0], 1),
            )

        return curr_sample

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype: mx.Dtype = mx.float32) -> HeartMuLa:
        """Load a HeartMuLa model from a converted MLX checkpoint directory.

        Expects:
            ``path/config.json``  — model configuration
            ``path/weights.safetensors``  — MLX weights

        Args:
            dtype: Target dtype for all model parameters (e.g. ``mx.bfloat16``).
        """
        path = Path(path)
        config = HeartMuLaConfig.from_json(path / "config.json")
        model = cls(config)
        loaded: Any = mx.load(str(path / "weights.safetensors"))
        # MLX may return either a dict of arrays, or (dict, metadata).
        if isinstance(loaded, tuple) and len(loaded) >= 1:
            loaded = loaded[0]
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected weights.safetensors to load as a dict, got {type(loaded)}")
        weights = cast(dict[str, mx.array], loaded)

        # Cast weights to target dtype before loading
        if dtype != mx.float32:
            weights = {k: v.astype(dtype) for k, v in weights.items()}

        model.load_weights(list(weights.items()))
        model._dtype = dtype
        mx.eval(model.parameters())
        return model


# Utility: scatter for indexed assignment  h[batch_idx, pos] = val
def _scatter_1d(
    h: mx.array,
    batch_indices: mx.array,
    positions: mx.array,
    values: mx.array,
) -> mx.array:
    """Emulate ``h[batch_indices, positions] = values`` for a [B, S, D] tensor.

    *positions* may be a scalar or [B] array.  *values*: [B, D] or [D].
    """
    # For now, iterate per-batch (small B during inference)
    parts = []
    for i in range(h.shape[0]):
        pos = int(positions[i].item()) if positions.ndim > 0 else int(positions.item())
        row = h[i]  # [S, D]
        row_before = row[:pos]
        row_after = row[pos + 1 :]
        val = values[i : i + 1] if values.ndim >= 2 else values[None]
        new_row = mx.concatenate([row_before, val, row_after], axis=0)
        parts.append(new_row[None])
    return mx.concatenate(parts, axis=0)
