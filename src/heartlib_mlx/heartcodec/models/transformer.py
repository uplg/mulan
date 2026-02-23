"""LlamaTransformer for flow-matching, ported to MLX.

This is the DiT-style transformer used inside HeartCodec's FlowMatching module.
It is *not* the autoregressive Llama used for HeartMuLa — that is a separate model.

Key differences from standard Llama:
  - Two-stage architecture (transformer_blocks → transformer_blocks_2)
  - PixArt-style AdaLayerNorm (ada_norm_single) for timestep conditioning
  - ProjectLayer using Conv1d for in/out projections
  - No KV-cache (this runs as a denoiser, not autoregressive)
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# RMSNorm  (MLX ships nn.RMSNorm — we use it directly)

# nn.RMSNorm(dims, eps) — compatible with legacy RMSNorm.


# Rotary Embedding


def _build_rope_sin_cos(seq_len: int, dim: int, base: float = 10000.0) -> tuple[mx.array, mx.array]:
    """Build sin/cos tables for rotary embedding.

    Returns sin, cos each of shape (seq_len, dim//2).
    """
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))  # (dim//2,)
    t = mx.arange(seq_len, dtype=mx.float32)  # (seq_len,)
    freqs = mx.outer(t, inv_freq)  # (seq_len, dim//2)
    return mx.sin(freqs), mx.cos(freqs)


def _apply_rope_interleaved(
    x: mx.array,
    sin: mx.array,
    cos: mx.array,
    rope_dim: int,
) -> mx.array:
    """Apply interleaved RoPE to x.

    x: (B, H, T, D)
    sin, cos: (T, rope_dim//2)

    Legacy code reshapes head to (..., rope_dim//2, 2), takes pairs (x1, x2),
    and computes [x1*cos - x2*sin, x1*sin + x2*cos].
    """
    head = x[..., :rope_dim]  # (B, H, T, rope_dim)
    tail = x[..., rope_dim:]  # (B, H, T, D - rope_dim)

    B, H, T, _ = head.shape
    half = rope_dim // 2
    head = head.reshape(B, H, T, half, 2)

    # sin, cos → (1, 1, T, half, 1)
    sin_ = sin[:T].reshape(1, 1, T, half, 1)
    cos_ = cos[:T].reshape(1, 1, T, half, 1)

    x1 = head[..., 0:1]  # (B, H, T, half, 1)
    x2 = head[..., 1:2]

    rot = mx.concatenate(
        [x1 * cos_ - x2 * sin_, x1 * sin_ + x2 * cos_], axis=-1
    )  # (B, H, T, half, 2)
    rot = rot.reshape(B, H, T, rope_dim)

    return mx.concatenate([rot, tail], axis=-1)


# LlamaAttention


class LlamaAttention(nn.Module):
    """Multi-head attention with RoPE (interleaved variant).

    Supports optional cross-attention when ``cross_attention_dim`` is set.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        bias: bool = False,
        dropout: float = 0.0,
        rope_dim: int | None = None,
        cross_attention_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim
        self.rope_dim = rope_dim if rope_dim is not None else head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.inner_dim, bias=bias)
        k_in = dim if cross_attention_dim is None else cross_attention_dim
        self.k_proj = nn.Linear(k_in, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(k_in, self.inner_dim, bias=bias)
        self.o_proj = nn.Linear(self.inner_dim, dim, bias=bias)

        # We'll lazily build sin/cos when we know seq_len
        self._rope_cache_len = 0
        self._sin: mx.array | None = None
        self._cos: mx.array | None = None

    def _ensure_rope(self, seq_len: int) -> tuple[mx.array, mx.array]:
        if seq_len > self._rope_cache_len:
            self._sin, self._cos = _build_rope_sin_cos(seq_len, self.rope_dim)
            self._rope_cache_len = seq_len
        assert self._sin is not None and self._cos is not None
        return self._sin, self._cos

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        if encoder_hidden_states is None:
            kv_source = x
            Tk = T
        else:
            kv_source = encoder_hidden_states
            Tk = kv_source.shape[1]

        k = self.k_proj(kv_source).reshape(B, Tk, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(kv_source).reshape(B, Tk, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        # RoPE
        rope_dim = min(self.rope_dim, self.head_dim)
        sin, cos = self._ensure_rope(max(T, Tk))
        q = _apply_rope_interleaved(q, sin, cos, rope_dim)
        k = _apply_rope_interleaved(k, sin, cos, rope_dim)

        # Scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, T, Tk)

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = mx.softmax(scores, axis=-1)
        out = weights @ v  # (B, H, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.inner_dim)
        return self.o_proj(out)


# LlamaMLP


class LlamaMLP(nn.Module):
    """SiLU-gated MLP matching legacy Llama style."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 256,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


# LlamaTransformerBlock


class LlamaTransformerBlock(nn.Module):
    """Transformer block with optional PixArt-style AdaLayerNorm single."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        mlp_multiple_of: int = 256,
        dropout: float = 0.0,
        attention_bias: bool = False,
        cross_attention_dim: int | None = None,
        use_ada_layer_norm_single: bool = False,
    ):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim, eps=1e-6)
        self.attn = LlamaAttention(
            dim,
            n_heads,
            head_dim,
            bias=attention_bias,
            dropout=dropout,
            rope_dim=head_dim,
            cross_attention_dim=None,
        )

        self.cross_attn: LlamaAttention | None = None
        if cross_attention_dim is not None:
            self.cross_attn_norm = nn.RMSNorm(dim, eps=1e-6)
            self.cross_attn = LlamaAttention(
                dim,
                n_heads,
                head_dim,
                bias=attention_bias,
                dropout=dropout,
                rope_dim=head_dim,
                cross_attention_dim=cross_attention_dim,
            )

        self.mlp_norm = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = LlamaMLP(dim, multiple_of=mlp_multiple_of)
        self.use_ada_layer_norm_single = use_ada_layer_norm_single
        if self.use_ada_layer_norm_single:
            self.scale_shift_table = mx.random.normal((6, dim)) / (dim**0.5)

    def __call__(
        self,
        x: mx.array,
        encoder_hidden_states: mx.array | None = None,
        attention_mask: mx.array | None = None,
        timestep: mx.array | None = None,
    ) -> mx.array:
        if self.use_ada_layer_norm_single:
            B = x.shape[0]
            # timestep: (B, 6*D)
            chunks = self.scale_shift_table[None] + timestep.reshape(B, 6, -1)  # (B, 6, D)
            shift_msa = chunks[:, 0:1, :]
            scale_msa = chunks[:, 1:2, :]
            gate_msa = chunks[:, 2:3, :]
            shift_mlp = chunks[:, 3:4, :]
            scale_mlp = chunks[:, 4:5, :]
            gate_mlp = chunks[:, 5:6, :]

            # Self-attention with modulation + gating
            norm_h = self.attn_norm(x)
            norm_h = norm_h * (1 + scale_msa) + shift_msa
            h = self.attn(norm_h, attention_mask=attention_mask)
            h = gate_msa * h
            x = x + h

            # MLP with modulation + gating
            norm_h = self.mlp_norm(x)
            norm_h = norm_h * (1 + scale_mlp) + shift_mlp
            h = self.mlp(norm_h)
            h = gate_mlp * h
            x = x + h
            return x
        else:
            h = self.attn(self.attn_norm(x), attention_mask=attention_mask)
            x = x + h
            h = self.mlp(self.mlp_norm(x))
            x = x + h
            return x


# ProjectLayer


class ProjectLayer(nn.Module):
    """Conv1d(kernel=k) → scale → Linear.

    Legacy uses PyTorch Conv1d (N,C,L) layout. We use MLX Conv1d (N,L,C).
    """

    def __init__(
        self,
        hidden_size: int,
        filter_size: int,
        kernel_size: int = 1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.ffn_1 = nn.Conv1d(hidden_size, filter_size, kernel_size, padding=kernel_size // 2)
        self.ffn_2 = nn.Linear(filter_size, filter_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)  — MLX Conv1d already expects (N, L, C)
        x = self.ffn_1(x)
        x = x * (self.kernel_size**-0.5)
        x = self.ffn_2(x)
        return x


# Timestep embeddings  (PixArt-style)


class TimestepEmbedding(nn.Module):
    """Linear → SiLU → Linear."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = nn.silu(x)
        x = self.linear_2(x)
        return x


class PixArtAlphaCombinedFlowEmbeddings(nn.Module):
    """Sinusoidal timestep → MLP embedding (PixArt flow-matching variant)."""

    def __init__(self, embedding_dim: int, size_emb_dim: int):
        super().__init__()
        self.flow_t_size = 512
        self.timestep_embedder = TimestepEmbedding(
            in_channels=self.flow_t_size, time_embed_dim=embedding_dim
        )

    @staticmethod
    def _timestep_embedding(
        timesteps: mx.array, dim: int, max_period: float = 10000.0, scale: float = 1000.0
    ) -> mx.array:
        """Sinusoidal embedding matching legacy ``timestep_embedding``."""
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)  # (half,)
        args = timesteps[:, None] * freqs[None] * scale  # (B, half)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)  # (B, dim)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, timestep: mx.array, hidden_dtype: mx.Dtype | None = None) -> mx.array:
        proj = self._timestep_embedding(timestep, self.flow_t_size)
        if hidden_dtype is not None:
            proj = proj.astype(hidden_dtype)
        return self.timestep_embedder(proj)


class AdaLayerNormSingleFlow(nn.Module):
    """PixArt-style adaptive layer norm for flow-matching timestep conditioning."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.emb = PixArtAlphaCombinedFlowEmbeddings(embedding_dim, size_emb_dim=embedding_dim // 3)
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def __call__(
        self,
        timestep: mx.array,
        hidden_dtype: mx.Dtype | None = None,
    ) -> tuple[mx.array, mx.array]:
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(nn.silu(embedded_timestep)), embedded_timestep


# LlamaTransformer  (two-stage DiT)


class LlamaTransformer(nn.Module):
    """Two-stage DiT transformer for flow-matching denoiser.

    Stage 1: ``transformer_blocks``   (inner_dim = heads * head_dim)
    Stage 2: ``transformer_blocks_2`` (inner_dim_2 = 2 * inner_dim)

    Connected via ``connection_proj`` that takes cat(hidden_states, stage1_out).
    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        out_channels: int,
        num_layers: int = 12,
        num_layers_2: int = 2,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        norm_type: str = "layer_norm",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        inner_dim_2 = inner_dim * 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_dim = inner_dim
        self.inner_dim_2 = inner_dim_2

        self.proj_in = ProjectLayer(in_channels, inner_dim, kernel_size=3)

        use_ada_single = norm_type == "ada_norm_single"

        self.transformer_blocks = [
            LlamaTransformerBlock(
                dim=inner_dim,
                n_heads=num_attention_heads,
                head_dim=attention_head_dim,
                dropout=dropout,
                attention_bias=False,
                cross_attention_dim=cross_attention_dim,
                use_ada_layer_norm_single=use_ada_single,
            )
            for _ in range(num_layers)
        ]

        self.transformer_blocks_2 = [
            LlamaTransformerBlock(
                dim=inner_dim_2,
                n_heads=num_attention_heads,
                head_dim=attention_head_dim * 2,
                dropout=dropout,
                attention_bias=False,
                cross_attention_dim=cross_attention_dim,
                use_ada_layer_norm_single=use_ada_single,
            )
            for _ in range(num_layers_2)
        ]

        self.connection_proj = ProjectLayer(in_channels + inner_dim, inner_dim_2, kernel_size=3)

        # LayerNorm without elementwise_affine → just use a function
        self._norm_out_eps = 1e-6
        self._norm_out_2_eps = 1e-6

        self.scale_shift_table = mx.random.normal((2, inner_dim)) / (inner_dim**0.5)
        self.scale_shift_table_2 = mx.random.normal((2, inner_dim_2)) / (inner_dim_2**0.5)

        self.proj_out = ProjectLayer(inner_dim_2, out_channels, kernel_size=3)

        self.adaln_single = AdaLayerNormSingleFlow(inner_dim)
        self.adaln_single_2 = AdaLayerNormSingleFlow(inner_dim_2)

    @staticmethod
    def _layer_norm(x: mx.array, eps: float) -> mx.array:
        """LayerNorm without learnable parameters (elementwise_affine=False)."""
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + eps)

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array | None = None,
    ) -> mx.array:
        s = self.proj_in(hidden_states)

        embedded_timestep: mx.array | None = None
        timestep_mod: mx.array | None = None
        if timestep is not None:
            timestep_mod, embedded_timestep = self.adaln_single(timestep, hidden_dtype=s.dtype)

        for blk in self.transformer_blocks:
            s = blk(s, timestep=timestep_mod)

        if embedded_timestep is None:
            embedded_timestep = mx.zeros((s.shape[0], s.shape[-1]), dtype=s.dtype)

        # Post-norm with AdaLN modulation
        shift, scale = mx.split(
            self.scale_shift_table[None] + embedded_timestep[:, None, :],
            indices_or_sections=2,
            axis=1,
        )  # each (B, 1, inner_dim)
        s = self._layer_norm(s, self._norm_out_eps)
        s = s * (1 + scale) + shift

        x = mx.concatenate([hidden_states, s], axis=-1)
        x = self.connection_proj(x)

        embedded_timestep_2: mx.array | None = None
        timestep_mod_2: mx.array | None = None
        if timestep is not None:
            timestep_mod_2, embedded_timestep_2 = self.adaln_single_2(
                timestep, hidden_dtype=x.dtype
            )

        for blk in self.transformer_blocks_2:
            x = blk(x, timestep=timestep_mod_2)

        if embedded_timestep_2 is None:
            embedded_timestep_2 = mx.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype)

        shift_2, scale_2 = mx.split(
            self.scale_shift_table_2[None] + embedded_timestep_2[:, None, :],
            indices_or_sections=2,
            axis=1,
        )
        x = self._layer_norm(x, self._norm_out_2_eps)
        x = x * (1 + scale_2) + shift_2

        return self.proj_out(x)
