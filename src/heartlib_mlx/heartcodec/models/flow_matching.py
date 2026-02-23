"""FlowMatching module ported to MLX.

Contains:
  - ResidualVQInference: Minimal re-implementation of vector_quantize_pytorch.ResidualVQ
    for inference only (codebook lookup + project_out).
  - FlowMatching: The complete flow-matching denoiser with RVQ conditioning.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .transformer import LlamaTransformer


# ---------------------------------------------------------------------------
# ResidualVQ  (inference-only)
# ---------------------------------------------------------------------------


class ResidualVQInference(nn.Module):
    """Minimal ResidualVQ that only supports ``get_output_from_indices``.

    Legacy ``vector_quantize_pytorch.ResidualVQ`` stores:
      - ``layers.{i}._codebook.embed``: shape ``(1, codebook_size, codebook_dim)``
      - ``project_in``: Linear(dim → codebook_dim)
      - ``project_out``: Linear(codebook_dim → dim)

    For inference we only need the codebook embeddings and ``project_out``.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int,
        num_quantizers: int,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers

        # Each layer stores a codebook of shape (1, codebook_size, codebook_dim).
        # We store them as a list of nn.Module-like containers so that
        # weight loading with the legacy key pattern works:
        #   vq_embed.layers.{i}._codebook.embed
        self.layers = [_CodebookLayer(codebook_size, codebook_dim) for _ in range(num_quantizers)]

        # project_out maps from codebook_dim back to dim
        requires_projection = codebook_dim != dim
        if requires_projection:
            self.project_in = nn.Linear(dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, dim)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

    def get_output_from_indices(self, indices: mx.array) -> mx.array:
        """Look up codebook vectors and sum across quantizers.

        Args:
            indices: (B, T, Q)  int codes, Q = num_quantizers

        Returns:
            (B, T, dim) quantized embeddings
        """
        B, T, Q = indices.shape

        # Gather codes for each quantizer and sum
        summed = mx.zeros((B, T, self.codebook_dim))
        for q in range(min(Q, self.num_quantizers)):
            # codebook: (1, codebook_size, codebook_dim) → squeeze to (codebook_size, codebook_dim)
            codebook = self.layers[q].codebook.embed.squeeze(0)  # (codebook_size, codebook_dim)
            idx = indices[:, :, q]  # (B, T)
            codes_q = codebook[idx]  # (B, T, codebook_dim)
            summed = summed + codes_q

        return self.project_out(summed)


class _CodebookLayer(nn.Module):
    """Container for a single VQ codebook.

    Legacy keys use ``layers.{i}._codebook.embed`` but MLX skips attributes
    starting with ``_``.  We store as ``codebook`` and the conversion script
    maps ``_codebook`` → ``codebook``.
    """

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = _Codebook(codebook_size, codebook_dim)


class _Codebook(nn.Module):
    """Holds the actual embedding table.

    Legacy stores: embed (1, codebook_size, codebook_dim),
    cluster_size, embed_avg, initted — we only need embed.
    """

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.embed = mx.zeros((1, codebook_size, codebook_dim))
        # These are stored in the checkpoint but not needed for inference.
        # We declare them so weight loading doesn't fail.
        self.cluster_size = mx.zeros((1, codebook_size))
        self.embed_avg = mx.zeros((1, codebook_size, codebook_dim))
        self.initted = mx.zeros((1,))


# ---------------------------------------------------------------------------
# FlowMatching
# ---------------------------------------------------------------------------


class FlowMatching(nn.Module):
    """Flow-matching denoiser with RVQ code conditioning.

    Inference path (``inference_codes``):
      1. Look up RVQ codes → quantized feature embedding
      2. Project with ``cond_feature_emb``
      3. Nearest-neighbour upsample ×2 in time
      4. Euler ODE solve with classifier-free guidance
    """

    def __init__(
        self,
        # RVQ parameters
        dim: int = 512,
        codebook_size: int = 8192,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        # Transformer (DiT) parameters
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
        # Legacy compat (unused in inference but present in config)
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
    ):
        super().__init__()

        self.vq_embed = ResidualVQInference(
            dim=dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            num_quantizers=num_quantizers,
        )
        self.cond_feature_emb = nn.Linear(dim, dim)
        self.zero_cond_embedding1 = mx.zeros((dim,))

        self.estimator = LlamaTransformer(
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            norm_type=norm_type,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            num_layers_2=num_layers_2,
            out_channels=out_channels,
        )

        self.latent_dim = out_channels

    def inference_codes(
        self,
        codes: list[mx.array],
        true_latents: mx.array,
        latent_length: int,
        incontext_length: int,
        guidance_scale: float = 2.0,
        num_steps: int = 20,
        disable_progress: bool = True,
        scenario: str = "start_seg",
    ) -> mx.array:
        """Generate latents from RVQ codes via ODE flow matching.

        Args:
            codes: list where codes[0] has shape (B, num_quantizers, T_codes)
            true_latents: (B, T_latent, latent_dim) — context latents
            latent_length: number of valid latent frames
            incontext_length: number of in-context frames to preserve
            guidance_scale: classifier-free guidance strength
            num_steps: number of Euler steps
            scenario: "start_seg" or "other_seg"

        Returns:
            (B, T_latent, latent_dim) generated latents
        """
        codes_bestrq_emb = codes[0]  # (B, Q, T_codes)
        B = codes_bestrq_emb.shape[0]

        # Codebook lookup: need (B, T, Q)
        quantized_feature_emb = self.vq_embed.get_output_from_indices(
            mx.transpose(codes_bestrq_emb, axes=(0, 2, 1))  # (B, T_codes, Q)
        )
        quantized_feature_emb = self.cond_feature_emb(quantized_feature_emb)  # (B, T, 512)

        # Nearest-neighbour upsample ×2 along time: (B, T, C) → (B, 2T, C)
        quantized_feature_emb = mx.repeat(quantized_feature_emb, repeats=2, axis=1)

        num_frames = quantized_feature_emb.shape[1]

        # Random noise
        latents = mx.random.normal((B, num_frames, self.latent_dim))

        # Build masks using vectorised comparisons (no .at[].add() allocations)
        frame_idx = mx.arange(num_frames)[None, :]  # [1, T], broadcasts over B
        latent_masks = mx.where(frame_idx < latent_length, 2, 0).astype(mx.int32)

        if scenario == "other_seg" and incontext_length > 0:
            latent_masks = mx.where(
                frame_idx < incontext_length,
                mx.array(1, dtype=mx.int32),
                latent_masks,
            )

        # Apply conditioning mask
        mask_gt_half = (latent_masks > 0)[..., None]  # (B, T, 1)
        mask_lt_half = ~mask_gt_half
        quantized_feature_emb = (
            mask_gt_half * quantized_feature_emb
            + mask_lt_half * self.zero_cond_embedding1[None, None, :]
        )

        # In-context latents
        mask_incontext = ((latent_masks > 0) & (latent_masks < 2))[..., None]  # True where mask==1
        incontext_latents = true_latents * mask_incontext.astype(true_latents.dtype)
        incontext_length_actual = int(
            ((latent_masks > 0) & (latent_masks < 2)).sum(axis=-1)[0].item()
        )

        additional_model_input = quantized_feature_emb
        temperature = 1.0
        t_span = mx.linspace(0.0, 1.0, num_steps + 1)

        latents = self._solve_euler(
            latents * temperature,
            incontext_latents,
            incontext_length_actual,
            t_span,
            additional_model_input,
            guidance_scale,
        )

        # Restore in-context latents
        if incontext_length_actual > 0:
            latents = mx.concatenate(
                [
                    incontext_latents[:, :incontext_length_actual, :],
                    latents[:, incontext_length_actual:, :],
                ],
                axis=1,
            )

        return latents

    def _solve_euler(
        self,
        x: mx.array,
        incontext_x: mx.array,
        incontext_length: int,
        t_span: mx.array,
        mu: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        """Fixed Euler ODE solver for flow matching."""
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        noise = x  # keep reference to initial noise

        # Precompute CFG constants used every step
        if guidance_scale > 1.0:
            mu_zeros = mx.zeros_like(mu)
        else:
            mu_zeros = mu  # unused placeholder, never read in else branch

        for step in range(1, t_span.shape[0]):
            # Blend in-context region
            if incontext_length > 0:
                blend = 1 - (1 - 1e-6) * t
                x = mx.concatenate(
                    [
                        blend * noise[:, :incontext_length, :]
                        + t * incontext_x[:, :incontext_length, :],
                        x[:, incontext_length:, :],
                    ],
                    axis=1,
                )

            if guidance_scale > 1.0:
                # Classifier-free guidance: run conditional + unconditional
                x_double = mx.concatenate([x, x], axis=0)
                ic_double = mx.concatenate([incontext_x, incontext_x], axis=0)
                mu_double = mx.concatenate([mu_zeros, mu], axis=0)
                combined = mx.concatenate([x_double, ic_double, mu_double], axis=2)
                t_input = mx.broadcast_to(t.reshape(1), (2,))

                dphi_dt = self.estimator(combined, timestep=t_input)
                dphi_uncond, dphi_cond = mx.split(dphi_dt, indices_or_sections=2, axis=0)
                dphi_dt = dphi_uncond + guidance_scale * (dphi_cond - dphi_uncond)
            else:
                combined = mx.concatenate([x, incontext_x, mu], axis=2)
                t_input = t.reshape(1)
                dphi_dt = self.estimator(combined, timestep=t_input)

            x = x + dt * dphi_dt
            t = t + dt

            if step < t_span.shape[0] - 1:
                dt = t_span[step + 1] - t

        # Evaluate once after the full ODE solve instead of per-step.
        # 10 Euler steps through the transformer build a manageable graph.
        mx.eval(x)
        return x
