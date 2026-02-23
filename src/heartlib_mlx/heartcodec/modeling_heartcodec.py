"""HeartCodec top-level model ported to MLX.

Combines ScalarModel (SQ codec) and FlowMatching into a single model
that can detokenize RVQ codes → audio waveform.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .configuration_heartcodec import HeartCodecConfig
from .models.flow_matching import FlowMatching
from .models.sq_codec import ScalarModel


class HeartCodec(nn.Module):
    """HeartCodec: audio codec with scalar VQ + flow-matching decoder.

    Primary inference method: ``detokenize(codes, ...)`` → waveform.
    """

    def __init__(self, config: HeartCodecConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate

        self.flow_matching = FlowMatching(
            dim=config.dim,
            codebook_size=config.codebook_size,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            use_cosine_sim=config.use_cosine_sim,
            codebook_dim=config.codebook_dim,
            num_quantizers=config.num_quantizers,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            norm_type=config.norm_type,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            num_layers_2=config.num_layers_2,
            out_channels=config.out_channels,
        )
        self.scalar_model = ScalarModel(
            num_bands=config.num_bands,
            sample_rate=config.sample_rate,
            causal=config.causal,
            num_samples=config.num_samples,
            downsample_factors=config.downsample_factors,
            downsample_kernel_sizes=config.downsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            latent_hidden_dim=config.latent_hidden_dim,
            default_kernel_size=config.default_kernel_size,
            delay_kernel_size=config.delay_kernel_size,
            init_channel=config.init_channel,
            res_kernel_size=config.res_kernel_size,
        )

    def detokenize(
        self,
        codes: mx.array,
        duration: float = 29.76,
        num_steps: int = 10,
        disable_progress: bool = False,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Convert RVQ codes to audio waveform.

        Args:
            codes: (num_quantizers, T_codes) int codes — note: no batch dim
            duration: segment duration in seconds
            num_steps: number of Euler ODE steps
            guidance_scale: classifier-free guidance strength

        Returns:
            (num_bands, num_samples) audio waveform
        """
        codes = codes[None]  # add batch dim → (1, Q, T_codes)

        first_latent = mx.random.normal((codes.shape[0], int(duration * 25), 256))  # (B, T, 256)

        first_latent_length = 0
        first_latent_codes_length = 0
        min_samples = int(duration * 12.5)
        hop_samples = min_samples // 93 * 80
        ovlp_samples = min_samples - hop_samples
        ovlp_frames = ovlp_samples * 2
        codes_len = codes.shape[-1]
        target_len = int((codes_len - first_latent_codes_length) / 12.5 * self.sample_rate)

        # Pad codes if too short
        if codes_len < min_samples:
            while codes.shape[-1] < min_samples:
                codes = mx.concatenate([codes, codes], axis=-1)
            codes = codes[:, :, :min_samples]
        codes_len = codes.shape[-1]

        if (codes_len - ovlp_frames) % hop_samples > 0:
            len_codes = (
                math.ceil((codes_len - ovlp_samples) / float(hop_samples)) * hop_samples
                + ovlp_samples
            )
            while codes.shape[-1] < len_codes:
                codes = mx.concatenate([codes, codes], axis=-1)
            codes = codes[:, :, :len_codes]

        latent_length = int(duration * 25)
        latent_list: list[mx.array] = []

        for sinx in range(0, codes.shape[-1] - hop_samples + 1, hop_samples):
            codes_input = [codes[:, :, sinx : sinx + min_samples]]

            if sinx == 0 or ovlp_frames == 0:
                incontext_length = first_latent_length
                latents = self.flow_matching.inference_codes(
                    codes_input,
                    first_latent,
                    latent_length,
                    incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    disable_progress=disable_progress,
                    scenario="other_seg",
                )
                latent_list.append(latents)
            else:
                true_latent = latent_list[-1][:, -ovlp_frames:, :]
                len_add = latent_length - true_latent.shape[1]
                incontext_length = true_latent.shape[1]
                true_latent = mx.concatenate(
                    [
                        true_latent,
                        mx.random.normal((true_latent.shape[0], len_add, true_latent.shape[-1])),
                    ],
                    axis=1,
                )
                latents = self.flow_matching.inference_codes(
                    codes_input,
                    true_latent,
                    latent_length,
                    incontext_length,
                    guidance_scale=guidance_scale,
                    num_steps=num_steps,
                    disable_progress=disable_progress,
                    scenario="other_seg",
                )
                latent_list.append(latents)
            # NOTE: inference_codes already evals its result internally
            # (via _solve_euler), so latents are materialized for the next
            # segment's overlap context.

        # Remove first_latent_length prefix from first segment
        latent_list[0] = latent_list[0][:, first_latent_length:, :]

        # Decode each latent segment to audio and overlap-add
        min_samples_audio = int(duration * self.sample_rate)
        hop_samples_audio = min_samples_audio // 93 * 80
        ovlp_samples_audio = min_samples_audio - hop_samples_audio

        output: mx.array | None = None
        # Precompute crossfade window (constant for all segments)
        if ovlp_samples_audio > 0:
            _ov_win = mx.linspace(0, 1, ovlp_samples_audio)[None, :]
            _ov_win_inv = 1 - _ov_win
        else:
            _ov_win = None
            _ov_win_inv = None

        for i, latent in enumerate(latent_list):
            # latent: (B, T, 256)
            # Reshape: (B, T, 2, 128) → permute → (B, 2, T, 128) → (2B, T, 128)
            B, T_lat, F_lat = latent.shape
            latent = latent.reshape(B, T_lat, 2, F_lat // 2)
            latent = latent.transpose(0, 2, 1, 3)  # (B, 2, T, 128)
            latent = latent.reshape(B * 2, T_lat, F_lat // 2)  # (2B, T, 128)

            # scalar_model.decode expects (N, L, C) — our MLX layout
            # Legacy does .transpose(1,2) because PyTorch uses (N,C,L)
            cur_output = self.scalar_model.decode(latent)  # (2B, L_out, num_bands)
            cur_output = cur_output.squeeze(-1)  # (2B, L_out) since num_bands=1

            cur_output = cur_output[:, :min_samples_audio]
            mx.eval(cur_output)  # Eval per segment to bound graph size for long audio

            # Remove batch dim if present
            if cur_output.ndim == 3:
                cur_output = cur_output[0]

            if output is None:
                output = cur_output
            else:
                if ovlp_samples_audio == 0:
                    output = mx.concatenate([output, cur_output], axis=-1)
                else:
                    # Linear crossfade with precomputed window
                    assert _ov_win is not None and _ov_win_inv is not None

                    # Blend overlap region
                    blended = (
                        output[:, -ovlp_samples_audio:] * _ov_win_inv
                        + cur_output[:, :ovlp_samples_audio] * _ov_win
                    )
                    output = mx.concatenate(
                        [
                            output[:, :-ovlp_samples_audio],
                            blended,
                            cur_output[:, ovlp_samples_audio:],
                        ],
                        axis=-1,
                    )

        assert output is not None
        output = output[:, :target_len]
        return output

    # -- loading helpers ----------------------------------------------------

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype: mx.Dtype = mx.float32) -> HeartCodec:
        """Load a HeartCodec model from a converted MLX checkpoint directory.

        Expects:
            ``path/config.json``  — model configuration
            ``path/weights.safetensors``  — MLX weights

        Args:
            dtype: Target dtype for all model parameters (e.g. ``mx.bfloat16``).
        """
        path = Path(path)
        config = HeartCodecConfig.from_json(path / "config.json")
        model = cls(config)
        weights = mx.load(str(path / "weights.safetensors"))

        # Cast weights to target dtype before loading
        if dtype != mx.float32:
            weights = {k: v.astype(dtype) for k, v in weights.items()}

        model.load_weights(list(weights.items()))
        mx.eval(model.parameters())
        return model
