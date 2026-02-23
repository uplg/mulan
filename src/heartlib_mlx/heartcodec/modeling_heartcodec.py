"""HeartCodec top-level model ported to MLX.

Combines ScalarModel (SQ codec) and FlowMatching into a single model
that can detokenize RVQ codes → audio waveform.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .configuration_heartcodec import HeartCodecConfig
from .models.flow_matching import FlowMatching
from .models.sq_codec import ScalarModel


@dataclass(frozen=True)
class AudioSegment:
    """A decoded audio segment yielded during streaming detokenization."""

    audio: mx.array  # (num_bands, num_new_samples) — only the NEW samples
    segment_index: int
    total_segments: int
    cumulative_samples: int  # total samples decoded so far (including this segment)


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

        num_segments = 0
        for sinx in range(0, codes.shape[-1] - hop_samples + 1, hop_samples):
            num_segments += 1
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

        # Remove first_latent_length prefix from first segment
        latent_list[0] = latent_list[0][:, first_latent_length:, :]
        min_samples_audio = int(duration * self.sample_rate)
        hop_samples_audio = min_samples_audio // 93 * 80
        ovlp_samples_audio = min_samples_audio - hop_samples_audio

        output: mx.array | None = None
        if ovlp_samples_audio > 0:
            _ov_win = mx.linspace(0, 1, ovlp_samples_audio)[None, :]
            _ov_win_inv = 1 - _ov_win
        else:
            _ov_win = None
            _ov_win_inv = None

        for i, latent in enumerate(latent_list):
            # Step 1: Reshape latent
            B, T_lat, F_lat = latent.shape
            latent = latent.reshape(B, T_lat, 2, F_lat // 2)
            latent = latent.transpose(0, 2, 1, 3)  # (B, 2, T, 128)
            latent = latent.reshape(B * 2, T_lat, F_lat // 2)  # (2B, T, 128)

            # Step 2: Decode with ScalarModel
            cur_output = self.scalar_model.decode(latent)  # (2B, L_out, num_bands)

            cur_output = cur_output.squeeze(-1)  # (2B, L_out) since num_bands=1
            cur_output = cur_output[:, :min_samples_audio]

            mx.eval(cur_output)  # Eval per segment to bound graph size for long audio

            # Step 3: Overlap-add
            if cur_output.ndim == 3:
                cur_output = cur_output[0]

            if output is None:
                output = cur_output
            else:
                if ovlp_samples_audio == 0:
                    output = mx.concatenate([output, cur_output], axis=-1)
                else:
                    assert _ov_win is not None and _ov_win_inv is not None
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

    def detokenize_streaming(
        self,
        codes: mx.array,
        duration: float = 29.76,
        num_steps: int = 10,
        disable_progress: bool = False,
        guidance_scale: float = 1.25,
    ) -> Iterator[AudioSegment]:
        """Stream-decode RVQ codes, yielding audio as each segment completes.

        Same algorithm as ``detokenize`` but yields an ``AudioSegment`` per
        codec segment so callers can convert / send audio incrementally.

        Args:
            codes: (num_quantizers, T_codes) int codes — no batch dim
            duration: codec segment duration in seconds (default 29.76)
            num_steps: Euler ODE steps per segment
            guidance_scale: classifier-free guidance strength

        Yields:
            ``AudioSegment`` with the *new* audio samples for each segment.
        """
        codes = codes[None]  # (1, Q, T)
        first_latent = mx.random.normal((codes.shape[0], int(duration * 25), 256))

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
        min_samples_audio = int(duration * self.sample_rate)
        hop_samples_audio = min_samples_audio // 93 * 80
        ovlp_samples_audio = min_samples_audio - hop_samples_audio

        if ovlp_samples_audio > 0:
            _ov_win = mx.linspace(0, 1, ovlp_samples_audio)[None, :]
            _ov_win_inv = 1 - _ov_win
        else:
            _ov_win = None
            _ov_win_inv = None

        # Count total segments
        total_segments = 0
        for _ in range(0, codes.shape[-1] - hop_samples + 1, hop_samples):
            total_segments += 1

        prev_latent: mx.array | None = None
        prev_tail: mx.array | None = None  # overlap tail from previous segment
        cumulative_samples = 0
        segment_idx = 0

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
            else:
                assert prev_latent is not None
                true_latent = prev_latent[:, -ovlp_frames:, :]
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

            prev_latent = latents

            # Remove prefix from first segment
            if sinx == 0:
                latents = latents[:, first_latent_length:, :]

            # Decode latent → audio
            B, T_lat, F_lat = latents.shape
            latents_r = latents.reshape(B, T_lat, 2, F_lat // 2)
            latents_r = latents_r.transpose(0, 2, 1, 3)
            latents_r = latents_r.reshape(B * 2, T_lat, F_lat // 2)

            cur_output = self.scalar_model.decode(latents_r)
            cur_output = cur_output.squeeze(-1)
            cur_output = cur_output[:, :min_samples_audio]
            mx.eval(cur_output)

            if cur_output.ndim == 3:
                cur_output = cur_output[0]

            # Overlap-add: yield only NEW samples (matching detokenize exactly).
            #
            # In the non-streaming version the accumulator grows as:
            #   seg 0: output = cur
            #   seg i: output = output[:-ovlp] + blend + cur[ovlp:]
            #
            # So the tail `ovlp_samples_audio` of each segment are NOT final
            # until blended with the next segment.  We withhold them and yield
            # only the "committed" samples.
            is_last = segment_idx == total_segments - 1

            if prev_tail is None:
                # First segment
                if ovlp_samples_audio > 0 and not is_last:
                    new_audio = cur_output[:, :-ovlp_samples_audio]
                else:
                    new_audio = cur_output
            elif ovlp_samples_audio == 0:
                new_audio = cur_output
            else:
                assert _ov_win is not None and _ov_win_inv is not None
                blended = prev_tail * _ov_win_inv + cur_output[:, :ovlp_samples_audio] * _ov_win
                if is_last:
                    # Last segment — emit blend + everything remaining
                    new_audio = mx.concatenate(
                        [blended, cur_output[:, ovlp_samples_audio:]], axis=-1
                    )
                else:
                    # Middle segment — emit blend + hop, withhold new tail
                    new_audio = mx.concatenate(
                        [blended, cur_output[:, ovlp_samples_audio:-ovlp_samples_audio]],
                        axis=-1,
                    )

            # Keep the overlap tail for the next iteration
            if ovlp_samples_audio > 0:
                prev_tail = cur_output[:, -ovlp_samples_audio:]

            mx.eval(new_audio)

            # Truncate to target_len (mirroring `output[:, :target_len]` in
            # the non-streaming path).  The padded codes produce extra audio
            # that would otherwise replay the beginning of the track.
            remaining = target_len - cumulative_samples
            if remaining <= 0:
                # Already emitted everything — skip this segment entirely.
                break
            if new_audio.shape[-1] > remaining:
                new_audio = new_audio[:, :remaining]

            cumulative_samples += new_audio.shape[-1]

            yield AudioSegment(
                audio=new_audio,
                segment_index=segment_idx,
                total_segments=total_segments,
                cumulative_samples=min(cumulative_samples, target_len),
            )
            segment_idx += 1

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
