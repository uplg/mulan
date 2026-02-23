"""Music generation pipeline for HeartMuLa + HeartCodec (MLX port).

Mirrors the legacy ``HeartMuLaGenPipeline`` but adapted for MLX:
- No device management (unified memory on Apple Silicon)
- No dtype/autocast — MLX handles precision natively
- Uses soundfile for audio I/O instead of torchaudio
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import soundfile as sf
from tokenizers import Tokenizer
from tqdm import tqdm

from ..heartcodec.modeling_heartcodec import HeartCodec
from ..heartmula.modeling_heartmula import HeartMuLa


# Config


@dataclass
class HeartMuLaGenConfig:
    """Generation-time config (token IDs, etc.)."""

    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> HeartMuLaGenConfig:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)


# Pipeline


class HeartMuLaGenPipeline:
    """End-to-end music generation: text → tokens → audio."""

    def __init__(
        self,
        heartmula: HeartMuLa,
        heartcodec: HeartCodec,
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
    ) -> None:
        self.mula = heartmula
        self.codec = heartcodec
        self.text_tokenizer = text_tokenizer
        self.config = config
        self._dtype: mx.Dtype = heartmula._dtype

        # Matches legacy constants
        self._parallel_number = 8 + 1  # audio_num_codebooks + 1 (text)
        self._muq_dim = 512

    def preprocess(self, inputs: dict[str, Any], cfg_scale: float) -> dict[str, Any]:
        """Tokenize tags + lyrics, build prompt tensors."""

        tags: str = inputs["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp:
                tags = fp.read()
        assert isinstance(tags, str), f"tags must be a string, got {type(tags)}"

        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids: list[int] = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        ref_audio = inputs.get("ref_audio", None)
        if ref_audio is not None:
            raise NotImplementedError("ref_audio is not supported yet.")
        muq_embed = mx.zeros([self._muq_dim], dtype=self._dtype)
        muq_idx = len(tags_ids)

        lyrics: str = inputs["lyrics"]
        if os.path.isfile(lyrics):
            with open(lyrics, encoding="utf-8") as fp:
                lyrics = fp.read()
        assert isinstance(lyrics, str), f"lyrics must be a string, got {type(lyrics)}"
        lyrics = lyrics.lower()

        lyrics_ids: list[int] = self.text_tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.config.text_bos_id:
            lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        # Build the tokens array in pure MLX.
        # Layout: [prompt_len, parallel_number] with text in the last column.
        tags_arr = mx.array(tags_ids, dtype=mx.int32)
        lyrics_arr = mx.array(lyrics_ids, dtype=mx.int32)

        # Last column: [tags_ids..., 0, lyrics_ids...]
        text_col = mx.concatenate(
            [
                tags_arr,
                mx.zeros((1,), dtype=mx.int32),
                lyrics_arr,
            ]
        )  # [prompt_len]

        # Other columns are all zeros; stack zeros + text column.
        zeros_cols = mx.zeros((prompt_len, self._parallel_number - 1), dtype=mx.int32)
        tokens = mx.concatenate([zeros_cols, text_col[:, None]], axis=1)  # [prompt_len, parallel]

        # Mask: only text channel (last column) is active.
        mask_col = mx.ones((prompt_len, 1), dtype=mx.bool_)
        mask_zeros = mx.zeros((prompt_len, self._parallel_number - 1), dtype=mx.bool_)
        tokens_mask = mx.concatenate([mask_zeros, mask_col], axis=1)

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(arr: mx.array) -> mx.array:
            arr = arr[None]  # add batch dim
            if cfg_scale != 1.0:
                arr = mx.concatenate([arr, arr], axis=0)
            return arr

        pos = mx.arange(prompt_len, dtype=mx.int32)

        return {
            "tokens": _cfg_cat(tokens),
            "tokens_mask": _cfg_cat(tokens_mask),
            "muq_embed": _cfg_cat(muq_embed),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(pos),
        }

    def _forward(
        self,
        model_inputs: dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        progress_callback: Any | None = None,
        cancel_check: Any | None = None,
    ) -> dict[str, Any]:
        """Autoregressive frame generation.

        Args:
            progress_callback: optional ``fn(frame_idx, total_frames)`` called each step.
            cancel_check: optional ``fn() -> bool`` — if it returns True, generation stops early.
        """
        prompt_tokens: mx.array = model_inputs["tokens"]
        prompt_tokens_mask: mx.array = model_inputs["tokens_mask"]
        continuous_segment: mx.array = model_inputs["muq_embed"]
        starts: list[int] = model_inputs["muq_idx"]
        starts_arr = mx.array(starts, dtype=mx.int32)
        prompt_pos: mx.array = model_inputs["pos"]
        frames: list[mx.array] = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        self.mula.setup_caches(bs_size)

        # First frame: encode full prompt
        curr_token = self.mula.generate_frame(
            tokens=prompt_tokens,
            tokens_mask=prompt_tokens_mask,
            input_pos=prompt_pos,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
            continuous_segments=continuous_segment,
            starts=starts_arr,
        )
        mx.eval(curr_token)
        frames.append(curr_token[0:1])

        # Pre-compute the constant mask (text channel inactive).
        _pad_mask = mx.concatenate(
            [
                mx.ones((1, 1, self._parallel_number - 1), dtype=mx.bool_),
                mx.zeros((1, 1, 1), dtype=mx.bool_),
            ],
            axis=-1,
        )
        mx.eval(_pad_mask)

        def _pad_audio_token(token: mx.array) -> tuple[mx.array, mx.array]:
            """Pad a [B, num_codebooks] token into [B, 1, parallel_number] with mask.

            Uses mx.pad (no external tensor dependency, ~2 us graph node).
            """
            padded = mx.pad(token, [(0, 0), (0, 1)])[:, None, :]  # [B, 1, parallel]
            return padded, _pad_mask

        max_audio_frames = max_audio_length_ms // 80

        for i in tqdm(range(max_audio_frames), desc="Generating"):
            if cancel_check is not None and cancel_check():
                break

            curr_token_padded, curr_token_mask = _pad_audio_token(curr_token)

            curr_token = self.mula.generate_frame(
                tokens=curr_token_padded,
                tokens_mask=curr_token_mask,
                input_pos=prompt_pos[:, -1:] + i + 1,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=None,
                starts=None,
            )
            mx.eval(curr_token)

            if progress_callback is not None:
                progress_callback(i + 1, max_audio_frames)

            # Check EOS
            if mx.any(curr_token[0:1, :] >= self.config.audio_eos_id).item():
                break
            frames.append(curr_token[0:1])

        # Stack frames: list of [1, num_codebooks] → [T, num_codebooks]
        # Then transpose to [num_codebooks, T] to match legacy format
        stacked = mx.concatenate(frames, axis=0)  # [T, num_codebooks]
        frames_out = stacked.T  # [num_codebooks, T]

        return {"frames": frames_out}

    def postprocess(self, model_outputs: dict[str, Any], save_path: str) -> None:
        frames: mx.array = model_outputs["frames"]
        wav = self.codec.detokenize(frames)
        # wav: [num_bands, num_samples] → [num_samples, num_bands]
        wav = wav.T.astype(mx.float32)
        mx.eval(wav)
        # soundfile consumes mx.array directly via the Python buffer protocol (zero-copy).
        sf.write(save_path, wav, 48000, subtype="PCM_16")

    def __call__(self, inputs: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_kwargs)
        model_outputs = self._forward(model_inputs, **forward_kwargs)
        self.postprocess(model_outputs, **postprocess_kwargs)
        return model_outputs

    def _sanitize_parameters(
        self, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        preprocess_kwargs = {"cfg_scale": kwargs.get("cfg_scale", 1.5)}
        forward_kwargs = {
            "max_audio_length_ms": kwargs.get("max_audio_length_ms", 120_000),
            "temperature": kwargs.get("temperature", 1.0),
            "topk": kwargs.get("topk", 50),
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
            "progress_callback": kwargs.get("progress_callback"),
            "cancel_check": kwargs.get("cancel_check"),
        }
        postprocess_kwargs = {
            "save_path": kwargs.get("save_path", "output.wav"),
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str | Path,
        version: str = "3B",
        dtype: mx.Dtype = mx.float32,
    ) -> HeartMuLaGenPipeline:
        """Load the full pipeline from a checkpoint directory.

        Expected layout::

            pretrained_path/
              HeartMuLa-oss-{version}/
                config.json
                weights.safetensors
              HeartCodec-oss/
                config.json
                weights.safetensors
              tokenizer.json
              gen_config.json

        Args:
            dtype: Target dtype for model parameters (e.g. ``mx.bfloat16``).
        """
        pretrained_path = Path(pretrained_path)

        heartmula_path = pretrained_path / f"HeartMuLa-oss-{version}"
        heartcodec_path = pretrained_path / "HeartCodec-oss"
        tokenizer_path = pretrained_path / "tokenizer.json"
        gen_config_path = pretrained_path / "gen_config.json"

        if not heartmula_path.exists():
            raise FileNotFoundError(f"HeartMuLa checkpoint not found at {heartmula_path}")
        if not heartcodec_path.exists():
            raise FileNotFoundError(f"HeartCodec checkpoint not found at {heartcodec_path}")
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")
        if not gen_config_path.is_file():
            raise FileNotFoundError(f"gen_config.json not found at {gen_config_path}")

        dtype_name = {mx.float32: "float32", mx.float16: "float16", mx.bfloat16: "bfloat16"}.get(
            dtype, str(dtype)
        )
        print(f"Loading HeartMuLa ({dtype_name})...")
        mula = HeartMuLa.from_pretrained(heartmula_path, dtype=dtype)
        print("Loading HeartCodec (float32)...")
        codec = HeartCodec.from_pretrained(heartcodec_path)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

        print("Pipeline ready.")
        return cls(
            heartmula=mula,
            heartcodec=codec,
            text_tokenizer=tokenizer,
            config=gen_config,
        )
