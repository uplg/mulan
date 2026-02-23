"""HeartCodec configuration - pure Python dataclass (no HuggingFace dependency)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class HeartCodecConfig:
    """Configuration for the HeartCodec audio codec model."""

    model_type: str = "heartcodec"

    # RVQ config
    dim: int = 512
    codebook_size: int = 8192
    decay: float = 0.9
    commitment_weight: float = 1.0
    threshold_ema_dead_code: int = 2
    use_cosine_sim: bool = False
    codebook_dim: int = 32
    num_quantizers: int = 8

    # Diffusion transformer config
    attention_head_dim: int = 64
    in_channels: int = 1024
    norm_type: str = "ada_norm_single"
    num_attention_heads: int = 24
    num_layers: int = 24
    num_layers_2: int = 6
    out_channels: int = 256

    # SQ codec config
    num_bands: int = 1
    sample_rate: int = 48000
    causal: bool = True
    num_samples: int = 2
    downsample_factors: list[int] = field(default_factory=lambda: [3, 4, 4, 4, 5])
    downsample_kernel_sizes: list[int] = field(default_factory=lambda: [6, 8, 8, 8, 10])
    upsample_factors: list[int] = field(default_factory=lambda: [5, 4, 4, 4, 3])
    upsample_kernel_sizes: list[int] = field(default_factory=lambda: [10, 8, 8, 8, 6])
    latent_hidden_dim: int = 128
    default_kernel_size: int = 7
    delay_kernel_size: int = 5
    init_channel: int = 64
    res_kernel_size: int = 7

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> HeartCodecConfig:
        """Load config from a JSON file (e.g. ``config.json`` in checkpoint)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Filter out keys that are not fields of this dataclass
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)
