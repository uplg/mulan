"""HeartMuLa configuration - pure Python dataclass (no HuggingFace dependency)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class HeartMuLaConfig:
    """Configuration for the HeartMuLa music generation model."""

    model_type: str = "heartmula"

    backbone_flavor: str = "llama-3B"
    decoder_flavor: str = "llama-300M"
    text_vocab_size: int = 128_256
    audio_vocab_size: int = 8197
    audio_num_codebooks: int = 8
    muq_dim: int = 512

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> HeartMuLaConfig:
        """Load config from a JSON file (e.g. ``config.json`` in checkpoint)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

@dataclass(frozen=True)
class LlamaFlavorSpec:
    """Immutable spec for a Llama-family transformer variant."""

    num_layers: int
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    max_seq_len: int
    intermediate_dim: int
    norm_eps: float = 1e-5
    rope_base: int = 500_000
    scale_factor: int = 32

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


FLAVORS: dict[str, LlamaFlavorSpec] = {
    "llama-3B": LlamaFlavorSpec(
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=8192,
        intermediate_dim=8192,
    ),
    "llama-300M": LlamaFlavorSpec(
        num_layers=3,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
    ),
    "llama-7B": LlamaFlavorSpec(
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        max_seq_len=8192,
        intermediate_dim=14336,
    ),
    "llama-400M": LlamaFlavorSpec(
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        embed_dim=3072,
        max_seq_len=2048,
        intermediate_dim=8192,
    ),
}
