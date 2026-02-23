"""Copy exact example script from Heartlib repo.
Generates 4min of audio with pure argmax (no sampling noise).

Usage:
    python tests/exact_heartlib_pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "e2e_outputs"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    import heartlib_mlx.heartmula.modeling_heartmula as hmm
    from heartlib_mlx.pipelines.music_generation import HeartMuLaGenPipeline

    print("Loading MLX pipeline (bfloat16)...")
    pipe = HeartMuLaGenPipeline.from_pretrained(str(ROOT / "ckpt-mlx"), dtype=mx.bfloat16)
    # process lyrics
    lyrics = Path(ROOT / "assets" / "lyrics.txt")
    if os.path.isfile(lyrics):
        with open(lyrics, encoding="utf-8") as fp:
            lyrics = fp.read()

    assert isinstance(
        lyrics, str
    ), f"lyrics must be a string, but got {type(lyrics)}"

    lyrics = lyrics.lower()

    pipe(
        {
            "tags": "piano,happy",
            "lyrics": lyrics,
        },
        max_audio_length_ms=240_000,
        save_path=str(OUT_DIR / "final_exact_heartlib.wav"),
        temperature=1.0,
        topk=50,
        cfg_scale=1.5,
    )
    print(f"\nSaved to {OUT_DIR / 'final_exact_heartlib.wav'}")


if __name__ == "__main__":
    main()
