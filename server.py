#!/usr/bin/env python3
"""
FastAPI server for MuLa(n)

Usage:
    uv run server.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, FastAPI, HTTPException, Path as PathParam, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Paths
ROOT_DIR = Path(__file__).parent
CKPT_MLX = ROOT_DIR / "ckpt-mlx"
OUTPUT_DIR = ROOT_DIR / "outputs"
STATIC_DIR = ROOT_DIR / "web-ui" / "dist"

# Add src to path so heartlib_mlx can be imported
_src_path = ROOT_DIR / "src"
if _src_path.exists():
    sys.path.insert(0, str(_src_path))

OUTPUT_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("heartmula")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Pydantic models
class GenerateRequest(BaseModel):
    """Request body for music generation."""

    model_config = {
        "json_schema_extra": {
            "example": {
                "lyrics": "[verse]\nHello world...",
                "styles": "electronic,ambient",
                "duration": 30,
                "title": "My Song",
            }
        }
    }

    lyrics: str = ""
    styles: str = Field(default="electronic,ambient,instrumental", min_length=1)
    duration: int = Field(default=10, ge=5, le=240, description="Duration in seconds")
    cfg_scale: float = Field(default=1.5, ge=0.0, le=10.0)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    topk: int = Field(default=50, ge=1, le=500)
    title: str = ""
    language: str = ""


class GenerateResponse(BaseModel):
    filename: str
    frames: int
    duration: float
    time: float


class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    is_generating: bool


class ProgressResponse(BaseModel):
    progress: float
    message: str
    is_generating: bool
    current_request: dict | None = None
    started_at: float | None = None


class SongInfo(BaseModel):
    model_config = {"from_attributes": True}

    filename: str
    title: str
    styles: str
    lyrics: str
    duration: float
    created_at: float
    album_art: str | None = None


class SongsResponse(BaseModel):
    songs: list[SongInfo]


class SurpriseLyricsResponse(BaseModel):
    title: str
    styles: str
    lyrics: str


class CancelResponse(BaseModel):
    status: str
    message: str


# Generation state (typed dataclass instead of bare dict)
@dataclass
class GenerationState:
    """Mutable state for the current generation process."""

    progress: float = 0.0
    message: str = "Ready"
    is_generating: bool = False
    cancelled: bool = False
    current_request: dict | None = field(default=None)
    started_at: float | None = None

    def reset(self) -> None:
        """Reset after generation completes or fails."""
        self.is_generating = False
        self.current_request = None
        self.started_at = None


gen_state = GenerationState()

SURPRISE_LYRICS = [
    {
        "title": "Digital Dreams",
        "styles": "electronic,synthwave,dreamy",
        "lyrics": """[verse]
Neon lights are calling me tonight
Through the digital rain I find my way
Binary stars align in perfect sight
In this virtual world I want to stay

[chorus]
Digital dreams, electric streams
Nothing is ever what it seems
Lost in the code, finding my soul
In this machine I'm finally whole""",
    },
    {
        "title": "City Lights",
        "styles": "jazz,soul,smooth",
        "lyrics": """[verse]
Walking through these crowded streets at night
A million stories passing by
Every window holds a different light
Every corner hears a different cry

[chorus]
City lights, they shine so bright
Hiding all our fears from sight
In the noise I find my peace
In the chaos, sweet release""",
    },
]


# Model loading
def _load_pipeline():
    """Load the HeartMuLaGenPipeline (blocking, called once at startup)."""
    import mlx.core as mx
    from heartlib_mlx.pipelines import HeartMuLaGenPipeline

    logger.info("Loading pipeline from %s ...", CKPT_MLX)
    pipe = HeartMuLaGenPipeline.from_pretrained(CKPT_MLX, dtype=mx.bfloat16)
    logger.info("Pipeline loaded!")
    return pipe


# Metadata helpers
def _save_song_metadata(
    filename: str, title: str, styles: str, lyrics: str, duration: float
) -> None:
    """Save metadata JSON alongside the audio file (atomic write)."""
    meta_path = OUTPUT_DIR / f"{Path(filename).stem}.json"
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    metadata = {
        "filename": filename,
        "title": title or filename,
        "styles": styles,
        "lyrics": lyrics,
        "duration": duration,
        "created_at": time.time(),
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=True, separators=(",", ":"))
    os.replace(tmp_path, meta_path)


_HAS_FFMPEG: bool | None = None


def _check_ffmpeg() -> bool:
    """Check whether ffmpeg is available on PATH (cached)."""
    global _HAS_FFMPEG  # noqa: PLW0603
    if _HAS_FFMPEG is None:
        _HAS_FFMPEG = shutil.which("ffmpeg") is not None
        if not _HAS_FFMPEG:
            logger.warning("ffmpeg not found — MP3 conversion disabled, serving WAV files")
    return _HAS_FFMPEG


def _convert_wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> bool:
    """Convert a WAV file to MP3 using ffmpeg. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                bitrate,
                "-q:a",
                "2",
                str(mp3_path),
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            logger.error("ffmpeg failed: %s", result.stderr.decode(errors="replace")[:500])
            return False
        return True
    except (OSError, subprocess.TimeoutExpired) as err:
        logger.error("ffmpeg conversion error: %s", err)
        return False


def _load_song_metadata(audio_path: Path) -> SongInfo:
    """Load metadata for an audio file, or create basic info from file."""
    meta_path = audio_path.with_suffix(".json")

    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            return SongInfo.model_validate(data)
        except (json.JSONDecodeError, ValueError) as err:
            logger.warning("Corrupt metadata for %s, using fallback: %s", audio_path.name, err)

    return SongInfo(
        filename=audio_path.name,
        title=audio_path.stem,
        styles="",
        lyrics="",
        duration=0.0,
        created_at=audio_path.stat().st_mtime,
    )


# Generation (sync, runs in executor)
def _generate_music_sync(
    pipeline,
    lyrics: str,
    styles: str,
    duration: int,
    cfg_scale: float,
    temperature: float,
    topk: int,
    title: str,
    language: str = "",
) -> tuple[str, int, float, float]:
    """Synchronous music generation using the pipeline."""
    gen_state.cancelled = False
    gen_state.progress = 0.05
    gen_state.message = "Preparing..."

    normalized_styles = ",".join(t.strip() for t in styles.split(",") if t.strip())
    tags = f"{language.strip().lower()},{normalized_styles}" if language else normalized_styles

    start_time = time.time()

    def on_progress(frame_idx: int, total_frames: int) -> None:
        progress = 0.1 + 0.75 * frame_idx / total_frames
        elapsed = time.time() - start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_idx) / fps if fps > 0 else 0
        gen_state.progress = progress
        gen_state.message = (
            f"Generating ({frame_idx + 1}/{total_frames} frames, {fps:.1f} f/s, ETA {eta:.0f}s)"
        )

    def is_cancelled() -> bool:
        return gen_state.cancelled

    filename = f"generation_{uuid.uuid4().hex[:8]}.wav"
    output_path = OUTPUT_DIR / filename

    gen_state.progress = 0.1
    gen_state.message = "Generating..."

    # Decompose the pipeline to update progress between stages.
    preprocess_kw = {"cfg_scale": cfg_scale}
    model_inputs = pipeline.preprocess({"tags": tags, "lyrics": lyrics}, **preprocess_kw)

    model_outputs = pipeline._forward(
        model_inputs,
        max_audio_length_ms=duration * 1000,
        temperature=temperature,
        topk=topk,
        cfg_scale=cfg_scale,
        progress_callback=on_progress,
        cancel_check=is_cancelled,
    )

    gen_state.progress = 0.90
    gen_state.message = "Decoding audio..."

    pipeline.postprocess(model_outputs, save_path=str(output_path))

    gen_state.progress = 0.95
    gen_state.message = "Saving file..."

    frames_arr = model_outputs["frames"]
    num_frames = int(frames_arr.shape[1])
    actual_duration = num_frames / 12.5  # frame_rate = 12.5

    # Convert WAV → MP3 for faster delivery (~10x smaller)
    served_filename = filename
    if _check_ffmpeg():
        mp3_path = output_path.with_suffix(".mp3")
        if _convert_wav_to_mp3(output_path, mp3_path):
            served_filename = mp3_path.name
            logger.info(
                "Converted to MP3: %.1f MB → %.1f MB",
                output_path.stat().st_size / 1e6,
                mp3_path.stat().st_size / 1e6,
            )

    _save_song_metadata(served_filename, title, styles, lyrics, actual_duration)

    elapsed = time.time() - start_time
    gen_state.progress = 1.0
    gen_state.message = "Complete!"

    return served_filename, num_frames, actual_duration, elapsed

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the ML pipeline on startup, clean up on shutdown."""
    app.state.pipeline = _load_pipeline()
    yield
    app.state.pipeline = None

app = FastAPI(
    title="HeartMuLa MLX API",
    description="AI music generation powered by MLX on Apple Silicon",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api", tags=["api"])

@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Check API and model status."""
    return StatusResponse(
        status="ok",
        model_loaded=app.state.pipeline is not None,
        is_generating=gen_state.is_generating,
    )


@router.get("/progress", response_model=ProgressResponse)
async def get_progress() -> ProgressResponse:
    """Get current generation progress."""
    return ProgressResponse(
        progress=gen_state.progress,
        message=gen_state.message,
        is_generating=gen_state.is_generating,
        current_request=gen_state.current_request,
        started_at=gen_state.started_at,
    )


@router.get("/songs", response_model=SongsResponse)
async def get_songs() -> SongsResponse:
    """List all generated songs."""
    # Prefer MP3 files; fall back to WAV if no MP3 exists for a given stem
    mp3_stems = {p.stem for p in OUTPUT_DIR.glob("*.mp3")}
    audio_files = list(OUTPUT_DIR.glob("*.mp3"))
    audio_files.extend(p for p in OUTPUT_DIR.glob("*.wav") if p.stem not in mp3_stems)
    audio_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    songs = [_load_song_metadata(p) for p in audio_files]
    return SongsResponse(songs=songs)


@router.get("/surprise-lyrics", response_model=SurpriseLyricsResponse)
async def get_surprise_lyrics() -> SurpriseLyricsResponse:
    """Return random lyrics for the 'Surprise Me' feature."""
    song = random.choice(SURPRISE_LYRICS)
    return SurpriseLyricsResponse(
        title=song["title"],
        styles=song["styles"],
        lyrics=song["lyrics"].strip(),
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={status.HTTP_409_CONFLICT: {"description": "Generation already in progress"}},
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Start music generation."""
    if gen_state.is_generating:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Generation already in progress",
        )

    gen_state.is_generating = True
    gen_state.progress = 0.0
    gen_state.message = "Starting..."
    gen_state.current_request = {
        "title": request.title,
        "styles": request.styles,
        "lyrics": (request.lyrics[:100] + "..." if len(request.lyrics) > 100 else request.lyrics),
        "duration": request.duration,
    }
    gen_state.started_at = time.time()

    try:
        loop = asyncio.get_event_loop()
        filename, frames, duration, elapsed = await loop.run_in_executor(
            None,
            _generate_music_sync,
            app.state.pipeline,
            request.lyrics,
            request.styles,
            request.duration,
            request.cfg_scale,
            request.temperature,
            request.topk,
            request.title,
            request.language,
        )
        return GenerateResponse(filename=filename, frames=frames, duration=duration, time=elapsed)
    finally:
        gen_state.reset()


@router.post("/cancel", response_model=CancelResponse)
async def cancel_generation() -> CancelResponse:
    """Cancel the current generation."""
    if not gen_state.is_generating:
        return CancelResponse(status="no_generation", message="No generation in progress")
    gen_state.cancelled = True
    gen_state.message = "Cancelling..."
    return CancelResponse(status="cancelling", message="Generation will be cancelled")


@router.get(
    "/audio/{filename}",
    responses={status.HTTP_404_NOT_FOUND: {"description": "Audio file not found"}},
)
async def get_audio(
    filename: Annotated[str, PathParam(pattern=r"^[\w\-]+\.(wav|mp3)$")],
) -> FileResponse:
    """Serve a generated audio file (WAV or MP3)."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found")
    media_type = "audio/mpeg" if file_path.suffix == ".mp3" else "audio/wav"
    return FileResponse(file_path, media_type=media_type)


app.include_router(router)

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
