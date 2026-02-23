"""Lightweight profiling helpers for HeartMuLa MLX inference.

Usage::

    from heartlib_mlx.profiler import Profiler

    prof = Profiler()

    with prof.section("preprocess"):
        model_inputs = pipeline.preprocess(inputs, cfg_scale=1.5)

    with prof.section("forward"):
        model_outputs = pipeline._forward(model_inputs, ...)

    with prof.section("postprocess"):
        pipeline.postprocess(model_outputs, save_path="out.wav")

    prof.summary()  # prints a clean table to stderr
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import mlx.core as mx

logger = logging.getLogger("heartmula.profiler")


@dataclass
class _Entry:
    """A single recorded timing."""

    name: str
    wall_s: float


@dataclass
class Profiler:
    """Accumulates wall-clock timings for named sections.

    Each ``section`` context manager calls ``mx.eval()`` on entry *and* exit
    so that GPU work from a previous section doesn't bleed into the next
    measurement.  This gives honest per-section timings at the cost of a
    sync barrier — use only for profiling, not production.
    """

    _entries: list[_Entry] = field(default_factory=list)

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        """Time a named section (wall-clock, GPU-synchronised)."""
        mx.synchronize()
        t0 = time.perf_counter()
        yield
        mx.synchronize()
        elapsed = time.perf_counter() - t0
        self._entries.append(_Entry(name=name, wall_s=elapsed))

    @property
    def total_s(self) -> float:
        return sum(e.wall_s for e in self._entries)

    def summary(self) -> str:
        """Return a formatted summary table and log it."""
        if not self._entries:
            return "(no sections recorded)"

        total = self.total_s
        # Column widths
        max_name = max(len(e.name) for e in self._entries)
        col_name = max(max_name, 7)  # "section"

        lines: list[str] = []
        header = f"{'section':<{col_name}}  {'time':>8}  {'pct':>5}"
        sep = "-" * len(header)
        lines.append(sep)
        lines.append(header)
        lines.append(sep)

        for entry in self._entries:
            pct = entry.wall_s / total * 100 if total > 0 else 0
            lines.append(f"{entry.name:<{col_name}}  {entry.wall_s:>7.2f}s  {pct:>4.1f}%")

        lines.append(sep)
        lines.append(f"{'total':<{col_name}}  {total:>7.2f}s")
        lines.append(sep)

        text = "\n".join(lines)
        logger.info("Profiler summary:\n%s", text)
        return text

    def reset(self) -> None:
        self._entries.clear()
