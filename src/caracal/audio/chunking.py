"""Chunk planning for transcription (docs/04 §3).

Pure boundary arithmetic: cut on the silence closest to each target boundary, or hard-cut
at the target when there is no usable silence. Chunks carry a small overlap so words on a
boundary are not lost; the merge step removes the duplicated words later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

_SEARCH_BEHIND_S = 120.0
_TAIL_MERGE_S = 30.0


@dataclass(frozen=True, slots=True)
class Chunk:
    """One transcription window, in seconds from the start of the source audio."""

    index: int
    start_s: float
    end_s: float


def _midpoint(silence: tuple[float, float]) -> float:
    return (silence[0] + silence[1]) / 2.0


def plan_chunks(
    duration_s: float,
    silences: Sequence[tuple[float, float]],
    *,
    target_s: float = 600.0,
    max_s: float = 900.0,
    overlap_s: float = 1.0,
) -> list[Chunk]:
    """Plan transcription chunks that cover ``[0, duration_s]``.

    A source no longer than ``max_s`` becomes one chunk. Longer sources get a boundary at
    the midpoint of the silence nearest ``previous + target_s`` within
    ``[previous + target_s - 120, previous + max_s]``; without a silence the cut is hard at
    ``previous + target_s``. Consecutive chunks overlap by ``overlap_s`` except that the
    first starts at zero. A tail shorter than 30 s is merged into the previous chunk.
    """
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    if target_s <= 0 or max_s <= 0:
        raise ValueError("target_s and max_s must be positive")
    if overlap_s < 0:
        raise ValueError("overlap_s must be non-negative")
    if duration_s <= max_s:
        return [Chunk(index=0, start_s=0.0, end_s=duration_s)]

    boundaries = [0.0]
    previous = 0.0
    while duration_s - previous > max_s:
        target = previous + target_s
        low = target - _SEARCH_BEHIND_S
        high = previous + max_s
        candidates: list[float] = []
        for silence in silences:
            if silence[0] >= silence[1]:
                continue
            midpoint = _midpoint(silence)
            if low <= midpoint <= high:
                candidates.append(midpoint)
        boundary = min(candidates, key=lambda midpoint: abs(midpoint - target)) if candidates else target
        if boundary <= previous:
            boundary = min(previous + target_s, duration_s)
        if boundary >= duration_s:
            break
        boundaries.append(boundary)
        previous = boundary

    if len(boundaries) > 1 and duration_s - boundaries[-1] < _TAIL_MERGE_S:
        # The final sliver is too short to be useful on its own; fold it into the last chunk.
        boundaries.pop()

    chunks: list[Chunk] = []
    for index, boundary in enumerate(boundaries):
        start = 0.0 if index == 0 else boundary - overlap_s
        end = boundaries[index + 1] if index + 1 < len(boundaries) else duration_s
        chunks.append(Chunk(index=index, start_s=start, end_s=end))
    return chunks
