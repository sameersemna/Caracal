"""Scripted diarizer for unit tests (docs/07 §4).

The model boundary is the only thing faked: :class:`FakeDiarizer` returns pre-built turns
and embeddings without importing pyannote or torch, and records every call so a test can
assert the stage passed the normalized audio and the caller hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from caracal.speakers.diarize import DiarizationResult


@dataclass
class FakeDiarizer:
    """A deterministic diarizer: return ``result`` (or raise ``error``) for every call."""

    result: DiarizationResult
    available_flag: bool = True
    error: Exception | None = None
    calls: list[tuple[Path, int | None]] = field(default_factory=list)

    def available(self) -> bool:
        return self.available_flag

    async def run(self, wav: Path, *, num_speakers: int | None = None) -> DiarizationResult:
        self.calls.append((wav, num_speakers))
        if self.error is not None:
            raise self.error
        return self.result
