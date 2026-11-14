"""Speaker separation: pyannote diarization, word alignment and intro re-timing.

pyannote/torch are imported lazily inside :mod:`caracal.speakers.diarize` so the gate and
every deployment without the ``speakers`` extra can import this package (docs/01 §2).
"""

from __future__ import annotations

from caracal.speakers.align import assign_speakers, retime_outside_speech, speaker_stats
from caracal.speakers.diarize import DiarizationResult, Diarizer, Turn

__all__ = [
    "DiarizationResult",
    "Diarizer",
    "Turn",
    "assign_speakers",
    "retime_outside_speech",
    "speaker_stats",
]
