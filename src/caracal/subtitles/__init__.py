"""Subtitle cues: word-timestamp segmentation and SRT/WebVTT rendering (docs/04 §10)."""

from __future__ import annotations

from caracal.subtitles.formats import to_srt, to_vtt
from caracal.subtitles.segmenter import Cue, cues_from_transcript

__all__ = ["Cue", "cues_from_transcript", "to_srt", "to_vtt"]
