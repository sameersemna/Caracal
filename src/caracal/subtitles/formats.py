"""SRT and WebVTT rendering for :class:`~caracal.subtitles.segmenter.Cue` (docs/04 §10).

Timestamps are rounded to milliseconds; hours keep at least two digits (``100:00:00,000``).
WebVTT text escapes ``&``, ``<`` and ``>`` and may carry a ``<v S1>`` voice tag when the
speaker is known.
"""

from __future__ import annotations

from collections.abc import Sequence

from caracal.subtitles.segmenter import Cue

_MS_PER_HOUR = 3_600_000
_MS_PER_MINUTE = 60_000
_MS_PER_SECOND = 1_000


def _total_ms(seconds: float) -> int:
    return max(0, round(seconds * _MS_PER_SECOND))


def _stamp(seconds: float, separator: str) -> str:
    hours, rest = divmod(_total_ms(seconds), _MS_PER_HOUR)
    minutes, rest = divmod(rest, _MS_PER_MINUTE)
    whole, millis = divmod(rest, _MS_PER_SECOND)
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{separator}{millis:03d}"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_srt(cues: Sequence[Cue]) -> str:
    """Render cues as SubRip; the result ends with exactly one newline."""
    blocks = [
        f"{cue.index}\n{_stamp(cue.start, ',')} --> {_stamp(cue.end, ',')}\n" + "\n".join(cue.lines)
        for cue in cues
    ]
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def to_vtt(cues: Sequence[Cue], *, voice_tags: bool = True) -> str:
    """Render cues as WebVTT; the result ends with exactly one newline."""
    blocks: list[str] = []
    for cue in cues:
        lines = [_escape(line) for line in cue.lines]
        if voice_tags and cue.speaker:
            lines[0] = f"<v {_escape(cue.speaker)}>{lines[0]}"
        blocks.append(f"{_stamp(cue.start, '.')} --> {_stamp(cue.end, '.')}\n" + "\n".join(lines))
    if not blocks:
        return "WEBVTT\n"
    return "WEBVTT\n\n" + "\n\n".join(blocks) + "\n"
