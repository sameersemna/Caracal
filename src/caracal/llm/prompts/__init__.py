"""Prompt loading with safe substitution, DATA-block wrapping and transcript rendering.

Templates use ``{placeholder}`` tokens. Substitution is a two-phase ``str.replace`` so braces
inside substituted DATA are never interpreted as placeholders (``str.format`` would do that).

The transcript is rendered as one line per segment, ``[seg:12] S2 03:14.2 | text``, so every
claim can cite an addressable evidence id (docs/04 §1, §6). Metadata uses the same
line-addressed shape without a timestamp: ``[meta:title] ...``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from caracal.asr import Segment

_PROMPTS_DIR = Path(__file__).resolve().parent
_GUARD = "_guard"
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
_UNKNOWN_SPEAKER = "S?"


def _read(name: str) -> str:
    if _NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid prompt name: {name!r}")
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def load_prompt(name: str, **variables: object) -> str:
    """Load ``<name>.md``, substitute ``{placeholders}`` and prepend the guard block.

    Raises :class:`KeyError` when a placeholder in the template has no variable. Values are
    inserted verbatim; braces inside them are data, not placeholders.
    """
    template = _read(name)
    for match in _PLACEHOLDER_RE.finditer(template):
        if match.group(1) not in variables:
            raise KeyError(match.group(1))

    rendered = template
    sentinels: dict[str, str] = {}
    for index, (key, value) in enumerate(variables.items()):
        sentinel = f"\x00{index}\x00"
        rendered = rendered.replace("{" + key + "}", sentinel)
        sentinels[sentinel] = str(value)
    for sentinel, value in sentinels.items():
        rendered = rendered.replace(sentinel, value)

    guard = _read(_GUARD).rstrip()
    return f"{guard}\n\n{rendered.lstrip()}"


def _sanitise(text: str) -> str:
    """Strip the DATA-block delimiters so data can never close the block it lives in."""
    return text.replace("<<<", "").replace(">>>", "")


def wrap_data(label: str, text: str) -> str:
    """Wrap untrusted text in a DATA block that it cannot close (docs/06 S9)."""
    return f"<<<DATA:{label}>>>\n{_sanitise(text)}\n<<<END:{label}>>>"


def format_timestamp(seconds: float) -> str:
    """Render a segment start as ``mm:ss.s``, or ``h:mm:ss.s`` past one hour."""
    deciseconds = round(max(seconds, 0.0) * 10)
    hours, remainder = divmod(deciseconds, 36_000)
    minutes, remainder = divmod(remainder, 600)
    whole_seconds, tenths = divmod(remainder, 10)
    if hours:
        return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{tenths}"
    return f"{minutes:02d}:{whole_seconds:02d}.{tenths}"


def _speaker_label(speaker: str | None) -> str:
    if speaker is None:
        return _UNKNOWN_SPEAKER
    stripped = speaker.strip()
    return stripped or _UNKNOWN_SPEAKER


def render_transcript_lines(segments: Iterable[Segment]) -> str:
    """Render transcript segments as citable ``[seg:<id>] S<n> mm:ss.s | text`` lines."""
    return "\n".join(
        f"[seg:{segment.id}] {_speaker_label(segment.speaker)} "
        f"{format_timestamp(segment.start)} | {segment.text}"
        for segment in segments
    )


def render_metadata(meta: Mapping[str, str]) -> str:
    """Render metadata as ``[meta:<field>] value`` lines the model can cite."""
    return "\n".join(f"[meta:{key}] {_sanitise(value)}" for key, value in meta.items())


__all__ = [
    "format_timestamp",
    "load_prompt",
    "render_metadata",
    "render_transcript_lines",
    "wrap_data",
]
