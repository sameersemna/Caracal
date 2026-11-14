"""Platform caption parsing and whisper↔caption agreement (docs/04 §4).

YouTube auto captions are rolling: a word-timed cue with inline ``<hh:mm:ss.mmm>`` tags is
followed by plain duplicate cues that repeat the same text. When inline tags exist anywhere
only the tagged cues count, so each spoken word appears once. Manual captions have plain,
multi-line cues and are spread evenly over the cue duration.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from caracal.asr import Transcript

_TIMING_RE = re.compile(r"(?P<start>\d{1,2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}\.\d{3})")
_INLINE_RE = re.compile(r"<(?P<ts>\d{1,2}:\d{2}:\d{2}\.\d{3})>")
_TAG_RE = re.compile(r"<[^>]*>")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_MIN_AGREEMENT_WORDS = 20


@dataclass(frozen=True, slots=True)
class CaptionWord:
    """One caption word on the source timeline."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class _Cue:
    start: float
    end: float
    lines: tuple[str, ...]


def _timestamp(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def _cues(text: str) -> list[_Cue]:
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[_Cue] = []
    for block in normalised.split("\n\n"):
        lines = block.split("\n")
        if not lines:
            continue
        first = lines[0].strip().upper()
        if first.startswith(("NOTE", "STYLE", "REGION")):
            continue
        timing_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        match = _TIMING_RE.search(lines[timing_index])
        if match is None:
            continue
        cues.append(
            _Cue(
                start=_timestamp(match.group("start")),
                end=_timestamp(match.group("end")),
                lines=tuple(lines[timing_index + 1 :]),
            )
        )
    return cues


def _plain_text(lines: Sequence[str]) -> str:
    text = " ".join(_TAG_RE.sub("", line) for line in lines)
    return html.unescape(text)


def _tagged_words(cue: _Cue) -> list[CaptionWord]:
    tagged = [line for line in cue.lines if _INLINE_RE.search(line)]
    if not tagged:
        return []
    content = " ".join(tagged)
    boundaries: list[float] = [cue.start]
    fragments: list[str] = []
    position = 0
    for match in _INLINE_RE.finditer(content):
        fragments.append(content[position : match.start()])
        boundaries.append(_timestamp(match.group("ts")))
        position = match.end()
    fragments.append(content[position:])
    boundaries.append(cue.end)

    words: list[CaptionWord] = []
    for index, fragment in enumerate(fragments):
        word = _plain_text([fragment]).strip()
        if not word:
            continue
        start = boundaries[index]
        end = boundaries[index + 1]
        words.append(CaptionWord(text=word, start=start, end=min(end, cue.end)))
    return words


def _plain_words(cue: _Cue) -> list[CaptionWord]:
    tokens = _plain_text(cue.lines).split()
    if not tokens:
        return []
    span = max(0.0, cue.end - cue.start)
    step = span / len(tokens)
    return [
        CaptionWord(text=token, start=cue.start + index * step, end=cue.start + (index + 1) * step)
        for index, token in enumerate(tokens)
    ]


def parse_vtt(text: str) -> list[CaptionWord]:
    """Parse a WebVTT file into words, preferring inline word-timed cues when present."""
    cues = _cues(text)
    tagged = any(_INLINE_RE.search(line) for cue in cues for line in cue.lines)
    words: list[CaptionWord] = []
    for cue in cues:
        if tagged:
            if _INLINE_RE.search(" ".join(cue.lines)):
                words.extend(_tagged_words(cue))
        else:
            words.extend(_plain_words(cue))
    words.sort(key=lambda word: (word.start, word.end))
    return words


def normalise_word(text: str) -> str:
    """NFKC, lower-case, bracket tags, punctuation and music marks removed."""
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = _BRACKET_RE.sub("", folded)
    return "".join(character for character in folded if character.isalnum())


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, start=1):
        current = [row]
        for column, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _buckets(items: Sequence[tuple[float, str]], window_s: float) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for midpoint, token in items:
        key = int(midpoint // window_s)
        result.setdefault(key, []).append(token)
    return result


def caption_agreement(
    transcript: Transcript,
    words: Sequence[CaptionWord],
    *,
    window_s: float = 60.0,
) -> float | None:
    """Word-level agreement between the transcript and platform captions, or ``None``.

    Both sides are restricted to their overlapping time range and compared per ``window_s``
    bucket, so the edit distance never becomes one global O(n·m) matrix.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")

    reference: list[tuple[float, str]] = []
    for caption_word in words:
        token = normalise_word(caption_word.text)
        if token:
            reference.append(((caption_word.start + caption_word.end) / 2.0, token))
    hypothesis: list[tuple[float, str]] = []
    for segment in transcript.segments:
        for word in segment.words:
            token = normalise_word(word.text)
            if token:
                hypothesis.append(((word.start + word.end) / 2.0, token))

    if not reference or not hypothesis:
        return None

    start = max(min(item[0] for item in reference), min(item[0] for item in hypothesis))
    end = min(max(item[0] for item in reference), max(item[0] for item in hypothesis))
    reference = [item for item in reference if start <= item[0] <= end]
    hypothesis = [item for item in hypothesis if start <= item[0] <= end]
    if len(reference) < _MIN_AGREEMENT_WORDS or len(hypothesis) < _MIN_AGREEMENT_WORDS:
        return None

    reference_buckets = _buckets(reference, window_s)
    hypothesis_buckets = _buckets(hypothesis, window_s)
    errors = 0
    total = 0
    for key in sorted(set(reference_buckets) | set(hypothesis_buckets)):
        expected = reference_buckets.get(key, [])
        actual = hypothesis_buckets.get(key, [])
        total += len(expected)
        errors += _edit_distance(expected, actual)
    if total == 0:
        return None
    return min(1.0, max(0.0, 1.0 - errors / total))
