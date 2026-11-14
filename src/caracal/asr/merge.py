"""Chunk merge and hallucination filters (docs/04 §4).

Whisper results are chunk-local; merging shifts them onto the source timeline, keeps each
word from the chunk whose timeline centre owns it, trims/rebuilds segments and drops
silence hallucinations and repeated text.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import replace

from caracal.asr import Segment, Transcript, WhisperResult, join_words
from caracal.audio.chunking import Chunk
from caracal.errors import CaracalError, ErrorCode

_NO_SPEECH_LIMIT = 0.8
_LOW_LOGPROB_LIMIT = -1.0
_REPEAT_LIMIT = 3


def _midpoint(start: float, end: float) -> float:
    return (start + end) / 2.0


def _cut(earlier: Chunk, later: Chunk) -> float | None:
    """Midpoint of the overlap of two adjacent chunks, or ``None`` when they do not overlap."""
    if later.start_s > earlier.end_s:
        return None
    return (later.start_s + earlier.end_s) / 2.0


def normalise_text(text: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace (repetition detection)."""
    normalised = unicodedata.normalize("NFKC", text).lower()
    kept = "".join(char if char.isalnum() or char.isspace() else " " for char in normalised)
    return " ".join(kept.split())


def _kept_segment(
    segment: Segment,
    shift: float,
    lower_cut: float | None,
    upper_cut: float | None,
) -> Segment | None:
    """Shift a segment, keep the words owned by this chunk and preserve its text.

    A segment that loses no words keeps whisper's own ``text`` verbatim; only a trimmed
    segment is rebuilt from its words with :func:`join_words`.
    """
    shifted = tuple(replace(word, start=word.start + shift, end=word.end + shift) for word in segment.words)
    if not shifted:
        midpoint = _midpoint(segment.start, segment.end) + shift
        if lower_cut is not None and midpoint < lower_cut:
            return None
        if upper_cut is not None and midpoint >= upper_cut:
            return None
        return replace(
            segment,
            start=segment.start + shift,
            end=segment.end + shift,
            text=segment.text.strip(),
        )

    kept = tuple(
        word
        for word in shifted
        if (lower_cut is None or _midpoint(word.start, word.end) >= lower_cut)
        and (upper_cut is None or _midpoint(word.start, word.end) < upper_cut)
    )
    if not kept:
        return None
    text = segment.text.strip() if len(kept) == len(shifted) else join_words(kept)
    return replace(
        segment,
        start=min(word.start for word in kept),
        end=max(word.end for word in kept),
        text=text,
        words=kept,
    )


def _dominant_language(segments: Sequence[Segment]) -> str | None:
    speech: dict[str, float] = {}
    for segment in segments:
        if segment.lang is None:
            continue
        speech[segment.lang] = speech.get(segment.lang, 0.0) + max(0.0, segment.end - segment.start)
    if not speech:
        return None
    return max(speech, key=speech.__getitem__)


def _apply_filters(segments: Sequence[Segment]) -> tuple[list[Segment], dict[str, int]]:
    dropped = {"no_speech": 0, "repetition": 0, "empty": 0}
    kept: list[Segment] = []
    last_norm: str | None = None
    run = 0
    for segment in segments:
        if segment.no_speech_prob > _NO_SPEECH_LIMIT and segment.avg_logprob < _LOW_LOGPROB_LIMIT:
            dropped["no_speech"] += 1
            continue
        if not segment.text.strip():
            dropped["empty"] += 1
            continue
        norm = normalise_text(segment.text)
        if norm == last_norm:
            run += 1
        else:
            last_norm = norm
            run = 1
        if norm and run >= _REPEAT_LIMIT:
            dropped["repetition"] += 1
            continue
        kept.append(replace(segment, id=len(kept)))
    return kept, dropped


def merge_chunks(results: Sequence[WhisperResult], chunks: Sequence[Chunk]) -> Transcript:
    """Merge chunk-local results into one absolute-timeline :class:`Transcript`."""
    if len(results) != len(chunks):
        raise CaracalError(
            ErrorCode.INTERNAL,
            "whisper results and chunks do not line up",
            {"results": len(results), "chunks": len(chunks)},
        )

    combined: list[Segment] = []
    for index, (result, chunk) in enumerate(zip(results, chunks, strict=True)):
        lower_cut = _cut(chunks[index - 1], chunk) if index > 0 else None
        upper_cut = _cut(chunk, chunks[index + 1]) if index + 1 < len(chunks) else None
        for raw_segment in result.segments:
            trimmed = _kept_segment(raw_segment, chunk.start_s, lower_cut, upper_cut)
            if trimmed is None:
                continue
            combined.append(
                replace(
                    trimmed,
                    lang=result.language_code,
                    lang_prob=result.language_prob,
                )
            )

    combined.sort(key=lambda segment: (segment.start, segment.end, segment.id))
    kept, dropped = _apply_filters(combined)
    return Transcript(
        segments=tuple(kept),
        language_code=_dominant_language(kept),
        dropped=dropped,
    )
