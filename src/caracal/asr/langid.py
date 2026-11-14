"""Language spans, low-confidence re-checks and the report ``languages`` summary (docs/04 §4).

Each chunk carries one detection; :func:`language_spans` turns the merged transcript into
contiguous spans, and :func:`refine_spans` re-detects 30 s windows of a long, low-confidence
chunk so code-switching inside a chunk is not hidden behind the chunk-level result.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import replace

from caracal.asr import LANGUAGE_NAMES, LangSpan, Segment, Transcript, WhisperResult

_UNKNOWN = "und"
_UNKNOWN_NAME = "Unknown"
_LOW_CONFIDENCE = 0.5
_RECHECK_BELOW = 0.6
_RECHECK_MIN_DURATION_S = 60.0
_WINDOW_S = 30.0


def _span_code(segment: Segment, min_conf: float) -> str:
    if segment.lang is None:
        return _UNKNOWN
    if segment.lang_prob < min_conf:
        return _UNKNOWN
    return segment.lang


def language_spans(transcript: Transcript, *, min_conf: float = 0.5) -> list[LangSpan]:
    """Group consecutive segments that share a language code into spans."""
    spans: list[LangSpan] = []
    current_code: str | None = None
    span_start = 0.0
    span_end = 0.0
    span_conf = 1.0
    for segment in transcript.segments:
        code = _span_code(segment, min_conf)
        if current_code is not None and code == current_code:
            span_end = segment.end
            span_conf = min(span_conf, segment.lang_prob)
            continue
        if current_code is not None:
            spans.append(LangSpan(code=current_code, start=span_start, end=span_end, confidence=span_conf))
        current_code = code
        span_start = segment.start
        span_end = segment.end
        span_conf = segment.lang_prob
    if current_code is not None:
        spans.append(LangSpan(code=current_code, start=span_start, end=span_end, confidence=span_conf))
    return spans


def _dominant(segments: Sequence[Segment]) -> str | None:
    speech: dict[str, float] = {}
    for segment in segments:
        if segment.lang is None:
            continue
        speech[segment.lang] = speech.get(segment.lang, 0.0) + max(0.0, segment.end - segment.start)
    if not speech:
        return None
    return max(speech, key=speech.__getitem__)


async def refine_spans(
    transcript: Transcript,
    chunk_results: Sequence[tuple[float, WhisperResult]],
    *,
    recheck: Callable[[float, float], Awaitable[WhisperResult]],
    max_calls: int = 6,
) -> Transcript:
    """Re-detect windows of long, low-confidence chunks and relabel the segments inside them.

    ``chunk_results`` holds ``(chunk_start_s, result)`` pairs so windows can be placed on the
    absolute timeline. At most ``max_calls`` calls to ``recheck`` are made in total.
    """
    updates: dict[int, tuple[str, float]] = {}
    calls = 0
    for chunk_start, result in chunk_results:
        if calls >= max_calls:
            break
        if result.language_prob >= _RECHECK_BELOW or result.duration <= _RECHECK_MIN_DURATION_S:
            continue
        chunk_end = chunk_start + result.duration
        window_start = chunk_start
        while window_start < chunk_end and calls < max_calls:
            window_end = min(window_start + _WINDOW_S, chunk_end)
            detected = await recheck(window_start, window_end)
            calls += 1
            if detected.language_prob >= _RECHECK_BELOW and detected.language_code is not None:
                for position, segment in enumerate(transcript.segments):
                    midpoint = (segment.start + segment.end) / 2.0
                    if window_start <= midpoint < window_end:
                        updates[position] = (detected.language_code, detected.language_prob)
            window_start = window_end

    if not updates:
        return transcript
    segments = tuple(
        replace(segment, lang=updates[position][0], lang_prob=updates[position][1])
        if position in updates
        else segment
        for position, segment in enumerate(transcript.segments)
    )
    return Transcript(
        segments=segments,
        language_code=_dominant(segments),
        dropped=transcript.dropped,
    )


def languages_summary(
    transcript: Transcript, *, min_conf: float = _LOW_CONFIDENCE
) -> list[dict[str, object]]:
    """Build the report ``languages`` items: code, name, share, spans and confidence."""
    spans = language_spans(transcript, min_conf=min_conf)
    if not spans:
        return []

    speech: dict[str, float] = {}
    for segment in transcript.segments:
        code = _span_code(segment, min_conf)
        speech[code] = speech.get(code, 0.0) + max(0.0, segment.end - segment.start)
    total = sum(speech.values())
    if total <= 0:
        return []

    grouped: dict[str, list[LangSpan]] = {}
    for span in spans:
        grouped.setdefault(span.code, []).append(span)

    items: list[dict[str, object]] = []
    for code, code_spans in grouped.items():
        name = _UNKNOWN_NAME if code == _UNKNOWN else LANGUAGE_NAMES.get(code, code)
        items.append(
            {
                "code": code,
                "name": name,
                "share": speech.get(code, 0.0) / total,
                "spans": [{"start": span.start, "end": span.end} for span in code_spans],
                "confidence": min(span.confidence for span in code_spans),
            }
        )
    return items
