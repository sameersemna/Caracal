"""Chunk merge and hallucination filters (docs/04 §4).

Whisper results are chunk-local; merging shifts them onto the source timeline, keeps each
word from the chunk whose timeline centre owns it, trims/rebuilds segments and drops
silence hallucinations, repeated text and the training-data credits whisper invents over
music and silence (``"Subtitles by the Amara.org community"``, ``"中文字幕志愿者"`` …). A
credit phrase that recurs is a decoding loop, so every occurrence is dropped wherever it
sits; a lone occurrence is judged by position and silence.
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
#: Credits are only plausible over the intro/outro or over non-speech; a speaker can say
#: "thank you for watching" in the middle of a talk and mean it.
_EDGE_WINDOW_S = 30.0

#: Normalised training-data credit phrases whisper hallucinates over music and silence.
_CREDIT_PHRASES: tuple[str, ...] = (
    "字幕志愿者",
    "中文字幕",
    "字幕由",
    "请不吝点赞",
    "ご視聴ありがとうございました",
    "untertitel der amara org community",
    "untertitel im auftrag des zdf",
    "untertitelung im auftrag",
    "sous titres réalisés par",
    "subtitles by the amara org community",
    "thanks for watching",
    "thank you for watching",
)

#: Chinese "subscribe + share + tip" begging appears together or not at all.
_CREDIT_TOKENS: tuple[str, ...] = ("订阅", "转发", "打赏")


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


_NORMALISED_CREDITS: tuple[str, ...] = tuple(normalise_text(phrase) for phrase in _CREDIT_PHRASES)
_NORMALISED_CREDIT_TOKENS: tuple[str, ...] = tuple(normalise_text(token) for token in _CREDIT_TOKENS)


def credit_matches(normalised: str) -> frozenset[str]:
    """The credit patterns present in ``normalised`` text (empty when it is not a credit)."""
    keys = {phrase for phrase in _NORMALISED_CREDITS if phrase in normalised}
    if normalised and all(token in normalised for token in _NORMALISED_CREDIT_TOKENS):
        keys.add("+".join(_NORMALISED_CREDIT_TOKENS))
    return frozenset(keys)


def is_hallucinated_credit(normalised: str) -> bool:
    """True when normalised segment text is a known whisper credit hallucination."""
    return bool(credit_matches(normalised))


def _at_edge(segment: Segment, duration_s: float) -> bool:
    """True when the segment lies wholly in the first or the last 30 s of the audio."""
    return segment.end <= _EDGE_WINDOW_S or segment.start >= duration_s - _EDGE_WINDOW_S


def _in_non_speech(segment: Segment, silences: Sequence[tuple[float, float]]) -> bool:
    """True when the segment's midpoint falls in a span measured as silence (music)."""
    midpoint = _midpoint(segment.start, segment.end)
    return any(start <= midpoint <= end for start, end in silences)


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


def _survives_speech_filter(segment: Segment) -> bool:
    """True when the segment is not a high-no-speech hallucination and has visible text."""
    if segment.no_speech_prob > _NO_SPEECH_LIMIT and segment.avg_logprob < _LOW_LOGPROB_LIMIT:
        return False
    return bool(segment.text.strip())


def _is_looped_credit(segments: Sequence[Segment]) -> bool:
    """True when one credit phrase recurs: whisper conditioned on its own hallucination.

    A single match can be a real sentence at the head of a talk ("thank you for watching"),
    so position and silence decide it. The same phrase in two or more segments is a decoding
    loop: the invented line comes back in every window regardless of where it lands. Two
    *different* one-off credit phrases are each judged by the position rule, not treated as
    a loop.
    """
    counts: dict[str, int] = {}
    for segment in segments:
        for key in credit_matches(normalise_text(segment.text)):
            counts[key] = counts.get(key, 0) + 1
    return any(count >= 2 for count in counts.values())


def _apply_filters(
    segments: Sequence[Segment],
    *,
    duration_s: float,
    silences: Sequence[tuple[float, float]],
) -> tuple[list[Segment], dict[str, int]]:
    dropped = {"no_speech": 0, "repetition": 0, "empty": 0, "hallucinated_credit": 0}
    looped_credit = _is_looped_credit([segment for segment in segments if _survives_speech_filter(segment)])
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
        at_edge = _at_edge(segment, duration_s)
        if is_hallucinated_credit(norm) and (looped_credit or at_edge or _in_non_speech(segment, silences)):
            dropped["hallucinated_credit"] += 1
            continue
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


def merge_chunks(
    results: Sequence[WhisperResult],
    chunks: Sequence[Chunk],
    *,
    duration_s: float | None = None,
    silences: Sequence[tuple[float, float]] = (),
) -> Transcript:
    """Merge chunk-local results into one absolute-timeline :class:`Transcript`.

    ``duration_s`` is the source audio length (default: the end of the last chunk); it bounds
    the intro/outro window in which hallucinated credits are dropped. ``silences`` are the
    non-speech spans measured by ``normalize``; a credit inside one is dropped wherever it
    sits. A caller that does not know them (the transcribe stage) passes none, and only the
    intro/outro rule applies.
    """
    if len(results) != len(chunks):
        raise CaracalError(
            ErrorCode.INTERNAL,
            "whisper results and chunks do not line up",
            {"results": len(results), "chunks": len(chunks)},
        )
    audio_duration = (
        duration_s if duration_s is not None else max((chunk.end_s for chunk in chunks), default=0.0)
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
    kept, dropped = _apply_filters(combined, duration_s=audio_duration, silences=silences)
    return Transcript(
        segments=tuple(kept),
        language_code=_dominant_language(kept),
        dropped=dropped,
    )
