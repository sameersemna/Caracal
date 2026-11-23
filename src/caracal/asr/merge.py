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
from collections.abc import Collection, Sequence
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
    """Midpoint of the overlap of two adjacent chunks, or ``None`` when they do not overlap.

    Two chunks that merely touch (``later.start_s == earlier.end_s``) do not overlap: there
    is nothing for a midpoint rule to arbitrate, so the whole of both is kept. This is how a
    stitched recovery span (one chunk exactly over the credit span) sits between the region
    pieces before and after it without a midpoint cut eating its first or last words.
    """
    if later.start_s >= earlier.end_s:
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
    *,
    preserve_bounds: bool = False,
) -> Segment | None:
    """Shift a segment, keep the words owned by this chunk and preserve its text.

    A segment that loses no words keeps whisper's own ``text`` verbatim; only a trimmed
    segment is rebuilt from its words with :func:`join_words`. ``preserve_bounds`` keeps the
    segment's own start/end when nothing is trimmed: a stitched recovery span already clamped
    them at the window seams, and recomputing from coarse CJK word times would reintroduce
    the overlap the stitch removed (docs/04 §4, card 49).
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
    if preserve_bounds and len(kept) == len(shifted):
        return replace(
            segment,
            start=segment.start + shift,
            end=segment.end + shift,
            text=text,
            words=kept,
        )
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


#: Credit recovery re-transcribes a credit span in windows of this length, overlapping by
#: ``_RECOVERY_OVERLAP_S`` (docs/04 §4). The measured 30 s decode window opens with ~10 s of
#: music, then speech; 10 s windows catch that speech without re-entering the music.
_RECOVERY_SIZE_S = 10.0
_RECOVERY_OVERLAP_S = 1.0


def credit_spans(
    result: WhisperResult,
    chunk: Chunk,
    *,
    min_s: float = 8.0,
) -> list[tuple[float, float]]:
    """Absolute ``(start, end)`` of each credit segment of one chunk longer than ``min_s``.

    A credit that lasts longer than ``min_s`` cannot be an intro/outro card (the measured one
    spans whisper's whole 30 s decode window): it was hallucinated over music, the speech under
    it may still be there and :func:`recovery_windows` says how to get it back. Bounds are
    snapped to 0.1 s: whisper
    reports the measured 30 s window's credit as ``0.0-29.98`` and the recovery windows are
    derived from the window, not from that float noise.
    """
    spans: list[tuple[float, float]] = []
    for segment in result.segments:
        if segment.end - segment.start <= min_s:
            continue
        if not is_hallucinated_credit(normalise_text(segment.text)):
            continue
        spans.append(
            (
                round(chunk.start_s + segment.start, 1),
                round(chunk.start_s + segment.end, 1),
            )
        )
    return spans


def recovery_windows(
    start: float,
    end: float,
    *,
    size: float = _RECOVERY_SIZE_S,
    overlap: float = _RECOVERY_OVERLAP_S,
) -> list[tuple[float, float]]:
    """The ``size``-long windows that cover ``[start, end]`` with ``overlap`` (docs/04 §4).

    The first windows start at ``start, start + (size - overlap), …`` while the *next* window
    still fits; the last window ends at ``end`` and starts at ``max(last_start, end - size)``,
    so ``(0, 30)`` yields ``[(0, 10), (9, 19), (18, 28), (20, 30)]`` and no audio at the tail is
    lost. A span no longer than ``size`` is one window.
    """
    if end - start <= size:
        return [(start, end)]
    windows: list[tuple[float, float]] = []
    cursor = start
    while cursor + size < end:
        windows.append((cursor, cursor + size))
        cursor += size - overlap
    final_start = max(windows[-1][0] if windows else start, end - size)
    windows.append((final_start, end))
    return windows


def drop_window_credits(
    results: Sequence[WhisperResult],
    chunks: Sequence[Chunk],
    *,
    duration_s: float,
    silences: Sequence[tuple[float, float]] = (),
) -> list[WhisperResult]:
    """Drop whole credit segments from recovery-window results, before they are merged.

    A recovery window that holds only music comes back as the credit again. The overlap rule
    trims a segment to the words the neighbouring window did not take, which can cut
    ``中文字幕志愿者`` down to ``中文`` — text that no longer matches a credit and would then
    survive into the transcript. So each window segment is judged *whole* at absolute time by
    the same rule as :func:`_apply_filters` (a recurring credit anywhere, or a one-off at an
    edge or inside measured silence) and the rejected segments are removed here. The surviving
    results are merged normally afterwards.
    """
    if len(results) != len(chunks):
        raise CaracalError(
            ErrorCode.INTERNAL,
            "whisper results and chunks do not line up",
            {"results": len(results), "chunks": len(chunks)},
        )
    absolute: list[tuple[int, int, Segment]] = []
    for result_index, (result, chunk) in enumerate(zip(results, chunks, strict=True)):
        for segment_index, segment in enumerate(result.segments):
            absolute.append(
                (
                    result_index,
                    segment_index,
                    replace(
                        segment,
                        start=segment.start + chunk.start_s,
                        end=segment.end + chunk.start_s,
                    ),
                )
            )
    segments = [segment for _result, _position, segment in absolute]
    looped_credit = _is_looped_credit([segment for segment in segments if _survives_speech_filter(segment)])
    dropped = {
        (result_index, segment_index)
        for result_index, segment_index, segment in absolute
        if is_hallucinated_credit(normalise_text(segment.text))
        and (looped_credit or _at_edge(segment, duration_s) or _in_non_speech(segment, silences))
    }
    return [
        replace(
            result,
            segments=tuple(
                segment
                for segment_index, segment in enumerate(result.segments)
                if (result_index, segment_index) not in dropped
            ),
        )
        for result_index, result in enumerate(results)
    ]


#: A segment that ends within this distance of a window's end is truncated or invented
#: (``切换了警方的手机`` at 18.66-19.00), so it never sets the seam (docs/04 §4).
_SEAM_EPSILON_S = 0.2


def _absolute_segment(segment: Segment, shift: float, index: int) -> Segment:
    """One window segment on the source timeline, words shifted with it."""
    return replace(
        segment,
        id=index,
        start=segment.start + shift,
        end=segment.end + shift,
        words=tuple(replace(word, start=word.start + shift, end=word.end + shift) for word in segment.words),
    )


def stitch_windows(
    results: Sequence[WhisperResult],
    chunks: Sequence[Chunk],
) -> list[Segment]:
    """Stitch the recovery windows of one credit span at segment level (docs/04 §4).

    ``results`` are the windows in chronological order, already passed through
    :func:`drop_window_credits`, with window-local times; ``chunks`` are their source spans.
    For consecutive windows A, B the seam ``t`` is the end of A's last segment that ends more
    than :data:`_SEAM_EPSILON_S` before A's end (a segment touching the window end is
    truncated or invented), or ``B.start`` when A has no such segment. A keeps the segments
    ending at or before ``t``; B keeps those whose midpoint is at or after ``t``. Every kept
    segment is whole -- no words are trimmed -- and the last window's segments are all kept,
    because the audio after the credit span continues from there.

    Whisper times the same word differently in two windows, so a kept B segment can start a
    little before the seam (or before A's last kept segment ends). Only its reported
    start/end are pulled up to the previous segment's end; its words and text stay whole.
    The returned segments are absolute, ordered and non-overlapping.
    """
    if len(results) != len(chunks):
        raise CaracalError(
            ErrorCode.INTERNAL,
            "whisper results and chunks do not line up",
            {"results": len(results), "chunks": len(chunks)},
        )
    stitched: list[Segment] = []
    previous_end: float | None = None
    lower_bound: float | None = None
    for index, (result, chunk) in enumerate(zip(results, chunks, strict=True)):
        segments = [
            _absolute_segment(segment, chunk.start_s, position)
            for position, segment in enumerate(result.segments)
        ]
        if lower_bound is not None:
            segments = [
                segment for segment in segments if _midpoint(segment.start, segment.end) >= lower_bound
            ]
        if index + 1 < len(chunks):
            reliable = [segment for segment in segments if segment.end < chunk.end_s - _SEAM_EPSILON_S]
            seam = max(segment.end for segment in reliable) if reliable else chunks[index + 1].start_s
            segments = [segment for segment in segments if segment.end <= seam]
            lower_bound = seam
        for segment in segments:
            if previous_end is not None and segment.start < previous_end:
                segment = replace(segment, start=previous_end)
            stitched.append(replace(segment, id=len(stitched)))
            previous_end = segment.end
    return stitched


def merge_chunks(
    results: Sequence[WhisperResult],
    chunks: Sequence[Chunk],
    *,
    duration_s: float | None = None,
    silences: Sequence[tuple[float, float]] = (),
    preserve_bounds: Collection[int] = (),
) -> Transcript:
    """Merge chunk-local results into one absolute-timeline :class:`Transcript`.

    ``duration_s`` is the source audio length (default: the end of the last chunk); it bounds
    the intro/outro window in which hallucinated credits are dropped. ``silences`` are the
    non-speech spans measured by ``normalize``; a credit inside one is dropped wherever it
    sits. A caller that does not know them (the transcribe stage) passes none, and only the
    intro/outro rule applies. ``preserve_bounds`` lists chunk indices whose segments already
    carry deliberate start/end times (a stitched recovery span) and must not have them
    recomputed from whisper's coarse word times.
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
            trimmed = _kept_segment(
                raw_segment,
                chunk.start_s,
                lower_cut,
                upper_cut,
                preserve_bounds=index in preserve_bounds,
            )
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
