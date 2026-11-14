"""Cue segmentation from word timestamps (docs/04 §10).

The flat word list of a transcript is split into subtitle cues of at most ``max_lines``
lines and ``max_chars`` characters per line, 1-7 s long, with at least ``min_gap`` seconds
between consecutive cues. Speaker changes, long pauses and size limits always break; once a
cue is long enough, sentence punctuation then clause punctuation are preferred break points.
Words are never split and two speakers never share a cue.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from caracal.asr import Transcript, Word, join_words

_SENTENCE_ENDINGS = frozenset(".?!…。！？")  # noqa: RUF001 - CJK punctuation is intentional
_CLAUSE_ENDINGS = frozenset(",;:、，")  # noqa: RUF001 - CJK punctuation is intentional
_MIN_CUE_CHARS = 15
_BREAK_PAUSE_S = 1.0
_PUNCTUATION_BALANCE_SLACK = 8


@dataclass(frozen=True, slots=True)
class Cue:
    """One subtitle cue: ``word_span`` is the inclusive first/last global word index."""

    index: int
    start: float
    end: float
    lines: list[str]
    speaker: str | None
    word_span: tuple[int, int]


def _flatten(transcript: Transcript) -> list[Word]:
    return [word for segment in transcript.segments for word in segment.words]


def _ends_with(word: Word, endings: frozenset[str]) -> bool:
    return bool(word.text) and word.text[-1] in endings


def _line_slices(words: Sequence[Word], max_chars: int) -> tuple[int, ...]:
    """Split point(s) that render ``words`` as at most two balanced lines."""
    if len(words) < 2 or len(join_words(words)) <= max_chars:
        return ()
    best: tuple[int, int] | None = None
    fallback: tuple[int, int] | None = None
    for split in range(1, len(words)):
        left = join_words(words[:split])
        right = join_words(words[split:])
        widest = max(len(left), len(right))
        candidate = (split, abs(len(left) - len(right)))
        if fallback is None or widest < fallback[1]:
            fallback = (split, widest)
        if len(left) <= max_chars and len(right) <= max_chars:
            if best is None or candidate[1] < best[1]:
                best = candidate
    if best is None:
        if fallback is None:
            return ()
        return (fallback[0],)

    balanced = best[1]
    punctuated: tuple[int, int] | None = None
    for split in range(1, len(words)):
        if not _ends_with(words[split - 1], _SENTENCE_ENDINGS | _CLAUSE_ENDINGS):
            continue
        left = join_words(words[:split])
        right = join_words(words[split:])
        if len(left) > max_chars or len(right) > max_chars:
            continue
        imbalance = abs(len(left) - len(right))
        if imbalance <= balanced + _PUNCTUATION_BALANCE_SLACK and (
            punctuated is None or imbalance < punctuated[1]
        ):
            punctuated = (split, imbalance)
    chosen = punctuated if punctuated is not None else best
    return (chosen[0],)


def _packed_lines(words: Sequence[Word], max_chars: int) -> list[list[Word]]:
    """Greedy word-wrap: the fewest lines of at most ``max_chars`` (a long word gets its own)."""
    lines: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and len(join_words([*current, word])) > max_chars:
            lines.append(current)
            current = [word]
        else:
            current = [*current, word]
    if current:
        lines.append(current)
    return lines


def _renders_within(words: Sequence[Word], max_chars: int, max_lines: int) -> bool:
    if len(join_words(words)) > max_lines * max_chars:
        return False
    return len(_packed_lines(words, max_chars)) <= max_lines


def _lines(words: Sequence[Word], max_chars: int) -> list[str]:
    splits = _line_slices(words, max_chars)
    if not splits:
        packed = _packed_lines(words, max_chars)
        if len(packed) <= 1:
            return [join_words(words)]
        return [join_words(line) for line in packed]
    split = splits[0]
    return [join_words(words[:split]), join_words(words[split:])]


def _break_before(
    cue_words: Sequence[Word],
    word: Word,
    *,
    max_chars: int,
    max_lines: int,
    max_dur: float,
    min_dur: float,
) -> bool:
    """True when ``word`` must start a new cue (hard break, then preferred break)."""
    previous = cue_words[-1]
    if word.speaker != previous.speaker:
        return True
    if word.start - previous.end > _BREAK_PAUSE_S:
        return True
    first = cue_words[0]
    if word.end - first.start > max_dur:
        return True
    if not _renders_within([*cue_words, word], max_chars, max_lines):
        return True

    cue_dur = previous.end - first.start
    cue_len = len(join_words(cue_words))
    if cue_dur < min_dur or cue_len < _MIN_CUE_CHARS:
        return False
    if _ends_with(previous, _SENTENCE_ENDINGS):
        return True
    return _ends_with(previous, _CLAUSE_ENDINGS) and cue_len >= max_chars - _PUNCTUATION_BALANCE_SLACK


def _spans(
    words: Sequence[Word],
    *,
    max_chars: int,
    max_lines: int,
    max_dur: float,
    min_dur: float,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(words)):
        if _break_before(
            words[start:index],
            words[index],
            max_chars=max_chars,
            max_lines=max_lines,
            max_dur=max_dur,
            min_dur=min_dur,
        ):
            spans.append((start, index - 1))
            start = index
    if words:
        spans.append((start, len(words) - 1))
    return spans


def _timings(
    words: Sequence[Word],
    spans: Sequence[tuple[int, int]],
    *,
    min_dur: float,
    min_gap: float,
    max_dur: float,
) -> list[tuple[float, float]]:
    starts = [words[first].start for first, _ in spans]
    ends: list[float] = []
    for position, (_first, last) in enumerate(spans):
        start = starts[position]
        end = words[last].end
        if end - start < min_dur:
            end = start + min_dur
        if position + 1 < len(spans):
            end = min(end, starts[position + 1] - min_gap)
        end = max(end, start)
        end = min(end, start + max_dur)
        ends.append(end)
    return list(zip(starts, ends, strict=True))


def wrap_lines(
    text: str,
    *,
    max_chars: int = 42,
    max_lines: int = 2,
    break_chars: bool = False,
) -> tuple[list[str], bool]:
    """Wrap already-translated text into subtitle lines (docs/04 §10).

    Word languages wrap at spaces; ``break_chars=True`` (CJK) wraps at characters. The
    normal case is one or two balanced lines. When two lines cannot hold the text the
    function keeps every character and reports ``overlong=True``; the caller counts those
    cues, because the fixed cue timing cannot be widened.
    """
    cleaned = "".join(text.split()) if break_chars else " ".join(text.split())
    if not cleaned:
        return [], False
    if len(cleaned) <= max_chars:
        return [cleaned], False

    if break_chars:
        best: tuple[int, int] | None = None
        for split in range(max(1, len(cleaned) - max_chars), min(len(cleaned), max_chars) + 1):
            left = cleaned[:split]
            right = cleaned[split:]
            if len(left) <= max_chars and len(right) <= max_chars:
                balance = abs(len(left) - len(right))
                if best is None or balance < best[1]:
                    best = (split, balance)
        if best is not None:
            return [cleaned[: best[0]], cleaned[best[0] :]], False
        lines = [cleaned[index : index + max_chars] for index in range(0, len(cleaned), max_chars)]
        return lines, len(lines) > max_lines or any(len(line) > max_chars for line in lines)

    tokens = cleaned.split()
    best = None
    for split in range(1, len(tokens)):
        left = " ".join(tokens[:split])
        right = " ".join(tokens[split:])
        if len(left) <= max_chars and len(right) <= max_chars:
            balance = abs(len(left) - len(right))
            if best is None or balance < best[1]:
                best = (split, balance)
    if best is not None:
        left = " ".join(tokens[: best[0]])
        right = " ".join(tokens[best[0] :])
        return [left, right], False

    packed: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if current and len(" ".join([*current, token])) > max_chars:
            packed.append(current)
            current = [token]
        else:
            current = [*current, token]
    if current:
        packed.append(current)
    lines = [" ".join(line) for line in packed]
    overlong = len(lines) > max_lines or any(len(line) > max_chars for line in lines)
    return lines, overlong


def cues_from_transcript(
    transcript: Transcript,
    *,
    max_chars: int = 42,
    max_lines: int = 2,
    max_dur: float = 7.0,
    min_dur: float = 1.0,
    min_gap: float = 0.08,
) -> list[Cue]:
    """Turn the transcript's flat word list into ordered, non-overlapping cues."""
    words = _flatten(transcript)
    spans = _spans(
        words,
        max_chars=max_chars,
        max_lines=max_lines,
        max_dur=max_dur,
        min_dur=min_dur,
    )
    timings = _timings(words, spans, min_dur=min_dur, min_gap=min_gap, max_dur=max_dur)
    cues: list[Cue] = []
    for number, ((first, last), (start, end)) in enumerate(zip(spans, timings, strict=True), start=1):
        cue_words = words[first : last + 1]
        cues.append(
            Cue(
                index=number,
                start=start,
                end=end,
                lines=_lines(cue_words, max_chars),
                speaker=cue_words[0].speaker,
                word_span=(first, last),
            )
        )
    return cues
