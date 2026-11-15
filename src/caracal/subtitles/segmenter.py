"""Cue segmentation from word timestamps (docs/04 §10).

The transcript's flat token list is first grouped into whole words (:func:`word_groups`):
whisper tokens are sub-word pieces, so breaking on a token would cut "Tagesschau" after
"Tag". Cue breaks, line breaks and the character limit then all act on those groups, so a
line or cue boundary can never fall inside a word. Cues have at most ``max_lines`` lines and
``max_chars`` characters per line, are 1-7 s long, with at least ``min_gap`` seconds between
consecutive cues. Speaker changes, long pauses and size limits always break; once a cue is
long enough, sentence punctuation then clause punctuation are preferred break points. A
single group longer than ``max_chars`` stays whole on its own line, and two speakers never
share a cue.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from caracal.asr import Transcript, Word, WordGroup, join_words, word_groups

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


def _text(words: Sequence[Word], groups: Sequence[WordGroup]) -> str:
    """Render whole groups, preserving whisper's own spacing between their tokens."""
    tokens = [word for group in groups for word in words[group.tokens[0] : group.tokens[1] + 1]]
    return join_words(tokens)


def _ends_with(group: WordGroup, endings: frozenset[str]) -> bool:
    return bool(group.text) and group.text[-1] in endings


def _line_group_slices(groups: Sequence[WordGroup], words: Sequence[Word], max_chars: int) -> tuple[int, ...]:
    """Split point(s) that render ``groups`` as at most two balanced lines."""
    if len(groups) < 2 or len(_text(words, groups)) <= max_chars:
        return ()
    best: tuple[int, int] | None = None
    fallback: tuple[int, int] | None = None
    for split in range(1, len(groups)):
        left = _text(words, groups[:split])
        right = _text(words, groups[split:])
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
    for split in range(1, len(groups)):
        if not _ends_with(groups[split - 1], _SENTENCE_ENDINGS | _CLAUSE_ENDINGS):
            continue
        left = _text(words, groups[:split])
        right = _text(words, groups[split:])
        if len(left) > max_chars or len(right) > max_chars:
            continue
        imbalance = abs(len(left) - len(right))
        if imbalance <= balanced + _PUNCTUATION_BALANCE_SLACK and (
            punctuated is None or imbalance < punctuated[1]
        ):
            punctuated = (split, imbalance)
    chosen = punctuated if punctuated is not None else best
    return (chosen[0],)


def _packed_group_lines(
    groups: Sequence[WordGroup], words: Sequence[Word], max_chars: int
) -> list[list[WordGroup]]:
    """Greedy group-wrap: the fewest lines of at most ``max_chars`` (a long word gets its own)."""
    lines: list[list[WordGroup]] = []
    current: list[WordGroup] = []
    for group in groups:
        if current and len(_text(words, [*current, group])) > max_chars:
            lines.append(current)
            current = [group]
        else:
            current = [*current, group]
    if current:
        lines.append(current)
    return lines


def _renders_within(
    groups: Sequence[WordGroup], words: Sequence[Word], max_chars: int, max_lines: int
) -> bool:
    if len(_text(words, groups)) > max_lines * max_chars:
        return False
    return len(_packed_group_lines(groups, words, max_chars)) <= max_lines


def _lines(groups: Sequence[WordGroup], words: Sequence[Word], max_chars: int) -> list[str]:
    splits = _line_group_slices(groups, words, max_chars)
    if not splits:
        packed = _packed_group_lines(groups, words, max_chars)
        if len(packed) <= 1:
            return [_text(words, groups)]
        return [_text(words, line) for line in packed]
    split = splits[0]
    return [_text(words, groups[:split]), _text(words, groups[split:])]


def _break_before(
    cue_groups: Sequence[WordGroup],
    group: WordGroup,
    words: Sequence[Word],
    *,
    max_chars: int,
    max_lines: int,
    max_dur: float,
    min_dur: float,
) -> bool:
    """True when ``group`` must start a new cue (hard break, then preferred break)."""
    previous = cue_groups[-1]
    if group.speaker != previous.speaker:
        return True
    if group.start - previous.end > _BREAK_PAUSE_S:
        return True
    first = cue_groups[0]
    if group.end - first.start > max_dur:
        return True
    if not _renders_within([*cue_groups, group], words, max_chars, max_lines):
        return True

    cue_dur = previous.end - first.start
    cue_len = len(_text(words, cue_groups))
    if cue_dur < min_dur or cue_len < _MIN_CUE_CHARS:
        return False
    if _ends_with(previous, _SENTENCE_ENDINGS):
        return True
    return _ends_with(previous, _CLAUSE_ENDINGS) and cue_len >= max_chars - _PUNCTUATION_BALANCE_SLACK


def _spans(
    groups: Sequence[WordGroup],
    words: Sequence[Word],
    *,
    max_chars: int,
    max_lines: int,
    max_dur: float,
    min_dur: float,
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(groups)):
        if _break_before(
            groups[start:index],
            groups[index],
            words,
            max_chars=max_chars,
            max_lines=max_lines,
            max_dur=max_dur,
            min_dur=min_dur,
        ):
            spans.append((start, index - 1))
            start = index
    if groups:
        spans.append((start, len(groups) - 1))
    return spans


def _timings(
    groups: Sequence[WordGroup],
    spans: Sequence[tuple[int, int]],
    *,
    min_dur: float,
    min_gap: float,
    max_dur: float,
) -> list[tuple[float, float]]:
    starts = [groups[first].start for first, _ in spans]
    ends: list[float] = []
    for position, (_first, last) in enumerate(spans):
        start = starts[position]
        end = groups[last].end
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
    """Turn the transcript's flat token list into ordered, non-overlapping cues."""
    words = _flatten(transcript)
    groups = word_groups(words)
    spans = _spans(
        groups,
        words,
        max_chars=max_chars,
        max_lines=max_lines,
        max_dur=max_dur,
        min_dur=min_dur,
    )
    timings = _timings(groups, spans, min_dur=min_dur, min_gap=min_gap, max_dur=max_dur)
    cues: list[Cue] = []
    for number, ((first, last), (start, end)) in enumerate(zip(spans, timings, strict=True), start=1):
        cue_groups = groups[first : last + 1]
        cues.append(
            Cue(
                index=number,
                start=start,
                end=end,
                lines=_lines(cue_groups, words, max_chars),
                speaker=cue_groups[0].speaker,
                word_span=(cue_groups[0].tokens[0], cue_groups[-1].tokens[1]),
            )
        )
    return cues
