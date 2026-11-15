"""Cue segmentation from word timestamps (docs/04 §10).

The transcript's flat token list is first grouped into whole words (:func:`word_groups`):
whisper tokens are sub-word pieces, so breaking on a token would cut "Tagesschau" after
"Tag". Cue breaks, line breaks and the character limit then all act on those groups, so a
line or cue boundary can never fall inside a word — even when whisper cut a word at a chunk
boundary (``couldn'`` | ``'t``), because grouping spans segments. Cues have at most
``max_lines`` lines and ``max_chars`` characters per line — except that a cue whose
characters are at least half Han/Kana/Hangul/Thai uses :data:`_CJK_LINE_CHARS` (subtitle
guidelines are ~16 characters, not 42 columns, for those scripts) — are 1-7 s long, with at
least ``min_gap`` seconds between consecutive cues. Speaker changes, long pauses and size
limits always break. Once a cue is long enough, a whisper segment boundary is a preferred
break point in every script, just like sentence punctuation, and a hard one for CJK/Thai text
that carries no sentence punctuation (whisper often emits none there, so the segment boundary
is the only sentence signal). Sentence punctuation then clause punctuation are the other
preferred points. A single group longer than the line limit stays whole on its own line, and
two speakers never share a cue.
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
#: Subtitle guidelines for scripts written without word spaces: ~16 characters per line.
_CJK_LINE_CHARS = 16

#: Han, Kana, Hangul and Thai blocks (a cue dominated by them wraps at characters).
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0E7F),  # Thai
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x2E80, 0x2FDF),  # CJK radicals, Kangxi radicals
    (0x3000, 0x30FF),  # CJK punctuation, Hiragana, Katakana
    (0x3130, 0x318F),  # Hangul compatibility Jamo
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xA960, 0xA97F),  # Hangul Jamo extended A
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFE30, 0xFE4F),  # CJK compatibility forms
    (0xFF00, 0xFFEF),  # half/fullwidth forms (halfwidth Katakana included)
    (0x20000, 0x3FFFF),  # CJK unified ideographs extension B and beyond
)


@dataclass(frozen=True, slots=True)
class Cue:
    """One subtitle cue: ``word_span`` is the inclusive first/last global word index."""

    index: int
    start: float
    end: float
    lines: list[str]
    speaker: str | None
    word_span: tuple[int, int]


def _is_cjk_char(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _is_cjk_heavy(text: str) -> bool:
    """True when at least half of the rendered characters are Han/Kana/Hangul/Thai."""
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return False
    return sum(1 for char in visible if _is_cjk_char(char)) * 2 >= len(visible)


def _line_chars(text: str, max_chars: int) -> int:
    """Characters per line for ``text``: the CJK guideline, else the caller's limit."""
    return _CJK_LINE_CHARS if _is_cjk_heavy(text) else max_chars


def _words_and_groups(transcript: Transcript) -> tuple[list[Word], list[WordGroup], set[int]]:
    """Flatten the transcript, group it into words and locate the segment boundaries.

    Returns the flat words, the groups with global token indices, and the global token index
    at which each whisper segment starts. Grouping is global so a word whisper cut across a
    chunk boundary (``couldn'`` | ``'t``) stays one word; such a group contains a segment
    start that cannot be a break point, and :func:`_spans` only sees the boundary when it
    falls on a group's first token.
    """
    words: list[Word] = []
    segment_starts: set[int] = set()
    offset = 0
    for segment in transcript.segments:
        if not segment.words:
            continue
        segment_starts.add(offset)
        words.extend(segment.words)
        offset += len(segment.words)
    return words, word_groups(words), segment_starts


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
    new_segment: bool,
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
    candidate = [*cue_groups, group]
    if not _renders_within(candidate, words, _line_chars(_text(words, candidate), max_chars), max_lines):
        return True

    cue_text = _text(words, cue_groups)
    if new_segment and _is_cjk_heavy(cue_text) and not _ends_with(previous, _SENTENCE_ENDINGS):
        # Whisper usually writes CJK/Thai without punctuation: the segment boundary is the
        # only sentence signal, so it breaks even a short cue.
        return True

    cue_dur = previous.end - first.start
    cue_len = len(cue_text)
    if cue_dur < min_dur or cue_len < _MIN_CUE_CHARS:
        return False
    if _ends_with(previous, _SENTENCE_ENDINGS) or new_segment:
        return True
    return (
        _ends_with(previous, _CLAUSE_ENDINGS)
        and cue_len >= _line_chars(cue_text, max_chars) - _PUNCTUATION_BALANCE_SLACK
    )


def _spans(
    groups: Sequence[WordGroup],
    words: Sequence[Word],
    segment_starts: set[int],
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
            new_segment=groups[index].tokens[0] in segment_starts,
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
    words, groups, segment_starts = _words_and_groups(transcript)
    spans = _spans(
        groups,
        words,
        segment_starts,
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
                lines=_lines(cue_groups, words, _line_chars(_text(words, cue_groups), max_chars)),
                speaker=cue_groups[0].speaker,
                word_span=(cue_groups[0].tokens[0], cue_groups[-1].tokens[1]),
            )
        )
    return cues
