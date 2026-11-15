"""Cue segmentation: real merged transcript invariants plus targeted break rules."""

from __future__ import annotations

import itertools
import re

import pytest
from tests.fakes.whisper import load_fixture

from caracal.asr import Segment, Transcript, WhisperResult, Word, join_words, word_groups
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.subtitles.segmenter import Cue, cues_from_transcript

_SENTENCE_ENDINGS = frozenset(".?!…。！？")  # noqa: RUF001 - CJK punctuation is intentional
_CLAUSE_ENDINGS = frozenset(",;:、，")  # noqa: RUF001 - CJK punctuation is intentional
_ENDINGS = _SENTENCE_ENDINGS | _CLAUSE_ENDINGS
_ARTEFACT_RE = re.compile(r" [,.?!;:]|\w[.?!,][A-Za-z]")
_MAX_CHARS = 42
_MAX_LINES = 2
_MAX_DUR = 7.0
_MIN_GAP = 0.08
_CJK_MAX_CHARS = 16


def _word(
    text: str,
    start: float,
    end: float,
    *,
    speaker: str | None = None,
    space_before: bool = True,
) -> Word:
    return Word(text=text, start=start, end=end, prob=0.9, speaker=speaker, space_before=space_before)


def _segment(index: int, words: tuple[Word, ...]) -> Segment:
    return Segment(
        id=index,
        start=words[0].start,
        end=words[-1].end,
        text=join_words(words),
        words=words,
        avg_logprob=0.0,
        no_speech_prob=0.0,
        lang="en",
    )


def _transcript(*word_groups: tuple[Word, ...]) -> Transcript:
    return Transcript(
        segments=tuple(_segment(index, group) for index, group in enumerate(word_groups)),
        language_code="en",
        dropped={"no_speech": 0, "repetition": 0, "empty": 0},
    )


def _real(name: str) -> WhisperResult:
    return parse_whisper_response(load_fixture(name))


def _merged_pair() -> Transcript:
    chunks = [Chunk(index=0, start_s=0.0, end_s=61.0), Chunk(index=1, start_s=60.0, end_s=120.0)]
    return merge_chunks([_real("talk_c0"), _real("talk_c1")], chunks)


def _flat(transcript: Transcript) -> list[Word]:
    return [word for segment in transcript.segments for word in segment.words]


def _segment_starts(transcript: Transcript) -> set[int]:
    """Global token indices at which a new whisper segment begins."""
    starts: set[int] = set()
    offset = 0
    for segment in transcript.segments:
        if segment.words:
            starts.add(offset)
        offset += len(segment.words)
    return starts


def _segment_of(transcript: Transcript) -> list[int]:
    """For every global token index, the ordinal of the whisper segment that owns it."""
    owners: list[int] = []
    for ordinal, segment in enumerate(transcript.segments):
        owners.extend([ordinal] * len(segment.words))
    return owners


def _ends_at_punctuation_or_hard_break(cues: list[Cue], words: list[Word], segment_starts: set[int]) -> bool:
    """An independent reading of the accept rule: punctuation, or a boundary the rules force."""
    cue = cues[0]
    last = words[cue.word_span[1]]
    if last.text and last.text[-1] in _ENDINGS:
        return True
    if len(cues) == 1:
        return True
    following = cues[1]
    first = words[following.word_span[0]]
    if following.word_span[0] in segment_starts:  # fix 2: a segment boundary is a preferred break
        return True
    if first.speaker != last.speaker:
        return True
    if first.start - last.end > 1.0:
        return True
    if first.end - cue.start > _MAX_DUR:
        return True
    return len(join_words(words[cue.word_span[0] : following.word_span[1] + 1])) > _MAX_LINES * _MAX_CHARS


def test_real_merged_transcript_satisfies_every_cue_invariant() -> None:
    transcript = _merged_pair()
    words = _flat(transcript)
    segment_starts = _segment_starts(transcript)
    cues = cues_from_transcript(transcript)

    assert len(words) == 404
    assert len(cues) > 20
    assert [cue.index for cue in cues] == list(range(1, len(cues) + 1))
    # Fix 2 adds segment-boundary breaks to this pair (was 36 cues, now 43); 10 cues now end
    # without punctuation at a segment boundary (1 before the fix).
    assert len(cues) == 43
    assert (
        sum(
            1
            for previous, cue in itertools.pairwise(cues)
            if cue.word_span[0] in segment_starts and words[previous.word_span[1]].text[-1] not in _ENDINGS
        )
        == 10
    )

    for position, cue in enumerate(cues):
        assert 1 <= len(cue.lines) <= _MAX_LINES
        for line in cue.lines:
            assert len(line) <= _MAX_CHARS or len(line.split()) == 1
        assert cue.end > cue.start
        assert cue.end - cue.start <= _MAX_DUR + 1e-9
        assert " ".join(cue.lines) == join_words(words[cue.word_span[0] : cue.word_span[1] + 1])
        assert _ARTEFACT_RE.search(" ".join(cue.lines)) is None
        if position + 1 < len(cues):
            following = cues[position + 1]
            assert cue.end <= following.start
            assert following.start - cue.end >= _MIN_GAP - 1e-6

    starts = [cue.start for cue in cues]
    assert starts == sorted(starts)
    assert all(left < right for left, right in itertools.pairwise(starts))

    spans = [cue.word_span for cue in cues]
    covered = [index for first, last in spans for index in range(first, last + 1)]
    assert covered == list(range(len(words)))  # no word lost or duplicated, spans in order

    ending_well = sum(
        1
        for position, cue in enumerate(cues)
        if _ends_at_punctuation_or_hard_break(cues[position:], words, segment_starts)
    )
    assert ending_well / len(cues) >= 0.8


def test_speaker_change_always_breaks_and_never_mixes_speakers() -> None:
    transcript = _transcript(
        (
            _word("one", 0.0, 0.4, speaker="S1"),
            _word("two", 0.5, 0.9, speaker="S1"),
            _word("three", 1.0, 1.4, speaker="S2"),
            _word("four", 1.5, 1.9, speaker="S2"),
            _word("five", 2.0, 2.4, speaker="S1"),
        )
    )
    cues = cues_from_transcript(transcript)

    assert [cue.speaker for cue in cues] == ["S1", "S2", "S1"]
    assert [cue.lines for cue in cues] == [["one two"], ["three four"], ["five"]]
    all_words = transcript.segments[0].words
    for cue in cues:
        first, last = cue.word_span
        assert {word.speaker for word in all_words[first : last + 1]} == {cue.speaker}


def test_pause_longer_than_one_second_starts_a_new_cue() -> None:
    transcript = _transcript(
        (
            _word("one", 0.0, 0.5),
            _word("two", 0.6, 1.0),
            _word("three", 2.5, 3.0),
        )
    )
    cues = cues_from_transcript(transcript)

    assert [cue.lines for cue in cues] == [["one two"], ["three"]]
    assert cues[1].start == 2.5
    assert cues[1].end == 3.5  # extended to min_dur, no following cue to stop it


def test_max_duration_splits_a_long_run_of_words() -> None:
    words = tuple(_word(f"w{index}", index * 0.9, (index + 1) * 0.9) for index in range(10))
    cues = cues_from_transcript(_transcript(words))

    assert len(cues) >= 2
    assert all(cue.end - cue.start <= _MAX_DUR for cue in cues)
    covered = [index for cue in cues for index in range(cue.word_span[0], cue.word_span[1] + 1)]
    assert covered == list(range(10))


def test_sentence_punctuation_ends_a_long_enough_cue() -> None:
    words = (
        _word("This", 0.0, 0.3),
        _word("is", 0.4, 0.6),
        _word("a", 0.7, 0.8),
        _word("sentence.", 0.9, 1.4),
        _word("Next", 1.5, 1.8),
        _word("one.", 1.9, 2.4),
    )
    cues = cues_from_transcript(_transcript(words))

    assert [cue.lines for cue in cues] == [["This is a sentence."], ["Next one."]]


def test_clause_punctuation_ends_a_cue_near_the_line_limit() -> None:
    words = (
        _word("alpha", 0.0, 0.3),
        _word("beta", 0.4, 0.7),
        _word("gamma", 0.8, 1.1),
        _word("delta", 1.2, 1.5),
        _word("epsilon", 1.6, 1.9),
        _word("zeta,", 2.0, 2.3),
        _word("eta", 2.4, 2.7),
    )
    cues = cues_from_transcript(_transcript(words))

    assert cues[0].lines == ["alpha beta gamma delta epsilon zeta,"]
    assert cues[1].lines == ["eta"]


def test_two_line_split_balances_and_prefers_a_punctuation_boundary() -> None:
    words = (
        _word("The", 0.0, 0.3),
        _word("quick", 0.4, 0.7),
        _word("brown", 0.8, 1.1),
        _word("fox", 1.2, 1.5),
        _word("jumps,", 1.6, 1.9),
        _word("over", 2.0, 2.3),
        _word("the", 2.4, 2.7),
        _word("lazy", 2.8, 3.1),
        _word("dog", 3.2, 3.5),
        _word("today.", 3.6, 3.9),
    )
    cues = cues_from_transcript(_transcript(words))

    assert len(cues) == 1
    assert cues[0].lines == ["The quick brown fox jumps,", "over the lazy dog today."]
    assert all(len(line) <= _MAX_CHARS for line in cues[0].lines)


def test_a_single_word_longer_than_a_line_stays_on_its_own_line() -> None:
    long_word = "antidisestablishmentarianism" * 2
    words = (_word(long_word, 0.0, 1.5), _word("next", 1.6, 1.9))
    cues = cues_from_transcript(_transcript(words))

    assert cues[0].lines == [long_word, "next"]


def test_real_chinese_cues_break_between_characters_without_a_space_signal() -> None:
    """Verdict 42 fix 1 + verdict 43 fixes 1-2: CJK lines hold 16 characters, cues follow segments.

    Real behaviour: every CJK character is breakable, a cue line never exceeds 16 characters
    (two lines), and a whisper segment boundary is a hard break because the text carries no
    sentence punctuation. The trailing credit is dropped by the merge filter, and no line may
    contain a U+FFFD byte fragment.
    """
    result = _real("zh_speech40")
    transcript = merge_chunks([result], [Chunk(index=0, start_s=0.0, end_s=result.duration)])
    words = _flat(transcript)
    owners = _segment_of(transcript)
    cues = cues_from_transcript(transcript)

    assert not any(word.space_before for word in words)
    assert len(cues) == 6  # was 5 before fixes 1-2
    assert cues[0].lines == ["欧洲阿里亚纳火箭"]
    for cue in cues:
        assert 1 <= len(cue.lines) <= _MAX_LINES
        for line in cue.lines:
            assert len(line) <= _CJK_MAX_CHARS
            assert "\ufffd" not in line
        first, last = cue.word_span
        assert len({owners[index] for index in range(first, last + 1)}) == 1  # never two segments
    rendered = "".join(line for cue in cues for line in cue.lines)
    assert rendered == "".join(segment.text for segment in transcript.segments)
    assert "字幕" not in rendered  # the "中文字幕:CJ" credit at 30 s is not a subtitle
    covered = [index for cue in cues for index in range(cue.word_span[0], cue.word_span[1] + 1)]
    assert covered == list(range(len(words)))  # no character lost or duplicated


def test_min_duration_is_extended_but_never_up_to_the_next_cue() -> None:
    transcript = _transcript(
        (
            _word("hi", 0.0, 0.2),
            _word("there", 1.5, 1.7),
        )
    )
    cues = cues_from_transcript(transcript)

    assert cues[0].end == 1.0  # min_dur reached, 1.5 - 0.08 is further away
    assert cues[1].start == 1.5


def test_a_gap_smaller_than_min_gap_pulls_the_earlier_end_back() -> None:
    transcript = _transcript(
        (
            _word("first sentence.", 0.0, 1.0),  # 15 chars: sentence break is preferred
            _word("second", 1.04, 1.6),
        )
    )
    cues = cues_from_transcript(transcript)

    assert len(cues) == 2
    assert cues[0].end == pytest.approx(0.96)
    assert cues[1].start - cues[0].end == pytest.approx(0.08)


def test_real_tagesschau_cues_and_lines_break_only_between_word_groups() -> None:
    payload = load_fixture("tagesschau_seg0")
    fixture_words = tuple(
        Word(
            text=item["text"],
            start=item["start"],
            end=item["end"],
            prob=item["prob"],
            speaker=item.get("speaker"),
            space_before=item["space_before"],
        )
        for item in payload["words"]
    )
    transcript = Transcript(
        segments=(
            Segment(
                id=payload["id"],
                start=payload["start"],
                end=payload["end"],
                text=payload["text"],
                words=fixture_words,
                avg_logprob=0.0,
                no_speech_prob=0.0,
                lang="de",
                speaker=payload.get("speaker"),
            ),
        ),
        language_code="de",
        dropped={"no_speech": 0, "repetition": 0, "empty": 0},
    )
    words = _flat(transcript)
    groups = word_groups(words)
    cues = cues_from_transcript(transcript, max_chars=20)

    assert len(cues) > 1
    assert any(len(cue.lines) == 2 for cue in cues)  # a wrap is forced and exercised

    # A cue boundary never falls inside "Tagesschau" or "Sekunden.".
    for cue in cues[1:]:
        assert words[cue.word_span[0]].space_before is True

    starts = {group.tokens[0] for group in groups}
    ends = {group.tokens[1] for group in groups}
    for cue in cues:
        first, last = cue.word_span
        assert first in starts
        assert last in ends
        # Every rendered whitespace-delimited word is one whole group, in order: no line
        # boundary can sit inside a group.
        expected = [group.text for group in groups if first <= group.tokens[0] and group.tokens[1] <= last]
        assert " ".join(cue.lines).split() == expected

    rendered = " ".join(" ".join(cue.lines) for cue in cues)
    assert "Tagesschau" in rendered
    assert "Tag esschau" not in rendered
    assert "Sekunden." in rendered


def test_a_word_split_into_tokens_and_longer_than_a_line_stays_whole() -> None:
    long_word = "a" * 30 + "b" * 30
    words = (
        _word(long_word[:30], 0.0, 0.5, space_before=False),
        _word(long_word[30:], 0.5, 1.0, space_before=False),
        _word("next", 1.1, 1.4),
    )
    cues = cues_from_transcript(_transcript(words))

    assert cues[0].lines == [long_word, "next"]
    assert cues[0].word_span == (0, 2)


def _chars(text: str, start: float = 0.0, step: float = 0.1) -> tuple[Word, ...]:
    """One Word per character, without a space signal — whisper's CJK/Thai tokenisation."""
    return tuple(
        Word(
            text=char,
            start=start + index * step,
            end=start + (index + 1) * step,
            prob=0.99,
            speaker=None,
            space_before=False,
        )
        for index, char in enumerate(text)
    )


def test_cjk_line_limit_is_sixteen_characters_while_latin_keeps_forty_two() -> None:
    """Fix 1: a cue at least half Han characters wraps at 16 per line, Latin still at 42."""
    chinese = "从法属圭亚纳的欧洲航天局的发射场发射携带物资开赴国际空间站"
    cjk_cues = cues_from_transcript(_transcript(_chars(chinese)))
    assert len(cjk_cues) == 1
    assert len(cjk_cues[0].lines) == 2
    assert [len(line) for line in cjk_cues[0].lines] == [14, 15]
    assert all(len(line) <= _CJK_MAX_CHARS for line in cjk_cues[0].lines)
    assert "".join(cjk_cues[0].lines) == chinese

    latin_words = (
        _word("abcdefghij", 0.0, 0.3),
        _word("klmnopqrst", 0.4, 0.7),
        _word("uvwxyzabcd", 0.8, 1.1),
        _word("efghijklmn", 1.2, 1.5),
        _word("opqrstuvwx", 1.6, 1.9),
        _word("yzabcdefgh", 2.0, 2.3),
    )
    latin_cues = cues_from_transcript(_transcript(latin_words))
    assert len(latin_cues) == 1
    assert len(latin_cues[0].lines) == 2
    assert all(len(line) <= _MAX_CHARS for line in latin_cues[0].lines)
    assert max(len(line) for line in latin_cues[0].lines) > _CJK_MAX_CHARS


def test_segment_boundary_is_a_preferred_break_for_latin_text() -> None:
    """Fix 2: without punctuation, a long-enough cue still breaks at the segment boundary."""
    long_first = _transcript(
        (
            _word("this", 0.0, 0.3),
            _word("is", 0.4, 0.5),
            _word("a", 0.6, 0.7),
            _word("long", 0.8, 1.0),
            _word("cue", 1.0, 1.2),
        ),
        (
            _word("and", 1.25, 1.4),
            _word("this", 1.4, 1.6),
            _word("is", 1.6, 1.7),
            _word("another", 1.7, 2.0),
        ),
    )
    owners = _segment_of(long_first)
    cues = cues_from_transcript(long_first)
    assert [cue.lines for cue in cues] == [["this is a long cue"], ["and this is another"]]
    assert {owners[index] for index in cues[0].word_span} == {0}
    assert {owners[index] for index in cues[1].word_span} == {1}

    # Preferred, not hard: a cue below the minimum length still spans both segments.
    short_first = _transcript(
        (_word("hi", 0.0, 0.2), _word("there", 0.3, 0.6)),
        (_word("friend", 0.7, 1.0),),
    )
    short_owners = _segment_of(short_first)
    short_cues = cues_from_transcript(short_first)
    assert len(short_cues) == 1
    assert short_cues[0].lines == ["hi there friend"]
    assert {short_owners[index] for index in short_cues[0].word_span} == {0, 1}


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("欧洲阿里亚纳火箭", "携带物资开赴国际空间站"),
        ("สวัสดีครับ", "ขอบคุณมากครับ"),
    ],
)
def test_segment_boundary_is_a_hard_break_for_cjk_without_sentence_punctuation(
    first: str, second: str
) -> None:
    """Fix 2: CJK/Thai text has no punctuation to break on, so the segment boundary is hard."""
    second_start = len(first) * 0.1 + 0.05  # within the 1 s pause that would break anyway
    transcript = _transcript(_chars(first, 0.0), _chars(second, second_start))
    owners = _segment_of(transcript)
    assert len(join_words(_flat(transcript))) <= _MAX_LINES * _CJK_MAX_CHARS  # would fit one cue

    cues = cues_from_transcript(transcript)

    assert len(cues) == 2
    assert {owners[index] for index in cues[0].word_span} == {0}
    assert {owners[index] for index in cues[1].word_span} == {1}
    assert "".join(cues[0].lines) == first
    assert "".join(cues[1].lines) == second


def test_a_word_cut_across_a_segment_boundary_stays_one_word() -> None:
    """Fix 2 must not break inside a word whisper cut at a chunk boundary ("couldn'" | "'t")."""
    transcript = _transcript(
        (_word("I", 0.0, 0.2), _word("couldn", 0.2, 0.6)),
        (
            _word("'t", 0.6, 0.7, space_before=False),
            _word("go", 0.8, 0.9),
            _word("back", 1.0, 1.2),
        ),
    )
    owners = _segment_of(transcript)
    # A 10-character line limit forces a wrap: with the word cut in two, the split would fall
    # between "couldn" and "'t".
    cues = cues_from_transcript(transcript, max_chars=10)

    assert len(cues) == 1
    assert cues[0].lines == ["I couldn't", "go back"]
    assert {owners[index] for index in cues[0].word_span} == {0, 1}
    assert all(cue.word_span[0] != 2 for cue in cues)  # the mid-word segment start never breaks
