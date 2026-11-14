"""Cue segmentation: real merged transcript invariants plus targeted break rules."""

from __future__ import annotations

import itertools
import re

import pytest
from tests.fakes.whisper import load_fixture

from caracal.asr import Segment, Transcript, WhisperResult, Word, join_words
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


def _ends_at_punctuation_or_hard_break(cues: list[Cue], words: list[Word]) -> bool:
    """An independent reading of the accept rule: punctuation, or a boundary the rules force."""
    cue = cues[0]
    last = words[cue.word_span[1]]
    if last.text and last.text[-1] in _ENDINGS:
        return True
    if len(cues) == 1:
        return True
    following = cues[1]
    first = words[following.word_span[0]]
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
    cues = cues_from_transcript(transcript)

    assert len(words) == 404
    assert len(cues) > 20
    assert [cue.index for cue in cues] == list(range(1, len(cues) + 1))

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
        1 for position, cue in enumerate(cues) if _ends_at_punctuation_or_hard_break(cues[position:], words)
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


def test_cjk_punctuation_breaks_without_inserting_spaces() -> None:
    words = (
        _word("一二三四五六七八", 0.0, 0.6, space_before=False),
        _word("九十一二三四。", 0.7, 1.4, space_before=False),
        _word("再見", 1.5, 1.9, space_before=False),
    )
    cues = cues_from_transcript(_transcript(words))

    assert cues[0].lines == ["一二三四五六七八九十一二三四。"]
    assert cues[1].lines == ["再見"]


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
