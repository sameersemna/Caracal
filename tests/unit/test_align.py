"""Word speaker assignment, segment splitting, stats and intro re-timing (docs/04 §5)."""

from __future__ import annotations

import pytest

from caracal.asr import Segment, Transcript, Word
from caracal.speakers.align import assign_speakers, retime_outside_speech, speaker_stats
from caracal.speakers.diarize import Turn


def _word(text: str, start: float, end: float, *, space_before: bool = True) -> Word:
    return Word(text=text, start=start, end=end, prob=0.9, space_before=space_before)


def _segment(seg_id: int, words: list[Word]) -> Segment:
    return Segment(
        id=seg_id,
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(word.text for word in words),
        words=tuple(words),
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        lang="en",
    )


def _transcript(*segments: Segment) -> Transcript:
    return Transcript(segments=segments, language_code="en", dropped={})


def _speakers(transcript: Transcript) -> list[str | None]:
    return [word.speaker for segment in transcript.segments for word in segment.words]


def test_word_midpoint_selects_the_exclusive_turn_that_overlaps_it() -> None:
    first = _word("one", 0.2, 0.8, space_before=False)
    second = _word("two", 2.9, 3.3)
    transcript = _transcript(_segment(0, [first, second]))

    aligned = assign_speakers(
        transcript,
        [Turn(0.0, 2.0, "S1"), Turn(2.8, 4.0, "S2")],
    )

    assert _speakers(aligned) == ["S1", "S2"]
    assert [segment.speaker for segment in aligned.segments] == ["S1", "S2"]


def test_word_in_no_turn_uses_nearest_turn_within_one_second_else_none() -> None:
    near = _word("near", 2.4, 2.8, space_before=False)
    far = _word("far", 3.5, 3.7)
    transcript = _transcript(_segment(0, [near, far]))

    aligned = assign_speakers(transcript, [Turn(1.0, 2.0, "S1")])

    assert _speakers(aligned) == ["S1", None]


def test_segment_is_split_at_a_mid_segment_speaker_change_without_spacing_artefacts() -> None:
    words = [
        _word("hello", 0.0, 0.4, space_before=False),
        _word("there", 0.4, 0.9),
        _word("friend", 1.2, 1.7),
    ]
    transcript = _transcript(_segment(0, words))

    aligned = assign_speakers(
        transcript,
        [Turn(0.0, 1.0, "S1"), Turn(1.1, 2.0, "S2")],
    )

    assert [segment.text for segment in aligned.segments] == ["hello there", "friend"]
    assert [segment.id for segment in aligned.segments] == [0, 1]
    assert [segment.speaker for segment in aligned.segments] == ["S1", "S2"]
    assert (aligned.segments[0].start, aligned.segments[0].end) == (0.0, 0.9)
    assert (aligned.segments[1].start, aligned.segments[1].end) == (1.2, 1.7)


def test_speaker_stats_shares_sum_to_one_and_first_appearance_is_the_earliest_word() -> None:
    transcript = _transcript(
        _segment(
            0,
            [
                _word("a", 0.0, 0.5, space_before=False),
                _word("b", 0.5, 1.0),
                _word("c", 1.2, 1.8),
            ],
        )
    )
    aligned = assign_speakers(
        transcript,
        [Turn(0.0, 1.0, "S1"), Turn(1.1, 2.0, "S2")],
    )

    stats = speaker_stats(aligned)

    assert set(stats) == {"S1", "S2"}
    assert stats["S1"] == {
        "talk_time_s": 1.0,
        "share": 0.625,
        "first_seen_s": 0.0,
        "words": 2,
    }
    assert stats["S2"] == {
        "talk_time_s": 0.6,
        "share": 0.375,
        "first_seen_s": 1.2,
        "words": 1,
    }
    assert sum(float(stats[label]["share"]) for label in stats) == pytest.approx(1.0)


def test_retime_moves_intro_words_into_the_first_turn_and_preserves_order() -> None:
    intro = [
        _word("and", 0.18, 0.48, space_before=False),
        _word("so", 0.6, 0.9),
        _word("my", 1.0, 1.3),
        _word("fellow", 3.8, 4.2),
    ]
    inside = [_word("americans", 13.4, 13.9, space_before=False), _word("today", 14.0, 14.5)]
    transcript = _transcript(_segment(0, intro), _segment(1, inside))

    retimed = retime_outside_speech(transcript, [Turn(13.2, 20.0, "S1")])

    moved = retimed.segments[0]
    starts = [word.start for word in moved.words]
    assert starts == pytest.approx([13.2, 13.25, 13.3, 13.35])
    assert starts == sorted(starts)
    assert all(13.2 <= start < 13.4 for start in starts)
    assert all(word.end - word.start >= 0.05 for word in moved.words)
    assert all(
        abs(word.start - before.start) <= 30.0 for word, before in zip(moved.words, intro, strict=False)
    )
    assert (moved.start, moved.end) == (13.2, 13.75)
    assert [(word.text, word.start, word.end) for word in retimed.segments[1].words] == [
        ("americans", 13.4, 13.9),
        ("today", 14.0, 14.5),
    ]


def test_retime_moves_words_sitting_in_a_long_gap_into_the_following_turn() -> None:
    words = [
        _word("before", 0.5, 1.0, space_before=False),
        _word("gap", 3.0, 3.5),
        _word("inside", 6.2, 6.8),
    ]
    transcript = _transcript(_segment(0, words))

    retimed = retime_outside_speech(transcript, [Turn(0.0, 2.0, "S1"), Turn(6.0, 9.0, "S2")])

    timings = [(word.text, word.start) for word in retimed.segments[0].words]
    assert timings == [("before", 0.5), ("gap", 6.0), ("inside", 6.2)]


def test_retime_never_shifts_a_word_further_than_max_shift() -> None:
    word = _word("intro", 0.0, 0.4, space_before=False)
    transcript = _transcript(_segment(0, [word]))

    retimed = retime_outside_speech(
        transcript,
        [Turn(40.0, 45.0, "S1")],
        max_shift_s=30.0,
    )

    assert [(item.start, item.end) for item in retimed.segments[0].words] == [(0.0, 0.4)]


def test_retime_is_a_no_op_when_every_word_is_already_inside_a_turn() -> None:
    transcript = _transcript(
        _segment(0, [_word("hello", 1.0, 1.4, space_before=False), _word("world", 1.4, 1.9)])
    )

    retimed = retime_outside_speech(transcript, [Turn(0.0, 5.0, "S1")])

    assert retimed.to_dict() == transcript.to_dict()
