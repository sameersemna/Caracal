"""Caption parsing (manual + rolling auto) and whisper↔caption agreement on the real talk."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from tests.fakes.whisper import load_fixture

from caracal.asr import Segment, Transcript, WhisperResult, Word
from caracal.asr.captions import CaptionWord, caption_agreement, normalise_word, parse_vtt
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk

CAPTIONS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "captions"
AUTO = CAPTIONS / "talk_en_auto.vtt"
MANUAL = CAPTIONS / "talk_en_manual.vtt"


def _real(name: str) -> WhisperResult:
    return parse_whisper_response(load_fixture(name))


def _talk_transcript() -> Transcript:
    chunks = [Chunk(index=0, start_s=0.0, end_s=61.0), Chunk(index=1, start_s=60.0, end_s=120.0)]
    return merge_chunks([_real("talk_c0"), _real("talk_c1")], chunks)


def _occurrences(haystack: list[str], needle: list[str]) -> int:
    return sum(
        1
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    )


def test_auto_captions_are_word_timed_and_not_duplicated_by_rolling_cues() -> None:
    words = parse_vtt(AUTO.read_text(encoding="utf-8"))
    starts = [word.start for word in words]
    assert all(left < right for left, right in itertools.pairwise(starts))

    tokens = [normalise_word(word.text) for word in words]
    phrase = ["a", "few", "years", "ago", "i", "broke", "into", "my", "own"]
    assert _occurrences(tokens, phrase) == 1

    assert words[0].text == "a"
    assert words[0].start == pytest.approx(13.280)
    own = tokens.index("own")
    assert tokens[own + 1] == "house"


def test_manual_captions_are_spread_evenly_over_each_cue() -> None:
    words = parse_vtt(MANUAL.read_text(encoding="utf-8"))

    assert [word.text for word in words[:10]] == [
        "A",
        "few",
        "years",
        "ago,",
        "I",
        "broke",
        "into",
        "my",
        "own",
        "house.",
    ]
    step = (15.800 - 13.240) / 10
    assert words[0].start == pytest.approx(13.240)
    assert words[0].end == pytest.approx(13.240 + step)
    assert words[9].start == pytest.approx(13.240 + 9 * step)
    assert words[9].end == pytest.approx(15.800)
    starts = [word.start for word in words]
    assert all(left < right for left, right in itertools.pairwise(starts))


def test_notes_styles_regions_and_cue_settings_are_ignored() -> None:
    text = """WEBVTT

NOTE a comment
that spans two lines

STYLE
::cue { color: red }

REGION
id:fred

00:00:01.000 --> 00:00:03.000 align:start position:0%
Hello <i>world</i> &amp; friends
"""
    words = parse_vtt(text)
    assert [word.text for word in words] == ["Hello", "world", "&", "friends"]
    assert [word.start for word in words] == [
        pytest.approx(1.0),
        pytest.approx(1.5),
        pytest.approx(2.0),
        pytest.approx(2.5),
    ]
    assert words[-1].end == pytest.approx(3.0)


def test_tagged_cues_win_over_rolling_plain_cues() -> None:
    text = """WEBVTT

00:00:01.000 --> 00:00:02.000
hello world

00:00:02.000 --> 00:00:04.000
hello<00:00:02.500><c> world</c>
"""
    words = parse_vtt(text)
    assert [(word.text, word.start, word.end) for word in words] == [
        ("hello", 2.0, 2.5),
        ("world", 2.5, 4.0),
    ]


def test_normalise_word_strips_punctuation_music_and_bracket_tags() -> None:
    assert normalise_word("Don't!") == "dont"
    assert normalise_word("[Music]") == ""
    assert normalise_word("♪") == ""
    assert normalise_word("\uff14\uff10\u00b0") == "40"


def test_agreement_with_real_auto_captions_is_at_least_eighty_percent() -> None:
    transcript = _talk_transcript()
    captions = parse_vtt(AUTO.read_text(encoding="utf-8"))

    agreement = caption_agreement(transcript, captions)
    assert agreement is not None
    assert agreement >= 0.80


def test_agreement_with_shuffled_captions_is_low() -> None:
    transcript = _talk_transcript()
    captions = parse_vtt(AUTO.read_text(encoding="utf-8"))
    texts = [word.text for word in captions]
    texts.reverse()
    shuffled = [
        CaptionWord(text=text, start=word.start, end=word.end)
        for text, word in zip(texts, captions, strict=True)
    ]

    agreement = caption_agreement(transcript, shuffled)
    assert agreement is not None
    assert agreement < 0.3


def test_agreement_is_none_when_either_side_has_too_few_words() -> None:
    transcript = Transcript(
        segments=(
            Segment(
                id=0,
                start=0.0,
                end=1.0,
                text="too short",
                words=(Word(text="too", start=0.0, end=0.5, prob=1.0),),
                avg_logprob=0.0,
                no_speech_prob=0.0,
                lang="en",
            ),
        ),
        language_code="en",
        dropped={},
    )
    captions = [
        CaptionWord(text=f"word{index}", start=float(index), end=float(index) + 0.5) for index in range(30)
    ]
    assert caption_agreement(transcript, captions) is None


def test_token_split_transcript_scores_the_same_as_whole_words() -> None:
    halves = [(f"word{index}a", f"word{index}b") for index in range(25)]
    captions = [
        CaptionWord(text=left + right, start=float(index), end=float(index) + 1.0)
        for index, (left, right) in enumerate(halves)
    ]

    whole_tokens = tuple(
        Word(text=left + right, start=float(index), end=float(index) + 1.0, prob=1.0, space_before=True)
        for index, (left, right) in enumerate(halves)
    )
    split_tokens = tuple(
        word
        for index, (left, right) in enumerate(halves)
        for word in (
            Word(text=left, start=float(index), end=float(index) + 0.5, prob=1.0, space_before=True),
            Word(text=right, start=float(index) + 0.5, end=float(index) + 1.0, prob=1.0, space_before=False),
        )
    )

    def transcript(words: tuple[Word, ...]) -> Transcript:
        return Transcript(
            segments=(
                Segment(
                    id=0,
                    start=0.0,
                    end=25.0,
                    text=" ".join(word.text for word in words if word.space_before),
                    words=words,
                    avg_logprob=0.0,
                    no_speech_prob=0.0,
                    lang="en",
                ),
            ),
            language_code="en",
            dropped={},
        )

    whole = caption_agreement(transcript(whole_tokens), captions)
    split = caption_agreement(transcript(split_tokens), captions)

    assert whole == pytest.approx(1.0)
    assert split == pytest.approx(1.0)
    assert split == whole


def test_agreement_is_one_for_identical_words() -> None:
    captions = [
        CaptionWord(text=f"word{index}", start=float(index), end=float(index) + 0.5) for index in range(30)
    ]
    transcript = Transcript(
        segments=(
            Segment(
                id=0,
                start=0.0,
                end=30.0,
                text=" ".join(word.text for word in captions),
                words=tuple(
                    Word(text=word.text, start=word.start, end=word.end, prob=1.0, space_before=True)
                    for word in captions
                ),
                avg_logprob=0.0,
                no_speech_prob=0.0,
                lang="en",
            ),
        ),
        language_code="en",
        dropped={},
    )
    assert caption_agreement(transcript, captions) == pytest.approx(1.0)
