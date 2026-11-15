"""Chunk merge: real overlapping chunk pair, filters, word-less segments and renumbering."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import replace

import pytest
from tests.fakes.whisper import load_fixture

from caracal.asr import Segment, Transcript, WhisperResult, Word, join_words
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.errors import CaracalError, ErrorCode

_PAIR_CHUNKS = (Chunk(index=0, start_s=0.0, end_s=61.0), Chunk(index=1, start_s=60.0, end_s=120.0))
_PAIR_CUT = (60.0 + 61.0) / 2.0
# Whisper's known spacing artefacts: a detached contraction suffix or detached punctuation.
_ARTEFACT_RE = re.compile(r" '(re|s|t|ve|ll|d|m)\b| [,.?!;:]")


def _real(name: str) -> WhisperResult:
    return parse_whisper_response(load_fixture(name))


def _word(text: str, start: float, end: float, *, space_before: bool = True) -> Word:
    return Word(text=text, start=start, end=end, prob=0.9, space_before=space_before)


def _segment(
    index: int,
    start: float,
    end: float,
    text: str,
    *,
    words: tuple[Word, ...] = (),
    no_speech: float = 0.0,
    avg_logprob: float = 0.0,
) -> Segment:
    return Segment(
        id=index,
        start=start,
        end=end,
        text=text,
        words=words,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech,
        lang="en",
    )


def _result(segments: tuple[Segment, ...], *, code: str | None = "en", prob: float = 0.99) -> WhisperResult:
    return WhisperResult(
        language_code=code,
        language_prob=prob,
        language_probs={code: prob} if code else {},
        duration=segments[-1].end if segments else 0.0,
        segments=segments,
    )


def _expected_pair_words() -> list[tuple[str, float]]:
    c0 = _real("talk_c0")
    c1 = _real("talk_c1")
    cut = (60.0 + 61.0) / 2.0
    expected: list[tuple[str, float]] = []
    for segment in c0.segments:
        for word in segment.words:
            if (word.start + word.end) / 2.0 < cut:
                expected.append((word.text, round(word.start, 6)))
    for segment in c1.segments:
        for word in segment.words:
            start = word.start + 60.0
            if (start + word.end + 60.0) / 2.0 >= cut:
                expected.append((word.text, round(start, 6)))
    return expected


def test_real_overlap_pair_keeps_each_word_once() -> None:
    chunks = [Chunk(index=0, start_s=0.0, end_s=61.0), Chunk(index=1, start_s=60.0, end_s=120.0)]
    transcript = merge_chunks([_real("talk_c0"), _real("talk_c1")], chunks)

    words = [(word.text, round(word.start, 6)) for segment in transcript.segments for word in segment.words]
    assert words == _expected_pair_words()

    all_words = [word for segment in transcript.segments for word in segment.words]
    assert (
        sum(1 for word in all_words if word.text == "'t" and abs((word.start + word.end) / 2.0 - 60.54) < 0.1)
        == 1
    )
    assert (
        sum(
            1
            for word in all_words
            if word.text == "couldn" and abs((word.start + word.end) / 2.0 - 59.615) < 0.1
        )
        == 1
    )
    assert [segment.id for segment in transcript.segments] == list(range(len(transcript.segments)))
    starts = [segment.start for segment in transcript.segments]
    assert starts == sorted(starts)
    assert transcript.dropped == {"no_speech": 0, "repetition": 0, "empty": 0, "hallucinated_credit": 0}
    assert transcript.language_code == "en"
    assert {segment.lang for segment in transcript.segments} == {"en"}
    assert all(segment.lang_prob > 0.99 for segment in transcript.segments)


def test_segments_are_trimmed_to_their_kept_words() -> None:
    chunks = [Chunk(index=0, start_s=0.0, end_s=10.0), Chunk(index=1, start_s=9.0, end_s=20.0)]
    first = _segment(
        0,
        5.0,
        10.0,
        "alpha beta",
        words=(
            _word("alpha", 5.0, 8.0),  # midpoint 6.5 < 9.5
            _word("beta", 9.0, 10.0),  # midpoint 9.5 >= 9.5 -> belongs to chunk 1
        ),
    )
    second = _segment(
        0,
        0.0,
        2.0,
        "beta gamma",
        words=(
            _word("beta", 0.0, 1.0),  # absolute 9.0-10.0, midpoint 9.5
            _word("gamma", 1.0, 2.0),  # absolute 10.0-11.0
        ),
    )
    transcript = merge_chunks([_result((first,)), _result((second,))], chunks)

    assert [
        (segment.text, round(segment.start, 3), round(segment.end, 3)) for segment in transcript.segments
    ] == [
        ("alpha", 5.0, 8.0),
        ("beta gamma", 9.0, 11.0),
    ]
    assert [word.text for segment in transcript.segments for word in segment.words] == [
        "alpha",
        "beta",
        "gamma",
    ]


def test_no_speech_hallucination_is_dropped() -> None:
    good = _segment(
        0,
        0.0,
        2.0,
        "hello there",
        words=(_word("hello", 0.0, 1.0), _word("there", 1.0, 2.0)),
    )
    hallucination = _segment(1, 2.0, 4.0, "silence text", no_speech=0.95, avg_logprob=-1.5)
    transcript = merge_chunks([_result((good, hallucination))], [Chunk(index=0, start_s=0.0, end_s=4.0)])

    assert [segment.text for segment in transcript.segments] == ["hello there"]
    assert transcript.dropped["no_speech"] == 1
    assert transcript.dropped["empty"] == 0


def test_repeated_text_is_dropped_from_the_third_consecutive_time() -> None:
    segments = tuple(_segment(index, index * 2.0, index * 2.0 + 1.0, "Thank you!") for index in range(4))
    transcript = merge_chunks([_result(segments)], [Chunk(index=0, start_s=0.0, end_s=8.0)])

    assert [segment.text for segment in transcript.segments] == ["Thank you!", "Thank you!"]
    assert transcript.dropped["repetition"] == 2


def test_empty_text_is_dropped() -> None:
    empty = _segment(0, 0.0, 1.0, "   ")
    punctuation_only = _segment(1, 1.0, 2.0, "?", words=(_word("?", 1.0, 2.0),))
    transcript = merge_chunks([_result((empty, punctuation_only))], [Chunk(index=0, start_s=0.0, end_s=2.0)])

    assert [segment.text for segment in transcript.segments] == ["?"]
    assert transcript.dropped == {"no_speech": 0, "repetition": 0, "empty": 1, "hallucinated_credit": 0}


def test_real_chinese_music_intro_credits_are_dropped() -> None:
    """zh_news40 is 40 s of music intro where whisper invents "中文字幕志愿者 李宗盛" twice."""
    transcript = merge_chunks([_real("zh_news40")], [Chunk(index=0, start_s=0.0, end_s=40.0)])

    assert transcript.segments == ()
    assert transcript.dropped["hallucinated_credit"] == 2
    assert transcript.dropped["no_speech"] == 0


def test_real_chinese_speech_keeps_every_segment_but_the_trailing_credit() -> None:
    result = _real("zh_speech40")
    transcript = merge_chunks([result], [Chunk(index=0, start_s=0.0, end_s=result.duration)])

    assert [segment.text for segment in transcript.segments] == [
        "欧洲阿里亚纳火箭从法属圭亚纳的欧洲航天局的发射场发射",
        "携带物资开赴国际空间站",
        "这是欧洲航天计划1979年开始以来的第200次飞行任务",
        "火箭运载的飞行器负责运送重20吨的水、粮食、氧气、零件和实验室工具到国际空间站",
    ]
    assert transcript.dropped["hallucinated_credit"] == 1


_CREDIT_PHRASES = (
    "中文字幕志愿者 李宗盛",
    "中文字幕:CJ",
    "字幕由 Amara.org 社群提供",
    "请不吝点赞 订阅 转发 打赏 支持我们",
    "ご視聴ありがとうございました",
    "Untertitel der Amara.org-Community",
    "Untertitel im Auftrag des ZDF",
    "Untertitelung im Auftrag des ZDF",
    "Sous-titres réalisés par la communauté d'Amara.org",
    "Subtitles by the Amara.org community",
    "Thanks for watching!",
    "Thank you for watching",
)


@pytest.mark.parametrize("phrase", _CREDIT_PHRASES)
def test_credit_phrase_in_the_intro_window_is_dropped(phrase: str) -> None:
    segment = _segment(0, 0.0, 4.0, phrase, words=(_word(phrase, 0.0, 4.0),))
    transcript = merge_chunks([_result((segment,))], [Chunk(index=0, start_s=0.0, end_s=4.0)])

    assert transcript.segments == ()
    assert transcript.dropped["hallucinated_credit"] == 1


@pytest.mark.parametrize("phrase", _CREDIT_PHRASES)
def test_the_same_phrase_in_the_middle_of_a_talk_is_kept(phrase: str) -> None:
    segment = _segment(0, 300.0, 304.0, phrase, words=(_word(phrase, 300.0, 304.0),))
    transcript = merge_chunks([_result((segment,))], [Chunk(index=0, start_s=0.0, end_s=600.0)])

    assert [segment.text for segment in transcript.segments] == [phrase]
    assert transcript.dropped["hallucinated_credit"] == 0


def test_credit_in_the_outro_window_is_dropped() -> None:
    segment = _segment(0, 570.5, 578.0, "Thanks for watching", words=(_word("Thanks", 570.5, 578.0),))
    transcript = merge_chunks([_result((segment,))], [Chunk(index=0, start_s=0.0, end_s=600.0)])

    assert transcript.segments == ()
    assert transcript.dropped["hallucinated_credit"] == 1


def test_credit_inside_a_measured_silence_span_is_dropped_mid_file() -> None:
    segment = _segment(0, 300.0, 304.0, "Subtitles by the Amara.org community")
    transcript = merge_chunks(
        [_result((segment,))],
        [Chunk(index=0, start_s=0.0, end_s=600.0)],
        duration_s=600.0,
        silences=[(298.0, 306.0)],
    )

    assert transcript.segments == ()
    assert transcript.dropped["hallucinated_credit"] == 1


def test_chinese_subscribe_begging_needs_all_three_verbs() -> None:
    only_subscribe = _segment(0, 0.0, 4.0, "请订阅这个频道", words=(_word("请订阅这个频道", 0.0, 4.0),))
    transcript = merge_chunks([_result((only_subscribe,))], [Chunk(index=0, start_s=0.0, end_s=4.0)])

    assert [segment.text for segment in transcript.segments] == ["请订阅这个频道"]
    assert transcript.dropped["hallucinated_credit"] == 0


def test_a_plain_thank_you_is_not_a_credit() -> None:
    """A plain "Thank you!" at the head is the repetition fixture, not a credit phrase."""
    segments = tuple(_segment(index, index * 2.0, index * 2.0 + 1.0, "Thank you!") for index in range(2))
    transcript = merge_chunks([_result(segments)], [Chunk(index=0, start_s=0.0, end_s=4.0)])

    assert [segment.text for segment in transcript.segments] == ["Thank you!", "Thank you!"]
    assert transcript.dropped["hallucinated_credit"] == 0


def test_wordless_segments_are_kept_by_their_own_midpoint() -> None:
    chunks = [Chunk(index=0, start_s=0.0, end_s=5.0), Chunk(index=1, start_s=4.0, end_s=10.0)]
    c0_kept = _segment(0, 3.0, 4.0, "kept by midpoint")
    c0_late = _segment(1, 4.8, 5.0, "too late")
    c1_early = _segment(0, 0.0, 0.4, "too early")
    c1_kept = _segment(1, 0.5, 1.5, "after the cut")
    transcript = merge_chunks([_result((c0_kept, c0_late)), _result((c1_early, c1_kept))], chunks)

    assert [segment.text for segment in transcript.segments] == ["kept by midpoint", "after the cut"]
    assert [(segment.start, segment.end) for segment in transcript.segments] == [(3.0, 4.0), (4.5, 5.5)]


def test_language_code_is_the_one_with_most_speech_time() -> None:
    short_en = _segment(0, 0.0, 2.0, "english here", words=(_word("english", 0.0, 2.0),))
    long_de = _segment(0, 0.0, 9.0, "deutsch hier", words=(_word("deutsch", 0.0, 9.0),))
    chunks = [Chunk(index=0, start_s=0.0, end_s=10.0), Chunk(index=1, start_s=10.0, end_s=20.0)]
    transcript = merge_chunks(
        [_result((short_en,), code="en", prob=0.9), _result((long_de,), code="de", prob=0.8)],
        chunks,
    )

    assert transcript.language_code == "de"
    assert [segment.lang for segment in transcript.segments] == ["en", "de"]


def test_result_and_chunk_count_mismatch_is_internal() -> None:
    with pytest.raises(CaracalError) as excinfo:
        merge_chunks([_result(())], [])
    assert excinfo.value.code is ErrorCode.INTERNAL


def test_empty_inputs_produce_an_empty_transcript() -> None:
    transcript = merge_chunks([], [])
    assert transcript == Transcript(
        segments=(),
        language_code=None,
        dropped={"no_speech": 0, "repetition": 0, "empty": 0, "hallucinated_credit": 0},
    )


def test_models_serialise_to_json() -> None:
    transcript = merge_chunks([_real("jfk")], [Chunk(index=0, start_s=0.0, end_s=11.0)])
    payload = json.loads(json.dumps(transcript.to_dict()))
    assert payload["language_code"] == "en"
    assert payload["segments"][0]["words"][0]["text"] == "And"
    assert payload["dropped"] == {"no_speech": 0, "repetition": 0, "empty": 0, "hallucinated_credit": 0}
    assert payload["segments"][0]["lang_prob"] == pytest.approx(0.9352319240570068)


def _owned_words(
    segment: Segment,
    shift: float,
    lower_cut: float | None,
    upper_cut: float | None,
) -> tuple[Word, ...]:
    """The words this chunk owns for ``segment``, computed independently of ``merge``."""
    shifted = tuple(replace(word, start=word.start + shift, end=word.end + shift) for word in segment.words)
    return tuple(
        word
        for word in shifted
        if (lower_cut is None or (word.start + word.end) / 2.0 >= lower_cut)
        and (upper_cut is None or (word.start + word.end) / 2.0 < upper_cut)
    )


def _pair_sources() -> list[tuple[Segment, float, float | None, float | None]]:
    """(raw segment, shift, lower cut, upper cut) for the recorded two-chunk pair."""
    sources: list[tuple[Segment, float, float | None, float | None]] = []
    for result, chunk in zip([_real("talk_c0"), _real("talk_c1")], _PAIR_CHUNKS, strict=True):
        lower_cut = _PAIR_CUT if chunk.index > 0 else None
        upper_cut = _PAIR_CUT if chunk.index == 0 else None
        sources.extend((segment, chunk.start_s, lower_cut, upper_cut) for segment in result.segments)
    return sources


def test_untrimmed_segments_keep_whisper_text_verbatim() -> None:
    """Every segment that loses no words keeps whisper's own text, punctuation intact."""
    untrimmed = [
        segment
        for segment, shift, lower_cut, upper_cut in _pair_sources()
        if segment.words and len(_owned_words(segment, shift, lower_cut, upper_cut)) == len(segment.words)
    ]
    assert len(untrimmed) == 47  # 49 raw segments; only the two at the 60.5 s seam are trimmed

    transcript = merge_chunks([_real("talk_c0"), _real("talk_c1")], list(_PAIR_CHUNKS))
    merged = Counter(segment.text for segment in transcript.segments)
    verbatim = Counter(segment.text.strip() for segment in untrimmed)

    assert len(transcript.segments) == 49
    assert not verbatim - merged  # every untrimmed text survived verbatim
    assert sum((merged - verbatim).values()) == 2  # only the two seam segments were rebuilt
    assert "A few years ago, I broke into my own house." in merged
    assert "I couldn't go back to my friend Jeff's house for the night" not in merged


def test_merge_does_not_add_punctuation_spacing_artefacts() -> None:
    """The rebuilt text has no more detached punctuation/contractions than whisper's own."""
    raw_texts = [segment.text.strip() for name in ("talk_c0", "talk_c1") for segment in _real(name).segments]
    transcript = merge_chunks([_real("talk_c0"), _real("talk_c1")], list(_PAIR_CHUNKS))

    raw_hits = [text for text in raw_texts if _ARTEFACT_RE.search(text)]
    merged_hits = [segment.text for segment in transcript.segments if _ARTEFACT_RE.search(segment.text)]

    assert raw_hits == ["and ..."]
    assert len(merged_hits) <= len(raw_hits)
    detached = re.compile(r" '(re|s|t|ve|ll|d|m)\b")
    assert not any(detached.search(segment.text) for segment in transcript.segments)


def test_trimmed_segment_is_rebuilt_with_whisper_spacing() -> None:
    """A synthetic seam cut after "'m": only the kept words are joined, keeping "I'm" attached."""
    chunks = [Chunk(index=0, start_s=0.0, end_s=10.0), Chunk(index=1, start_s=7.0, end_s=20.0)]  # cut 8.5
    segment = _segment(
        0,
        5.0,
        10.0,
        "I'm here,",
        words=(
            _word("I", 5.0, 6.0, space_before=False),
            _word("'m", 6.0, 7.0, space_before=False),
            _word("here", 8.0, 9.0, space_before=True),
            _word(",", 9.0, 9.5, space_before=False),
        ),
    )
    transcript = merge_chunks([_result((segment,)), _result(())], chunks)

    assert [(trimmed.text, trimmed.start, trimmed.end) for trimmed in transcript.segments] == [
        ("I'm", 5.0, 7.0)
    ]
    assert [word.text for word in transcript.segments[0].words] == ["I", "'m"]


def test_untrimmed_synthetic_segment_keeps_its_original_text() -> None:
    """No cut at all: the segment text is not rerendered from words."""
    segment = _segment(
        0,
        5.0,
        10.0,
        "I'm here,",
        words=(
            _word("I", 5.0, 6.0, space_before=False),
            _word("'m", 6.0, 7.0, space_before=False),
            _word("here", 8.0, 9.0, space_before=True),
            _word(",", 9.0, 9.5, space_before=False),
        ),
    )
    transcript = merge_chunks([_result((segment,))], [Chunk(index=0, start_s=0.0, end_s=10.0)])

    assert [kept.text for kept in transcript.segments] == ["I'm here,"]


def test_join_words_keeps_contractions_and_attached_punctuation() -> None:
    words = (
        _word("I", 0.0, 0.1, space_before=False),
        _word("'m", 0.1, 0.2, space_before=False),
        _word("here", 0.2, 0.3, space_before=True),
        _word(",", 0.3, 0.4, space_before=False),
    )
    assert join_words(words) == "I'm here,"
    assert join_words(()) == ""


def test_join_words_keeps_the_space_across_a_real_segment_seam() -> None:
    """Cue building joins words across segment boundaries: the seam keeps its space.

    Regression for the old "first word of a segment has no leading space" rule, which
    produced ``totalcatastrophe?`` / ``effects.That's`` at live chunk seams.
    """
    result = _real("talk_c0")
    joins = [
        (left.words[-1], right.words[0], join_words((left.words[-1], right.words[0])))
        for left, right in zip(result.segments, result.segments[1:], strict=False)
    ]

    assert joins[0][2] == ". I"  # segment 0 ends with ".", segment 1 starts with "I"
    assert joins[4][2] == "40 degrees"  # segment 4 ends with "40", segment 5 starts with "degrees"
    for last, first, seam in joins:
        if last.text[-1].isalnum() and first.text[0].isalnum():
            assert seam == f"{last.text} {first.text}"  # never totalcatastrophe / effects.That's


def test_join_words_attaches_a_detached_punctuation_token() -> None:
    """Whisper's isolated "..." token joins the preceding word instead of rendering "and ..."."""
    words = (
        _word("and", 0.0, 0.1, space_before=False),
        _word("...", 0.1, 0.2, space_before=True),
    )
    assert join_words(words) == "and..."


def test_join_words_attaches_a_closing_quote_after_punctuation() -> None:
    """Verdict 13 carry-forward: a cue line started with ``."`` because it got a space."""
    words = [
        _word("He", 0.0, 0.1, space_before=False),
        _word("said", 0.1, 0.2, space_before=True),
        _word('."', 0.2, 0.3, space_before=True),
        _word("Then", 0.3, 0.4, space_before=True),
    ]

    assert join_words(words) == 'He said." Then'
