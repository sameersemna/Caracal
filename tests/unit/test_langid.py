"""Language spans, low-confidence refinement and the report ``languages`` summary."""

from __future__ import annotations

import json

import pytest

from caracal.asr import Segment, Transcript, WhisperResult
from caracal.asr.langid import language_spans, languages_summary, refine_spans


def _segment(index: int, start: float, end: float, *, lang: str | None, prob: float = 0.99) -> Segment:
    return Segment(
        id=index,
        start=start,
        end=end,
        text=f"segment {index}",
        words=(),
        avg_logprob=0.0,
        no_speech_prob=0.0,
        lang=lang,
        lang_prob=prob,
    )


def _transcript(*segments: Segment) -> Transcript:
    return Transcript(segments=segments, language_code=segments[0].lang if segments else None, dropped={})


def test_consecutive_segments_with_the_same_language_form_one_span() -> None:
    transcript = _transcript(
        _segment(0, 0.0, 2.0, lang="en", prob=0.9),
        _segment(1, 2.0, 4.0, lang="en", prob=0.8),
        _segment(2, 4.0, 6.0, lang="de", prob=0.95),
        _segment(3, 6.0, 7.0, lang="en", prob=0.7),
    )

    spans = language_spans(transcript)
    assert [(span.code, span.start, span.end) for span in spans] == [
        ("en", 0.0, 4.0),
        ("de", 4.0, 6.0),
        ("en", 6.0, 7.0),
    ]
    assert [span.confidence for span in spans] == [0.8, 0.95, 0.7]


def test_span_below_the_confidence_floor_becomes_und() -> None:
    transcript = _transcript(
        _segment(0, 0.0, 2.0, lang="en", prob=0.9),
        _segment(1, 2.0, 4.0, lang="de", prob=0.4),
    )

    spans = language_spans(transcript, min_conf=0.5)
    assert [(span.code, span.confidence) for span in spans] == [("en", 0.9), ("und", 0.4)]


def test_unknown_language_is_und() -> None:
    transcript = _transcript(_segment(0, 0.0, 2.0, lang=None, prob=0.99))
    assert [span.code for span in language_spans(transcript)] == ["und"]


async def test_refine_relabels_windows_and_respects_max_calls() -> None:
    transcript = _transcript(
        _segment(0, 10.0, 20.0, lang="en", prob=0.4),
        _segment(1, 40.0, 70.0, lang="en", prob=0.4),
        _segment(2, 80.0, 90.0, lang="en", prob=0.4),
    )
    chunk_results = [
        (
            0.0,
            WhisperResult(
                language_code="en", language_prob=0.4, language_probs={"en": 0.4}, duration=120.0, segments=()
            ),
        )
    ]
    calls: list[tuple[float, float]] = []

    async def recheck(start: float, end: float) -> WhisperResult:
        calls.append((start, end))
        if start < 30.0:
            return WhisperResult(
                language_code="en", language_prob=0.4, language_probs={"en": 0.4}, duration=30.0, segments=()
            )
        return WhisperResult(
            language_code="de", language_prob=0.9, language_probs={"de": 0.9}, duration=30.0, segments=()
        )

    refined = await refine_spans(transcript, chunk_results, recheck=recheck, max_calls=2)

    assert calls == [(0.0, 30.0), (30.0, 60.0)]
    assert [segment.lang for segment in refined.segments] == ["en", "de", "en"]
    assert [segment.lang_prob for segment in refined.segments] == [0.4, 0.9, 0.4]
    assert refined.language_code == "de"


async def test_refine_skips_confident_and_short_chunks() -> None:
    transcript = _transcript(_segment(0, 0.0, 90.0, lang="en", prob=0.9))
    calls: list[tuple[float, float]] = []

    async def recheck(start: float, end: float) -> WhisperResult:
        calls.append((start, end))
        raise AssertionError("no recheck expected")

    confident = WhisperResult(
        language_code="en", language_prob=0.9, language_probs={"en": 0.9}, duration=120.0, segments=()
    )
    short = WhisperResult(
        language_code="en", language_prob=0.3, language_probs={"en": 0.3}, duration=45.0, segments=()
    )
    refined = await refine_spans(transcript, [(0.0, confident), (120.0, short)], recheck=recheck)
    assert refined is transcript
    assert calls == []


def test_languages_summary_shares_sum_to_one() -> None:
    transcript = _transcript(
        _segment(0, 0.0, 10.0, lang="en", prob=0.9),
        _segment(1, 10.0, 15.0, lang="de", prob=0.8),
        _segment(2, 15.0, 20.0, lang="en", prob=0.4),
    )

    summary = languages_summary(transcript)
    assert [item["code"] for item in summary] == ["en", "de", "und"]
    assert [item["name"] for item in summary] == ["English", "German", "Unknown"]
    assert sum(float(item["share"]) for item in summary) == pytest.approx(1.0, abs=1e-6)
    assert [float(item["share"]) for item in summary] == [
        pytest.approx(0.5),
        pytest.approx(0.25),
        pytest.approx(0.25),
    ]
    assert summary[0]["spans"] == [{"start": 0.0, "end": 10.0}]
    assert summary[2]["spans"] == [{"start": 15.0, "end": 20.0}]
    assert [float(item["confidence"]) for item in summary] == [
        pytest.approx(0.9),
        pytest.approx(0.8),
        pytest.approx(0.4),
    ]


def test_empty_transcript_has_no_languages() -> None:
    assert languages_summary(_transcript()) == []


def test_lang_span_serialises_to_json() -> None:
    spans = language_spans(_transcript(_segment(0, 0.0, 2.0, lang="en", prob=0.9)))
    assert json.loads(json.dumps(spans[0].to_dict())) == {
        "code": "en",
        "start": 0.0,
        "end": 2.0,
        "confidence": 0.9,
    }
