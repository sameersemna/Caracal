"""Live planner + content analysis against the real gemma4 (judge-run, never by the gate).

The transcript is the recorded two-chunk TED talk (``talk_c0`` + ``talk_c1``) merged exactly as
the pipeline merges whisper results, so the evidence ids and timestamps are the real ones.
"""

from __future__ import annotations

import httpx
import pytest

from caracal.analysis.content import ContentAnalysis, analyse_content
from caracal.analysis.planner import PlanInputs, plan
from caracal.asr import Segment, Transcript
from caracal.asr.langid import languages_summary
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.config import get_settings
from caracal.llm.client import LLMClient
from caracal.pipeline.stages import JobOptions
from tests.fakes.whisper import load_fixture

pytestmark = pytest.mark.live

_META = {
    "title": "How to stay calm when you know you'll be stressed",
    "channel": "TED",
}
_HINTS = ["TED talk"]


def _ted_transcript() -> Transcript:
    chunks = [
        Chunk(index=0, start_s=0.0, end_s=61.0),
        Chunk(index=1, start_s=60.0, end_s=120.0),
    ]
    return merge_chunks(
        [
            parse_whisper_response(load_fixture("talk_c0")),
            parse_whisper_response(load_fixture("talk_c1")),
        ],
        chunks,
    )


def _music_transcript() -> Transcript:
    segments = tuple(
        Segment(
            id=index,
            start=float(index) * 10.0,
            end=float(index + 1) * 10.0,
            text="🎵🎵🎵",
            words=(),
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            lang="nn",
            lang_prob=0.9,
        )
        for index in range(6)
    )
    return Transcript(segments=segments, language_code="nn", dropped={})


def _allowed(transcript: Transcript) -> set[str]:
    allowed = {f"seg:{segment.id}" for segment in transcript.segments}
    allowed.update(f"meta:{key}" for key in _META)
    allowed.update(f"hint:{index}" for index in range(len(_HINTS)))
    return allowed


def _assert_evidence_resolves(values: list[str], allowed: set[str]) -> None:
    assert values, "an analysed item must cite evidence"
    assert set(values) <= allowed, values


async def test_live_planner_and_content_on_the_merged_ted_transcript() -> None:
    settings = get_settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")

    transcript = _ted_transcript()
    duration = max(segment.end for segment in transcript.segments)
    async with httpx.AsyncClient() as http:
        client = LLMClient(settings, http)
        planned = await plan(
            PlanInputs(
                duration=duration,
                speech_ratio=0.95,
                options=JobOptions(depth="standard"),
                transcript=transcript,
                languages=languages_summary(transcript),
                metadata=_META,
                diarizer_available=True,
                search_configured=True,
                default_target_langs=settings.CARACAL_DEFAULT_TARGET_LANGS,
                search_max_queries=settings.SEARCH_MAX_QUERIES_PER_JOB,
                report_language=settings.CARACAL_REPORT_LANGUAGE,
            ),
            llm=client,
        )
        analysis = await analyse_content(
            transcript,
            _META,
            _HINTS,
            JobOptions(depth="standard"),
            settings,
            llm=client,
        )

    assert planned.music_detected is False
    assert planned.run["content"] is True
    assert planned.content_profile.likely_genre.strip()

    assert isinstance(analysis, ContentAnalysis)
    assert analysis.summary.one_line.strip()
    assert analysis.context.genre.strip()
    assert len(analysis.entities) >= 3, analysis.entities
    assert any(claim.checkworthy for claim in analysis.claims), analysis.claims
    assert set(analysis.llm_meta.completeness) == {"chapters", "entities", "claims", "topics"}
    assert analysis.llm_meta.completeness["entities"] == len(analysis.entities)
    for section in ("chapters", "entities", "claims", "topics"):
        if not getattr(analysis, section):
            assert section in analysis.llm_meta.reasked

    allowed = _allowed(transcript)
    for entity in analysis.entities:
        _assert_evidence_resolves(entity.evidence, allowed)
    for claim in analysis.claims:
        _assert_evidence_resolves(claim.evidence, allowed)
    for topic in analysis.topics:
        _assert_evidence_resolves(topic.evidence, allowed)
    for chapter in analysis.chapters:
        _assert_evidence_resolves(chapter.evidence, allowed)
    _assert_evidence_resolves(analysis.summary.evidence, allowed)
    _assert_evidence_resolves(analysis.context.evidence, allowed)


async def test_live_music_rule_on_a_synthetic_music_transcript() -> None:
    settings = get_settings()
    transcript = _music_transcript()

    planned = await plan(
        PlanInputs(
            duration=60.0,
            speech_ratio=1.0,
            options=JobOptions(depth="standard"),
            transcript=transcript,
            languages=languages_summary(transcript),
            metadata={},
            diarizer_available=True,
            search_configured=True,
            default_target_langs=settings.CARACAL_DEFAULT_TARGET_LANGS,
            report_language=settings.CARACAL_REPORT_LANGUAGE,
        ),
        llm=None,
    )

    assert planned.music_detected is True
    # The plan runs after diarization, so it never skips the diarize stage.
    assert planned.run["diarize"] is True
    assert "diarize" not in planned.reasons
    assert planned.run["align"] is False
    assert planned.run["voiceprint"] is False
    assert planned.run["identity"] is False
    assert planned.run["verify"] is False
    assert planned.run["translate"] is False
    assert planned.reasons["align"] == "music detected"
    assert planned.languages is not None
    assert {language["code"] for language in planned.languages} == {"und"}


async def test_live_map_reduce_with_a_small_context_budget_is_complete() -> None:
    settings = get_settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")
    settings = settings.model_copy(update={"LLM_CONTEXT_CHARS": 1500})

    transcript = _ted_transcript()
    async with httpx.AsyncClient() as http:
        client = LLMClient(settings, http)
        analysis = await analyse_content(
            transcript,
            _META,
            _HINTS,
            JobOptions(depth="standard"),
            settings,
            llm=client,
        )

    assert analysis.summary.one_line.strip()
    assert analysis.context.genre.strip()
    assert analysis.llm_meta.calls >= 2
    # The deterministic merge plus the summary/context call and completeness re-ask must
    # deliver every section, not the empty reduce reply verdict 25 measured.
    assert set(analysis.llm_meta.completeness) == {"chapters", "entities", "claims", "topics"}
    assert all(count > 0 for count in analysis.llm_meta.completeness.values()), analysis.llm_meta
