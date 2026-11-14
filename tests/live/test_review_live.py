"""Live self-review against the real gemma4 (judge-run, never by the gate).

The draft is a TED-like report built from the recorded Levitin transcript. The judge plants
one inconsistency — the summary calls the speaker "Jim Nantz" while ``speakers[0].name`` is
"Daniel Levitin" — and expects the reviewer to notice it. A clean draft must stay quiet.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest

from caracal.analysis.review import MAX_FLAGS, review_report
from caracal.asr import Transcript
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.config import get_settings
from caracal.llm.client import LLMClient
from caracal.report.model import (
    AudioInfo,
    Claim,
    Context,
    QualityInfo,
    Report,
    ReportSource,
    SpeakerIdentity,
    Summary,
    TranscriptRef,
    Verification,
)
from tests.fakes.whisper import load_fixture

pytestmark = pytest.mark.live

JOB_ID = uuid.UUID("6f5c5f4e-0d5b-4a3e-9d1e-2f6b0f3f9a10")


def _ted_transcript() -> Transcript:
    """The recorded two-chunk TED talk, merged exactly as the pipeline merges whisper."""
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


def _draft(*, summary_one_line: str, speaker_name: str) -> Report:
    return Report(
        job_id=JOB_ID,
        created_at=datetime(2026, 11, 14, 12, 0, tzinfo=UTC),
        pipeline_version="0.1.0",
        source=ReportSource(
            kind="platform",
            url="https://www.youtube.com/watch?v=8jPQjjsBbIc",
            title="How to stay calm when you know you'll be stressed",
            uploader="TED",
            channel="TED",
            duration_s=120.0,
            original_format="webm",
            sha256="a3f1" + "0" * 60,
        ),
        audio=AudioInfo(
            duration_s=120.0,
            sample_rate=16000,
            channels=1,
            speech_ratio=0.91,
            music_detected=False,
            quality="good",
        ),
        speakers=[
            SpeakerIdentity(
                label="S1",
                talk_time_s=119.0,
                share=1.0,
                first_seen_s=0.18,
                name=speaker_name,
                confidence=0.75,
                role="speaker",
                description="adult male, calm, tells a personal story",
                evidence=["seg:0"],
            )
        ],
        summary=Summary(
            one_line=summary_one_line,
            short=(
                "The speaker tells the story of breaking into his own house on a cold "
                "Montreal night and turns it into advice about anticipating stress."
            ),
            detailed=(
                "On a cold Montreal night the speaker broke a window to get into his own "
                "house, set off the alarm and was held by the police until a neighbour "
                "confirmed who he was. He uses the story to explain why stress narrows "
                "thinking and how to prepare for stressful moments."
            ),
            evidence=["seg:0", "seg:7"],
        ),
        context=Context(
            genre="talk",
            setting="TED stage",
            audience="general public",
            purpose="explain a practical technique",
            tone=["calm", "humorous"],
            recorded_when="2015",
            recorded_where="TED stage",
            confidence=0.8,
            evidence=["seg:1"],
        ),
        claims=[
            Claim(
                id="c1",
                text="Minus 40 is where the Celsius and Fahrenheit scales meet.",
                speaker="S1",
                seg_ids=["seg:3"],
                kind="fact",
                checkworthy=True,
                evidence=["seg:3"],
            )
        ],
        verification=[
            Verification(
                target="claim:c1",
                verdict="supported",
                confidence=0.9,
                rationale="Multiple reference sources confirm the scales meet at -40.",
            )
        ],
        transcript_ref=TranscriptRef(segments=0, words=0),
        quality=QualityInfo(asr_avg_logprob=-0.1, low_confidence_segments=0),
    )


def _check(result_flags: list, allowed_prefixes: tuple[str, ...]) -> bool:
    return any(
        flag.field == prefix or flag.field.startswith(f"{prefix}.")
        for flag in result_flags
        for prefix in allowed_prefixes
    )


async def test_live_review_flags_a_planted_speaker_inconsistency() -> None:
    settings = get_settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")

    transcript = _ted_transcript()
    draft = _draft(
        summary_one_line="The speaker is sports broadcaster Jim Nantz.",
        speaker_name="Daniel Levitin",
    )
    async with httpx.AsyncClient() as http:
        client = LLMClient(settings, http)
        result = await review_report(draft, transcript=transcript, llm=client)

    assert result.llm_meta.calls == 1
    assert len(result.flags) <= MAX_FLAGS
    assert result.flags, f"the reviewer missed the planted inconsistency: {result.llm_meta}"
    assert _check(result.flags, ("summary", "speakers")), result.flags
    assert not any("seg:" in flag.message for flag in result.flags)


async def test_live_review_is_quiet_on_a_clean_draft() -> None:
    settings = get_settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")

    transcript = _ted_transcript()
    draft = _draft(
        summary_one_line=("Neuroscientist Daniel Levitin explains how to prepare for stressful moments."),
        speaker_name="Daniel Levitin",
    )
    async with httpx.AsyncClient() as http:
        client = LLMClient(settings, http)
        result = await review_report(draft, transcript=transcript, llm=client)

    assert len(result.flags) <= 2, result.flags
