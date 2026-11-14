"""LLM self-review of a draft report (docs/04 §11, card T3.4).

The review is deliberately narrow: it consumes a report view with no evidence payload plus a
sampled transcript, and emits at most ten validated flags. These tests pin the sampling, the
field-path rule, message cleaning, the cap and the prompt's JSON-mode shape against a scripted
Ollama; the real-model behaviour is covered by ``tests/live/test_review_live.py``.
"""

from __future__ import annotations

import itertools
import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from tests.fakes.ollama import FakeOllama, json_response

from caracal.analysis.review import (
    MAX_FLAGS,
    MAX_MESSAGE_CHARS,
    SAMPLE_LINES,
    ReviewResult,
    clean_message,
    review_report,
    sample_segments,
    valid_field_path,
)
from caracal.asr import Segment, Transcript
from caracal.config import Settings
from caracal.llm.client import LLMClient
from caracal.report.model import (
    AudioInfo,
    Claim,
    Context,
    QualityInfo,
    Report,
    ReportSource,
    ReviewFlag,
    SpeakerIdentity,
    Summary,
    TranscriptRef,
    Verification,
)

JOB_ID = uuid.UUID("6f5c5f4e-0d5b-4a3e-9d1e-2f6b0f3f9a10")
CREATED_AT = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def _segment(seg_id: int, text: str = "words", speaker: str = "S1") -> Segment:
    return Segment(
        id=seg_id,
        start=seg_id * 2.0,
        end=seg_id * 2.0 + 1.5,
        text=text,
        words=(),
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        lang="en",
        speaker=speaker,
    )


def _transcript(count: int = 40) -> Transcript:
    return Transcript(
        segments=tuple(_segment(index) for index in range(count)),
        language_code="en",
        dropped={},
    )


def _report(
    *,
    one_line: str = "Neuroscientist Daniel Levitin explains how to prepare for stress.",
    speaker_name: str = "Daniel Levitin",
) -> Report:
    return Report(
        job_id=JOB_ID,
        created_at=CREATED_AT,
        pipeline_version="0.1.0",
        source=ReportSource(
            kind="platform",
            title="How to stay calm when you know you'll be stressed",
            channel="TED",
            duration_s=80.0,
            original_format="webm",
            sha256="a3f1" + "0" * 60,
        ),
        audio=AudioInfo(
            duration_s=80.0,
            sample_rate=16000,
            channels=1,
            speech_ratio=0.91,
            music_detected=False,
            quality="good",
        ),
        speakers=[
            SpeakerIdentity(
                label="S1",
                talk_time_s=79.0,
                share=1.0,
                first_seen_s=0.1,
                name=speaker_name,
                confidence=0.75,
                role="speaker",
                description="adult male, calm",
                evidence=["seg:0"],
            )
        ],
        summary=Summary(
            one_line=one_line,
            short="Levitin tells a story about breaking into his own house.",
            detailed="Levitin turns a cold Montreal night into advice about stress.",
            evidence=["seg:0"],
        ),
        context=Context(
            genre="talk",
            setting="stage",
            audience="general public",
            purpose="explain",
            tone=["calm"],
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
                verdict="contradicted",
                confidence=0.8,
                rationale="Sources disagree.",
            )
        ],
        transcript_ref=TranscriptRef(segments=40, words=40),
        quality=QualityInfo(asr_avg_logprob=-0.1, low_confidence_segments=0),
        evidence_index={"seg:0": {"kind": "seg", "ref": "0", "excerpt": "words"}},
    )


def test_sample_segments_includes_first_and_last_evenly_spaced() -> None:
    segments = _transcript(100).segments

    sampled = sample_segments(segments)

    assert len(sampled) == SAMPLE_LINES
    assert sampled[0].id == 0
    assert sampled[-1].id == 99
    ids = [segment.id for segment in sampled]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
    assert max(right - left for left, right in itertools.pairwise(ids)) <= 5


def test_sample_segments_keeps_a_short_transcript_whole() -> None:
    segments = _transcript(12).segments

    assert sample_segments(segments) == list(segments)
    assert sample_segments(()) == []


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("summary", True),
        ("speakers", True),
        ("quality.caption_agreement", True),
        ("speakers.S1.name", True),
        ("claims.claim_3", True),
        ("Summary", False),
        ("transcript.text", False),
        ("nonsense", False),
        ("speakers..name", False),
        ("speakers.S1.", False),
        ("", False),
    ],
)
def test_valid_field_path(field: str, expected: bool) -> None:
    assert valid_field_path(field) is expected


def test_clean_message_strips_evidence_ids_and_truncates() -> None:
    cleaned = clean_message("The summary contradicts claim c1 [seg:12, web:3] but meta:title agrees.")

    assert "seg:12" not in cleaned
    assert "web:3" not in cleaned
    assert "meta:title" not in cleaned
    assert cleaned.startswith("The summary contradicts claim c1")
    assert "\n" not in clean_message("one\ntwo   three")
    assert len(clean_message("x" * 500)) == MAX_MESSAGE_CHARS


def _client(settings: Settings, fake: FakeOllama) -> tuple[LLMClient, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=fake.transport())
    return LLMClient(settings, http), http


async def test_review_report_uses_json_mode_validates_dedupes_and_caps_flags(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    transcript = _transcript(40)
    raw_flags = [
        {
            "severity": "warn",
            "field": "summary",
            "message": "Summary says X but the claim is contradicted [seg:3].",
        },
        {
            "severity": "warn",
            "field": "summary",
            "message": "Summary says X but the claim is contradicted [seg:3].",
        },
        {
            "severity": "error",
            "field": "speakers.S1.name",
            "message": "The transcript names a different speaker.",
        },
        {"severity": "info", "field": "transcript.text", "message": "Unknown section, dropped."},
        {"severity": "info", "field": "summary", "message": "y" * 400},
    ]
    for index in range(12):
        raw_flags.append(
            {
                "severity": "info",
                "field": "entities",
                "message": f"entity note {index}",
            }
        )
    fake = FakeOllama([json_response({"flags": raw_flags})])
    client, http = _client(settings, fake)
    try:
        result = await review_report(_report(), transcript=transcript, llm=client)
    finally:
        await http.aclose()

    assert isinstance(result, ReviewResult)
    assert 0 < len(result.flags) <= MAX_FLAGS
    assert all(isinstance(flag, ReviewFlag) for flag in result.flags)
    assert all(valid_field_path(flag.field) for flag in result.flags)
    assert all(len(flag.message) <= MAX_MESSAGE_CHARS for flag in result.flags)
    assert not any("seg:" in flag.message for flag in result.flags)
    assert len({(flag.field, flag.message) for flag in result.flags}) == len(result.flags)
    assert result.llm_meta.raw_flags == len(raw_flags)
    assert result.llm_meta.dropped_flags > 0
    assert result.llm_meta.kept_flags == len(result.flags)

    body = fake.body()
    assert body["format"] == "json"
    prompt = body["messages"][1]["content"]
    report_block = prompt.split("<<<DATA:report>>>", 1)[1].split("<<<END:report>>>", 1)[0]
    assert json.loads(report_block)["summary"]["one_line"].startswith("Neuroscientist")
    assert "evidence" not in json.dumps(json.loads(report_block))
    assert "evidence_index" not in report_block
    assert "key_quotes" not in report_block
    assert "sources" not in report_block
    transcript_block = prompt.split("<<<DATA:transcript>>>", 1)[1].split("<<<END:transcript>>>", 1)[0]
    lines = transcript_block.strip().splitlines()
    assert len(lines) == SAMPLE_LINES
    assert lines[0].startswith("[seg:0]")
    assert lines[-1].startswith("[seg:39]")


async def test_review_report_keeps_an_empty_clean_review(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    fake = FakeOllama([json_response({"flags": []})])
    client, http = _client(settings, fake)
    try:
        result = await review_report(_report(), transcript=_transcript(5), llm=client)
    finally:
        await http.aclose()

    assert result.flags == []
    assert result.llm_meta.calls == 1
    assert result.llm_meta.sampled == 5
