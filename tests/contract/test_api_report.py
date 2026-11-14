"""``GET /v1/jobs/{id}/report`` contract (docs/03 §2, §5; card T2.7)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from caracal.pipeline.stages import JobOptions
from caracal.report.model import (
    AudioInfo,
    Context,
    Language,
    LanguageSpan,
    QualityInfo,
    Report,
    ReportSource,
    Summary,
    TranscriptRef,
)
from tests.contract.conftest import AUTH, Harness

URL = "https://example.com/talk"


def _report(job_id: uuid.UUID) -> Report:
    """A small but complete report, exactly the shape the assembler stores."""
    return Report(
        job_id=job_id,
        created_at=datetime(2026, 11, 14, 12, 0, tzinfo=UTC),
        pipeline_version="0.1.0",
        source=ReportSource(
            kind="platform",
            url=URL,
            title="A talk",
            duration_s=12.5,
            original_format="webm",
            sha256="a" * 64,
        ),
        audio=AudioInfo(
            duration_s=12.5,
            sample_rate=16000,
            channels=1,
            loudness_lufs=-18.0,
            speech_ratio=0.9,
            music_detected=False,
            quality="good",
        ),
        languages=[
            Language(
                code="en",
                name="English",
                share=1.0,
                spans=[LanguageSpan(start=0.0, end=12.5)],
                confidence=0.99,
            )
        ],
        summary=Summary(
            one_line="A short talk.",
            short="A short talk about one idea.",
            detailed="A short talk about one idea, with an example.",
            evidence=["seg:0"],
        ),
        context=Context(genre="talk", tone=["calm"], evidence=["seg:0"], confidence=0.7),
        transcript_ref=TranscriptRef(
            segments=1,
            words=4,
            links={
                "json": f"/v1/jobs/{job_id}/transcript?format=json",
                "txt": f"/v1/jobs/{job_id}/transcript?format=txt",
                "srt": f"/v1/jobs/{job_id}/subtitles/source.srt",
                "vtt": f"/v1/jobs/{job_id}/subtitles/source.vtt",
            },
        ),
        quality=QualityInfo(
            asr_avg_logprob=-0.1,
            low_confidence_segments=0,
            caption_agreement=None,
            stages={"assemble": "done"},
        ),
        evidence_index={},
    )


async def _job(h: Harness, *, status: str = "done") -> uuid.UUID:
    record = await h.repos.jobs.create(
        source_kind="platform",
        source_url=URL,
        title="A talk",
        options=JobOptions().model_dump(),
    )
    await h.repos.jobs.set_status(record.id, status)
    return record.id


async def _stored_job(h: Harness) -> uuid.UUID:
    job_id = await _job(h)
    report = _report(job_id)
    await h.repos.reports.put(job_id, report.model_dump(mode="json"), "1.0")
    return job_id


async def test_report_json_is_the_stored_report(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _stored_job(h)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/report", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == _report(job_id).model_dump(mode="json")
    assert response.json()["schema_version"] == "1.0"
    assert response.json()["summary"]["one_line"] == "A short talk."


async def test_report_markdown_is_served_as_utf8_markdown(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _stored_job(h)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/report?format=md", headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/markdown; charset=utf-8"
    assert response.text.startswith("# A talk")
    assert "## Summary" in response.text


async def test_report_before_assembly_is_409_job_not_ready(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h, status="running")

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/report", headers=AUTH)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_ready"


async def test_unknown_job_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{uuid.uuid4()}/report", headers=AUTH)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_failed_job_without_a_report_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h, status="failed")

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/report", headers=AUTH)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_invalid_report_format_is_422_invalid_input(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _stored_job(h)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/report?format=pdf", headers=AUTH)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
