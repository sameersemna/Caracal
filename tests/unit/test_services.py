"""The shared service layer REST and MCP both call (docs/03 §0, card 15)."""

from __future__ import annotations

import asyncio
import base64
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.fakes.redis import RecordingRedis
from tests.fakes.repo import make_fake_repos
from tests.repo_contract import Repos

from caracal.api.schemas import JobSourceIn
from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.pipeline.stages import JobOptions, Stage
from caracal.queue import JobQueue, Keys, publish_event
from caracal.services import MAX_BASE64_DECODED_BYTES, JobService
from caracal.storage import JobFiles

URL = "https://example.com/talk"


def _build(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
) -> tuple[JobService, Repos, RecordingRedis, Path]:
    settings = make_settings()
    repos = make_fake_repos()
    redis = RecordingRedis()
    keys = Keys(settings.REDIS_PREFIX)
    data_dir = tmp_path / "data"
    service = JobService(settings, repos, redis, keys, JobQueue(redis, keys), data_dir)
    return service, repos, redis, data_dir


def _write_artifacts(data_dir: Path, job_id: uuid.UUID) -> JobFiles:
    files = JobFiles(data_dir, job_id)
    files.ensure()
    transcript = {
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 12.5,
                "text": "my fellow americans",
                "words": [],
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
                "lang": "en",
            }
        ],
        "language_code": "en",
        "dropped": {},
    }
    files.write_json(files.transcript_json, transcript)
    files.write_text(files.transcript_txt, "[00:00] my fellow americans\n")
    files.write_json(
        files.stage(Stage.LANGID),
        {
            "status": "done",
            "reason": None,
            "output": {
                "languages": [{"code": "en", "name": "English", "share": 1.0, "confidence": 0.93}],
                "caption_agreement": None,
                "caption_lang": None,
            },
        },
    )
    files.write_json(
        files.stage(Stage.NORMALIZE),
        {"status": "done", "reason": None, "output": {"stats": {"duration_s": 12.5}}},
    )
    files.write_text(files.subtitle("source", "srt"), "1\n00:00:00,000 --> 00:00:12,500\nhello\n")
    files.write_text(files.subtitle("source", "vtt"), "WEBVTT\n\n00:00:00.000 --> 00:00:12.500\nhello\n")
    return files


async def _done_job(repos: Repos) -> uuid.UUID:
    job = await repos.jobs.create(
        source_kind="platform",
        source_url=URL,
        title="A talk",
        options=JobOptions().model_dump(),
    )
    await repos.jobs.set_status(job.id, "done")
    return job.id


async def test_create_job_persists_enqueues_and_returns_links(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, redis, _data_dir = _build(make_settings, tmp_path)

    created = await service.create_job(
        JobSourceIn(url=URL),
        JobOptions(),
        ident_hash="client-a",
    )

    assert created.status == "queued"
    assert created.deduplicated is False
    assert created.links.report == f"http://127.0.0.1:11404/v1/jobs/{created.job_id}/report"
    assert created.links.subtitles == {}

    job = await repos.jobs.get(created.job_id)
    assert job is not None
    assert job.source_kind == "platform"
    assert job.source_url == URL
    assert job.options == JobOptions().model_dump()
    entries = await redis.xrange(service.keys.jobs_stream())
    assert len(entries) == 1
    assert entries[0][1][b"job_id"].decode() == str(created.job_id)
    assert entries[0][1][b"kind"].decode() == "pipeline"


async def test_create_job_deduplicates_a_recent_done_job(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, _redis, _data_dir = _build(make_settings, tmp_path)
    existing = await repos.jobs.create(
        source_kind="platform",
        source_url=URL,
        options=JobOptions().model_dump(),
    )
    await repos.jobs.set_status(existing.id, "done")

    created = await service.create_job(JobSourceIn(url=URL), JobOptions(), ident_hash="client-a")

    assert created.deduplicated is True
    assert created.job_id == existing.id
    assert created.status == "done"
    jobs, _ = await repos.jobs.list()
    assert len(jobs) == 1


async def test_create_job_rejects_a_consumed_upload(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, _redis, _data_dir = _build(make_settings, tmp_path)
    upload = await repos.uploads.create(filename="a.wav", size_bytes=1, sha256="sha", path="/x/a.wav")
    other = await repos.jobs.create(source_kind="upload", upload_id=upload.id, options={})
    await repos.uploads.consume(upload.id, other.id)

    with pytest.raises(CaracalError) as excinfo:
        await service.create_job(
            JobSourceIn(upload_id=upload.id),
            JobOptions(),
            ident_hash="client-a",
        )
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


async def test_base64_oversize_is_rejected_before_decoding(
    make_settings: Callable[..., Settings],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _repos, _redis, data_dir = _build(make_settings, tmp_path)
    calls: list[str] = []

    def spy(data: str, *args: Any, **kwargs: Any) -> bytes:
        calls.append(data)
        raise AssertionError("b64decode must not be reached for an oversize payload")

    monkeypatch.setattr("caracal.services.base64.b64decode", spy)
    oversize = "A" * (MAX_BASE64_DECODED_BYTES * 4 // 3 + 8)

    with pytest.raises(CaracalError) as excinfo:
        await service.create_upload_from_base64(oversize, "big.wav")

    assert excinfo.value.code is ErrorCode.TOO_LARGE
    assert excinfo.value.details["max_decoded_bytes"] == MAX_BASE64_DECODED_BYTES
    assert calls == []
    assert not (data_dir / "uploads").exists()


async def test_base64_invalid_and_valid_payloads(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, _redis, data_dir = _build(make_settings, tmp_path)

    with pytest.raises(CaracalError) as excinfo:
        await service.create_upload_from_base64("not base64!!", "x.wav")
    assert excinfo.value.code is ErrorCode.INVALID_INPUT

    wav = b"RIFF$\x00\x00\x00WAVEfmt "
    created = await service.create_upload_from_base64(base64.b64encode(wav).decode(), "../x.wav")
    assert created.filename == "x.wav"
    assert created.size_bytes == len(wav)
    record = await repos.uploads.get(created.upload_id)
    assert record is not None
    assert Path(record.path).read_bytes() == wav  # noqa: ASYNC240 - test-local file
    assert data_dir / "uploads" in Path(record.path).parents


async def test_transcript_and_subtitles_share_artifact_semantics(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, _redis, data_dir = _build(make_settings, tmp_path)
    job_id = await _done_job(repos)
    _write_artifacts(data_dir, job_id)

    payload = await service.transcript(job_id, "json")
    assert isinstance(payload, dict)
    assert payload["job_id"] == str(job_id)
    assert payload["language_code"] == "en"
    assert [language["code"] for language in payload["languages"]] == ["en"]

    text = await service.transcript(job_id, "txt")
    assert text == "[00:00] my fellow americans\n"

    srt = await service.transcript(job_id, "srt")
    assert isinstance(srt, str)
    assert srt.startswith("1\n00:00:00,000 --> ")

    content, media_type = await service.subtitles(job_id, "source", "vtt")
    assert media_type == "text/vtt; charset=utf-8"
    assert content.startswith(b"WEBVTT")

    with pytest.raises(CaracalError) as excinfo:
        await service.subtitles(job_id, "de", "srt")
    assert excinfo.value.code is ErrorCode.NOT_FOUND


async def test_transcript_missing_while_running_is_job_not_ready(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, _redis, _data_dir = _build(make_settings, tmp_path)
    job = await repos.jobs.create(source_kind="platform", source_url=URL, options={})

    with pytest.raises(CaracalError) as excinfo:
        await service.transcript(job.id, "json")
    assert excinfo.value.code is ErrorCode.JOB_NOT_READY


async def test_analyze_summary_reads_duration_languages_and_preview(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, _redis, data_dir = _build(make_settings, tmp_path)
    job_id = await _done_job(repos)
    _write_artifacts(data_dir, job_id)

    summary = await service.analyze_summary(job_id)

    assert summary.title == "A talk"
    assert summary.duration_s == 12.5
    assert [language["code"] for language in summary.languages] == ["en"]
    assert summary.transcript_preview == "[00:00] my fellow americans\n"


async def test_wait_for_job_reports_progress_and_returns_done(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, redis, _data_dir = _build(make_settings, tmp_path)
    job = await repos.jobs.create(source_kind="platform", source_url=URL, options={})
    seen: list[float] = []

    async def on_progress(progress: Any) -> None:
        seen.append(progress.pct)

    waiter = asyncio.create_task(service.wait_for_job(job.id, 2.0, on_progress))
    await asyncio.sleep(0.02)
    await publish_event(
        redis,
        service.keys,
        job.id,
        {"type": "stage", "stage": "transcribe", "status": "running", "pct": 40, "message": "asr"},
    )
    await asyncio.sleep(0.02)
    await repos.jobs.set_status(job.id, "done")
    await publish_event(redis, service.keys, job.id, {"type": "end"})

    status = await asyncio.wait_for(waiter, timeout=2.0)
    assert status.status == "done"
    assert seen == [40.0]


async def test_wait_for_job_returns_the_current_status_on_timeout(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, _redis, _data_dir = _build(make_settings, tmp_path)
    job = await repos.jobs.create(source_kind="platform", source_url=URL, options={})
    await repos.jobs.set_progress(job.id, {"pct": 25, "stage": "transcribe", "message": "asr"})

    status = await service.wait_for_job(job.id, 0.05)

    assert status.status == "queued"
    assert status.progress.pct == 25
    assert status.progress.stage == "transcribe"


async def test_cancel_or_delete_active_and_terminal(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, redis, data_dir = _build(make_settings, tmp_path)
    active = await repos.jobs.create(source_kind="platform", source_url=URL, options={})

    cancelled = await service.cancel_or_delete(active.id, delete_artifacts=False)
    assert cancelled is not None
    assert cancelled.status == "cancelling"
    assert await redis.exists(service.keys.cancel(active.id)) == 1

    done_id = await _done_job(repos)
    files = _write_artifacts(data_dir, done_id)
    assert files.root.is_dir()

    kept = await service.cancel_or_delete(done_id, delete_artifacts=False)
    assert kept is not None
    assert kept.status == "done"
    assert await repos.jobs.get(done_id) is not None

    deleted = await service.cancel_or_delete(done_id, delete_artifacts=True)
    assert deleted is None
    assert await repos.jobs.get(done_id) is None
    assert not files.root.exists()


async def test_list_jobs_filters_and_rejects_an_unknown_status(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    service, repos, _redis, _data_dir = _build(make_settings, tmp_path)
    first = await repos.jobs.create(source_kind="platform", source_url="https://e/1", options={})
    second = await repos.jobs.create(source_kind="direct", source_url="https://e/2", options={})
    await repos.jobs.set_status(second.id, "failed")

    page = await service.list_jobs("failed", 20, None)
    assert [item.job_id for item in page.items] == [second.id]
    assert page.items[0].source_kind == "direct"

    everything = await service.list_jobs(None, 20, None)
    assert {item.job_id for item in everything.items} == {first.id, second.id}

    with pytest.raises(CaracalError) as excinfo:
        await service.list_jobs("exploded", 20, None)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.details["allowed"] == [
        "queued",
        "running",
        "done",
        "failed",
        "cancelled",
    ]
