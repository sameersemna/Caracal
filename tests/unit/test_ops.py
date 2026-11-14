"""Operational tooling: doctor checks, retention policy and the metrics registry (docs/05 §3)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from tests.fakes.redis import RecordingRedis, all_keys
from tests.fakes.repo import force_finished_at, force_upload_created_at, make_fake_repos
from tests.repo_contract import Repos

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.ops import (
    DoctorCheck,
    DoctorProbes,
    MetricsRegistry,
    Probe,
    RetentionReport,
    doctor_exit_code,
    emit_checks,
    run_doctor,
    run_retention,
)
from caracal.queue import Keys
from caracal.storage import JobFiles, UploadFiles
from caracal.worker import Worker, WorkerIntervals

MakeSettings = Callable[..., Settings]
_METRIC_LINE = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?\d+(\.\d+)?")


async def _ok(_settings: Settings) -> Probe:
    return "ok", "ok"


def _probes(**overrides: Any) -> DoctorProbes:
    values: dict[str, Any] = {
        "postgres": _ok,
        "redis": _ok,
        "whisper": _ok,
        "ollama": _ok,
        "ffmpeg": _ok,
        "ytdlp": _ok,
        "pyannote": _ok,
        "search": lambda settings, deep: _ok(settings),
        "disk": _ok,
    }
    values.update(overrides)
    return DoctorProbes(**values)


async def test_doctor_all_ok_sets_required_flags_and_exit_zero(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    checks = await run_doctor(settings, _probes())

    assert [check.check for check in checks] == [
        "config",
        "postgres",
        "redis",
        "whisper",
        "ollama",
        "ffmpeg",
        "ytdlp",
        "pyannote",
        "search",
        "disk",
    ]
    assert all(check.status == "ok" for check in checks)
    required = {check.check for check in checks if check.required}
    assert required == {"config", "postgres", "redis", "whisper", "ollama", "ffmpeg"}
    assert doctor_exit_code(checks) == 0


async def test_doctor_required_failure_exits_one(make_settings: MakeSettings) -> None:
    async def down(_settings: Settings) -> Probe:
        return "fail", "connection refused"

    checks = await run_doctor(make_settings(), _probes(postgres=down))

    assert next(check for check in checks if check.check == "postgres").status == "fail"
    assert doctor_exit_code(checks) == 1


@pytest.mark.parametrize(
    ("name", "required"),
    [
        ("postgres", True),
        ("redis", True),
        ("whisper", True),
        ("ollama", True),
        ("ffmpeg", True),
        ("ytdlp", False),
        ("pyannote", False),
        ("disk", False),
    ],
)
async def test_each_probe_failure_is_reported_with_its_required_flag(
    name: str, required: bool, make_settings: MakeSettings
) -> None:
    async def down(_settings: Settings) -> Probe:
        return "fail", f"{name} is down"

    checks = await run_doctor(make_settings(), _probes(**{name: down}))

    check = next(item for item in checks if item.check == name)
    assert check.status == "fail"
    assert check.detail == f"{name} is down"
    assert check.required is required
    assert doctor_exit_code(checks) == (1 if required else 0)


async def test_search_probe_failure_is_reported(make_settings: MakeSettings) -> None:
    async def search(_settings: Settings, _deep: bool) -> Probe:
        return "fail", "exa: HTTP 401"

    checks = await run_doctor(make_settings(), _probes(search=search))

    check = next(item for item in checks if item.check == "search")
    assert (check.status, check.detail, check.required) == ("fail", "exa: HTTP 401", False)
    assert doctor_exit_code(checks) == 0


async def test_a_probe_that_raises_becomes_a_redacted_fail_line(make_settings: MakeSettings) -> None:
    async def explode(_settings: Settings) -> Probe:
        raise RuntimeError("cannot reach postgresql://caracal:hunter2@db:5432/caracal")

    checks = await run_doctor(make_settings(), _probes(redis=explode))

    check = next(item for item in checks if item.check == "redis")
    assert check.status == "fail"
    assert "RuntimeError" in check.detail
    assert "hunter2" not in check.detail
    assert "***" in check.detail


async def test_doctor_optional_failure_and_warning_do_not_change_the_exit_code(
    make_settings: MakeSettings,
) -> None:
    async def fail(_settings: Settings) -> Probe:
        return "fail", "node is missing"

    async def warn(_settings: Settings) -> Probe:
        return "warn", "10 GiB free"

    checks = await run_doctor(make_settings(), _probes(ytdlp=fail, disk=warn))

    assert next(check for check in checks if check.check == "ytdlp").status == "fail"
    assert next(check for check in checks if check.check == "disk").status == "warn"
    assert doctor_exit_code(checks) == 0


async def test_doctor_passes_the_deep_flag_to_the_search_probe(make_settings: MakeSettings) -> None:
    seen: list[bool] = []

    async def search(_settings: Settings, deep: bool) -> Probe:
        seen.append(deep)
        return "ok", f"deep={deep}"

    checks = await run_doctor(make_settings(), _probes(search=search), deep=True)

    assert seen == [True]
    assert next(check for check in checks if check.check == "search").detail == "deep=True"


def test_emit_checks_redacts_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    checks = [
        DoctorCheck(
            "postgres",
            "fail",
            "postgresql://caracal:hunter2@db:5432/caracal hf_abcdefghijklmnopqrstuvwx",
        )
    ]
    emit_checks(checks)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hunter2" not in captured.err
    assert "hf_abcdefghijklmnopqrstuvwx" not in captured.err
    assert "***" in captured.err


def test_emit_checks_json_is_one_object_per_line(capsys: pytest.CaptureFixture[str]) -> None:
    emit_checks([DoctorCheck("redis", "ok", "ready")], json_output=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err) == {"check": "redis", "status": "ok", "detail": "ready"}


def _job_files(data_dir: Path, job_id: Any) -> JobFiles:
    files = JobFiles(data_dir, job_id)
    files.ensure()
    files.write_text(files.audio_wav, "a" * 10)
    files.write_text(files.source_dir / "original.mp3", "s" * 5)
    files.write_text(files.chunk(0), "c" * 3)
    files.write_text(files.transcript_json, "{}")
    files.write_text(files.transcript_txt, "hello")
    files.write_text(files.subtitle("source", "srt"), "srt")
    files.write_text(files.subtitle("source", "vtt"), "vtt")
    files.write_text(files.report_json, "{}")
    files.write_text(files.report_md, "# report")
    files.write_text(files.stage("ingest"), "{}")
    return files


async def _finished_job(
    repos: Repos,
    *,
    finished_at: datetime,
    title: str = "talk",
) -> Any:
    job = await repos.jobs.create(
        source_kind="platform",
        source_url=f"https://example.com/{title}",
        options={},
    )
    await repos.jobs.set_status(job.id, "done")
    force_finished_at(repos, job.id, finished_at)
    return job


async def test_retention_deletes_only_old_job_media(make_settings: MakeSettings, tmp_path: Path) -> None:
    now = datetime(2026, 11, 14, 12, tzinfo=UTC)
    data_dir = tmp_path / "data"
    settings = make_settings(CARACAL_DATA_DIR=data_dir, CARACAL_RETENTION_DAYS=30)
    repos = make_fake_repos()
    old = await _finished_job(repos, finished_at=now - timedelta(days=40), title="old")
    young = await _finished_job(repos, finished_at=now - timedelta(days=1), title="young")
    old_files = _job_files(data_dir, old.id)
    young_files = _job_files(data_dir, young.id)

    report = await run_retention(settings, repos, now=now)

    assert report.media_files == 3
    assert report.media_bytes == 18
    assert report.jobs == 1
    assert not old_files.audio_wav.exists()
    assert not old_files.source_dir.exists()
    assert not old_files.chunks_dir.exists()
    for kept in (
        old_files.transcript_json,
        old_files.transcript_txt,
        old_files.subtitle("source", "srt"),
        old_files.subtitle("source", "vtt"),
        old_files.report_json,
        old_files.report_md,
        old_files.stage("ingest"),
    ):
        assert kept.is_file(), kept
    assert young_files.audio_wav.is_file()
    assert (young_files.source_dir / "original.mp3").is_file()
    assert young_files.chunk(0).is_file()


async def test_retention_dry_run_counts_but_deletes_nothing(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    now = datetime(2026, 11, 14, 12, tzinfo=UTC)
    data_dir = tmp_path / "data"
    settings = make_settings(CARACAL_DATA_DIR=data_dir, CARACAL_RETENTION_DAYS=30)
    repos = make_fake_repos()
    job = await _finished_job(repos, finished_at=now - timedelta(days=40))
    files = _job_files(data_dir, job.id)

    report = await run_retention(settings, repos, now=now, dry_run=True)

    assert report.dry_run is True
    assert (report.media_files, report.media_bytes, report.jobs) == (3, 18, 1)
    assert files.audio_wav.is_file()
    assert (files.source_dir / "original.mp3").is_file()
    assert files.chunk(0).is_file()


async def test_retention_deletes_only_stale_unconsumed_uploads(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    now = datetime(2026, 11, 14, 12, tzinfo=UTC)
    data_dir = tmp_path / "data"
    settings = make_settings(CARACAL_DATA_DIR=data_dir)
    repos = make_fake_repos()

    async def upload(filename: str, age: timedelta) -> Any:
        upload_id = uuid.uuid4()
        files = UploadFiles(data_dir, upload_id)
        files.ensure()
        path = files.file(filename)
        path.write_bytes(b"payload")
        record = await repos.uploads.create(
            upload_id=upload_id,
            filename=filename,
            size_bytes=7,
            sha256="sha",
            path=str(path),
        )
        force_upload_created_at(repos, upload_id, now - age)
        return record, files, path

    stale, stale_files, stale_path = await upload("stale.wav", timedelta(hours=25))
    fresh, _fresh_files, fresh_path = await upload("fresh.wav", timedelta(hours=2))
    consumed, consumed_files, consumed_path = await upload("consumed.wav", timedelta(hours=25))
    job = await repos.jobs.create(source_kind="upload", upload_id=consumed.id, options={})
    await repos.uploads.consume(consumed.id, job.id)

    report = await run_retention(settings, repos, now=now)

    assert report.uploads == 1
    assert report.upload_bytes == 7
    assert await repos.uploads.get(stale.id) is None
    assert not stale_path.exists()
    assert not stale_files.root.exists()
    assert await repos.uploads.get(fresh.id) is not None
    assert fresh_path.is_file()
    assert await repos.uploads.get(consumed.id) is not None
    assert consumed_path.is_file()
    assert consumed_files.root.exists()


async def test_retention_refuses_a_symlink_that_escapes_the_data_dir(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    now = datetime(2026, 11, 14, 12, tzinfo=UTC)
    data_dir = tmp_path / "data"
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"must survive")
    settings = make_settings(CARACAL_DATA_DIR=data_dir, CARACAL_RETENTION_DAYS=30)
    repos = make_fake_repos()
    job = await _finished_job(repos, finished_at=now - timedelta(days=40))
    files = JobFiles(data_dir, job.id)
    files.ensure()
    os.symlink(outside, files.audio_wav)

    with pytest.raises(CaracalError) as excinfo:
        await run_retention(settings, repos, now=now)

    assert excinfo.value.code is ErrorCode.INTERNAL
    assert "outside CARACAL_DATA_DIR" in excinfo.value.message
    assert outside.read_bytes() == b"must survive"


async def test_metrics_render_is_prometheus_text() -> None:
    redis = RecordingRedis()
    keys = Keys("caracal:")
    registry = MetricsRegistry(redis, keys)
    await registry.incr_job("done")
    await registry.incr_job("done")
    await registry.incr_job("failed")
    await registry.observe_stage("transcribe", 11.5)
    await registry.observe_stage("transcribe", 2.5)
    await registry.incr_llm_call("cloud", "tool", False)
    await registry.incr_llm_call("local", "json", True)
    await registry.incr_search_query("brave", cached=False)
    await registry.incr_search_query("brave", cached=True)
    await registry.add_search_cost(0.004)
    await registry.add_whisper_seconds(11.0)

    text = await registry.render()
    lines = text.splitlines()
    assert lines
    for line in lines:
        assert _METRIC_LINE.fullmatch(line), line
    assert 'caracal_jobs_total{status="done"} 2' in lines
    assert 'caracal_jobs_total{status="failed"} 1' in lines
    assert 'caracal_stage_seconds_count{stage="transcribe"} 2' in lines
    assert 'caracal_stage_seconds_sum{stage="transcribe"} 14' in lines
    assert 'caracal_llm_calls_total{endpoint="cloud",mode="tool",fallback="false"} 1' in lines
    assert 'caracal_llm_calls_total{endpoint="local",mode="json",fallback="true"} 1' in lines
    assert 'caracal_search_queries_total{provider="brave",cached="false"} 1' in lines
    assert 'caracal_search_queries_total{provider="brave",cached="true"} 1' in lines
    assert "caracal_search_cost_usd_total 0.004" in lines
    assert "caracal_whisper_audio_seconds_total 11" in lines
    assert all(key.startswith("caracal:") for key in await all_keys(redis))
    await redis.aclose()


async def test_worker_retention_pass_holds_the_singleton_lock(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis = RecordingRedis()
    keys = Keys("t3st:")
    repos = make_fake_repos()
    stop = asyncio.Event()
    seen: list[object] = []

    async def fake_retention(
        settings: Settings,
        repos_: Repos,
        *,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> RetentionReport:
        seen.append(await redis.get(keys.lock_retention()))
        stop.set()
        return RetentionReport(dry_run=False, jobs=0, media_files=0, media_bytes=0, uploads=0, upload_bytes=0)

    monkeypatch.setattr("caracal.worker.run_retention", fake_retention)
    worker = Worker(
        make_settings(REDIS_PREFIX="t3st:"),
        redis,
        repos,
        worker_id="worker-a",
        intervals=WorkerIntervals(retention_s=60.0),
    )
    try:
        await asyncio.wait_for(worker._retention_loop(stop), timeout=2.0)
    finally:
        await worker._http.aclose()
        await redis.aclose()

    assert seen, "retention did not run"
    assert isinstance(seen[0], bytes) and seen[0], "the retention lock was not held during the pass"


async def test_worker_records_job_and_stage_metrics(make_settings: MakeSettings, tmp_path: Path) -> None:
    redis = RecordingRedis()
    keys = Keys("t3st:")
    repos = make_fake_repos()
    job = await repos.jobs.create(source_kind="platform", source_url="https://example.com/talk", options={})
    await repos.jobs.upsert_stage(job.id, "transcribe", status="running")
    await repos.jobs.upsert_stage(job.id, "transcribe", status="done")
    worker = Worker(
        make_settings(CARACAL_DATA_DIR=tmp_path, REDIS_PREFIX="t3st:"),
        redis,
        repos,
        worker_id="worker-a",
        intervals=WorkerIntervals(),
    )
    try:
        await worker._record_metrics(job.id, "done")
        jobs_total = await redis.hget(keys.key("metrics", "jobs_total"), "done")
        stage_count = await redis.hget(keys.key("metrics", "stage_seconds_count"), "transcribe")
    finally:
        await worker._http.aclose()
        await redis.aclose()

    assert jobs_total == b"1"
    assert stage_count == b"1"


async def test_metrics_empty_registry_still_reports_global_totals() -> None:
    redis = RecordingRedis()
    try:
        text = await MetricsRegistry(redis, Keys("caracal:")).render()
    finally:
        await redis.aclose()
    assert text.splitlines() == [
        "caracal_search_cost_usd_total 0",
        "caracal_whisper_audio_seconds_total 0",
    ]


def test_revision_probe_maps_missing_and_stale_revisions_to_the_migrate_hint() -> None:
    from caracal.ops import _NOT_MIGRATED, _revision_probe

    assert _revision_probe(None, "abc") == ("fail", _NOT_MIGRATED)
    assert _revision_probe("old", "new") == ("fail", _NOT_MIGRATED)
    assert _revision_probe("abc", "abc") == (
        "ok",
        "postgres ok; pgvector visible; alembic at abc",
    )


async def test_probe_postgres_reports_an_unmigrated_database_as_one_actionable_line(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlalchemy as sa

    from caracal.ops import _NOT_MIGRATED, _probe_postgres

    class FakeEngine:
        async def dispose(self) -> None:
            return None

    class FailingScope:
        async def __aenter__(self) -> object:
            raise sa.exc.ProgrammingError(
                "SELECT version_num FROM alembic_version",
                {},
                Exception('relation "alembic_version" does not exist'),
            )

        async def __aexit__(self, *_args: object) -> bool:
            return False

    monkeypatch.setattr("caracal.ops.make_engine", lambda _settings: FakeEngine())
    monkeypatch.setattr("caracal.ops.session_scope", lambda _engine: FailingScope())

    assert await _probe_postgres(make_settings()) == ("fail", _NOT_MIGRATED)


async def test_probe_postgres_does_not_hide_an_unrelated_database_error(
    make_settings: MakeSettings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlalchemy as sa

    from caracal.ops import _probe_postgres

    class FakeEngine:
        async def dispose(self) -> None:
            return None

    class FailingScope:
        async def __aenter__(self) -> object:
            raise sa.exc.ProgrammingError("SELECT 1", {}, Exception("permission denied"))

        async def __aexit__(self, *_args: object) -> bool:
            return False

    monkeypatch.setattr("caracal.ops.make_engine", lambda _settings: FakeEngine())
    monkeypatch.setattr("caracal.ops.session_scope", lambda _engine: FailingScope())

    with pytest.raises(sa.exc.ProgrammingError):
        await _probe_postgres(make_settings())


async def test_metrics_records_stage_failures() -> None:
    redis = RecordingRedis()
    try:
        registry = MetricsRegistry(redis, Keys("caracal:"))
        await registry.incr_stage_failure("transcribe")
        await registry.incr_stage_failure("transcribe")
        await registry.incr_stage_failure("diarize")
        lines = (await registry.render()).splitlines()
    finally:
        await redis.aclose()

    assert 'caracal_stage_failures_total{stage="transcribe"} 2' in lines
    assert 'caracal_stage_failures_total{stage="diarize"} 1' in lines
