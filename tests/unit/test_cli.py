"""CLI surface: subcommands, entry points and the doctor command."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from tests.fakes.redis import RecordingRedis
from tests.fakes.repo import make_fake_repos

from caracal.app import AppDeps
from caracal.cli import _run_worker, _serve, main
from caracal.config import Settings, get_settings
from caracal.db.repo import PostgresJobRepo
from caracal.log import get_logger
from caracal.ops import DoctorCheck, RetentionReport
from caracal.queue import JobQueue, Keys
from caracal.services import JobService

SUBCOMMANDS = ("serve", "worker", "mcp-stdio", "migrate", "doctor", "retention")


def _fake_doctor(
    monkeypatch: pytest.MonkeyPatch,
    checks: list[DoctorCheck],
    calls: list[tuple[bool, bool]] | None = None,
) -> None:
    async def fake_run(settings: Settings, *, deep: bool = False) -> list[DoctorCheck]:
        if calls is not None:
            calls.append((settings.CARACAL_PORT == 11404, deep))
        return checks

    monkeypatch.setattr("caracal.cli.run_doctor", fake_run)


def _ok_checks() -> list[DoctorCheck]:
    return [
        DoctorCheck("config", "ok", "valid", required=True),
        DoctorCheck("postgres", "ok", "postgres ok", required=True),
        DoctorCheck("redis", "ok", "redis ok", required=True),
        DoctorCheck("whisper", "ok", "whisper ok", required=True),
        DoctorCheck("ollama", "ok", "ollama ok", required=True),
        DoctorCheck("ffmpeg", "ok", "ffmpeg ok", required=True),
        DoctorCheck("ytdlp", "ok", "yt-dlp ok"),
        DoctorCheck("pyannote", "warn", "pyannote missing"),
        DoctorCheck("search", "ok", "keys present"),
        DoctorCheck("disk", "ok", "100 GiB free"),
    ]


def _set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARACAL_API_TOKEN", "t" * 32)
    monkeypatch.setenv("DATABASE_URL", "postgresql://caracal:secret@db:5432/caracal")
    monkeypatch.setenv("REDIS_URL", "redis://:secret@redis:6379/0")


def _records(stderr: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in stderr.splitlines() if line.strip()]


def test_help_lists_all_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    for name in SUBCOMMANDS:
        assert name in captured.out


def test_mcp_stdio_builds_the_real_server_and_runs_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Card 15: ``mcp-stdio`` is a real entry point, not a stub (no stdout output)."""
    _set_required(monkeypatch)
    repos = make_fake_repos()
    redis = RecordingRedis()
    keys = Keys("caracal:")
    http_client = httpx.AsyncClient()

    async def db_ping() -> None:
        return None

    deps = AppDeps(
        repos=repos,
        redis=redis,
        http=http_client,
        keys=keys,
        queue=JobQueue(redis, keys),
        db_ping=db_ping,
    )
    built: dict[str, object] = {}

    class FakeEngine:
        async def dispose(self) -> None:
            built["disposed"] = True

    class FakeServer:
        async def run_stdio_async(self) -> None:
            built["ran"] = True

    def fake_build_mcp(service: JobService, settings: Settings) -> FakeServer:
        built["service"] = service
        built["settings"] = settings
        return FakeServer()

    monkeypatch.setattr("caracal.cli._build_real_deps", lambda settings: (deps, FakeEngine()))
    monkeypatch.setattr("caracal.cli.build_mcp", fake_build_mcp)

    assert main(["mcp-stdio"]) == 0
    assert built["ran"] is True
    assert built["disposed"] is True
    service = built["service"]
    assert isinstance(service, JobService)
    assert service.data_dir == get_settings().CARACAL_DATA_DIR
    captured = capsys.readouterr()
    assert captured.out == ""


def test_serve_command_runs_the_server(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    calls: list[Settings] = []

    def fake_serve(settings: Settings) -> int:
        calls.append(settings)
        return 0

    monkeypatch.setattr("caracal.cli._serve", fake_serve)

    assert main(["serve"]) == 0
    assert len(calls) == 1
    assert calls[0].CARACAL_PORT == 11404
    assert calls[0].CARACAL_HOST == "127.0.0.1"
    assert capsys.readouterr().out == ""


def test_serve_passes_uvicorn_the_documented_options(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(CARACAL_HOST="192.0.2.10", CARACAL_PORT=12000)
    app = object()
    built: list[Settings] = []
    run_calls: list[tuple[object, dict[str, Any]]] = []

    def fake_create_app(received: Settings) -> object:
        built.append(received)
        return app

    def fake_run(application: object, **kwargs: Any) -> None:
        run_calls.append((application, kwargs))

    monkeypatch.setattr("caracal.app.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", fake_run)

    assert _serve(settings) == 0
    assert built == [settings]
    assert run_calls == [
        (
            app,
            {
                "host": "192.0.2.10",
                "port": 12000,
                "workers": 1,
                "log_config": None,
                "proxy_headers": False,
            },
        )
    ]


def test_serve_rejects_invalid_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["serve"]) == 1
    records = _records(capsys.readouterr().err)
    invalid = next(record for record in records if record["msg"] == "config invalid")
    assert invalid["level"] == "ERROR"
    assert "CARACAL_API_TOKEN" in str(invalid["fields"])


def test_worker_command_runs_the_worker(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    calls: list[Settings] = []

    async def fake_run_worker(settings: Settings) -> int:
        calls.append(settings)
        return 0

    monkeypatch.setattr("caracal.cli._run_worker", fake_run_worker)

    assert main(["worker"]) == 0
    assert len(calls) == 1
    assert calls[0].CARACAL_PORT == 11404
    assert capsys.readouterr().out == ""


def test_run_worker_builds_datastores_and_stops_cleanly(
    make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings()

    class FakeRedis:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class FakeEngine:
        def __init__(self) -> None:
            self.disposed = False

        async def dispose(self) -> None:
            self.disposed = True

    built: dict[str, Any] = {}

    class FakeWorker:
        def __init__(self, settings: Settings, redis: Any, repos: Any) -> None:
            self.worker_id = "fake-worker"
            built["settings"] = settings
            built["redis"] = redis
            built["repos"] = repos

        async def run(self, stop: asyncio.Event) -> None:
            built["ran"] = True
            stop.set()

    fake_redis = FakeRedis()
    fake_engine = FakeEngine()
    monkeypatch.setattr("caracal.cli.make_redis", lambda settings: fake_redis)
    monkeypatch.setattr("caracal.cli.make_engine", lambda settings: fake_engine)
    monkeypatch.setattr("caracal.cli.Worker", FakeWorker)

    assert asyncio.run(_run_worker(settings)) == 0

    assert built["ran"] is True
    assert built["settings"] is settings
    assert built["redis"] is fake_redis
    assert isinstance(built["repos"].jobs, PostgresJobRepo)
    assert fake_redis.closed is True
    assert fake_engine.disposed is True


def test_no_command_prints_usage_and_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err.lower()


def test_doctor_reports_every_check_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    _fake_doctor(monkeypatch, _ok_checks())
    assert main(["doctor"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 10
    assert lines[0].startswith("config: ok - ")
    assert any(line.startswith("postgres: ok - ") for line in lines)
    assert any(line.startswith("pyannote: warn - ") for line in lines)


def test_doctor_json_output_is_one_object_per_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    _fake_doctor(monkeypatch, _ok_checks())
    assert main(["doctor", "--json"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    records = _records(captured.err)
    assert [record["check"] for record in records][:3] == ["config", "postgres", "redis"]
    assert records[0] == {"check": "config", "status": "ok", "detail": "valid"}


def test_doctor_passes_deep_flag_and_exits_one_on_required_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    calls: list[tuple[bool, bool]] = []
    checks = _ok_checks()
    checks[1] = DoctorCheck("postgres", "fail", "postgres down", required=True)
    _fake_doctor(monkeypatch, checks, calls)
    assert main(["doctor", "--deep"]) == 1
    assert calls == [(True, True)]
    assert "postgres: fail - postgres down" in capsys.readouterr().err


def test_doctor_non_required_failure_does_not_change_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    checks = _ok_checks()
    checks[6] = DoctorCheck("ytdlp", "fail", "node missing")
    checks[9] = DoctorCheck("disk", "warn", "10 GiB free")
    _fake_doctor(monkeypatch, checks)
    assert main(["doctor"]) == 0
    err = capsys.readouterr().err
    assert "ytdlp: fail - node missing" in err
    assert "disk: warn - 10 GiB free" in err


def test_doctor_reports_invalid_config_without_leaking_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CARACAL_API_TOKEN", "too-short")
    assert main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    line = captured.err.splitlines()[0]
    assert line.startswith("config: fail - invalid:")
    assert "CARACAL_API_TOKEN" in line
    assert "too-short" not in captured.err


def test_doctor_defaults_are_valid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    _fake_doctor(monkeypatch, _ok_checks())
    assert main(["doctor"]) == 0
    assert "config: ok" in capsys.readouterr().err


def test_doctor_registers_loaded_secrets_for_exact_match_redaction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    sample = "exa-registered-secret-value"
    monkeypatch.setenv("EXA_API_KEY", sample)
    _fake_doctor(monkeypatch, _ok_checks())
    assert main(["doctor"]) == 0
    capsys.readouterr()

    get_logger("caracal.test").info("search key %s", sample)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert sample not in captured.err
    assert "***" in captured.err


def test_retention_dry_run_prints_counts_and_bytes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    calls: list[tuple[bool, bool]] = []

    async def fake_retention(settings: Settings, repos: Any, *, dry_run: bool = False) -> RetentionReport:
        calls.append((settings.CARACAL_RETENTION_DAYS == 30, dry_run))
        return RetentionReport(
            dry_run=dry_run,
            jobs=2,
            media_files=3,
            media_bytes=1800,
            uploads=1,
            upload_bytes=42,
        )

    monkeypatch.setattr("caracal.cli.run_retention", fake_retention)

    assert main(["retention", "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert calls == [(True, True)]
    assert "would delete 3 job media files (1800 bytes) across 2 expired jobs" in captured.err
    assert "1 uploads (42 bytes)" in captured.err


def test_retention_reports_deletions_without_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    dry_runs: list[bool] = []

    async def fake_retention(settings: Settings, repos: Any, *, dry_run: bool = False) -> RetentionReport:
        dry_runs.append(dry_run)
        return RetentionReport(
            dry_run=dry_run, jobs=0, media_files=0, media_bytes=0, uploads=0, upload_bytes=0
        )

    monkeypatch.setattr("caracal.cli.run_retention", fake_retention)

    assert main(["retention"]) == 0
    assert dry_runs == [False]
    assert "retention: deleted 0 job media files (0 bytes)" in capsys.readouterr().err


def test_migrate_upgrades_head_programmatically(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    calls: list[tuple[Any, str]] = []

    def upgrade(config: Any, revision: str) -> None:
        calls.append((config, revision))

    monkeypatch.setattr("caracal.cli.command.upgrade", upgrade)

    assert main(["migrate"]) == 0

    assert len(calls) == 1
    config, revision = calls[0]
    assert revision == "head"
    script_location = config.get_main_option("script_location")
    assert script_location is not None
    assert script_location.replace("\\", "/").endswith("caracal/migrations")
    records = _records(capsys.readouterr().err)
    assert [record["msg"] for record in records] == ["applying migrations", "migrations applied"]
    assert all(record["level"] != "ERROR" for record in records)


def test_migrate_returns_one_and_logs_redacted_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("DATABASE_URL", "postgresql://caracal:topsecret@db:5432/caracal")
    get_settings.cache_clear()

    def explode(config: object, revision: str) -> None:
        raise RuntimeError("cannot connect to postgresql://caracal:topsecret@db:5432/caracal")

    monkeypatch.setattr("caracal.cli.command.upgrade", explode)

    assert main(["migrate"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    records = _records(captured.err)
    failed = next(record for record in records if record["msg"] == "migration failed")
    assert failed["level"] == "ERROR"
    assert "cannot connect" in str(failed["error"])
    assert "topsecret" not in captured.err


def test_migrate_rejects_invalid_configuration_without_running_alembic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def must_not_run(config: object, revision: str) -> None:
        raise AssertionError("alembic must not run without a valid configuration")

    monkeypatch.setattr("caracal.cli.command.upgrade", must_not_run)

    assert main(["migrate"]) == 1

    records = _records(capsys.readouterr().err)
    invalid = next(record for record in records if record["msg"] == "config invalid")
    assert invalid["level"] == "ERROR"
    assert "DATABASE_URL" in str(invalid["fields"])
