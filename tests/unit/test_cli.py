"""CLI surface: subcommands, stub exit codes and the doctor command."""

from __future__ import annotations

import json
from typing import Any

import pytest

from caracal.cli import main
from caracal.config import get_settings
from caracal.log import get_logger

SUBCOMMANDS = ("serve", "worker", "mcp-stdio", "migrate", "doctor")


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


@pytest.mark.parametrize("command", ["serve", "worker", "mcp-stdio"])
def test_unimplemented_subcommands_exit_two_with_stderr_message(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([command]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"{command}: available from a later milestone\n"


def test_no_command_prints_usage_and_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err.lower()


def test_doctor_reports_valid_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    assert main(["doctor"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    records = _records(captured.err)
    assert any(record["msg"] == "config valid" for record in records)
    assert all(record["level"] != "ERROR" for record in records)
    valid = next(record for record in records if record["msg"] == "config valid")
    assert valid["port"] == 11404
    assert valid["host"] == "127.0.0.1"


def test_doctor_reports_invalid_config_without_leaking_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    monkeypatch.setenv("CARACAL_API_TOKEN", "too-short")
    assert main(["doctor"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    records = _records(captured.err)
    invalid = next(record for record in records if record["msg"] == "config invalid")
    assert invalid["level"] == "ERROR"
    assert any("CARACAL_API_TOKEN" in field for field in invalid["fields"])
    assert "too-short" not in captured.err


def test_doctor_defaults_are_valid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    assert main(["doctor"]) == 0
    assert any(record["msg"] == "config valid" for record in _records(capsys.readouterr().err))


def test_doctor_registers_loaded_secrets_for_exact_match_redaction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _set_required(monkeypatch)
    sample = "exa-registered-secret-value"
    monkeypatch.setenv("EXA_API_KEY", sample)
    assert main(["doctor"]) == 0
    capsys.readouterr()

    get_logger("caracal.test").info("search key %s", sample)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert sample not in captured.err
    assert "***" in captured.err


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
