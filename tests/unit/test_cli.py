"""CLI surface: subcommands, stub exit codes and the doctor command."""

from __future__ import annotations

import json

import pytest

from caracal.cli import main
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


@pytest.mark.parametrize("command", ["serve", "worker", "mcp-stdio", "migrate"])
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
