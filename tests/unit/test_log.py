"""JSON logging and secret redaction (docs/06 S8)."""

from __future__ import annotations

import json

import pytest

from caracal.log import configure_logging, get_logger, redact


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Authorization: Bearer abc.def-123", "Authorization: Bearer ***"),
        ("authorization=bearer xyz", "authorization=bearer ***"),
        ("https://api.example/v1?api_key=supersecret&page=2", "https://api.example/v1?api_key=***&page=2"),
        ("token=abc123&next=1", "token=***&next=1"),
        ("key=deadbeef", "key=***"),
        ("api-key=deadbeef", "api-key=***"),
        (
            "postgresql+asyncpg://user:s3cr3t@host:5432/db",
            "postgresql+asyncpg://user:***@host:5432/db",
        ),
        ("redis://:pw@latitude:6379/0", "redis://:***@latitude:6379/0"),
    ],
)
def test_redact_masks_secrets(raw: str, expected: str) -> None:
    assert redact(raw) == expected


def test_redact_leaves_plain_text_untouched() -> None:
    assert redact("job 42 finished in 3 s") == "job 42 finished in 3 s"


def test_redact_is_idempotent() -> None:
    once = redact("Authorization: Bearer abc token=def")
    assert redact(once) == once


def _records(stderr: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in stderr.splitlines() if line.strip()]


def test_logger_writes_one_json_line_to_stderr_and_nothing_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    get_logger("caracal.test").info("job %s started", "j1", extra={"job_id": "j1", "attempt": 3})

    captured = capsys.readouterr()
    assert captured.out == ""
    records = _records(captured.err)
    assert len(records) == 1
    record = records[0]
    assert record["level"] == "INFO"
    assert record["logger"] == "caracal.test"
    assert record["msg"] == "job j1 started"
    assert record["job_id"] == "j1"
    assert record["attempt"] == 3
    assert str(record["ts"]).endswith("Z")


def test_formatter_redacts_message_and_string_extras(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    get_logger("caracal.test").info(
        "request failed with token=abc123",
        extra={"detail": "Authorization: Bearer topsecret"},
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    record = _records(captured.err)[0]
    assert record["msg"] == "request failed with token=***"
    assert record["detail"] == "Authorization: Bearer ***"


def test_formatter_redacts_dsn_in_extra(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    get_logger("caracal.test").info("db", extra={"dsn": "postgresql://u:hunter2@db:5432/caracal"})

    record = _records(capsys.readouterr().err)[0]
    assert record["dsn"] == "postgresql://u:***@db:5432/caracal"


def test_configure_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("DEBUG")
    configure_logging("INFO")
    get_logger("caracal.test").info("once")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(_records(captured.err)) == 1


def test_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING")
    logger = get_logger("caracal.test")
    logger.info("dropped")
    logger.warning("kept")

    records = _records(capsys.readouterr().err)
    assert [record["msg"] for record in records] == ["kept"]


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("LOUD")


# --- verdict 01, attack A2: every leaked secret form must be masked ----------------------


def _emit(
    capsys: pytest.CaptureFixture[str],
    *,
    message: str = "msg",
    extra: dict[str, object] | None = None,
    secrets: tuple[str, ...] = (),
) -> str:
    configure_logging("INFO", secrets=secrets)
    get_logger("caracal.test").info(message, extra={} if extra is None else extra)
    captured = capsys.readouterr()
    assert captured.out == ""
    return captured.err


def _assert_masked(stderr: str, secret: str) -> None:
    assert secret not in stderr
    assert "***" in stderr


def test_a2_extra_api_key_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    _assert_masked(_emit(capsys, extra={"api_key": "api-key-value-111"}), "api-key-value-111")


def test_a2_extra_token_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    _assert_masked(_emit(capsys, extra={"token": "token-value-222"}), "token-value-222")


def test_a2_extra_password_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    _assert_masked(_emit(capsys, extra={"password": "password-value-333"}), "password-value-333")


def test_a2_message_access_token_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    _assert_masked(
        _emit(capsys, message="access_token=access-token-444"),
        "access-token-444",
    )


def test_a2_message_x_api_key_header_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    _assert_masked(_emit(capsys, message="x-api-key: x-api-key-555"), "x-api-key-555")


def test_a2_message_x_subscription_token_header_is_masked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _assert_masked(
        _emit(capsys, message="X-Subscription-Token: subscription-token-666"),
        "subscription-token-666",
    )


def test_a2_hf_token_shape_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "hf_" + "a" * 24
    _assert_masked(_emit(capsys, message=f"token {secret}"), secret)


def test_a2_message_password_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    _assert_masked(_emit(capsys, message="password=password-777"), "password-777")


def test_a2_nested_x_api_key_header_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    _assert_masked(
        _emit(capsys, extra={"headers": {"x-api-key": "nested-api-key-888"}}),
        "nested-api-key-888",
    )


def test_sk_token_shape_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    secret = "sk-" + "b" * 24
    _assert_masked(_emit(capsys, message=f"key {secret}"), secret)


def test_json_name_value_form_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    _assert_masked(
        _emit(capsys, message='{"api_key": "json-secret-999"}'),
        "json-secret-999",
    )


def test_registered_setting_secret_value_is_masked(capsys: pytest.CaptureFixture[str]) -> None:
    sample = "registered-secret-value"
    _assert_masked(
        _emit(capsys, message=f"using {sample}", secrets=(sample,)),
        sample,
    )


def test_registered_secret_shorter_than_eight_characters_is_ignored(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stderr = _emit(capsys, message="value short is fine", secrets=("short",))
    assert "short" in stderr


def test_token_count_extra_is_not_masked(capsys: pytest.CaptureFixture[str]) -> None:
    record = _records(_emit(capsys, message="tokens counted", extra={"token_count": 5}))[0]
    assert record["token_count"] == 5


@pytest.mark.parametrize("key", ["token_usage", "prompt_tokens_count", "token_limit"])
def test_token_metric_suffixes_are_not_masked(capsys: pytest.CaptureFixture[str], key: str) -> None:
    record = _records(_emit(capsys, message="metrics", extra={key: 7}))[0]
    assert record[key] == 7
