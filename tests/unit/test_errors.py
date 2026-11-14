"""Error codes, HTTP mapping and the error envelope (docs/03 §4)."""

from __future__ import annotations

import pytest

from caracal.errors import HTTP_STATUS, CaracalError, ErrorCode

EXPECTED_STATUS = {
    "unauthorized": 401,
    "invalid_input": 422,
    "not_found": 404,
    "source_blocked": 400,
    "source_unavailable": 424,
    "too_large": 413,
    "unsupported_media": 415,
    "dependency_down": 503,
    "job_not_ready": 409,
    "rate_limited": 429,
    "internal": 500,
}


def test_error_code_values_match_docs_03() -> None:
    assert {code.value for code in ErrorCode} == set(EXPECTED_STATUS)


@pytest.mark.parametrize(("value", "status"), sorted(EXPECTED_STATUS.items()))
def test_http_status_mapping(value: str, status: int) -> None:
    code = ErrorCode(value)
    assert HTTP_STATUS[code] == status


def test_every_error_code_has_a_status() -> None:
    assert set(HTTP_STATUS) == set(ErrorCode)


def test_caracal_error_defaults() -> None:
    error = CaracalError(ErrorCode.NOT_FOUND, "job not found")
    assert isinstance(error, Exception)
    assert error.code is ErrorCode.NOT_FOUND
    assert error.message == "job not found"
    assert error.details == {}
    assert error.retryable is False
    assert str(error) == "job not found"


def test_caracal_error_to_dict() -> None:
    error = CaracalError(
        ErrorCode.SOURCE_BLOCKED,
        "private address",
        {"host": "127.0.0.1", "attempt": 2},
        retryable=True,
    )
    assert error.retryable is True
    assert error.to_dict() == {
        "error": {
            "code": "source_blocked",
            "message": "private address",
            "details": {"host": "127.0.0.1", "attempt": 2},
        }
    }


def test_details_are_copied_not_shared() -> None:
    details: dict[str, object] = {"field": "url"}
    error = CaracalError(ErrorCode.INVALID_INPUT, "bad", details)
    details["field"] = "changed"
    assert error.details == {"field": "url"}


def test_retryable_is_keyword_only() -> None:
    error = CaracalError(ErrorCode.DEPENDENCY_DOWN, "redis down", retryable=True)
    assert error.retryable is True
    with pytest.raises(TypeError):
        CaracalError(ErrorCode.DEPENDENCY_DOWN, "redis down", None, True)  # type: ignore[misc]
