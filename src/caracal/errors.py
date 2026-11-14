"""Error codes and the exception type used across Caracal (docs/03 §4)."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Stable machine-readable error codes returned by the API and MCP tools."""

    UNAUTHORIZED = "unauthorized"
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    SOURCE_BLOCKED = "source_blocked"
    SOURCE_UNAVAILABLE = "source_unavailable"
    TOO_LARGE = "too_large"
    UNSUPPORTED_MEDIA = "unsupported_media"
    DEPENDENCY_DOWN = "dependency_down"
    JOB_NOT_READY = "job_not_ready"
    INTERNAL = "internal"


HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.INVALID_INPUT: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.SOURCE_BLOCKED: 400,
    ErrorCode.SOURCE_UNAVAILABLE: 424,
    ErrorCode.TOO_LARGE: 413,
    ErrorCode.UNSUPPORTED_MEDIA: 415,
    ErrorCode.DEPENDENCY_DOWN: 503,
    ErrorCode.JOB_NOT_READY: 409,
    ErrorCode.INTERNAL: 500,
}


class CaracalError(Exception):
    """A failure with a stable error code, human message and optional structured details."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, object] | None = None,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, object] = dict(details) if details is not None else {}
        self.retryable = retryable

    def to_dict(self) -> dict[str, object]:
        """Serialise to the API error envelope: ``{"error": {...}}``."""
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "details": self.details,
            }
        }
