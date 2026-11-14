"""Structured JSON logging and secret redaction (docs/06 S8).

Logs go to **stderr** only: stdout is the MCP stdio channel. Messages and every string
extra are passed through :func:`redact` before they are serialised.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime

_DSN_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)(?P<user>[^:/@\s]*):(?P<password>[^@\s]+)@")
_SCHEME_RE = re.compile(r"(?i)\b(bearer|basic|digest)(\s+)([^\s,;\"']+)")
_SECRET_KEY_MARKERS = (
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "api-key",
    "apikey",
    "cookie",
)
_TOKEN_COUNT_SUFFIXES = ("_count", "_counts", "_usage", "_limit")
_CREDENTIAL_NAME = r"\"?(?:[A-Za-z0-9_.\-]*?(?:token|key|secret|password)|authorization|cookie)\"?"
_NAME_VALUE_RE = re.compile(
    rf"(?i)(?P<prefix>{_CREDENTIAL_NAME}\s*[:=]\s*)"
    r"(?P<value>(?!bearer\b|basic\b|digest\b)(?:\"[^\"]*\"|'[^']*'|[^\s,;&)}}\]]+))"
)
_TOKEN_SHAPE_RE = re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,})")

_STANDARD_ATTRS = frozenset(logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}

_HANDLER: logging.Handler | None = None
_REGISTERED_SECRETS: tuple[str, ...] = ()


def _key_is_secret(key: str) -> bool:
    """True when an extra/dict key names a credential (docs/06 S8a)."""
    lowered = key.strip().lower()
    if lowered == "key":
        return True
    for marker in _SECRET_KEY_MARKERS:
        if marker not in lowered:
            continue
        # ``token_count`` / ``token_usage`` are metrics, not credentials.
        if marker == "token" and lowered.endswith(_TOKEN_COUNT_SUFFIXES):
            continue
        return True
    return False


def _mask_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return f"{value[0]}***{value[0]}"
    return "***"


def redact(value: str) -> str:
    """Mask DSN passwords, auth schemes, ``name=value``/``name: value`` secrets, token shapes
    and every registered Settings secret value."""
    result = _DSN_RE.sub(
        lambda match: f"{match.group('scheme')}{match.group('user')}:***@",
        value,
    )
    result = _SCHEME_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}***",
        result,
    )
    result = _NAME_VALUE_RE.sub(
        lambda match: f"{match.group('prefix')}{_mask_value(match.group('value'))}",
        result,
    )
    result = _TOKEN_SHAPE_RE.sub("***", result)
    for secret in _REGISTERED_SECRETS:
        result = result.replace(secret, "***")
    return result


def _redact_nested(value: object) -> object:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {
            key: "***" if _key_is_secret(str(key)) else _redact_nested(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_nested(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON line with ``ts``, ``level``, ``logger`` and ``msg``."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = "***" if _key_is_secret(key) else _redact_nested(value)
        if record.exc_info:
            payload["exc_info"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.strip().upper())
    if resolved is None:
        raise ValueError(f"unknown log level: {level!r}")
    return resolved


def configure_logging(level: str | int = "INFO", secrets: Iterable[str] = ()) -> None:
    """Install the JSON stderr handler on the root logger (idempotent).

    ``secrets`` registers exact values (Settings ``SecretStr`` contents) that are masked
    wherever they appear; values shorter than 8 characters are ignored.
    """
    global _HANDLER, _REGISTERED_SECRETS
    _REGISTERED_SECRETS = tuple(
        sorted({secret for secret in secrets if len(secret) >= 8}, key=len, reverse=True)
    )
    root = logging.getLogger()
    if _HANDLER is not None and _HANDLER in root.handlers:
        root.removeHandler(_HANDLER)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(_coerce_level(level))
    _HANDLER = handler


def get_logger(name: str) -> logging.Logger:
    """Return the named stdlib logger; the root handler writes JSON lines to stderr."""
    return logging.getLogger(name)
