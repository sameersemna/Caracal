"""Structured JSON logging and secret redaction (docs/06 S8).

Logs go to **stderr** only: stdout is the MCP stdio channel. Messages and every string
extra are passed through :func:`redact` before they are serialised.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime

_DSN_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)(?P<user>[^:/@\s]*):(?P<password>[^@\s]+)@")
_BEARER_RE = re.compile(r"(?i)\b(bearer)(\s+)([^\s,;\"']+)")
_KEY_VALUE_RE = re.compile(r"(?i)\b(api[_-]?key|token|key)=([^&\s,;\"']+)")

_STANDARD_ATTRS = frozenset(logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}

_HANDLER: logging.Handler | None = None


def redact(value: str) -> str:
    """Mask bearer tokens, ``api_key=`` / ``token=`` / ``key=`` values and DSN passwords."""
    result = _DSN_RE.sub(
        lambda match: f"{match.group('scheme')}{match.group('user')}:***@",
        value,
    )
    result = _BEARER_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}***",
        result,
    )
    return _KEY_VALUE_RE.sub(lambda match: f"{match.group(1)}=***", result)


def _redact_nested(value: object) -> object:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redact_nested(item) for key, item in value.items()}
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
            payload[key] = _redact_nested(value)
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


def configure_logging(level: str | int = "INFO") -> None:
    """Install the JSON stderr handler on the root logger (idempotent)."""
    global _HANDLER
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
