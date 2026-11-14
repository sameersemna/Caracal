"""Command line entry point (docs/01 §3)."""

from __future__ import annotations

import argparse
import sys
from typing import cast

from pydantic import SecretStr, ValidationError

from caracal.config import Settings, get_settings
from caracal.log import configure_logging, get_logger

_COMMANDS = ("serve", "worker", "mcp-stdio", "migrate", "doctor")

_HELP = {
    "serve": "run the REST API and MCP HTTP server",
    "worker": "run the pipeline worker",
    "mcp-stdio": "run the MCP stdio server",
    "migrate": "apply database migrations",
    "doctor": "validate configuration (no network calls)",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="caracal",
        description="Caracal — audio intelligence API and MCP server",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="{" + ",".join(_COMMANDS) + "}",
    )
    for name in _COMMANDS:
        subparsers.add_parser(name, help=_HELP[name], description=_HELP[name])
    return parser


def _secret_values(settings: Settings) -> list[str]:
    """Every loaded ``SecretStr`` value, for exact-match redaction (docs/06 S8c)."""
    values: list[str] = []
    for field in type(settings).model_fields:
        value = getattr(settings, field)
        if isinstance(value, SecretStr):
            values.append(value.get_secret_value())
    return values


def _doctor() -> int:
    configure_logging("INFO")
    logger = get_logger("caracal.doctor")
    try:
        settings = get_settings()
    except ValidationError as exc:
        fields = sorted(
            {
                ".".join(str(part) for part in error["loc"]) or "<root>"
                for error in exc.errors(include_url=False)
            }
        )
        logger.error("config invalid", extra={"fields": fields})
        return 1
    configure_logging(settings.CARACAL_LOG_LEVEL, secrets=_secret_values(settings))
    logger.info(
        "config valid",
        extra={
            "host": settings.CARACAL_HOST,
            "port": settings.CARACAL_PORT,
            "data_dir": str(settings.CARACAL_DATA_DIR),
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        if isinstance(code, int):
            return code
        return 0 if code is None else 2

    command = cast("str | None", args.command)
    if command is None:
        parser.print_usage(sys.stderr)
        return 2
    if command == "doctor":
        return _doctor()

    sys.stderr.write(f"{command}: available from a later milestone\n")
    return 2
