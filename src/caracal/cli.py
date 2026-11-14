"""Command line entry point (docs/01 §3)."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import cast

from alembic import command
from alembic.config import Config as AlembicConfig
from pydantic import SecretStr, ValidationError
from redis import asyncio as redis_asyncio

from caracal.config import Settings, get_settings
from caracal.db.engine import make_engine
from caracal.db.repo import (
    PostgresJobRepo,
    PostgresReportRepo,
    PostgresSegmentRepo,
    PostgresUploadRepo,
)
from caracal.errors import CaracalError
from caracal.log import configure_logging, get_logger
from caracal.worker import RepoBundle, Worker

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

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


def _invalid_config_fields(exc: ValidationError) -> list[str]:
    return sorted(
        {".".join(str(part) for part in error["loc"]) or "<root>" for error in exc.errors(include_url=False)}
    )


def _doctor() -> int:
    configure_logging("INFO")
    logger = get_logger("caracal.doctor")
    try:
        settings = get_settings()
    except ValidationError as exc:
        logger.error("config invalid", extra={"fields": _invalid_config_fields(exc)})
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


def _migrate() -> int:
    """Apply ``alembic upgrade head`` programmatically (docs/01 §3)."""
    configure_logging("INFO")
    logger = get_logger("caracal.migrate")
    try:
        settings = get_settings()
    except ValidationError as exc:
        logger.error("config invalid", extra={"fields": _invalid_config_fields(exc)})
        return 1

    configure_logging(settings.CARACAL_LOG_LEVEL, secrets=_secret_values(settings))
    config = AlembicConfig()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    logger.info("applying migrations", extra={"revision": "head"})
    try:
        command.upgrade(config, "head")
    except Exception as exc:
        logger.error("migration failed", extra={"error": str(exc)})
        return 1
    logger.info("migrations applied", extra={"revision": "head"})
    return 0


async def _run_worker(settings: Settings) -> int:
    """Build the datastores and run the worker until SIGINT/SIGTERM."""
    logger = get_logger("caracal.worker")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for handled in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(handled, stop.set)
        except RuntimeError:
            logger.warning("signal handler unavailable", extra={"signal": handled})

    redis_client = redis_asyncio.from_url(settings.REDIS_URL.get_secret_value())
    engine = make_engine(settings)
    repos = RepoBundle(
        jobs=PostgresJobRepo(engine),
        uploads=PostgresUploadRepo(engine),
        reports=PostgresReportRepo(engine),
        segments=PostgresSegmentRepo(engine),
    )
    worker = Worker(settings, redis_client, repos)
    try:
        await worker.run(stop)
    finally:
        await redis_client.aclose()
        await engine.dispose()
    logger.info("worker stopped", extra={"worker_id": worker.worker_id})
    return 0


def _worker() -> int:
    configure_logging("INFO")
    logger = get_logger("caracal.worker")
    try:
        settings = get_settings()
    except ValidationError as exc:
        logger.error("config invalid", extra={"fields": _invalid_config_fields(exc)})
        return 1
    configure_logging(settings.CARACAL_LOG_LEVEL, secrets=_secret_values(settings))
    try:
        return asyncio.run(_run_worker(settings))
    except KeyboardInterrupt:
        return 0
    except CaracalError as exc:
        logger.error(
            "worker stopped with an error",
            extra={"code": exc.code.value, "error": exc.message},
        )
        return 1


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
    if command == "migrate":
        return _migrate()
    if command == "worker":
        return _worker()

    sys.stderr.write(f"{command}: available from a later milestone\n")
    return 2
