"""Command line entry point (docs/01 §3)."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import signal
import sys
from pathlib import Path
from typing import cast

from alembic import command
from alembic.config import Config as AlembicConfig
from pydantic import SecretStr, ValidationError
from redis import asyncio as redis_asyncio

from caracal.app import _build_real_deps
from caracal.config import Settings, get_settings
from caracal.db.engine import make_engine
from caracal.db.repo import (
    PostgresJobRepo,
    PostgresReportRepo,
    PostgresSegmentRepo,
    PostgresSpeakerRepo,
    PostgresUploadRepo,
)
from caracal.errors import CaracalError
from caracal.log import configure_logging, get_logger
from caracal.mcp.server import build_mcp
from caracal.services import JobService
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


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _diarization_status() -> dict[str, object]:
    """Report the optional speakers extra: pyannote importable and CUDA usable."""
    pyannote = _importable("pyannote.audio")
    torch_installed = _importable("torch")
    cuda = False
    if torch_installed:
        try:
            torch_module = importlib.import_module("torch")
            cuda = bool(torch_module.cuda.is_available())
        except Exception:
            cuda = False
    return {"pyannote": pyannote, "torch": torch_installed, "cuda": cuda}


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
            "diarization": _diarization_status(),
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
        speakers=PostgresSpeakerRepo(engine),
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


async def _run_mcp_stdio(settings: Settings) -> int:
    """Build real dependencies, then serve the MCP tools on stdin/stdout (docs/01 §3).

    Logs go to stderr (``configure_logging``); stdout belongs exclusively to the MCP
    protocol framing.
    """
    deps, engine = _build_real_deps(settings)
    service = JobService(
        settings,
        deps.repos,
        deps.redis,
        deps.keys,
        deps.queue,
        settings.CARACAL_DATA_DIR,
    )
    server = build_mcp(service, settings)
    try:
        await server.run_stdio_async()
    finally:
        await deps.http.aclose()
        await deps.redis.aclose()
        await engine.dispose()
    return 0


def _mcp_stdio() -> int:
    """Load settings, configure JSON logging on stderr, then run the stdio server."""
    configure_logging("INFO")
    logger = get_logger("caracal.mcp")
    try:
        settings = get_settings()
    except ValidationError as exc:
        logger.error("config invalid", extra={"fields": _invalid_config_fields(exc)})
        return 1
    configure_logging(settings.CARACAL_LOG_LEVEL, secrets=_secret_values(settings))
    try:
        return asyncio.run(_run_mcp_stdio(settings))
    except KeyboardInterrupt:
        return 0
    except CaracalError as exc:
        logger.error(
            "MCP stdio server stopped with an error",
            extra={"code": exc.code.value, "error": exc.message},
        )
        return 1


def _serve(settings: Settings) -> int:
    """Run the REST + MCP HTTP app with uvicorn in this process (docs/01 §3)."""
    import uvicorn

    from caracal.app import create_app

    uvicorn.run(
        create_app(settings),
        host=settings.CARACAL_HOST,
        port=settings.CARACAL_PORT,
        workers=1,
        log_config=None,
        proxy_headers=False,
    )
    return 0


def _serve_command() -> int:
    """Load settings, configure JSON logging, then run uvicorn in-process."""
    configure_logging("INFO")
    logger = get_logger("caracal.serve")
    try:
        settings = get_settings()
    except ValidationError as exc:
        logger.error("config invalid", extra={"fields": _invalid_config_fields(exc)})
        return 1
    configure_logging(settings.CARACAL_LOG_LEVEL, secrets=_secret_values(settings))
    try:
        return _serve(settings)
    except CaracalError as exc:
        logger.error(
            "server stopped with an error",
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
    if command == "serve":
        return _serve_command()
    if command == "mcp-stdio":
        return _mcp_stdio()
    return 2
