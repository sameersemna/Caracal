"""FastAPI application factory, dependency wiring and transport-level guards.

Docs: docs/03 §1 (envelope, auth), docs/06 S1 (bearer), S2 (Origin), docs/02 (settings).
The app is deliberately thin: it validates, enqueues and reads; the worker runs the pipeline.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
import sqlalchemy as sa
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis import asyncio as redis_asyncio
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from caracal import __version__
from caracal.api.routes import public_router, router
from caracal.config import Settings
from caracal.db.engine import make_engine, session_scope
from caracal.db.repo import (
    JobRepo,
    PostgresJobRepo,
    PostgresReportRepo,
    PostgresSegmentRepo,
    PostgresUploadRepo,
    ReportRepo,
    SegmentRepo,
    UploadRepo,
)
from caracal.errors import HTTP_STATUS, CaracalError, ErrorCode
from caracal.log import get_logger
from caracal.queue import JobQueue, Keys

if TYPE_CHECKING:
    from caracal.pipeline.stages import Repos

logger = get_logger("caracal.app")

MAX_JOBS_BODY_BYTES = 64 * 1024
DEFAULT_HEALTH_TIMEOUT_S = 2.0
DEFAULT_HEALTH_CACHE_TTL_S = 10.0


@dataclass
class AppDeps:
    """Everything the REST layer needs, injectable so tests never touch real services."""

    repos: Repos
    redis: Redis
    http: httpx.AsyncClient
    keys: Keys
    queue: JobQueue
    db_ping: Callable[[], Awaitable[None]]
    clock: Callable[[], float] = time.monotonic
    health_timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S
    health_cache_ttl_s: float = DEFAULT_HEALTH_CACHE_TTL_S
    health_cache: list[tuple[float, dict[str, object]]] = field(default_factory=list)


@dataclass(frozen=True)
class _RepoBundle:
    """Concrete repository bundle for the real application."""

    jobs: JobRepo
    uploads: UploadRepo
    reports: ReportRepo
    segments: SegmentRepo


def _envelope(
    status_code: int,
    code: ErrorCode,
    message: str,
    details: dict[str, object] | None = None,
) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=CaracalError(code, message, details).to_dict())


def _caracal_response(error: CaracalError) -> JSONResponse:
    headers: dict[str, str] = {}
    if error.code is ErrorCode.UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=HTTP_STATUS[error.code],
        content=error.to_dict(),
        headers=headers or None,
    )


def _origin_allowed(origin: str, public_base_url: str) -> bool:
    """Allow the public base URL's origin or any localhost origin (docs/06 S2)."""
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return False
    if parts.hostname in {"localhost", "127.0.0.1"}:
        return True
    base = urlsplit(public_base_url)
    base_origin = f"{base.scheme.lower()}://{base.netloc.lower()}"
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}" == base_origin


def _build_real_deps(settings: Settings) -> tuple[AppDeps, AsyncEngine]:
    engine = make_engine(settings)
    repos = _RepoBundle(
        jobs=PostgresJobRepo(engine),
        uploads=PostgresUploadRepo(engine),
        reports=PostgresReportRepo(engine),
        segments=PostgresSegmentRepo(engine),
    )
    redis_client = redis_asyncio.from_url(settings.REDIS_URL.get_secret_value())
    http_client = httpx.AsyncClient(timeout=DEFAULT_HEALTH_TIMEOUT_S)
    keys = Keys(settings.REDIS_PREFIX)
    queue = JobQueue(redis_client, keys)

    async def db_ping() -> None:
        async with session_scope(engine) as connection:
            await connection.execute(sa.text("SELECT 1"))

    deps = AppDeps(
        repos=repos,
        redis=redis_client,
        http=http_client,
        keys=keys,
        queue=queue,
        db_ping=db_ping,
    )
    return deps, engine


def create_app(settings: Settings, deps: AppDeps | None = None) -> FastAPI:
    """Build the FastAPI application; ``deps`` is injected by every test."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if deps is not None:
            yield
            return
        built, engine = _build_real_deps(settings)
        application.state.deps = built
        try:
            yield
        finally:
            await built.http.aclose()
            await built.redis.aclose()
            await engine.dispose()

    app = FastAPI(title="Caracal", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    if deps is not None:
        app.state.deps = deps

    @app.middleware("http")
    async def origin_guard(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path.startswith("/mcp"):
            origin = request.headers.get("origin")
            if origin is not None and not _origin_allowed(origin, settings.CARACAL_PUBLIC_BASE_URL):
                return _envelope(403, ErrorCode.UNAUTHORIZED, "origin not allowed")
        return await call_next(request)

    @app.middleware("http")
    async def jobs_body_guard(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "POST" and request.url.path == "/v1/jobs":
            declared = request.headers.get("content-length")
            if declared is not None:
                try:
                    length = int(declared)
                except ValueError:
                    length = 0
                if length > MAX_JOBS_BODY_BYTES:
                    return _envelope(413, ErrorCode.TOO_LARGE, "request body is too large")
        return await call_next(request)

    @app.exception_handler(CaracalError)
    async def caracal_error_handler(_request: Request, exc: CaracalError) -> JSONResponse:
        return _caracal_response(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "loc": list(error.get("loc", ())),
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
            for error in exc.errors()
        ]
        return _envelope(422, ErrorCode.INVALID_INPUT, "request validation failed", {"errors": details})

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_INPUT
        response = _envelope(exc.status_code, code, str(exc.detail))
        for name, value in (exc.headers or {}).items():
            response.headers[name] = value
        return response

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled request error", extra={"path": request.url.path})
        return _envelope(500, ErrorCode.INTERNAL, "internal error")

    app.include_router(public_router)
    app.include_router(router)
    return app


async def _probe(check: Awaitable[None], timeout_s: float) -> str:
    try:
        await asyncio.wait_for(check, timeout_s)
    except Exception as exc:  # health is a status report: every failure becomes "down"
        logger.warning("dependency health check failed", extra={"error": str(exc)})
        return "down"
    return "ok"


async def _redis_ping(redis: Redis) -> None:
    await redis.ping()


async def _http_probe(client: httpx.AsyncClient, url: str) -> None:
    response = await client.get(url)
    response.raise_for_status()


async def _health_payload(settings: Settings, deps: AppDeps) -> dict[str, object]:
    postgres, redis_status, whisper, ollama = await asyncio.gather(
        _probe(deps.db_ping(), deps.health_timeout_s),
        _probe(_redis_ping(deps.redis), deps.health_timeout_s),
        _probe(_http_probe(deps.http, f"{settings.WHISPER_URL.rstrip('/')}/"), deps.health_timeout_s),
        _probe(
            _http_probe(deps.http, f"{settings.OLLAMA_LOCAL_URL.rstrip('/')}/api/tags"),
            deps.health_timeout_s,
        ),
    )
    statuses: dict[str, object] = {
        "postgres": postgres,
        "redis": redis_status,
        "whisper": whisper,
        "ollama": ollama,
    }
    return {
        "ok": postgres == "ok" and redis_status == "ok",
        "version": __version__,
        "deps": statuses,
    }


async def health_payload(settings: Settings, deps: AppDeps) -> dict[str, object]:
    """Cached health check (10 s by default), used by ``GET /healthz``."""
    now = deps.clock()
    if deps.health_cache:
        checked_at, cached = deps.health_cache[0]
        if now - checked_at < deps.health_cache_ttl_s:
            return cached
    payload = await _health_payload(settings, deps)
    deps.health_cache.clear()
    deps.health_cache.append((now, payload))
    return payload


__all__ = ["MAX_JOBS_BODY_BYTES", "AppDeps", "create_app", "health_payload"]
