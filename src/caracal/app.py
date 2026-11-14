"""FastAPI application factory, dependency wiring and transport-level guards.

Docs: docs/03 §1 (envelope, auth), docs/06 S1 (bearer), S2 (Origin), docs/02 (settings).
The app is deliberately thin: it validates, enqueues and reads; the worker runs the pipeline.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
import sqlalchemy as sa
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from redis import asyncio as redis_asyncio
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from caracal import __version__
from caracal.api.auth import bearer_from_header, token_matches
from caracal.api.routes import public_router, router
from caracal.config import Settings
from caracal.db.engine import make_engine, session_scope
from caracal.db.repo import (
    JobRepo,
    PostgresJobRepo,
    PostgresReportRepo,
    PostgresSegmentRepo,
    PostgresSpeakerRepo,
    PostgresUploadRepo,
    ReportRepo,
    SegmentRepo,
    SpeakerRepo,
    UploadRepo,
)
from caracal.errors import HTTP_STATUS, CaracalError, ErrorCode
from caracal.log import get_logger
from caracal.mcp.server import HTTP_INLINE_BASE64_BYTES, build_mcp
from caracal.queue import JobQueue, Keys
from caracal.services import JobService

if TYPE_CHECKING:
    from caracal.pipeline.stages import Repos

logger = get_logger("caracal.app")

DEFAULT_V1_BODY_LIMIT_BYTES = 64 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
MCP_HTTP_BODY_LIMIT_BYTES = 1024 * 1024
DEFAULT_HEALTH_TIMEOUT_S = 2.0
DEFAULT_HEALTH_CACHE_TTL_S = 10.0

BodyLimits = Mapping[tuple[str, str], int]


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
    speakers: SpeakerRepo


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
    elif error.code is ErrorCode.RATE_LIMITED:
        retry_after = error.details.get("retry_after_s")
        if isinstance(retry_after, int):
            headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=HTTP_STATUS[error.code],
        content=error.to_dict(),
        headers=headers or None,
    )


def _is_v1_path(path: str) -> bool:
    """True for the REST API surface (``/healthz`` stays public)."""
    return path == "/v1" or path.startswith("/v1/")


def _is_mcp_path(path: str) -> bool:
    """True for the mounted streamable-HTTP MCP endpoint (``/mcp`` and ``/mcp/*``)."""
    return path == "/mcp" or path.startswith("/mcp/")


def _content_length(scope: Scope) -> int | None:
    declared = Headers(scope=scope).get("content-length")
    if declared is None:
        return None
    try:
        return int(declared)
    except ValueError:
        return None


async def _send_error(
    send: Send,
    error: CaracalError,
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
    status_code: int | None = None,
) -> None:
    """Send one error envelope without going through the routed application."""
    body = json.dumps(error.to_dict(), separators=(",", ":")).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
        *extra_headers,
    ]
    status = HTTP_STATUS[error.code] if status_code is None else status_code
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body, "more_body": False})


class OriginGuardMiddleware:
    """Reject foreign ``Origin`` headers on the MCP endpoint (docs/06 S2).

    Pure ASGI so the stack has no ``BaseHTTPMiddleware`` task groups: those would wrap an
    exception raised from ``receive`` in an ``ExceptionGroup`` and defeat the early 413.
    """

    def __init__(self, app: ASGIApp, *, public_base_url: str) -> None:
        self.app = app
        self.public_base_url = public_base_url

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            origin = Headers(scope=scope).get("origin")
            if origin is not None and not _origin_allowed(origin, self.public_base_url):
                await _send_error(
                    send,
                    CaracalError(ErrorCode.UNAUTHORIZED, "origin not allowed"),
                    status_code=403,
                )
                return
        await self.app(scope, receive, send)


class _BodyTooLarge(BaseException):
    """Raised from ``receive`` when a body crosses the route limit.

    It must not be an ``Exception``: FastAPI wraps exceptions raised while parsing a body
    into a 400, and the mounted MCP sub-app has a Starlette exception middleware that turns
    an ``HTTPException`` into a plain-text 413. A ``BaseException`` passes both layers
    untouched, so :class:`AuthBodyLimitMiddleware` can render the ``too_large`` envelope.
    """

    def __init__(self) -> None:
        super().__init__("request body is too large")


class AuthBodyLimitMiddleware:
    """Authenticate ``/v1`` and ``/mcp`` requests before the body is read, then cap the stream.

    Docs: docs/03 §1 (auth on everything but ``/healthz``), docs/06 S1 (bearer), docs/03 §4
    (``too_large``). This is a *pure* ASGI middleware on purpose: ``BaseHTTPMiddleware`` would
    let the routed app pull the body first, so an unauthenticated client could make us buffer
    an upload before answering 401.

    ``body_limits`` maps ``(METHOD, path)`` to a byte cap so individual REST routes can
    declare their own limit (e.g. ``POST /v1/uploads``); every other ``/v1`` route uses
    ``default_body_limit``. The MCP endpoint gets its own cap (``mcp_body_limit``, 1 MiB):
    larger audio must go through ``POST /v1/uploads``.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        token: str,
        body_limits: BodyLimits | None = None,
        default_body_limit: int = DEFAULT_V1_BODY_LIMIT_BYTES,
        mcp_body_limit: int = MCP_HTTP_BODY_LIMIT_BYTES,
    ) -> None:
        self.app = app
        self.token = token
        self.body_limits = dict(body_limits or {})
        self.default_body_limit = default_body_limit
        self.mcp_body_limit = mcp_body_limit

    def _limit(self, method: str, path: str) -> int:
        if _is_mcp_path(path):
            return self.mcp_body_limit
        return self.body_limits.get((method.upper(), path), self.default_body_limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if not (_is_v1_path(path) or _is_mcp_path(path)):
            await self.app(scope, receive, send)
            return

        header = Headers(scope=scope).get("authorization")
        candidate = bearer_from_header(header)
        if candidate is None or not token_matches(candidate, self.token):
            message = "missing authorization header" if header is None else "invalid or missing bearer token"
            await _send_error(
                send,
                CaracalError(ErrorCode.UNAUTHORIZED, message),
                extra_headers=((b"www-authenticate", b"Bearer"),),
            )
            return

        limit = self._limit(scope["method"], scope["path"])
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await _send_error(send, CaracalError(ErrorCode.TOO_LARGE, "request body is too large"))
            return

        received = 0
        response_started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                body_length = len(message.get("body", b""))
                received += body_length
                if received > limit:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if response_started:
                raise
            await _send_error(send, CaracalError(ErrorCode.TOO_LARGE, "request body is too large"))


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
        speakers=PostgresSpeakerRepo(engine),
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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def mcp_transport_security(settings: Settings) -> TransportSecuritySettings:
    """Allowed Host/Origin values for the MCP transport (docs/02 ``CARACAL_ALLOWED_HOSTS``)."""
    port = settings.CARACAL_PORT
    hosts = [
        f"localhost:{port}",
        f"127.0.0.1:{port}",
        "localhost:*",
        "127.0.0.1:*",
    ]
    try:
        base = urlsplit(settings.CARACAL_PUBLIC_BASE_URL)
    except ValueError:
        base = None
    if base is not None and base.hostname:
        hosts.append(base.hostname)
        if base.port is not None:
            hosts.append(f"{base.hostname}:{base.port}")
        elif base.scheme.lower() == "https":
            hosts.append(f"{base.hostname}:443")
        else:
            hosts.append(f"{base.hostname}:80")
    hosts.extend(settings.CARACAL_ALLOWED_HOSTS)
    allowed_hosts = _dedupe(hosts)
    # A browser Origin has no default port, so the bare loopback names are allowed too.
    origin_hosts = _dedupe([*allowed_hosts, "localhost", "127.0.0.1"])
    allowed_origins = _dedupe(
        [f"http://{host}" for host in origin_hosts] + [f"https://{host}" for host in origin_hosts]
    )
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def create_app(
    settings: Settings,
    deps: AppDeps | None = None,
    *,
    body_limits: BodyLimits | None = None,
) -> FastAPI:
    """Build the FastAPI application; ``deps`` is injected by every test.

    The MCP transport is wired here (its session manager needs the sub-app at mount time),
    so real dependencies are built eagerly and still closed by the lifespan. ``body_limits``
    overrides the 64 KiB cap for specific ``(METHOD, path)`` pairs; it is threaded into
    :class:`AuthBodyLimitMiddleware` so upload routes can declare their own.
    """
    engine: AsyncEngine | None = None
    if deps is None:
        deps, engine = _build_real_deps(settings)
    service = JobService(
        settings,
        deps.repos,
        deps.redis,
        deps.keys,
        deps.queue,
        settings.CARACAL_DATA_DIR,
    )
    mcp_server = build_mcp(service, settings, max_inline_bytes=HTTP_INLINE_BASE64_BYTES)
    mcp_app = mcp_server.streamable_http_app(
        streamable_http_path="/",
        stateless_http=True,
        json_response=True,
        max_request_body_size=MCP_HTTP_BODY_LIMIT_BYTES,
        transport_security=mcp_transport_security(settings),
    )
    owned_engine = engine

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        async with mcp_app.router.lifespan_context(mcp_app):
            try:
                yield
            finally:
                if owned_engine is not None:
                    await deps.http.aclose()
                    await deps.redis.aclose()
                    await owned_engine.dispose()

    app = FastAPI(title="Caracal", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.deps = deps
    app.state.service = service
    app.state.mcp_server = mcp_server

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
    async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
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
    app.mount("/mcp", mcp_app)
    app.add_middleware(OriginGuardMiddleware, public_base_url=settings.CARACAL_PUBLIC_BASE_URL)
    # The upload route has its own cap (docs/02 CARACAL_MAX_UPLOAD_MB) plus multipart
    # header/boundary overhead; explicit overrides win so tests can shrink it.
    effective_limits: dict[tuple[str, str], int] = {
        ("POST", "/v1/uploads"): settings.CARACAL_MAX_UPLOAD_MB * 1024 * 1024 + MULTIPART_OVERHEAD_BYTES,
    }
    if body_limits:
        effective_limits.update(body_limits)
    # Added last so it is the outermost user middleware (inside Starlette's error handling):
    # no /v1 or /mcp body byte is read before the bearer token has been checked.
    app.add_middleware(
        AuthBodyLimitMiddleware,
        token=settings.CARACAL_API_TOKEN.get_secret_value(),
        body_limits=effective_limits,
    )
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


__all__ = [
    "DEFAULT_V1_BODY_LIMIT_BYTES",
    "MCP_HTTP_BODY_LIMIT_BYTES",
    "AppDeps",
    "AuthBodyLimitMiddleware",
    "create_app",
    "health_payload",
    "mcp_transport_security",
]
