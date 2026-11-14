"""Shared harness for the REST contract tests.

Every test app is built through :func:`caracal.app.create_app` with injected fakes:
in-memory repositories, ``fakeredis`` and an ``httpx.MockTransport`` client. No test
touches a network, Postgres or a real Redis.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from caracal.app import AppDeps, create_app
from caracal.config import Settings
from caracal.queue import JobQueue, Keys
from tests.fakes.redis import RecordingRedis
from tests.fakes.repo import make_fake_repos
from tests.repo_contract import Repos

TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeClock:
    """A monotonic clock the health-cache tests can advance."""

    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass
class FakeUpstreams:
    """Scriptable whisper/Ollama responses behind ``httpx.MockTransport``."""

    whisper: str = "ok"
    ollama: str = "ok"
    calls: list[str] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(str(request.url))
        if request.url.path == "/api/tags":
            if self.ollama == "down":
                return httpx.Response(503, json={"error": "down"})
            return httpx.Response(200, json={"models": []})
        if self.whisper == "timeout":
            raise httpx.ReadTimeout("upstream timed out", request=request)
        if self.whisper == "down":
            return httpx.Response(500, text="boom")
        return httpx.Response(200, text="whisper.cpp")


@dataclass
class Harness:
    """One fully-injected application plus the state the tests inspect."""

    app: FastAPI
    settings: Settings
    deps: AppDeps
    repos: Repos
    redis: RecordingRedis
    keys: Keys
    queue: JobQueue
    upstreams: FakeUpstreams
    clock: FakeClock
    db: dict[str, bool]

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False),
            base_url="http://testserver",
        )


@pytest.fixture
def make_harness(make_settings: Callable[..., Settings]) -> Callable[..., Harness]:
    """Build an app with fakes; overrides let each test tune settings or the queue."""

    def _make(
        *,
        settings_overrides: dict[str, Any] | None = None,
        queue: Any = None,
    ) -> Harness:
        settings = make_settings(**(settings_overrides or {}))
        repos = make_fake_repos()
        redis = RecordingRedis()
        keys = Keys(settings.REDIS_PREFIX)
        job_queue = queue if queue is not None else JobQueue(redis, keys)
        upstreams = FakeUpstreams()
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(upstreams.handle))
        clock = FakeClock()
        db = {"down": False}

        async def db_ping() -> None:
            if db["down"]:
                raise RuntimeError("postgres is down")

        deps = AppDeps(
            repos=repos,
            redis=redis,
            http=http_client,
            keys=keys,
            queue=job_queue,
            db_ping=db_ping,
            clock=clock,
        )
        app = create_app(settings, deps)
        return Harness(
            app=app,
            settings=settings,
            deps=deps,
            repos=repos,
            redis=redis,
            keys=keys,
            queue=job_queue,
            upstreams=upstreams,
            clock=clock,
            db=db,
        )

    return _make
