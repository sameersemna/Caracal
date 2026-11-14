"""Live M1 pipeline: real whisper.cpp, ffmpeg, Postgres and Redis (docs/07 §3).

Run by the judge, never by the gate. Each test provisions a throwaway Postgres schema and a
throwaway Redis prefix, runs the whole M1 registry in-process against the real services,
and removes exactly those resources afterwards. The jfk fixture is public domain.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
import redis.asyncio as redis_asyncio
from alembic import command
from alembic.config import Config as AlembicConfig
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from caracal.asr.whisper import WhisperClient
from caracal.config import Settings, get_settings
from caracal.db.engine import make_engine
from caracal.db.repo import (
    JobRecord,
    PostgresJobRepo,
    PostgresReportRepo,
    PostgresSegmentRepo,
    PostgresUploadRepo,
)
from caracal.ingest.ssrf import SystemResolver
from caracal.pipeline.runner import RunnerDeps, run_pipeline
from caracal.pipeline.stages import JobOptions, Stage, m1_registry
from caracal.queue import Keys, is_cancelled, publish_event, whisper_lock
from caracal.storage import JobFiles, UploadFiles
from tests.repo_contract import Repos

pytestmark = pytest.mark.live

_AUDIO = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "jfk.wav"
_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "caracal" / "migrations"


def _alembic(direction: str) -> None:
    config = AlembicConfig()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    if direction == "upgrade":
        command.upgrade(config, "head")
    else:
        command.downgrade(config, "base")


async def _create_schema(url: str, schema: str) -> None:
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.begin() as connection:
            await connection.execute(CreateSchema(schema))
    finally:
        await engine.dispose()


async def _drop_schema(url: str, schema: str) -> None:
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.begin() as connection:
            await connection.execute(DropSchema(schema, cascade=True))
    finally:
        await engine.dispose()


def _settings_for(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    schema = f"caracal_t_{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("CARACAL_DB_SCHEMA", schema)
    monkeypatch.setenv("CARACAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REDIS_PREFIX", f"{schema}:")
    get_settings.cache_clear()
    return get_settings()


@dataclass
class _Live:
    settings: Settings
    repos: Repos
    redis: Redis
    keys: Keys
    http: httpx.AsyncClient
    whisper: WhisperClient
    engine: AsyncEngine


@pytest_asyncio.fixture
async def live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[_Live]:
    settings = _settings_for(monkeypatch, tmp_path)
    schema = settings.CARACAL_DB_SCHEMA
    url = settings.DATABASE_URL.get_secret_value()
    await _create_schema(url, schema)
    engine: AsyncEngine | None = None
    redis: Redis | None = None
    http: httpx.AsyncClient | None = None
    try:
        await asyncio.to_thread(_alembic, "upgrade")
        engine = make_engine(settings)
        repos = Repos(
            jobs=PostgresJobRepo(engine),
            uploads=PostgresUploadRepo(engine),
            reports=PostgresReportRepo(engine),
            segments=PostgresSegmentRepo(engine),
            cleanup=_noop_cleanup,
        )
        redis = redis_asyncio.from_url(settings.REDIS_URL.get_secret_value())
        keys = Keys(settings.REDIS_PREFIX)
        http = httpx.AsyncClient()
        whisper = WhisperClient(
            settings.WHISPER_URL,
            http,
            timeout_s=settings.WHISPER_TIMEOUT_S,
            lock=lambda: whisper_lock(redis, keys),
        )
        yield _Live(
            settings=settings,
            repos=repos,
            redis=redis,
            keys=keys,
            http=http,
            whisper=whisper,
            engine=engine,
        )
    finally:
        if redis is not None:
            async for key in redis.scan_iter(match=f"{settings.REDIS_PREFIX}*"):
                await redis.delete(key)
            await redis.aclose()
        if http is not None:
            await http.aclose()
        if engine is not None:
            await engine.dispose()
            await asyncio.to_thread(_alembic, "downgrade")
        await _drop_schema(url, schema)


async def _noop_cleanup() -> None:
    return None


def _deps(live: _Live) -> RunnerDeps:
    async def publish(job_id: uuid.UUID, event: dict[str, Any]) -> None:
        await publish_event(live.redis, live.keys, job_id, event)

    async def cancelled(job_id: uuid.UUID) -> bool:
        return await is_cancelled(live.redis, live.keys, job_id)

    async def owner() -> bool:
        return True

    return RunnerDeps(
        settings=live.settings,
        repos=live.repos,
        queue_events=publish,
        cancel_check=cancelled,
        still_owner=owner,
        whisper=live.whisper,
        resolver=SystemResolver(),
        redis=live.redis,
        keys=live.keys,
    )


async def _upload_jfk(live: _Live) -> JobRecord:
    upload_id = uuid.uuid4()
    files = UploadFiles(live.settings.CARACAL_DATA_DIR, upload_id)
    files.ensure()
    destination = files.file("jfk.wav")
    shutil.copy2(_AUDIO, destination)
    data = destination.read_bytes()
    await live.repos.uploads.create(
        upload_id=upload_id,
        filename="jfk.wav",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        path=str(destination),
    )
    job = await live.repos.jobs.create(
        source_kind="upload",
        upload_id=upload_id,
        options=JobOptions().model_dump(),
    )
    await live.repos.uploads.consume(upload_id, job.id)
    return job


async def _assert_done(live: _Live, job_id: uuid.UUID) -> JobFiles:
    stored = await live.repos.jobs.get(job_id)
    assert stored is not None
    assert stored.status == "done", stored.error
    files = JobFiles(live.settings.CARACAL_DATA_DIR, job_id)
    assert files.transcript_json.is_file()
    assert files.transcript_txt.is_file()
    assert files.subtitle("source", "srt").is_file()
    assert files.subtitle("source", "vtt").is_file()
    assert not files.chunks_dir.exists()
    assert not files.source_dir.exists()
    stages = {row.name: row for row in await live.repos.jobs.list_stages(job_id)}
    for name in ("ingest", "normalize", "chunk", "transcribe", "langid", "subtitles"):
        assert stages[name].status == "done"
    assert stages[Stage.DIARIZE.value].status == "skipped"
    return files


async def test_live_pipeline_on_a_jfk_upload(live: _Live) -> None:
    job = await _upload_jfk(live)

    status = await run_pipeline(job.id, _deps(live), m1_registry())

    assert status == "done"
    files = await _assert_done(live, job.id)
    transcript = files.transcript_txt.read_text(encoding="utf-8")
    assert "country" in transcript.lower()
    srt = files.subtitle("source", "srt").read_text(encoding="utf-8")
    assert "-->" in srt
    hits = await live.repos.segments.search("country")
    assert hits
    assert {hit.job_id for hit in hits} == {job.id}


@pytest.mark.skipif(not os.environ.get("CARACAL_LIVE_YT_URL"), reason="CARACAL_LIVE_YT_URL is not set")
async def test_live_pipeline_on_a_platform_url(live: _Live) -> None:
    url = os.environ["CARACAL_LIVE_YT_URL"]
    job = await live.repos.jobs.create(
        source_kind="platform",
        source_url=url,
        options=JobOptions().model_dump(),
    )

    status = await run_pipeline(job.id, _deps(live), m1_registry())

    assert status == "done"
    files = await _assert_done(live, job.id)
    assert files.transcript_txt.read_text(encoding="utf-8").strip()
