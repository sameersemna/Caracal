"""Live Postgres: migrations in a throwaway schema and the shared repository contract.

Run by the judge, never by the gate (``-m live``). Each test provisions
``caracal_t_<8 hex>``: create schema, ``alembic upgrade head``, run, then ``downgrade base`` and
``DROP SCHEMA ... CASCADE`` in a ``finally``. The database URL is the real ``DATABASE_URL`` from
Settings; only ``CARACAL_DB_SCHEMA`` is overridden.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from caracal.config import Settings, get_settings
from caracal.db.engine import make_engine
from caracal.db.repo import (
    PostgresJobRepo,
    PostgresReportRepo,
    PostgresSegmentRepo,
    PostgresUploadRepo,
)
from tests.repo_contract import CASES, Repos

pytestmark = pytest.mark.live

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "caracal" / "migrations"


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    """Override the unit-suite isolation: live tests need the real environment and ``.env``."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _provision_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("CARACAL_DB_SCHEMA", f"caracal_t_{uuid.uuid4().hex[:8]}")
    get_settings.cache_clear()
    return get_settings()


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


def _alembic(direction: str, revision: str) -> None:
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS_DIR))
    if direction == "upgrade":
        command.upgrade(config, revision)
    else:
        command.downgrade(config, revision)


async def _table_names(url: str, schema: str) -> set[str]:
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            rows = await connection.execute(
                sa.text("SELECT tablename FROM pg_tables WHERE schemaname = :schema").bindparams(
                    schema=schema
                )
            )
            return set(rows.scalars().all())
    finally:
        await engine.dispose()


async def _tsv_generation(url: str, schema: str) -> str | None:
    engine = create_async_engine(url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            value = await connection.scalar(
                sa.text(
                    "SELECT is_generated FROM information_schema.columns "
                    "WHERE table_schema = :schema AND table_name = 'segments' AND column_name = 'tsv'"
                ).bindparams(schema=schema)
            )
            return value if isinstance(value, str) else None
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def live_repos(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Repos]:
    settings = _provision_settings(monkeypatch)
    schema = settings.CARACAL_DB_SCHEMA
    url = settings.DATABASE_URL.get_secret_value()
    await _create_schema(url, schema)
    try:
        await asyncio.to_thread(_alembic, "upgrade", "head")
        engine = make_engine(settings)

        async def cleanup() -> None:
            async with engine.begin() as connection:
                await connection.execute(sa.text("TRUNCATE TABLE uploads, jobs CASCADE"))

        try:
            yield Repos(
                jobs=PostgresJobRepo(engine),
                uploads=PostgresUploadRepo(engine),
                reports=PostgresReportRepo(engine),
                segments=PostgresSegmentRepo(engine),
                cleanup=cleanup,
            )
        finally:
            await engine.dispose()
            await asyncio.to_thread(_alembic, "downgrade", "base")
    finally:
        await _drop_schema(url, schema)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.__name__)
async def test_postgres_repo_contract(live_repos: Repos, case: Callable[[Repos], Awaitable[None]]) -> None:
    await live_repos.cleanup()
    await case(live_repos)


async def test_upgrade_then_downgrade_removes_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _provision_settings(monkeypatch)
    schema = settings.CARACAL_DB_SCHEMA
    url = settings.DATABASE_URL.get_secret_value()
    await _create_schema(url, schema)
    try:
        await asyncio.to_thread(_alembic, "upgrade", "head")
        tables = await _table_names(url, schema)
        assert tables == {
            "alembic_version",
            "jobs",
            "job_stages",
            "uploads",
            "reports",
            "segments",
        }
        assert "voiceprints" not in tables
        assert await _tsv_generation(url, schema) == "ALWAYS"

        await asyncio.to_thread(_alembic, "downgrade", "base")
        assert await _table_names(url, schema) == {"alembic_version"}
    finally:
        await _drop_schema(url, schema)
