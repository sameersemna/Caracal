"""Async engine construction and transaction helpers (docs/01 §3)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from caracal.config import Settings


def search_path(schema: str) -> str:
    """Return the ``search_path`` that keeps ``schema`` first and ``public`` visible.

    ``public`` is always appended because objects installed there must resolve from every
    deployment schema: pgvector lives in ``public`` on latitude, and a schema-only search
    path hides the ``vector`` type (docs/05 §1, verdict 19).
    """
    if schema == "public":
        return "public"
    return f"{schema}, public"


def server_settings(schema: str) -> dict[str, str]:
    """Return asyncpg ``server_settings`` pinning ``search_path`` (see :func:`search_path`)."""
    return {"search_path": search_path(schema)}


def make_engine(settings: Settings) -> AsyncEngine:
    """Build the asyncpg engine, pinning every connection's ``search_path`` to the schema."""
    return create_async_engine(
        settings.DATABASE_URL.get_secret_value(),
        pool_pre_ping=True,
        connect_args={"server_settings": server_settings(settings.CARACAL_DB_SCHEMA)},
    )


@asynccontextmanager
async def session_scope(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """Yield an :class:`AsyncConnection` inside a transaction.

    Committing on clean exit and rolling back when the body raises keeps repository methods
    atomic without exposing engine details to callers.
    """
    async with engine.begin() as connection:
        yield connection
