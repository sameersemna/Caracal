"""Async engine construction and transaction helpers (docs/01 §3)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from caracal.config import Settings


def make_engine(settings: Settings) -> AsyncEngine:
    """Build the asyncpg engine, pinning every connection's ``search_path`` to the schema."""
    return create_async_engine(
        settings.DATABASE_URL.get_secret_value(),
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": settings.CARACAL_DB_SCHEMA}},
    )


@asynccontextmanager
async def session_scope(engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    """Yield an :class:`AsyncConnection` inside a transaction.

    Committing on clean exit and rolling back when the body raises keeps repository methods
    atomic without exposing engine details to callers.
    """
    async with engine.begin() as connection:
        yield connection
