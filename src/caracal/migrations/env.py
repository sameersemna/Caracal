"""Alembic environment: async asyncpg engine configured from Settings.

The URL and target schema come from :func:`caracal.config.get_settings`; nothing database
related is read from ``alembic.ini`` (docs/05 §1).
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from caracal.config import get_settings
from caracal.db.tables import metadata

config = context.config

target_metadata = metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection (``alembic upgrade --sql``)."""
    settings = get_settings()
    context.configure(
        url=settings.DATABASE_URL.get_secret_value(),
        target_metadata=target_metadata,
        version_table_schema=settings.CARACAL_DB_SCHEMA,
        include_schemas=True,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    settings = get_settings()
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=settings.CARACAL_DB_SCHEMA,
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    settings = get_settings()
    engine = create_async_engine(
        settings.DATABASE_URL.get_secret_value(),
        poolclass=pool.NullPool,
        connect_args={"server_settings": {"search_path": settings.CARACAL_DB_SCHEMA}},
    )
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
