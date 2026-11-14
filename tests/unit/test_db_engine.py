"""Engine ``search_path`` and the pgvector visibility guard (verdict 19, docs/05 §1).

pgvector is installed in ``public`` on latitude while Caracal tables live in
``CARACAL_DB_SCHEMA``. Every connection must therefore keep ``public`` on the search path,
both for the app engine and for alembic, or ``vector``/``vector_cosine_ops`` stop resolving.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import alembic
import pytest
from sqlalchemy.pool import NullPool

from caracal.config import Settings
from caracal.db import engine as engine_module
from caracal.db.engine import make_engine, search_path, server_settings

MakeSettings = Callable[..., Settings]

_MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "src" / "caracal" / "migrations"


def _load_voiceprints_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "caracal_0002_voiceprints", _MIGRATIONS_DIR / "versions" / "0002_voiceprints.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VOICEPRINTS_MIGRATION = _load_voiceprints_migration()


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ("public", "public"),
        ("caracal_t_x", "caracal_t_x, public"),
    ],
)
def test_search_path_keeps_public_visible(schema: str, expected: str) -> None:
    assert search_path(schema) == expected


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ("public", {"search_path": "public"}),
        ("caracal_t_x", {"search_path": "caracal_t_x, public"}),
    ],
)
def test_server_settings_are_exact(schema: str, expected: dict[str, str]) -> None:
    assert server_settings(schema) == expected


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ("public", "public"),
        ("caracal_t_x", "caracal_t_x, public"),
    ],
)
def test_make_engine_passes_exact_server_settings(
    make_settings: MakeSettings,
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
    expected: str,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_create_async_engine(url: str, **kwargs: Any) -> object:
        captured["url"] = url
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(engine_module, "create_async_engine", fake_create_async_engine)
    settings = make_settings(CARACAL_DB_SCHEMA=schema)

    assert make_engine(settings) is sentinel
    assert captured == {
        "url": settings.DATABASE_URL.get_secret_value(),
        "pool_pre_ping": True,
        "connect_args": {"server_settings": {"search_path": expected}},
    }


class _FakeAlembicContext:
    """Stand-in for ``alembic.context`` so ``env.py`` can be imported outside a migration.

    ``env.py`` runs its offline branch at import time when ``is_offline_mode()`` is true, so
    the fake absorbs ``configure``/``run_migrations`` and needs settings in the environment.
    """

    config: object | None = None

    def __init__(self) -> None:
        self.configures: list[dict[str, Any]] = []

    def is_offline_mode(self) -> bool:
        return True

    def configure(self, **kwargs: Any) -> None:
        self.configures.append(kwargs)

    def begin_transaction(self) -> contextlib.nullcontext[None]:
        return contextlib.nullcontext()

    def run_migrations(self) -> None:
        return None


class _FakeConnection:
    def __init__(self) -> None:
        self.ran: list[object] = []

    async def __aenter__(self) -> _FakeConnection:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def run_sync(self, fn: Callable[[_FakeConnection], None]) -> None:
        fn(self)
        self.ran.append(fn)


class _FakeEngine:
    def __init__(self) -> None:
        self.disposed = False

    def connect(self) -> _FakeConnection:
        return _FakeConnection()

    async def dispose(self) -> None:
        self.disposed = True


@pytest.fixture
def migrations_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    monkeypatch.setenv("CARACAL_API_TOKEN", "t" * 32)
    monkeypatch.setenv("DATABASE_URL", "postgresql://caracal:secret@db:5432/caracal")
    monkeypatch.setenv("REDIS_URL", "redis://:secret@redis:6379/0")
    monkeypatch.setattr(alembic, "context", _FakeAlembicContext())
    monkeypatch.delitem(sys.modules, "caracal.migrations.env", raising=False)
    module = importlib.import_module("caracal.migrations.env")
    try:
        yield module
    finally:
        sys.modules.pop("caracal.migrations.env", None)


@pytest.mark.parametrize(
    ("schema", "expected"),
    [
        ("public", "public"),
        ("caracal_t_x", "caracal_t_x, public"),
    ],
)
def test_migrations_env_pins_the_same_search_path(
    migrations_env: ModuleType,
    make_settings: MakeSettings,
    monkeypatch: pytest.MonkeyPatch,
    schema: str,
    expected: str,
) -> None:
    module = migrations_env
    settings = make_settings(CARACAL_DB_SCHEMA=schema)
    captured: dict[str, Any] = {}
    created: list[_FakeEngine] = []

    def fake_create_async_engine(url: str, **kwargs: Any) -> _FakeEngine:
        captured["url"] = url
        captured.update(kwargs)
        engine = _FakeEngine()
        created.append(engine)
        return engine

    monkeypatch.setattr(module, "get_settings", lambda: settings)
    monkeypatch.setattr(module, "create_async_engine", fake_create_async_engine)
    monkeypatch.setattr(module, "do_run_migrations", lambda connection: None)

    asyncio.run(module.run_async_migrations())

    assert captured == {
        "url": settings.DATABASE_URL.get_secret_value(),
        "poolclass": NullPool,
        "connect_args": {"server_settings": {"search_path": expected}},
    }
    assert len(created) == 1
    assert created[0].disposed is True


class _FakeBind:
    def __init__(self, resolved: object) -> None:
        self.resolved = resolved
        self.statements: list[object] = []

    def scalar(self, statement: object) -> object:
        self.statements.append(statement)
        return self.resolved


class _FakeMigrationContext:
    def __init__(self, *, as_sql: bool) -> None:
        self.as_sql = as_sql


class _FakeMigrationOp:
    def __init__(self, resolved: object, *, as_sql: bool = False) -> None:
        self.resolved = resolved
        self.as_sql = as_sql
        self.executed: list[str] = []
        self.bind = _FakeBind(resolved)
        self.bind_requested = False

    def execute(self, statement: str) -> None:
        self.executed.append(statement)

    def get_context(self) -> _FakeMigrationContext:
        return _FakeMigrationContext(as_sql=self.as_sql)

    def get_bind(self) -> _FakeBind:
        self.bind_requested = True
        return self.bind


def test_require_vector_type_raises_when_the_type_is_invisible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_op = _FakeMigrationOp(resolved=None)
    monkeypatch.setattr(VOICEPRINTS_MIGRATION, "op", fake_op)

    with pytest.raises(RuntimeError, match="pgvector not visible on search_path"):
        VOICEPRINTS_MIGRATION._require_vector_type()

    assert fake_op.executed == ["CREATE EXTENSION IF NOT EXISTS vector"]
    assert fake_op.bind_requested is True
    assert len(fake_op.bind.statements) == 1


def test_require_vector_type_accepts_a_resolvable_vector_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_op = _FakeMigrationOp(resolved="vector")
    monkeypatch.setattr(VOICEPRINTS_MIGRATION, "op", fake_op)

    VOICEPRINTS_MIGRATION._require_vector_type()

    assert fake_op.executed == ["CREATE EXTENSION IF NOT EXISTS vector"]
    assert fake_op.bind.statements


def test_require_vector_type_skips_the_check_for_offline_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_op = _FakeMigrationOp(resolved=None, as_sql=True)
    monkeypatch.setattr(VOICEPRINTS_MIGRATION, "op", fake_op)

    VOICEPRINTS_MIGRATION._require_vector_type()

    assert fake_op.executed == ["CREATE EXTENSION IF NOT EXISTS vector"]
    assert fake_op.bind_requested is False
