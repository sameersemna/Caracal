"""Shared test setup: isolate every test from the developer's shell and .env file."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from caracal.config import Settings, get_settings

_ENV_FIELDS = tuple(Settings.model_fields)

REQUIRED_ENV: dict[str, str] = {
    "CARACAL_API_TOKEN": "t" * 32,
    "DATABASE_URL": "postgresql://caracal:secret@db:5432/caracal",
    "REDIS_URL": "redis://:secret@redis:6379/0",
}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Clear Caracal env vars, ignore any real ``.env`` and reset the settings cache."""
    for name in _ENV_FIELDS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def make_settings() -> object:
    """Build ``Settings`` with valid required values and no ``.env`` file."""

    def _make(**overrides: object) -> Settings:
        data: dict[str, object] = {**REQUIRED_ENV, **overrides}
        return Settings(_env_file=None, **data)

    return _make
