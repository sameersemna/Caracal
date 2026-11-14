"""Live-test environment: the real shell/``.env`` is required, not the unit-suite isolation.

``tests/conftest.py`` clears every Caracal variable for the hermetic unit suite. Every live
test needs the real configuration, so this override replaces that fixture once for the whole
``tests/live`` directory and only resets the cached :class:`Settings`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from caracal.config import get_settings


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
