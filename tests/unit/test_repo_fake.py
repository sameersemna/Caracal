"""The repository contract executed against the in-memory fakes (docs/07 §3-§5)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from tests.fakes.repo import make_fake_repos
from tests.repo_contract import CASES, Repos


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.__name__)
async def test_fake_repo_contract(case: Callable[[Repos], Awaitable[None]]) -> None:
    repos = make_fake_repos()
    await repos.cleanup()
    await case(repos)
