"""Fake search providers, clocks and recorded-fixture helpers (docs/07 §4).

Tests replay the real Brave/Exa recordings through ``httpx.MockTransport``; router tests use
:class:`FakeProvider`, which scripts one outcome per call and records the call arguments.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import httpx

from caracal.search.base import SearchHit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "search"


def load_fixture(name: str) -> dict[str, Any]:
    """Load one recorded search response (without the ``.json`` suffix)."""
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"fixture {name} is not a JSON object")
    return payload


def fixture_response(name: str) -> httpx.Response:
    """A response built from a recorded fixture, including its measured headers."""
    fixture = load_fixture(name)
    headers = fixture.get("headers")
    return httpx.Response(
        cast(int, fixture["status"]),
        json=fixture["body"],
        headers=cast(dict[str, str], headers) if isinstance(headers, dict) else {},
    )


def auth_error_response(provider: str) -> httpx.Response:
    """The recorded invalid-key body for ``brave`` (422) or ``exa`` (401)."""
    fixture = load_fixture("auth_errors")
    entry = cast(dict[str, Any], fixture[provider])
    return httpx.Response(cast(int, entry["status"]), json=entry["body"])


def recording_handler(
    fixture: str,
    requests: list[httpx.Request],
) -> Callable[[httpx.Request], httpx.Response]:
    """A MockTransport handler that records the request and replays a fixture."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return fixture_response(fixture)

    return handler


def make_hit(
    provider: str,
    *,
    rank: int,
    url: str = "https://example.com/story",
    title: str = "Example",
    snippet: str = "Example snippet",
    query: str = "q",
    published: str | None = None,
    text: str | None = None,
) -> SearchHit:
    return SearchHit(
        provider=provider,
        query=query,
        rank=rank,
        url=url,
        title=title,
        snippet=snippet,
        published=published,
        text=text,
    )


class FakeProvider:
    """A provider with one scripted outcome per call; exceptions are raised verbatim."""

    def __init__(
        self,
        name: str,
        outcomes: Sequence[list[SearchHit] | Exception] = (),
        *,
        cost_per_call: float = 0.0,
    ) -> None:
        self.name = name
        self.outcomes: list[list[SearchHit] | Exception] = list(outcomes)
        self.cost_per_call = cost_per_call
        self.calls: list[tuple[str, int]] = []
        self.cost_usd = 0.0

    async def search(self, query: str, *, limit: int) -> list[SearchHit]:
        self.calls.append((query, limit))
        self.cost_usd += self.cost_per_call
        if not self.outcomes:
            return []
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return list(outcome)


class FakeClock:
    """A monotonic clock advanced explicitly or by :class:`RecordingSleep`."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingSleep:
    """An async sleep that records requested delays and optionally advances a fake clock."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)
        if self.clock is not None:
            self.clock.advance(seconds)
