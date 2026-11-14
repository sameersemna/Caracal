"""Exa provider: exact request, fixture parsing, cost accounting and error mapping (docs/04 §8)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
import pytest
from tests.fakes.search import auth_error_response, recording_handler

from caracal.search import ExaProvider, SearchError

_QUERY = "All Ears English podcast co-host Lindsay"
_KEY = "fake-exa-key-0123456789"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_exa_request_shape_and_parsed_fixture_hits() -> None:
    requests: list[httpx.Request] = []
    provider: ExaProvider
    async with _client(recording_handler("exa_lindsay", requests)) as http:
        provider = ExaProvider(_KEY, http)
        hits = await provider.search(_QUERY, limit=5)

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.host == "api.exa.ai"
    assert request.url.path == "/search"
    assert request.headers["x-api-key"] == _KEY
    body = request.read().decode("utf-8")
    assert '"query":"All Ears English podcast co-host Lindsay"' in body
    assert '"numResults":5' in body
    assert '"type":"auto"' in body
    assert '"contents":{"text":{"maxCharacters":2000}}' in body

    assert len(hits) == 5
    first = hits[0]
    assert first.provider == "exa"
    assert first.query == _QUERY
    assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]
    assert first.url == "https://www.allearsenglish.com/about-english-conversation-listening-podcast/"
    assert "About All Ears English" in first.snippet
    assert hits[3].published == "2025-04-08T09:00:00.000Z"
    for hit in hits:
        assert hit.text is not None
        assert len(hit.text) <= 2_000
        assert len(hit.snippet) <= 500
        assert len(hit.title) <= 300
    assert provider.cost_usd == 0.007


async def test_exa_cost_counter_accumulates_across_searches() -> None:
    async with _client(recording_handler("exa_lindsay", [])) as http:
        provider = ExaProvider(_KEY, http)
        await provider.search(_QUERY, limit=5)
        await provider.search(_QUERY, limit=5)

    assert provider.cost_usd == pytest.approx(0.014)


async def test_exa_recorded_invalid_key_is_auth() -> None:
    async with _client(lambda request: auth_error_response("exa")) as http:
        with pytest.raises(SearchError) as caught:
            await ExaProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "auth"
    assert caught.value.status == 401


async def test_exa_403_is_auth() -> None:
    async with _client(lambda request: httpx.Response(403, json={})) as http:
        with pytest.raises(SearchError) as caught:
            await ExaProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "auth"


async def test_exa_429_reports_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={}, headers={"retry-after": "1"})

    async with _client(handler) as http:
        with pytest.raises(SearchError) as caught:
            await ExaProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "rate_limited"
    assert caught.value.retry_after == 1.0


async def test_exa_503_is_unavailable() -> None:
    async with _client(lambda request: httpx.Response(503, json={})) as http:
        with pytest.raises(SearchError) as caught:
            await ExaProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "unavailable"


async def test_exa_connect_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with _client(handler) as http:
        with pytest.raises(SearchError) as caught:
            await ExaProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "unavailable"


async def test_exa_unparsable_body_is_bad_response() -> None:
    async with _client(lambda request: httpx.Response(200, content=b"not json")) as http:
        with pytest.raises(SearchError) as caught:
            await ExaProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "bad_response"


async def test_exa_drops_non_http_urls() -> None:
    body = {
        "results": [
            {"url": "mailto:someone@example.com", "title": "Mail", "text": "x"},
            {"url": "https://good.example/a", "title": "A", "text": "hello"},
        ],
        "costDollars": {"total": 0.001},
    }

    async with _client(lambda request: httpx.Response(200, json=body)) as http:
        hits = await ExaProvider(_KEY, http).search(_QUERY, limit=5)

    assert [hit.url for hit in hits] == ["https://good.example/a"]
    assert hits[0].rank == 1
    assert hits[0].text == "hello"


async def test_exa_never_logs_the_key_or_the_error_body(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)

    async with _client(lambda request: auth_error_response("exa")) as http:
        with pytest.raises(SearchError):
            await ExaProvider(_KEY, http).search(_QUERY, limit=5)

    assert _KEY not in caplog.text
    assert "Invalid API key" not in caplog.text
    assert "INVALID_API_KEY" not in caplog.text


async def test_exa_caps_text_title_and_snippet() -> None:
    body = {
        "results": [
            {
                "url": "https://example.com/x",
                "title": "T" * 400,
                "text": "word " * 1_000,
            }
        ]
    }

    async with _client(lambda request: httpx.Response(200, json=body)) as http:
        hits = await ExaProvider(_KEY, http).search(_QUERY, limit=1)

    assert len(hits[0].text or "") == 2_000
    assert len(hits[0].title) == 300
    assert len(hits[0].snippet) == 500
