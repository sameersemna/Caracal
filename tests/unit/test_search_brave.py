"""Brave provider: exact request, real-fixture parsing, hygiene and error mapping (docs/04 §8)."""

from __future__ import annotations

import logging
from collections.abc import Callable

import httpx
import pytest
from tests.fakes.search import auth_error_response, recording_handler

from caracal.search import BraveProvider, SearchError

_QUERY = "All Ears English podcast co-host Lindsay"
_KEY = "fake-brave-key-0123456789"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_brave_request_shape_and_parsed_fixture_hits() -> None:
    requests: list[httpx.Request] = []
    async with _client(recording_handler("brave_lindsay", requests)) as http:
        hits = await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert str(request.url.copy_with(query=None)) == "https://api.search.brave.com/res/v1/web/search"
    assert request.url.params["q"] == _QUERY
    assert request.url.params["count"] == "5"
    assert request.url.params["extra_snippets"] == "true"
    assert request.headers["x-subscription-token"] == _KEY
    assert request.headers["accept"] == "application/json"

    assert len(hits) == 5
    first = hits[0]
    assert first.provider == "brave"
    assert first.query == _QUERY
    assert [hit.rank for hit in hits] == [1, 2, 3, 4, 5]
    assert first.url == "https://podcasts.apple.com/us/podcast/all-ears-english-podcast/id751574016"
    assert first.title == "All Ears English Podcast - Podcast - Apple Podcasts"
    assert "Lindsay McMahon" in first.snippet
    assert "Michelle Kaplan's" in first.snippet
    assert " … " in first.snippet
    assert first.published == "2026-11-19T08:00:00"
    assert first.text is None

    for hit in hits:
        assert "<strong>" not in hit.snippet
        assert "&#x27;" not in hit.snippet
        assert len(hit.snippet) <= 500
        assert len(hit.title) <= 300
        assert hit.url.startswith("https://")


async def test_brave_cleans_strong_and_quot_entities_in_jfk_fixture() -> None:
    requests: list[httpx.Request] = []
    async with _client(recording_handler("brave_jfk_quote", requests)) as http:
        hits = await BraveProvider(_KEY, http).search("JFK ask not quote", limit=3)

    assert len(hits) == 5
    assert "<strong>" not in hits[0].snippet
    assert "ask not what your country can do for you--ask what you can do for your country" in hits[0].snippet
    assert all("&quot;" not in hit.snippet for hit in hits)
    assert '"My fellow Americans: ask not' in hits[2].snippet


@pytest.mark.parametrize("status", [401, 403])
async def test_brave_unauthorized_statuses_are_auth(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "nope"})

    async with _client(handler) as http:
        with pytest.raises(SearchError) as caught:
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "auth"
    assert caught.value.status == status


async def test_brave_recorded_422_subscription_token_invalid_is_auth() -> None:
    async with _client(lambda request: auth_error_response("brave")) as http:
        with pytest.raises(SearchError) as caught:
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "auth"
    assert caught.value.status == 422


async def test_brave_422_without_subscription_code_is_bad_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": {"code": "INVALID_QUERY"}})

    async with _client(handler) as http:
        with pytest.raises(SearchError) as caught:
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "bad_response"


async def test_brave_429_reports_retry_after_in_seconds() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "2.5"})

    async with _client(handler) as http:
        with pytest.raises(SearchError) as caught:
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "rate_limited"
    assert caught.value.retry_after == 2.5


async def test_brave_429_falls_back_to_x_ratelimit_reset_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={}, headers={"x-ratelimit-reset": "7, 1414445"})

    async with _client(handler) as http:
        with pytest.raises(SearchError) as caught:
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "rate_limited"
    assert caught.value.retry_after == 7.0


async def test_brave_5xx_is_unavailable() -> None:
    async with _client(lambda request: httpx.Response(503, json={})) as http:
        with pytest.raises(SearchError) as caught:
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "unavailable"
    assert caught.value.status == 503


async def test_brave_timeout_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with _client(handler) as http:
        with pytest.raises(SearchError) as caught:
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "unavailable"


async def test_brave_unparsable_body_is_bad_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>not json</html>")

    async with _client(handler) as http:
        with pytest.raises(SearchError) as caught:
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert caught.value.kind == "bad_response"


async def test_brave_drops_non_http_urls_and_keeps_rank_contiguous() -> None:
    body = {
        "web": {
            "results": [
                {"url": "https://good.example/a", "title": "A"},
                {"url": "ftp://bad.example/b", "title": "B"},
                {"url": "javascript:alert(1)", "title": "C"},
                {"url": "/relative", "title": "D"},
                {"url": "http://good.example/e", "title": "E"},
            ]
        }
    }

    async with _client(lambda request: httpx.Response(200, json=body)) as http:
        hits = await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert [hit.url for hit in hits] == ["https://good.example/a", "http://good.example/e"]
    assert [hit.rank for hit in hits] == [1, 2]


async def test_brave_never_logs_the_key_or_the_error_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    async with _client(lambda request: auth_error_response("brave")) as http:
        with pytest.raises(SearchError):
            await BraveProvider(_KEY, http).search(_QUERY, limit=5)

    assert _KEY not in caplog.text
    assert "SUBSCRIPTION_TOKEN_INVALID" not in caplog.text
    assert "provided subscription token is invalid" not in caplog.text


async def test_brave_caps_title_and_snippet_and_uses_two_extra_snippets() -> None:
    body = {
        "web": {
            "results": [
                {
                    "url": "https://example.com/x",
                    "title": "T" * 400,
                    "description": "short description",
                    "extra_snippets": [
                        "first-extra-" + "a" * 100,
                        "second-extra-" + "b" * 100,
                        "third-extra-unique-" + "c" * 100,
                    ],
                }
            ]
        }
    }

    async with _client(lambda request: httpx.Response(200, json=body)) as http:
        hits = await BraveProvider(_KEY, http).search(_QUERY, limit=1)

    assert len(hits[0].title) == 300
    assert len(hits[0].snippet) <= 500
    assert "first-extra-" in hits[0].snippet
    assert "second-extra-" in hits[0].snippet
    assert "third-extra-unique-" not in hits[0].snippet


async def test_brave_strips_control_characters() -> None:
    body = {
        "web": {
            "results": [
                {
                    "url": "https://example.com/x",
                    "title": "A\x00B\x07C\x85D",
                    "description": "line\x0bone\ttwo\rthree\x85four",
                }
            ]
        }
    }

    async with _client(lambda request: httpx.Response(200, json=body)) as http:
        hits = await BraveProvider(_KEY, http).search(_QUERY, limit=1)

    assert hits[0].title == "ABCD"
    assert hits[0].snippet == "lineone two threefour"
