"""Authentication (S1) and the public health endpoint (docs/03 §1, docs/06 S1)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.contract.conftest import AUTH, TOKEN, Harness


async def test_missing_authorization_header_is_401(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/v1/jobs")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize(
    "header",
    [
        f"Bearer {TOKEN} ",  # trailing space changes the token
        f"Basic {TOKEN}",
        TOKEN,
        "Bearer",
        "",
    ],
)
async def test_malformed_or_wrong_authorization_is_401(
    make_harness: Callable[..., Harness], header: str
) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/v1/jobs", headers={"Authorization": header})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "unauthorized"


async def test_wrong_token_is_401(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/v1/jobs", headers={"Authorization": "Bearer " + "x" * 32})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_unauthenticated_unknown_v1_path_is_401_before_routing(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/v1/does-not-exist")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "unauthorized"


async def test_correct_token_reaches_the_route(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/v1/jobs", headers=AUTH)
    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


async def test_healthz_is_public(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["version"] == "0.1.0"
    assert payload["deps"] == {
        "postgres": "ok",
        "redis": "ok",
        "whisper": "ok",
        "ollama": "ok",
    }


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "http://localhost",
        "http://localhost:4321",
        "https://localhost:8443",
        "http://127.0.0.1",
        "http://127.0.0.1:9999",
        "https://caracal.example:8443",
    ],
)
async def test_mcp_origin_allows_absent_localhost_and_public_origin(
    make_harness: Callable[..., Harness], origin: str | None
) -> None:
    # Card 15 mounts the real MCP app at /mcp and authenticates it, so the guard is exercised
    # through a real ``initialize`` request (with Host: testserver explicitly allowed).
    h = make_harness(
        settings_overrides={
            "CARACAL_PUBLIC_BASE_URL": "https://caracal.example:8443",
            "CARACAL_ALLOWED_HOSTS": "testserver",
        }
    )
    headers = {
        **AUTH,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if origin is not None:
        headers["Origin"] = origin
    initialize = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2026-07-28",
            "capabilities": {},
            "clientInfo": {"name": "contract", "version": "1"},
        },
    }
    async with h.app.router.lifespan_context(h.app):
        async with h.client() as client:
            response = await client.post("/mcp/", headers=headers, json=initialize)
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "caracal"


@pytest.mark.parametrize(
    "origin",
    [
        "https://evil.example",
        "https://caracal.example",
        "https://caracal.example:9999",
        "ftp://localhost:21",
        "null",
    ],
)
async def test_mcp_origin_rejects_foreign_origins_with_the_envelope(
    make_harness: Callable[..., Harness], origin: str
) -> None:
    h = make_harness(settings_overrides={"CARACAL_PUBLIC_BASE_URL": "https://caracal.example:8443"})

    @h.app.get("/mcp/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    async with h.client() as client:
        response = await client.get("/mcp/ping", headers={**AUTH, "Origin": origin})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "unauthorized"


async def test_origin_is_only_checked_for_mcp_paths(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/healthz", headers={"Origin": "https://evil.example"})
    assert response.status_code == 200
