"""MCP over streamable HTTP: auth-before-body, Host/Origin checks, real JSON-RPC (card 15).

Requests drive the real mounted MCP app through the ASGI transport, so the FastAPI
middleware stack (bearer + body cap + Origin guard), the SDK's DNS-rebinding protection and
the tool registry are all exercised together.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from caracal.pipeline.stages import JobOptions
from tests.contract.conftest import AUTH, Harness
from tests.contract.test_api_limits import CHUNK, _drive

PROTOCOL_VERSION = "2026-07-28"


def _initialize() -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "contract", "version": "1"},
        },
    }


def _headers(**extra: str) -> dict[str, str]:
    return {
        **AUTH,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        **extra,
    }


def _rpc_result(response: Any) -> dict[str, Any]:
    payload = response.json()
    assert "result" in payload, payload
    result = payload["result"]
    assert isinstance(result, dict)
    return result


async def test_mcp_without_bearer_is_401_before_reading_the_body(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    chunks = (b"x" * CHUNK for _ in range(4 * 1024 * 1024 // CHUNK))

    response, pulled_chunks, pulled_bytes = await _drive(
        h.app,
        "POST",
        "/mcp/",
        headers={"transfer-encoding": "chunked"},
        chunks=chunks,
    )

    assert response.status == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "unauthorized"
    assert pulled_chunks == 0
    assert pulled_bytes == 0


async def test_initialize_and_tools_call_get_job_over_http(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness(settings_overrides={"CARACAL_ALLOWED_HOSTS": "testserver"})
    job = await h.repos.jobs.create(
        source_kind="platform",
        source_url="https://example.com/talk",
        title="A talk",
        options=JobOptions().model_dump(),
    )

    async with h.app.router.lifespan_context(h.app):
        async with h.client() as client:
            initialized = await client.post("/mcp/", headers=_headers(), json=_initialize())
            assert initialized.status_code == 200
            server_info = _rpc_result(initialized)["serverInfo"]
            assert server_info["name"] == "caracal"

            notified = await client.post(
                "/mcp/",
                headers=_headers(),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            assert notified.status_code == 202

            called = await client.post(
                "/mcp/",
                headers=_headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "get_job", "arguments": {"job_id": str(job.id)}},
                },
            )

    assert called.status_code == 200
    result = _rpc_result(called)
    assert result["isError"] is False
    assert result["structuredContent"]["job_id"] == str(job.id)
    assert result["structuredContent"]["status"] == "queued"
    assert result["structuredContent"]["source"] == {
        "kind": "platform",
        "url": "https://example.com/talk",
        "title": "A talk",
    }


async def test_host_from_caracal_allowed_hosts_is_accepted(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness(
        settings_overrides={"CARACAL_ALLOWED_HOSTS": "promaxgb10-6116:11404"},
    )

    async with h.app.router.lifespan_context(h.app):
        async with h.client() as client:
            response = await client.post(
                "/mcp/",
                headers=_headers(Host="promaxgb10-6116:11404"),
                json=_initialize(),
            )

    assert response.status_code == 200
    assert _rpc_result(response)["serverInfo"]["name"] == "caracal"


async def test_unlisted_host_is_421(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()

    async with h.app.router.lifespan_context(h.app):
        async with h.client() as client:
            response = await client.post(
                "/mcp/",
                headers=_headers(Host="promaxgb10-6116:11404"),
                json=_initialize(),
            )

    assert response.status_code == 421
    assert "Invalid Host header" in response.text


async def test_foreign_origin_is_403_with_the_error_envelope(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness(settings_overrides={"CARACAL_ALLOWED_HOSTS": "testserver"})

    async with h.app.router.lifespan_context(h.app):
        async with h.client() as client:
            response = await client.post(
                "/mcp/",
                headers=_headers(Origin="https://evil.example"),
                json=_initialize(),
            )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "unauthorized"


async def test_mcp_body_over_one_mib_is_413(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    chunks = (b"x" * CHUNK for _ in range(4 * 1024 * 1024 // CHUNK))

    response, pulled_chunks, pulled_bytes = await _drive(
        h.app,
        "POST",
        "/mcp/",
        headers={**AUTH, "content-type": "application/json", "transfer-encoding": "chunked"},
        chunks=chunks,
    )

    assert response.status == 413
    assert response.json()["error"]["code"] == "too_large"
    assert pulled_bytes > 1024 * 1024
    assert pulled_chunks < 4 * 1024 * 1024 // CHUNK
    assert json.loads(response.body)["error"]["code"] == "too_large"


async def test_unknown_mcp_tool_is_an_error_result(make_harness: Callable[..., Harness]) -> None:
    h = make_harness(settings_overrides={"CARACAL_ALLOWED_HOSTS": "testserver"})

    async with h.app.router.lifespan_context(h.app):
        async with h.client() as client:
            await client.post("/mcp/", headers=_headers(), json=_initialize())
            response = await client.post(
                "/mcp/",
                headers=_headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "get_report", "arguments": {"job_id": str(uuid.uuid4())}},
                },
            )

    assert response.status_code == 200
    assert _rpc_result(response)["isError"] is True
