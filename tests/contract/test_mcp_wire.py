"""MCP results over the real streamable-HTTP transport (card 21, verdict 20).

The in-memory ``Client(server)`` path never runs the low-level server's ``CallToolResult``
validation, so a tool whose ``structuredContent`` is a JSON array passed the contract tests
and then failed for every real client ("Handler returned an invalid result"). These tests
drive the mounted ``/mcp`` app through the ASGI transport with plain JSON-RPC and require a
``result`` — the SDK turns a non-object payload into a JSON-RPC ``error``.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from typing import Any

from caracal.pipeline.stages import JobOptions, Stage
from caracal.storage import JobFiles
from tests.contract.conftest import AUTH, Harness

URL = "https://example.com/talk"
PROTOCOL_VERSION = "2026-07-28"
REQUIRED_TOOLS = frozenset(
    {"list_jobs", "get_job", "list_speakers", "enroll_speaker", "delete_speaker", "cancel_job"}
)


def _headers() -> dict[str, str]:
    return {
        **AUTH,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }


async def _initialise(client: Any) -> None:
    response = await client.post(
        "/mcp/",
        headers=_headers(),
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "contract", "version": "1"},
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "caracal"

    notified = await client.post(
        "/mcp/",
        headers=_headers(),
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert notified.status_code == 202


async def _call(
    client: Any,
    request_id: int,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """One ``tools/call``; a JSON-RPC error or a non-object payload fails the test."""
    response = await client.post(
        "/mcp/",
        headers=_headers(),
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "error" not in payload, (name, payload)
    assert payload.get("jsonrpc") == "2.0"
    assert payload.get("id") == request_id
    result = payload["result"]
    assert result["isError"] is False, (name, result)
    structured = result["structuredContent"]
    assert isinstance(structured, dict), (name, structured)
    return structured


def _unit(similarity: float) -> list[float]:
    rest = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return [similarity, rest, *([0.0] * 254)]


def _write_diarize(h: Harness, job_id: uuid.UUID, embeddings: dict[str, list[float]]) -> None:
    files = JobFiles(h.settings.CARACAL_DATA_DIR, job_id)
    files.ensure()
    files.write_json(
        files.stage(Stage.DIARIZE),
        {
            "status": "done",
            "reason": None,
            "output": {
                "num_speakers": len(embeddings),
                "turns": [],
                "exclusive_turns": [],
                "embeddings": embeddings,
            },
        },
    )


async def _job(h: Harness, *, status: str) -> uuid.UUID:
    record = await h.repos.jobs.create(
        source_kind="platform",
        source_url=URL,
        title="A talk",
        options=JobOptions().model_dump(),
    )
    await h.repos.jobs.set_status(record.id, status)
    return record.id


async def test_every_tool_returns_a_json_rpc_result_over_streamable_http(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness(settings_overrides={"CARACAL_ALLOWED_HOSTS": "testserver"})
    done_id = await _job(h, status="done")
    _write_diarize(h, done_id, {"S1": _unit(0.97)})
    queued_id = await _job(h, status="queued")
    called: set[str] = set()

    async with h.app.router.lifespan_context(h.app):
        async with h.client() as client:
            await _initialise(client)

            jobs = await _call(client, 2, "list_jobs", {})
            called.add("list_jobs")
            assert {item["job_id"] for item in jobs["items"]} == {str(done_id), str(queued_id)}
            assert jobs["next_cursor"] is None

            job = await _call(client, 3, "get_job", {"job_id": str(done_id)})
            called.add("get_job")
            assert job["job_id"] == str(done_id)
            assert job["status"] == "done"

            empty = await _call(client, 4, "list_speakers", {})
            called.add("list_speakers")
            assert empty == {"items": []}

            created = await _call(
                client,
                5,
                "enroll_speaker",
                {
                    "job_id": str(done_id),
                    "speaker_label": "S1",
                    "name": "Daniel  Levitin",
                    "notes": "Neuroscientist",
                },
            )
            called.add("enroll_speaker")
            assert created["name"] == "Daniel  Levitin"
            assert created["samples"] == 1
            assert uuid.UUID(created["id"])

            listed = await _call(client, 6, "list_speakers", {})
            assert listed == {"items": [created]}

            deleted = await _call(client, 7, "delete_speaker", {"speaker_id": created["id"]})
            called.add("delete_speaker")
            assert deleted == {"deleted": True, "voiceprints_removed": 1}

            cancelled = await _call(client, 8, "cancel_job", {"job_id": str(queued_id)})
            called.add("cancel_job")
            assert cancelled == {"job_id": str(queued_id), "status": "cancelling"}

    assert called == REQUIRED_TOOLS
