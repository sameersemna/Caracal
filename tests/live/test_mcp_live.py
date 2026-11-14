"""Live MCP check against a running ``caracal serve`` (docs/07 §3, card 15).

The judge runs this with ``CARACAL_LIVE_BASE_URL`` pointing at the real server and a real
worker consuming the queue. It uploads the public-domain jfk fixture through REST, then
drives ``analyze_audio`` and ``get_transcript`` over the streamable-HTTP MCP endpoint with
raw JSON-RPC (no SDK client dependency for the transport handshake).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from caracal.config import Settings, get_settings

pytestmark = pytest.mark.live

_AUDIO = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "jfk.wav"
_PROTOCOL_VERSION = "2026-07-28"


def _base_url() -> str:
    base = os.environ.get("CARACAL_LIVE_BASE_URL")
    if not base:
        pytest.skip("CARACAL_LIVE_BASE_URL is not set")
    return base.rstrip("/")


def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }


async def _initialize(client: httpx.AsyncClient, token: str, base: str) -> None:
    response = await client.post(
        f"{base}/mcp/",
        headers=_headers(token),
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "caracal-live", "version": "1"},
            },
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["result"]["serverInfo"]["name"] == "caracal"


async def _call(
    client: httpx.AsyncClient,
    token: str,
    base: str,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    response = await client.post(
        f"{base}/mcp/",
        headers=_headers(token),
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["isError"] is False, result
    structured = result.get("structuredContent")
    assert isinstance(structured, dict), result
    return structured


async def test_analyze_audio_reaches_done_and_serves_srt() -> None:
    get_settings.cache_clear()
    settings: Settings = get_settings()
    token = settings.CARACAL_API_TOKEN.get_secret_value()
    base = _base_url()

    async with httpx.AsyncClient(timeout=210.0) as client:
        with _AUDIO.open("rb") as handle:
            uploaded = await client.post(
                f"{base}/v1/uploads",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": (_AUDIO.name, handle, "audio/wav")},
            )
        assert uploaded.status_code == 201, uploaded.text
        upload_id = uploaded.json()["upload_id"]

        await _initialize(client, token, base)
        analyzed = await _call(
            client,
            token,
            base,
            "analyze_audio",
            {"upload_id": upload_id, "wait_seconds": 120},
        )

        assert analyzed["status"] == "done", analyzed
        summary = analyzed["summary"]
        assert summary["transcript_preview"]
        assert "country" in summary["transcript_preview"].lower()

        transcript = await _call(
            client,
            token,
            base,
            "get_transcript",
            {"job_id": analyzed["job_id"], "format": "srt"},
        )

    assert transcript["content"].strip()
    assert "-->" in transcript["content"]
