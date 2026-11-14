"""SSE stream contract: snapshot-first, no lost events, keep-alive and terminal end (docs/03 §2)."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable

import pytest

from caracal.api.routes import job_event_stream
from caracal.queue import publish_event
from tests.contract.conftest import AUTH, Harness

URL = "https://example.com/talk"


def _payload(chunk: str) -> dict[str, object]:
    _, _, encoded = chunk.partition("data: ")
    return json.loads(encoded)


async def _next_event(stream: AsyncIterator[str]) -> str:
    while True:
        chunk = await anext(stream)
        if not chunk.startswith(":"):
            return chunk


async def test_generator_snapshot_then_stage_then_end(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})
    stream = job_event_stream(h.settings, h.deps, job.id, ping_interval_s=1.0)

    snapshot = await anext(stream)
    assert snapshot.startswith("event: status\ndata: ")
    snapshot_payload = _payload(snapshot)
    assert snapshot_payload["job_id"] == str(job.id)
    assert snapshot_payload["status"] == "queued"

    await publish_event(
        h.redis,
        h.keys,
        job.id,
        {"type": "stage", "stage": "transcribe", "status": "running", "pct": 25, "message": "asr"},
    )
    event = await _next_event(stream)
    assert event.startswith("event: stage\ndata: ")
    assert _payload(event)["stage"] == "transcribe"
    assert _payload(event)["pct"] == 25

    await publish_event(h.redis, h.keys, job.id, {"type": "end"})
    end = await _next_event(stream)
    assert end.startswith("event: end\ndata: ")

    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_event_published_between_subscribe_and_snapshot_is_not_lost(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})

    async def publish_between() -> None:
        await publish_event(
            h.redis,
            h.keys,
            job.id,
            {"type": "stage", "stage": "ingest", "status": "done", "pct": 5},
        )

    stream = job_event_stream(
        h.settings,
        h.deps,
        job.id,
        ping_interval_s=1.0,
        after_subscribe=publish_between,
    )
    snapshot = await anext(stream)
    assert _payload(snapshot)["status"] == "queued"

    event = await _next_event(stream)
    assert event.startswith("event: stage\ndata: ")
    assert _payload(event)["stage"] == "ingest"


async def test_keep_alive_comment_when_idle(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})
    stream = job_event_stream(h.settings, h.deps, job.id, ping_interval_s=0.05)

    await anext(stream)
    ping = await asyncio.wait_for(anext(stream), timeout=1.0)
    assert ping == ": ping\n\n"
    await stream.aclose()


async def test_terminal_job_over_http_is_snapshot_then_end(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})
    await h.repos.jobs.set_status(job.id, "done")

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job.id}/events", headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    body = response.text
    assert body.count("event: status") == 1
    assert body.count("event: end") == 1
    assert body.index("event: status") < body.index("event: end")
    start_payload = _payload(body[body.index("event: status") : body.index("event: end")])
    assert start_payload["status"] == "done"


async def test_unknown_job_events_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{uuid.uuid4()}/events", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
