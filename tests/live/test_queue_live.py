"""Live Redis: queue, lock, events and cancellation under a throwaway prefix.

Run by the judge, never by the gate. Every key is created under ``caracal_t_<8 hex>:`` and
the ``finally`` block deletes exactly that prefix with ``SCAN``; no key outside it is touched.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis

from caracal.config import get_settings
from caracal.queue import (
    JobQueue,
    Keys,
    is_cancelled,
    make_redis,
    publish_event,
    rate_limit,
    set_cancel,
    whisper_lock,
)

pytestmark = pytest.mark.live


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    settings = get_settings()
    client: Redis = make_redis(settings)
    try:
        yield client
    finally:
        await client.aclose()


async def _scan(redis: Redis, match: str) -> list[str]:
    return [
        key.decode("utf-8") if isinstance(key, bytes) else key async for key in redis.scan_iter(match=match)
    ]


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


async def test_queue_lock_events_cancel_and_rate_limit(redis: Redis) -> None:
    prefix = f"caracal_t_{uuid.uuid4().hex[:8]}:"
    keys = Keys(prefix)
    queue = JobQueue(redis, keys)
    job_id = uuid.uuid4()
    try:
        await queue.ensure_group()
        entry_id = await queue.enqueue(job_id, "pipeline")

        deliveries = await queue.read("live-worker", count=1, block_ms=1_000)
        assert len(deliveries) == 1
        assert deliveries[0].job_id == job_id
        assert deliveries[0].entry_id == entry_id
        assert deliveries[0].times_delivered == 1

        assert await queue.touch("live-worker", entry_id) is True
        assert await queue.ack("live-worker", entry_id) is True
        assert await redis.xlen(keys.jobs_stream()) == 0

        async with whisper_lock(
            redis,
            keys,
            ttl_ms=5_000,
            renew_every_s=1.0,
            acquire_timeout_s=5.0,
        ):
            assert await redis.exists(keys.lock_whisper()) == 1
        assert await redis.exists(keys.lock_whisper()) == 0

        async with redis.pubsub() as pubsub:
            await pubsub.subscribe(keys.events(job_id))
            await pubsub.get_message(timeout=2)
            await publish_event(redis, keys, job_id, {"type": "status", "status": "running"})
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=2)
            assert message is not None
            assert json.loads(message["data"]) == {"type": "status", "status": "running"}

        assert (await is_cancelled(redis, keys, job_id)) is False
        await set_cancel(redis, keys, job_id)
        assert (await is_cancelled(redis, keys, job_id)) is True

        allowed, retry_after = await rate_limit(redis, keys, "live-ident", limit=1)
        assert (allowed, retry_after) == (True, 0)
        allowed, retry_after = await rate_limit(redis, keys, "live-ident", limit=1)
        assert allowed is False
        assert retry_after >= 1
    finally:
        async for key in redis.scan_iter(match=f"{prefix}*"):
            await redis.delete(key)
        assert await _scan(redis, f"{prefix}*") == []


async def test_steal_back_rejects_the_stale_owner(redis: Redis) -> None:
    """Verdict 04 A1 on real Redis: after B reclaims, A cannot touch or ack any more."""
    prefix = f"caracal_t_{uuid.uuid4().hex[:8]}:"
    keys = Keys(prefix)
    queue = JobQueue(redis, keys)
    job_id = uuid.uuid4()
    stream = keys.jobs_stream()
    try:
        await queue.ensure_group()
        entry_id = await queue.enqueue(job_id, "pipeline")

        first = await queue.read("live-worker-a", count=1, block_ms=1_000)
        assert [item.entry_id for item in first] == [entry_id]
        assert first[0].times_delivered == 1
        # Make sure the entry's idle time is measurably above XAUTOCLAIM's min-idle 0.
        await asyncio.sleep(0.02)

        stolen = await queue.reclaim("live-worker-b", min_idle_ms=0)
        assert [item.entry_id for item in stolen] == [entry_id]
        assert stolen[0].times_delivered == 2

        # A is still running and heartbeats: the owner check must refuse and change nothing.
        assert await queue.touch("live-worker-a", entry_id) is False
        pending = await redis.xpending_range(stream, queue.group, min="-", max="+", count=10)
        assert [(_text(row["message_id"]), _text(row["consumer"])) for row in pending] == [
            (entry_id, "live-worker-b")
        ]

        # A finishes late and acks: B's pending entry must survive untouched.
        assert await queue.ack("live-worker-a", entry_id) is False
        assert await redis.xlen(stream) == 1
        assert (await redis.xpending(stream, queue.group))["pending"] == 1
        pending = await redis.xpending_range(stream, queue.group, min="-", max="+", count=10)
        assert [_text(row["consumer"]) for row in pending] == ["live-worker-b"]

        # B still owns it and its ack succeeds.
        assert await queue.ack("live-worker-b", entry_id) is True
        assert await redis.xlen(stream) == 0
        assert (await redis.xpending(stream, queue.group))["pending"] == 0

        # A counter left without a TTL (crash between INCR and EXPIRE) is repaired.
        stale = keys.ratelimit("live-stale")
        await redis.set(stale, 4)
        assert await redis.ttl(stale) == -1
        assert await rate_limit(redis, keys, "live-stale", limit=10) == (True, 0)
        assert 1 <= await redis.ttl(stale) <= 60
    finally:
        async for key in redis.scan_iter(match=f"{prefix}*"):
            await redis.delete(key)
        assert await _scan(redis, f"{prefix}*") == []
