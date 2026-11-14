"""Live Redis: queue, lock, events and cancellation under a throwaway prefix.

Run by the judge, never by the gate. Every key is created under ``caracal_t_<8 hex>:`` and
the ``finally`` block deletes exactly that prefix with ``SCAN``; no key outside it is touched.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
import redis.asyncio as redis_asyncio
from redis.asyncio import Redis

from caracal.config import get_settings
from caracal.queue import (
    JobQueue,
    Keys,
    is_cancelled,
    publish_event,
    rate_limit,
    set_cancel,
    whisper_lock,
)

pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    """Override the unit-suite isolation: live tests need the real ``.env``."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    settings = get_settings()
    client: Redis = redis_asyncio.from_url(settings.REDIS_URL.get_secret_value())
    try:
        yield client
    finally:
        await client.aclose()


async def _scan(redis: Redis, match: str) -> list[str]:
    return [
        key.decode("utf-8") if isinstance(key, bytes) else key async for key in redis.scan_iter(match=match)
    ]


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

        await queue.touch("live-worker", entry_id)
        await queue.ack(entry_id)
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
