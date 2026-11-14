"""Redis Streams queue, keys, locks, events, cancellation and rate limiting (docs/05 §2)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
from tests.fakes.redis import RecordingRedis, all_keys, make_fake_redis

from caracal.errors import CaracalError, ErrorCode
from caracal.queue import (
    JobQueue,
    Keys,
    is_cancelled,
    publish_event,
    rate_limit,
    redis_lock,
    set_cancel,
    subscribe_events,
    whisper_lock,
)

PREFIX = "t3st:"


@pytest.fixture
def keys() -> Keys:
    return Keys(PREFIX)


@pytest.fixture
async def redis() -> AsyncIterator[RecordingRedis]:
    client = make_fake_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def queue(redis: RecordingRedis, keys: Keys) -> JobQueue:
    return JobQueue(redis, keys)


def test_key_prefixes_and_joins_parts(keys: Keys) -> None:
    assert keys.key("jobs") == "t3st:jobs"
    assert keys.key("worker", "host-1") == "t3st:worker:host-1"
    assert keys.jobs_stream() == "t3st:jobs"
    assert keys.worker("w1") == "t3st:worker:w1"
    assert keys.lock_whisper() == "t3st:lock:whisper"
    assert keys.lock_retention() == "t3st:lock:retention"
    job_id = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert keys.events(job_id) == "t3st:events:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert keys.cancel(job_id) == "t3st:cancel:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert keys.search("brave", "abc123") == "t3st:search:brave:abc123"
    assert keys.ratelimit("deadbeef") == "t3st:ratelimit:deadbeef"


@pytest.mark.parametrize(
    "part",
    ["", " ", "a b", "jobs*", "jobs?", "jobs[0]", "jobs]", "\tjobs"],
)
def test_key_rejects_empty_glob_and_whitespace_parts(keys: Keys, part: str) -> None:
    with pytest.raises(ValueError):
        keys.key("jobs", part)


async def test_enqueue_read_and_ack(queue: JobQueue, redis: RecordingRedis) -> None:
    await queue.ensure_group()
    job_id = uuid.uuid4()

    entry_id = await queue.enqueue(job_id, "pipeline")
    assert entry_id

    deliveries = await queue.read("worker-a", count=1, block_ms=10)
    assert len(deliveries) == 1
    delivery = deliveries[0]
    assert delivery.job_id == job_id
    assert delivery.kind == "pipeline"
    assert delivery.entry_id == entry_id
    assert delivery.times_delivered == 1

    await queue.touch("worker-a", entry_id)
    assert await queue.reclaim("worker-b", min_idle_ms=60_000) == []

    await queue.ack(entry_id)
    assert await redis.xlen(queue.keys.jobs_stream()) == 0
    pending = await redis.xpending(queue.keys.jobs_stream(), queue.group)
    assert pending["pending"] == 0


async def test_reclaim_reports_delivery_count(queue: JobQueue) -> None:
    await queue.ensure_group()
    job_id = uuid.uuid4()
    entry_id = await queue.enqueue(job_id, "pipeline")
    first = await queue.read("worker-a", count=1, block_ms=10)
    assert first[0].times_delivered == 1

    reclaimed = await queue.reclaim("worker-b", min_idle_ms=0)

    assert len(reclaimed) == 1
    assert reclaimed[0].entry_id == entry_id
    assert reclaimed[0].job_id == job_id
    assert reclaimed[0].times_delivered == 2


async def test_publish_and_subscribe_events(queue: JobQueue, redis: RecordingRedis, keys: Keys) -> None:
    job_id = uuid.uuid4()
    subscription = subscribe_events(redis, keys, job_id)
    assert isinstance(subscription, AsyncIterator)
    first = asyncio.create_task(anext(subscription))
    await asyncio.sleep(0.05)

    await publish_event(redis, keys, job_id, {"type": "status", "status": "running", "pct": 0.0})
    assert await asyncio.wait_for(first, timeout=1.0) == {
        "type": "status",
        "status": "running",
        "pct": 0.0,
    }

    second = asyncio.create_task(anext(subscription))
    await publish_event(redis, keys, job_id, {"type": "end", "status": "done", "pct": 100.0})
    assert await asyncio.wait_for(second, timeout=1.0) == {"type": "end", "status": "done", "pct": 100.0}

    await subscription.aclose()
    assert redis.published == [keys.events(job_id), keys.events(job_id)]


async def test_cancel_flag_round_trip(redis: RecordingRedis, keys: Keys) -> None:
    job_id = uuid.uuid4()
    assert await is_cancelled(redis, keys, job_id) is False
    await set_cancel(redis, keys, job_id)
    assert await is_cancelled(redis, keys, job_id) is True
    ttl = await redis.ttl(keys.cancel(job_id))
    assert 86_000 <= ttl <= 86_400


async def test_rate_limit_allows_then_blocks(redis: RecordingRedis, keys: Keys) -> None:
    assert await rate_limit(redis, keys, "ident", limit=2) == (True, 0)
    assert await rate_limit(redis, keys, "ident", limit=2) == (True, 0)

    allowed, retry_after = await rate_limit(redis, keys, "ident", limit=2)

    assert allowed is False
    assert 1 <= retry_after <= 60
    assert await redis.ttl(keys.ratelimit("ident")) > 0


async def test_whisper_lock_excludes_a_second_acquirer(redis: RecordingRedis, keys: Keys) -> None:
    async with whisper_lock(redis, keys, acquire_timeout_s=0.5):
        assert await redis.get(keys.lock_whisper()) is not None
        with pytest.raises(CaracalError) as excinfo:
            async with whisper_lock(redis, keys, acquire_timeout_s=0.05):
                raise AssertionError("the second acquirer must not enter")
    assert excinfo.value.code is ErrorCode.DEPENDENCY_DOWN
    assert await redis.get(keys.lock_whisper()) is None


async def test_lock_release_never_deletes_another_owners_token(redis: RecordingRedis, keys: Keys) -> None:
    async with redis_lock(redis, keys.lock_whisper(), acquire_timeout_s=0.5):
        await redis.set(keys.lock_whisper(), "someone-else")
    assert await redis.get(keys.lock_whisper()) == b"someone-else"


async def test_lock_renewal_extends_the_ttl(redis: RecordingRedis, keys: Keys) -> None:
    async with redis_lock(
        redis,
        keys.lock_whisper(),
        ttl_ms=1000,
        renew_every_s=0.05,
        acquire_timeout_s=0.5,
    ):
        await asyncio.sleep(0.35)
        ttl = await redis.pttl(keys.lock_whisper())
        assert ttl > 700, f"renewal should keep the TTL high, got {ttl}"


async def test_lock_can_be_acquired_after_release(redis: RecordingRedis, keys: Keys) -> None:
    async with redis_lock(redis, keys.lock_retention(), acquire_timeout_s=0.5):
        pass
    async with redis_lock(redis, keys.lock_retention(), acquire_timeout_s=0.5):
        assert await redis.get(keys.lock_retention()) is not None


async def test_every_key_and_channel_uses_the_prefix(
    redis: RecordingRedis, keys: Keys, queue: JobQueue
) -> None:
    job_id = uuid.uuid4()
    await queue.ensure_group()
    entry_id = await queue.enqueue(job_id, "pipeline")
    await queue.read("worker-a", count=1, block_ms=10)
    await queue.touch("worker-a", entry_id)
    await queue.ack(entry_id)

    await set_cancel(redis, keys, job_id)
    async with whisper_lock(redis, keys, acquire_timeout_s=0.5):
        pass
    await rate_limit(redis, keys, "ident", limit=5)
    await publish_event(redis, keys, job_id, {"type": "status", "status": "running"})

    keys_seen = await all_keys(redis)
    assert keys_seen, "the test must have created at least one key"
    assert all(key.startswith(PREFIX) for key in keys_seen), keys_seen
    assert redis.published and all(channel.startswith(PREFIX) for channel in redis.published)
    assert "XADD" in redis.command_names()
