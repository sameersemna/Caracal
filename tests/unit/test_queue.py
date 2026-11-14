"""Redis Streams queue, keys, locks, events, cancellation and rate limiting (docs/05 §2)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable

import pytest
from tests.fakes.redis import RecordingRedis, all_keys, make_fake_redis, text

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.queue import (
    JobQueue,
    Keys,
    is_cancelled,
    make_redis,
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
    assert keys.lock_translate(job_id) == ("t3st:lock:translate:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
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

    assert await queue.touch("worker-a", entry_id) is True
    assert await queue.reclaim("worker-b", min_idle_ms=60_000) == []

    assert await queue.ack("worker-a", entry_id) is True
    assert await redis.xlen(queue.keys.jobs_stream()) == 0
    pending = await redis.xpending(queue.keys.jobs_stream(), queue.group)
    assert pending["pending"] == 0


async def test_enqueue_carries_a_payload_round_trip(queue: JobQueue, redis: RecordingRedis) -> None:
    """Verdict 35 fix 1: a translate entry carries its own languages, never a shared file."""
    await queue.ensure_group()
    job_id = uuid.uuid4()

    await queue.enqueue(job_id, "translate", payload={"target_langs": ["de", "ar"]})
    await queue.enqueue(job_id, "pipeline")

    deliveries = await queue.read("worker-a", count=2, block_ms=10)
    assert [delivery.kind for delivery in deliveries] == ["translate", "pipeline"]
    assert deliveries[0].payload == {"target_langs": ["de", "ar"]}
    assert deliveries[1].payload == {}

    entries = await redis.xrange(queue.keys.jobs_stream())
    raw = {key: value for key, value in entries[0][1].items()}
    assert text(raw[b"job_id"]) == str(job_id)
    assert raw[b"payload"] == b'{"target_langs":["de","ar"]}'


async def test_malformed_entry_payload_is_a_caracal_error(queue: JobQueue, redis: RecordingRedis) -> None:
    await queue.ensure_group()
    await redis.xadd(
        queue.keys.jobs_stream(),
        {"job_id": str(uuid.uuid4()), "kind": "translate", "payload": "{not json"},
    )

    with pytest.raises(CaracalError) as excinfo:
        await queue.read("worker-a", count=1, block_ms=10)

    assert excinfo.value.code is ErrorCode.INTERNAL
    assert excinfo.value.message == "malformed queue entry"


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


async def _pending_owner(redis: RecordingRedis, queue: JobQueue) -> str:
    rows = await redis.xpending_range(
        queue.keys.jobs_stream(),
        queue.group,
        min="-",
        max="+",
        count=10,
    )
    assert len(rows) == 1
    return text(rows[0]["consumer"])


async def test_steal_back_touch_by_the_old_owner_is_refused_and_keeps_the_new_owner(
    queue: JobQueue, redis: RecordingRedis
) -> None:
    """Verdict 04 A1: a stalled consumer must not take the entry back from its new owner."""
    await queue.ensure_group()
    job_id = uuid.uuid4()
    entry_id = await queue.enqueue(job_id, "pipeline")
    await queue.read("worker-a", count=1, block_ms=10)
    stolen = await queue.reclaim("worker-b", min_idle_ms=0)
    assert stolen[0].times_delivered == 2
    assert await _pending_owner(redis, queue) == "worker-b"

    assert await queue.touch("worker-a", entry_id) is False

    assert await _pending_owner(redis, queue) == "worker-b"


async def test_steal_back_ack_by_the_old_owner_leaves_the_new_owners_entry_pending(
    queue: JobQueue, redis: RecordingRedis
) -> None:
    """Verdict 04 A1: XACK must not let a stale consumer delete the new owner's entry."""
    await queue.ensure_group()
    job_id = uuid.uuid4()
    entry_id = await queue.enqueue(job_id, "pipeline")
    await queue.read("worker-a", count=1, block_ms=10)
    await queue.reclaim("worker-b", min_idle_ms=0)

    assert await queue.ack("worker-a", entry_id) is False

    assert await redis.xlen(queue.keys.jobs_stream()) == 1
    assert await _pending_owner(redis, queue) == "worker-b"
    pending = await redis.xpending(queue.keys.jobs_stream(), queue.group)
    assert pending["pending"] == 1

    assert await queue.ack("worker-b", entry_id) is True
    assert await redis.xlen(queue.keys.jobs_stream()) == 0
    pending = await redis.xpending(queue.keys.jobs_stream(), queue.group)
    assert pending["pending"] == 0


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


async def test_rate_limit_increments_and_sets_the_window_in_one_command(
    redis: RecordingRedis, keys: Keys
) -> None:
    """Verdict 04 A2: INCR and EXPIRE must be one atomic script, never two round-trips."""
    redis.commands.clear()

    assert await rate_limit(redis, keys, "atomic", limit=5) == (True, 0)

    assert redis.command_names() == ["EVAL"]
    assert await redis.get(keys.ratelimit("atomic")) == b"1"
    assert 1 <= await redis.ttl(keys.ratelimit("atomic")) <= 60


async def test_rate_limit_repairs_a_counter_left_without_a_ttl(redis: RecordingRedis, keys: Keys) -> None:
    """Verdict 04 A2: a counter seeded by a crashed INCR (TTL -1) must get a window."""
    key = keys.ratelimit("stale")
    await redis.set(key, 4)
    assert await redis.ttl(key) == -1

    assert await rate_limit(redis, keys, "stale", limit=10) == (True, 0)

    assert await redis.get(key) == b"5"
    assert 1 <= await redis.ttl(key) <= 60


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
    assert await queue.touch("worker-a", entry_id) is True
    assert await queue.ack("worker-a", entry_id) is True

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


def test_make_redis_sets_a_block_outliving_socket_timeout(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_from_url(url: str, **kwargs: object) -> object:
        captured["url"] = url
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("caracal.queue.redis_asyncio.from_url", fake_from_url)

    assert make_redis(settings) is sentinel
    assert captured == {
        "url": settings.REDIS_URL.get_secret_value(),
        "socket_timeout": 15.0,
        "socket_connect_timeout": 5.0,
        "health_check_interval": 30,
        "retry_on_timeout": True,
    }


def test_make_redis_scales_the_socket_timeout_with_the_block(
    make_settings: Callable[..., Settings],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings()
    captured: dict[str, object] = {}

    def fake_from_url(url: str, **kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("caracal.queue.redis_asyncio.from_url", fake_from_url)

    make_redis(settings, block_ms=9000)

    assert captured["socket_timeout"] == 19.0
