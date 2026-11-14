"""Redis keys, the Redis Streams job queue, locks, events, cancellation and rate limiting.

The Redis database is shared with other applications (docs/05 §2). Every key and channel is
built by :class:`Keys`, which prefixes ``REDIS_PREFIX`` and rejects glob/whitespace characters
in any part, so this module can never touch a foreign key.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from redis import asyncio as redis_asyncio
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.log import get_logger

logger = get_logger("caracal.queue")

_STREAM_MAXLEN = 10_000
_CANCEL_TTL_S = 86_400
_RATE_WINDOW_S = 60
_DEFAULT_LOCK_TTL_MS = 960_000
_FORBIDDEN_KEY_CHARS = frozenset("*?[]")
_ACQUIRE_POLL_S = 0.05

#: The consumer blocks for ``block_ms``; the socket must outlive that block or redis-py
#: raises ``TimeoutError`` at the blocking deadline (verdict 25, measured: BLOCK 5000 timed
#: out at 5.01 s with the implicit 5 s read timeout).
_DEFAULT_BLOCK_MS = 5000
_SOCKET_TIMEOUT_GRACE_S = 10.0
_SOCKET_CONNECT_TIMEOUT_S = 5.0
_HEALTH_CHECK_INTERVAL_S = 30

# Release only when we still own the lock; never delete another owner's key.
_RELEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

# Renew only when we still own the lock.
_RENEW_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

# Heartbeat an entry only when ``ARGV[2]`` is still the pending owner (docs/05 §2).
# KEYS[1] = stream, ARGV[1] = group, ARGV[2] = consumer, ARGV[3] = entry id.
_TOUCH_LUA = """
local p = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[3], ARGV[3], 1)
if p[1] == nil or p[1][2] ~= ARGV[2] then return 0 end
redis.call('XCLAIM', KEYS[1], ARGV[1], ARGV[2], 0, ARGV[3], 'JUSTID')
return 1
"""

# Acknowledge only when ``ARGV[2]`` is still the pending owner: a stale consumer must never
# remove the entry a newer owner is working on (docs/05 §2).
_ACK_LUA = """
local p = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[3], ARGV[3], 1)
if p[1] == nil or p[1][2] ~= ARGV[2] then return 0 end
redis.call('XACK', KEYS[1], ARGV[1], ARGV[3])
redis.call('XDEL', KEYS[1], ARGV[3])
return 1
"""

# One round-trip for count and window: a counter can never be left without a TTL when the
# connection drops between INCR and EXPIRE (docs/05 §2). ARGV[1] = window seconds.
_RATE_LIMIT_LUA = """
local count = redis.call('INCR', KEYS[1])
local ttl = redis.call('TTL', KEYS[1])
if ttl == -1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
    ttl = tonumber(ARGV[1])
end
return {count, ttl}
"""


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _redis_error(operation: str, exc: RedisError) -> CaracalError:
    return CaracalError(
        ErrorCode.DEPENDENCY_DOWN,
        f"redis {operation} failed",
        {"operation": operation},
        retryable=True,
    )


def make_redis(settings: Settings, *, block_ms: int = _DEFAULT_BLOCK_MS) -> Redis:
    """The one place that builds a Redis client (verdict 25 D4).

    ``XREADGROUP BLOCK`` holds the connection for ``block_ms``; redis-py's implicit 5 s
    read timeout would fire first and the worker would log ``queue read failed`` every
    five seconds. The socket timeout is therefore the block time plus a 10 s grace.
    """
    if block_ms < 0:
        raise ValueError("block_ms must not be negative")
    return redis_asyncio.from_url(
        settings.REDIS_URL.get_secret_value(),
        socket_timeout=block_ms / 1000.0 + _SOCKET_TIMEOUT_GRACE_S,
        socket_connect_timeout=_SOCKET_CONNECT_TIMEOUT_S,
        health_check_interval=_HEALTH_CHECK_INTERVAL_S,
        retry_on_timeout=True,
    )


class Keys:
    """Build every Caracal Redis key and channel name from one prefix."""

    def __init__(self, prefix: str) -> None:
        if not prefix:
            raise ValueError("Redis prefix must not be empty")
        self.prefix = prefix

    def key(self, *parts: str) -> str:
        for part in parts:
            if not isinstance(part, str) or not part:
                raise ValueError("Redis key parts must be non-empty strings")
            if any(char.isspace() for char in part):
                raise ValueError("Redis key parts must not contain whitespace")
            if any(char in _FORBIDDEN_KEY_CHARS for char in part):
                raise ValueError("Redis key parts must not contain glob characters")
        return f"{self.prefix}{':'.join(parts)}"

    def jobs_stream(self) -> str:
        return self.key("jobs")

    def worker(self, worker_id: str) -> str:
        return self.key("worker", worker_id)

    def lock_whisper(self) -> str:
        return self.key("lock", "whisper")

    def lock_retention(self) -> str:
        return self.key("lock", "retention")

    def events(self, job_id: uuid.UUID) -> str:
        return self.key("events", str(job_id))

    def cancel(self, job_id: uuid.UUID) -> str:
        return self.key("cancel", str(job_id))

    def search(self, provider: str, qhash: str) -> str:
        return self.key("search", provider, qhash)

    def search_budget(self, job_id: uuid.UUID | str) -> str:
        return self.key("search", "budget", str(job_id))

    def ratelimit(self, ident_hash: str) -> str:
        return self.key("ratelimit", ident_hash)


@dataclass(frozen=True)
class Delivery:
    """One stream entry handed to a consumer."""

    entry_id: str
    job_id: uuid.UUID
    kind: str
    times_delivered: int


def _fields(raw: Mapping[object, object]) -> dict[str, str]:
    return {_text(key): _text(value) for key, value in raw.items()}


def _delivery(entry_id: object, raw: Mapping[object, object], times_delivered: int) -> Delivery:
    fields = _fields(raw)
    try:
        job_id = uuid.UUID(fields["job_id"])
        kind = fields["kind"]
    except (KeyError, ValueError) as exc:
        raise CaracalError(
            ErrorCode.INTERNAL,
            "malformed queue entry",
            {"entry_id": _text(entry_id)},
        ) from exc
    return Delivery(
        entry_id=_text(entry_id),
        job_id=job_id,
        kind=kind,
        times_delivered=times_delivered,
    )


class JobQueue:
    """Our own Redis Streams queue: one stream, one consumer group (docs/05 §2)."""

    def __init__(self, redis: Redis, keys: Keys, group: str = "workers") -> None:
        self.redis = redis
        self.keys = keys
        self.group = group

    async def ensure_group(self) -> None:
        """Create the consumer group (and stream) at id ``0`` so pre-existing jobs are read."""
        try:
            await self.redis.xgroup_create(
                self.keys.jobs_stream(),
                self.group,
                id="0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise _redis_error("xgroup_create", exc) from exc
        except RedisError as exc:
            raise _redis_error("xgroup_create", exc) from exc

    async def enqueue(self, job_id: uuid.UUID, kind: str) -> str:
        """Append a job to the stream, trimming it approximately to 10 000 entries."""
        try:
            entry_id = await self.redis.xadd(
                self.keys.jobs_stream(),
                {
                    "job_id": str(job_id),
                    "kind": kind,
                    "enqueued_at": datetime.now(UTC).isoformat(timespec="milliseconds"),
                },
                maxlen=_STREAM_MAXLEN,
                approximate=True,
            )
        except RedisError as exc:
            raise _redis_error("xadd", exc) from exc
        return _text(entry_id)

    async def read(self, consumer: str, *, count: int, block_ms: int) -> list[Delivery]:
        """Read new entries for ``consumer``; ``block_ms`` bounds the wait."""
        try:
            raw = await self.redis.xreadgroup(
                self.group,
                consumer,
                {self.keys.jobs_stream(): ">"},
                count=count,
                block=block_ms,
            )
        except RedisError as exc:
            raise _redis_error("xreadgroup", exc) from exc
        if not raw:
            return []
        deliveries: list[Delivery] = []
        streams = cast("Any", raw)
        for _stream, entries in streams.items() if isinstance(streams, dict) else streams:
            for entry_id, fields in entries:
                deliveries.append(_delivery(entry_id, cast("Mapping[object, object]", fields), 1))
        return deliveries

    async def touch(self, consumer: str, entry_id: str) -> bool:
        """Reset an entry's idle time, but only while ``consumer`` still owns it.

        Returns ``True`` when the entry is still pending for ``consumer`` (and was re-claimed
        for it), ``False`` when another consumer owns it or it is no longer pending.
        """
        try:
            result = await self.redis.eval(
                _TOUCH_LUA,
                1,
                self.keys.jobs_stream(),
                self.group,
                consumer,
                entry_id,
            )
        except RedisError as exc:
            raise _redis_error("touch", exc) from exc
        return int(cast("int", result)) == 1

    async def ack(self, consumer: str, entry_id: str) -> bool:
        """Acknowledge and delete an entry, but only while ``consumer`` still owns it.

        Returns ``True`` when this consumer's pending entry was acknowledged and deleted,
        ``False`` (doing nothing) when another consumer owns it or it is no longer pending.
        """
        try:
            result = await self.redis.eval(
                _ACK_LUA,
                1,
                self.keys.jobs_stream(),
                self.group,
                consumer,
                entry_id,
            )
        except RedisError as exc:
            raise _redis_error("ack", exc) from exc
        return int(cast("int", result)) == 1

    async def reclaim(self, consumer: str, *, min_idle_ms: int) -> list[Delivery]:
        """Take over entries idle longer than ``min_idle_ms`` (dead consumer recovery)."""
        stream = self.keys.jobs_stream()
        claimed: list[tuple[str, Mapping[object, object]]] = []
        cursor = "0-0"
        try:
            while True:
                result = await self.redis.xautoclaim(
                    stream,
                    self.group,
                    consumer,
                    min_idle_ms,
                    start_id=cursor,
                    count=100,
                )
                cursor = _text(result[0])
                entries = result[1] or []
                for entry_id, raw_fields in entries:
                    claimed.append((_text(entry_id), cast("Mapping[object, object]", raw_fields)))
                if cursor == "0-0":
                    break
            pending = await self.redis.xpending_range(stream, self.group, min="-", max="+", count=10_000)
        except RedisError as exc:
            raise _redis_error("xautoclaim", exc) from exc

        delivered: dict[str, int] = {}
        for row in pending or []:
            delivered[_text(row["message_id"])] = int(row["times_delivered"])
        return [
            _delivery(entry_id, raw_fields, delivered.get(entry_id, 1)) for entry_id, raw_fields in claimed
        ]


async def _acquire(
    redis: Redis,
    key: str,
    token: str,
    *,
    ttl_ms: int,
    acquire_timeout_s: float | None,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = None if acquire_timeout_s is None else loop.time() + acquire_timeout_s
    while True:
        try:
            acquired = await redis.set(key, token, nx=True, px=ttl_ms)
        except RedisError as exc:
            raise _redis_error("set", exc) from exc
        if acquired:
            return
        if deadline is not None and loop.time() >= deadline:
            raise CaracalError(
                ErrorCode.DEPENDENCY_DOWN,
                "timed out acquiring redis lock",
                {"key": key},
                retryable=True,
            )
        await asyncio.sleep(_ACQUIRE_POLL_S)


async def _renew(redis: Redis, key: str, token: str, ttl_ms: int, every_s: float) -> None:
    while True:
        await asyncio.sleep(every_s)
        try:
            extended = await redis.eval(_RENEW_LUA, 1, key, token, ttl_ms)
        except RedisError as exc:
            logger.warning("redis lock renewal failed", extra={"key": key, "error": str(exc)})
            continue
        if int(cast("int", extended)) == 0:
            logger.warning("redis lock lost before release", extra={"key": key})
            return


async def _release(redis: Redis, key: str, token: str) -> None:
    try:
        await redis.eval(_RELEASE_LUA, 1, key, token)
    except RedisError as exc:
        raise _redis_error("eval", exc) from exc


@asynccontextmanager
async def redis_lock(
    redis: Redis,
    key: str,
    *,
    ttl_ms: int = _DEFAULT_LOCK_TTL_MS,
    renew_every_s: float = 60.0,
    acquire_timeout_s: float | None = None,
) -> AsyncIterator[None]:
    """Own ``key`` for the duration of the block, renewing the TTL and releasing safely."""
    token = uuid.uuid4().hex
    await _acquire(redis, key, token, ttl_ms=ttl_ms, acquire_timeout_s=acquire_timeout_s)
    renewal = asyncio.create_task(_renew(redis, key, token, ttl_ms, renew_every_s))
    try:
        yield
    finally:
        renewal.cancel()
        with suppress(asyncio.CancelledError):
            await renewal
        await _release(redis, key, token)


def whisper_lock(
    redis: Redis,
    keys: Keys,
    *,
    ttl_ms: int = _DEFAULT_LOCK_TTL_MS,
    renew_every_s: float = 60.0,
    acquire_timeout_s: float | None = None,
) -> AbstractAsyncContextManager[None]:
    """The whisper.cpp single-slot lock (`<prefix>lock:whisper`)."""
    return redis_lock(
        redis,
        keys.lock_whisper(),
        ttl_ms=ttl_ms,
        renew_every_s=renew_every_s,
        acquire_timeout_s=acquire_timeout_s,
    )


async def publish_event(redis: Redis, keys: Keys, job_id: uuid.UUID, event: dict[str, Any]) -> None:
    """Publish one JSON progress event on the job's channel."""
    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    try:
        await redis.publish(keys.events(job_id), payload)
    except RedisError as exc:
        raise _redis_error("publish", exc) from exc


async def subscribe_events(redis: Redis, keys: Keys, job_id: uuid.UUID) -> AsyncIterator[dict[str, Any]]:
    """Yield decoded events from a job's channel until the caller stops iterating."""
    channel = keys.events(job_id)
    async with redis.pubsub() as pubsub:
        await pubsub.subscribe(channel)
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message is None:
                continue
            data = message.get("data")
            text = data.decode("utf-8") if isinstance(data, bytes) else str(data)
            payload = json.loads(text)
            if isinstance(payload, dict):
                yield cast("dict[str, Any]", payload)


async def set_cancel(redis: Redis, keys: Keys, job_id: uuid.UUID) -> None:
    """Raise the job's cancel flag for one day."""
    try:
        await redis.set(keys.cancel(job_id), "1", ex=_CANCEL_TTL_S)
    except RedisError as exc:
        raise _redis_error("set", exc) from exc


async def is_cancelled(redis: Redis, keys: Keys, job_id: uuid.UUID) -> bool:
    """True when the job's cancel flag is set."""
    try:
        return bool(await redis.exists(keys.cancel(job_id)))
    except RedisError as exc:
        raise _redis_error("exists", exc) from exc


async def rate_limit(
    redis: Redis,
    keys: Keys,
    ident_hash: str,
    *,
    limit: int,
    window_s: int = _RATE_WINDOW_S,
) -> tuple[bool, int]:
    """Fixed-window counter; returns ``(allowed, retry_after_s)``.

    Increment and window setup run as one Lua script, so a counter can never survive without
    a TTL (which would lock that token+IP out permanently).
    """
    key = keys.ratelimit(ident_hash)
    try:
        raw = await redis.eval(_RATE_LIMIT_LUA, 1, key, window_s)
    except RedisError as exc:
        raise _redis_error("rate_limit", exc) from exc
    count, ttl = (int(value) for value in cast("list[int]", raw))
    if count <= limit:
        return True, 0
    retry_after = ttl if ttl > 0 else window_s
    return False, max(1, retry_after)
