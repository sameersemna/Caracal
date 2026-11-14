"""Search router: provider order, fallback, Redis cache, budget and rate limiting (docs/04 §8).

The router is the only place that knows the provider priority and the per-job query budget.
Providers stay dumb HTTP adapters; every cross-cutting concern (cache, budget, token bucket,
auth disabling) lives here so the verify stage sees one stable ``search()`` call.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from redis.asyncio import Redis

from caracal.config import Settings
from caracal.log import get_logger
from caracal.queue import Keys
from caracal.search.base import SearchError, SearchHit, SearchProvider, normalise_query

logger = get_logger("caracal.search.router")

_BUDGET_TTL_S = 86_400
_MAX_RETRY_S = 5.0
_DEFAULT_RETRY_S = 1.0
_PROVIDER_RATES: dict[str, float] = {"brave": 5.0, "exa": 2.0}
_DEFAULT_RATE = 2.0


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one router search."""

    hits: list[SearchHit]
    provider_used: str | None
    cached: bool
    cost_usd: float


@dataclass
class _TokenBucket:
    """A small in-process token bucket; capacity is one second of requests."""

    rate: float
    capacity: float
    tokens: float
    updated: float | None = None

    @classmethod
    def for_rate(cls, rate: float) -> _TokenBucket:
        return cls(rate=rate, capacity=rate, tokens=rate)

    def _refill(self, now: float) -> None:
        if self.updated is None:
            self.updated = now
            return
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = now

    def wait_time(self, now: float) -> float:
        """Seconds until a token is available (0 when one is available now)."""
        self._refill(now)
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.rate

    def consume(self, now: float) -> None:
        """Take one token; after a caller-supplied wait, grant one regardless of clock."""
        self._refill(now)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
        else:
            self.tokens = 0.0
            self.updated = now


def _cost(provider: SearchProvider) -> float:
    value = getattr(provider, "cost_usd", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _cache_hash(query: str, limit: int) -> str:
    return hashlib.sha1(f"{query}\n{limit}".encode(), usedforsecurity=False).hexdigest()


def _decode(raw: object) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _hits_from_cache(raw: object) -> list[SearchHit]:
    document = json.loads(_decode(raw))
    if not isinstance(document, list):
        raise ValueError("cached search is not a list")
    return [SearchHit.from_dict(item) for item in document]


class SearchRouter:
    """Ordered, cached and budgeted search across the configured providers."""

    def __init__(
        self,
        settings: Settings,
        redis: Redis,
        keys: Keys,
        providers: Sequence[SearchProvider],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._redis = redis
        self._keys = keys
        self._providers = {provider.name: provider for provider in providers}
        self._clock = clock
        self._sleep = sleep
        self._disabled: set[str] = set()
        self._auth_logged: set[str] = set()
        self._buckets: dict[str, _TokenBucket] = {
            name: _TokenBucket.for_rate(_PROVIDER_RATES.get(name, _DEFAULT_RATE)) for name in self._providers
        }

    def _has_credentials(self, name: str) -> bool:
        if name == "brave":
            return self._settings.BRAVE_API_KEY is not None
        if name == "exa":
            return self._settings.EXA_API_KEY is not None
        return True

    def _active_providers(self) -> list[SearchProvider]:
        ordered: list[SearchProvider] = []
        for name in self._settings.SEARCH_PROVIDERS:
            provider = self._providers.get(name)
            if provider is None or name in self._disabled or not self._has_credentials(name):
                continue
            ordered.append(provider)
        return ordered

    async def _write_cache(self, provider: str, qhash: str, hits: list[SearchHit]) -> None:
        ttl = self._settings.SEARCH_CACHE_TTL_S
        if ttl <= 0:
            return
        payload = json.dumps([hit.to_dict() for hit in hits], ensure_ascii=False)
        await self._redis.set(self._keys.search(provider, qhash), payload, ex=ttl)

    async def _read_cache(self, provider: str, qhash: str) -> list[SearchHit] | None:
        key = self._keys.search(provider, qhash)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            hits = _hits_from_cache(raw)
        except (TypeError, ValueError):
            return None
        return hits or None

    async def _consume_budget(self, job_id: str | uuid.UUID) -> bool:
        key = self._keys.search_budget(job_id)
        count = await self._redis.incr(key)
        await self._redis.expire(key, _BUDGET_TTL_S)
        if count > self._settings.SEARCH_MAX_QUERIES_PER_JOB:
            logger.info(
                "search.budget_exhausted",
                extra={"job_id": str(job_id), "budget": self._settings.SEARCH_MAX_QUERIES_PER_JOB},
            )
            return False
        return True

    async def _wait_for_token(self, provider: str) -> None:
        bucket = self._buckets.setdefault(
            provider, _TokenBucket.for_rate(_PROVIDER_RATES.get(provider, _DEFAULT_RATE))
        )
        wait = bucket.wait_time(self._clock())
        if wait > 0.0:
            await self._sleep(wait)
        bucket.consume(self._clock())

    def _handle_failure(self, provider: str, error: SearchError) -> None:
        if error.kind != "auth":
            return
        self._disabled.add(provider)
        if provider in self._auth_logged:
            return
        self._auth_logged.add(provider)
        logger.warning("search.auth_failed", extra={"provider": provider, "status": error.status})

    async def _attempt(
        self, provider: SearchProvider, query: str, limit: int
    ) -> tuple[list[SearchHit], float]:
        before = _cost(provider)
        await self._wait_for_token(provider.name)
        try:
            hits = await provider.search(query, limit=limit)
        except SearchError as error:
            if error.kind == "rate_limited":
                delay = min(
                    error.retry_after if error.retry_after is not None else _DEFAULT_RETRY_S, _MAX_RETRY_S
                )
                await self._sleep(max(0.0, delay))
                try:
                    hits = await provider.search(query, limit=limit)
                except SearchError as retry_error:
                    self._handle_failure(provider.name, retry_error)
                    return [], _cost(provider) - before
            else:
                self._handle_failure(provider.name, error)
                return [], _cost(provider) - before
        return hits, _cost(provider) - before

    async def search(self, query: str, *, job_id: str | uuid.UUID, limit: int = 5) -> SearchResult:
        """Search the providers in configured order, using cache and budget."""
        normalised = normalise_query(query)
        if not normalised:
            return SearchResult(hits=[], provider_used=None, cached=False, cost_usd=0.0)

        providers = self._active_providers()
        if not providers:
            return SearchResult(hits=[], provider_used=None, cached=False, cost_usd=0.0)

        qhash = _cache_hash(normalised, limit)
        for provider in providers:
            cached = await self._read_cache(provider.name, qhash)
            if cached is not None:
                return SearchResult(hits=cached, provider_used=provider.name, cached=True, cost_usd=0.0)

        if not await self._consume_budget(job_id):
            return SearchResult(hits=[], provider_used=None, cached=False, cost_usd=0.0)

        total_cost = 0.0
        for provider in self._active_providers():
            hits, cost = await self._attempt(provider, normalised, limit)
            total_cost += cost
            if hits:
                await self._write_cache(provider.name, qhash, hits)
                return SearchResult(
                    hits=hits,
                    provider_used=provider.name,
                    cached=False,
                    cost_usd=total_cost,
                )

        return SearchResult(hits=[], provider_used=None, cached=False, cost_usd=total_cost)


__all__ = ["SearchResult", "SearchRouter"]
