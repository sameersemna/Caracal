"""Live search checks (judge-run, never by the gate): real Brave/Exa and the router cache.

Requires ``BRAVE_API_KEY`` and ``EXA_API_KEY`` in the real environment. Redis is used with a
throwaway prefix and every key is deleted afterwards.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from redis.asyncio import Redis

from caracal.config import Settings, get_settings
from caracal.queue import Keys, make_redis
from caracal.search import BraveProvider, ExaProvider, SearchHit, SearchRouter

pytestmark = pytest.mark.live

_QUERY = "All Ears English podcast co-host Lindsay"


def _keys(settings: Settings) -> Keys:
    return Keys(f"{settings.REDIS_PREFIX}live-{uuid.uuid4().hex[:8]}:")


async def _cleanup(redis: Redis, keys: Keys) -> None:
    found = [key async for key in redis.scan_iter(match=f"{keys.prefix}*")]
    if found:
        await redis.delete(*found)


def _mentions_mcmahon(hits: list[SearchHit]) -> bool:
    return any("McMahon" in hit.snippet or (hit.text is not None and "McMahon" in hit.text) for hit in hits)


@pytest.mark.live
async def test_live_providers_return_lindsay_hits() -> None:
    settings = get_settings()
    if settings.BRAVE_API_KEY is None or settings.EXA_API_KEY is None:
        pytest.skip("BRAVE_API_KEY and EXA_API_KEY are required")

    async with httpx.AsyncClient() as http:
        brave_hits = await BraveProvider(settings.BRAVE_API_KEY.get_secret_value(), http).search(
            _QUERY, limit=5
        )
        exa_hits = await ExaProvider(settings.EXA_API_KEY.get_secret_value(), http).search(_QUERY, limit=5)

    assert brave_hits, "Brave returned no hits"
    assert exa_hits, "Exa returned no hits"
    assert _mentions_mcmahon(brave_hits)
    assert _mentions_mcmahon(exa_hits)


@pytest.mark.live
async def test_live_router_prefers_brave_and_serves_the_second_call_from_cache() -> None:
    settings = get_settings()
    if settings.BRAVE_API_KEY is None or settings.EXA_API_KEY is None:
        pytest.skip("BRAVE_API_KEY and EXA_API_KEY are required")

    redis: Redis = make_redis(settings)
    keys = _keys(settings)
    try:
        router_settings = settings.model_copy(update={"SEARCH_PROVIDERS": ["brave", "exa"]})
        async with httpx.AsyncClient() as http:
            providers = [
                BraveProvider(settings.BRAVE_API_KEY.get_secret_value(), http),
                ExaProvider(settings.EXA_API_KEY.get_secret_value(), http),
            ]
            router = SearchRouter(router_settings, redis, keys, providers)

            first = await router.search(_QUERY, job_id=uuid.uuid4())
            second = await router.search(_QUERY, job_id=uuid.uuid4())

        assert first.provider_used == "brave"
        assert first.cached is False
        assert first.hits
        assert _mentions_mcmahon(first.hits)
        assert second.cached is True
        assert second.provider_used == "brave"
        assert second.hits == first.hits
        assert second.cost_usd == 0.0
    finally:
        await _cleanup(redis, keys)
        await redis.aclose()
