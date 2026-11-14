"""Router: order, fallback, auth disabling, retry, cache, budget, cost and token bucket."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator

import pytest
from tests.fakes.redis import RecordingRedis, make_fake_redis
from tests.fakes.search import FakeClock, FakeProvider, RecordingSleep, make_hit

from caracal.config import Settings
from caracal.queue import Keys
from caracal.search import SearchError, SearchHit, SearchRouter

PREFIX = "t3st:"


def _settings(**overrides: object) -> Settings:
    data: dict[str, object] = {
        "CARACAL_API_TOKEN": "t" * 32,
        "DATABASE_URL": "postgresql://caracal:secret@db:5432/caracal",
        "REDIS_URL": "redis://:secret@redis:6379/0",
        "SEARCH_PROVIDERS": ["first", "second"],
        "SEARCH_CACHE_TTL_S": 3_600,
        "SEARCH_MAX_QUERIES_PER_JOB": 24,
    }
    data.update(overrides)
    return Settings(_env_file=None, **data)


@pytest.fixture
async def redis() -> AsyncIterator[RecordingRedis]:
    client = make_fake_redis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def keys() -> Keys:
    return Keys(PREFIX)


def _one_hit(provider: str) -> list[SearchHit]:
    return [make_hit(provider, rank=1, url=f"https://example.com/{provider}")]


async def test_router_uses_configured_order_and_normalises_the_query(
    redis: RecordingRedis, keys: Keys
) -> None:
    first = FakeProvider("first", [_one_hit("first")])
    second = FakeProvider("second", [_one_hit("second")])
    router = SearchRouter(_settings(), redis, keys, [second, first])

    result = await router.search("  spaced\t query   here ", job_id="job-1", limit=3)

    assert result.provider_used == "first"
    assert result.cached is False
    assert result.hits == _one_hit("first")
    assert first.calls == [("spaced query here", 3)]
    assert second.calls == []


@pytest.mark.parametrize(
    "error",
    [
        SearchError("auth", "no"),
        SearchError("unavailable", "down"),
        SearchError("bad_response", "garbage"),
    ],
)
async def test_router_falls_through_on_error_kinds(
    redis: RecordingRedis, keys: Keys, error: SearchError
) -> None:
    first = FakeProvider("first", [error])
    second = FakeProvider("second", [_one_hit("second")])
    router = SearchRouter(_settings(), redis, keys, [first, second])

    result = await router.search("query", job_id="job-1")

    assert result.provider_used == "second"
    assert result.hits == _one_hit("second")
    assert len(first.calls) == 1
    assert len(second.calls) == 1


async def test_router_falls_through_on_empty_results(redis: RecordingRedis, keys: Keys) -> None:
    first = FakeProvider("first", [[]])
    second = FakeProvider("second", [_one_hit("second")])
    router = SearchRouter(_settings(), redis, keys, [first, second])

    result = await router.search("query", job_id="job-1")

    assert result.provider_used == "second"
    assert len(first.calls) == 1


async def test_router_retries_rate_limited_once_then_succeeds(redis: RecordingRedis, keys: Keys) -> None:
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    first = FakeProvider(
        "first",
        [SearchError("rate_limited", "slow", retry_after=2.0), _one_hit("first")],
    )
    second = FakeProvider("second", [_one_hit("second")])
    router = SearchRouter(_settings(), redis, keys, [first, second], clock=clock, sleep=sleep)

    result = await router.search("query", job_id="job-1")

    assert result.provider_used == "first"
    assert len(first.calls) == 2
    assert sleep.delays == [2.0]
    assert second.calls == []


async def test_router_retries_rate_limited_once_then_falls_through(redis: RecordingRedis, keys: Keys) -> None:
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    first = FakeProvider(
        "first",
        [
            SearchError("rate_limited", "slow", retry_after=9.0),
            SearchError("rate_limited", "still slow", retry_after=1.0),
        ],
    )
    second = FakeProvider("second", [_one_hit("second")])
    router = SearchRouter(_settings(), redis, keys, [first, second], clock=clock, sleep=sleep)

    result = await router.search("query", job_id="job-1")

    assert result.provider_used == "second"
    assert len(first.calls) == 2
    assert sleep.delays == [5.0]


async def test_router_auth_failure_disables_provider_and_logs_once(
    redis: RecordingRedis, keys: Keys, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    first = FakeProvider("first", [SearchError("auth", "bad key")])
    second = FakeProvider("second", [_one_hit("second")])
    router = SearchRouter(_settings(), redis, keys, [first, second])

    await router.search("first query", job_id="job-1")
    await router.search("second query", job_id="job-1")

    assert len(first.calls) == 1
    assert len(second.calls) == 2
    assert caplog.text.count("search.auth_failed") == 1


async def test_router_cache_hit_skips_providers_and_budget(redis: RecordingRedis, keys: Keys) -> None:
    first = FakeProvider("first", [_one_hit("first")])
    second = FakeProvider("second", [_one_hit("second")])
    router = SearchRouter(_settings(), redis, keys, [first, second])

    await router.search("cache  query", job_id="job-1")
    cached = await router.search(" cache query ", job_id="job-1")

    assert cached.cached is True
    assert cached.provider_used == "first"
    assert cached.hits == _one_hit("first")
    assert cached.cost_usd == 0.0
    assert len(first.calls) == 1
    assert len(second.calls) == 0
    budget_key = keys.search_budget("job-1")
    assert await redis.get(budget_key) == b"1"
    cache_keys = [key async for key in redis.scan_iter(match=f"{PREFIX}search:first:*")]
    assert len(cache_keys) == 1
    ttl = await redis.ttl(cache_keys[0])
    assert 0 < ttl <= 3_600


async def test_router_budget_exhaustion_after_n_calls(
    redis: RecordingRedis, keys: Keys, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    provider = FakeProvider("first", [_one_hit("first")] * 3)
    router = SearchRouter(_settings(SEARCH_MAX_QUERIES_PER_JOB=2), redis, keys, [provider])

    first = await router.search("one", job_id="job-1")
    second = await router.search("two", job_id="job-1")
    third = await router.search("three", job_id="job-1")

    assert first.provider_used == "first"
    assert second.provider_used == "first"
    assert third.hits == []
    assert third.provider_used is None
    assert len(provider.calls) == 2
    assert await redis.get(keys.search_budget("job-1")) == b"3"
    assert 0 < await redis.ttl(keys.search_budget("job-1")) <= 86_400
    assert "search.budget_exhausted" in caplog.text


async def test_router_sums_provider_costs(redis: RecordingRedis, keys: Keys) -> None:
    first = FakeProvider("first", [[]], cost_per_call=0.01)
    second = FakeProvider("second", [_one_hit("second")], cost_per_call=0.02)
    router = SearchRouter(_settings(), redis, keys, [first, second])

    result = await router.search("query", job_id="job-1")

    assert result.provider_used == "second"
    assert result.cost_usd == pytest.approx(0.03)
    assert first.cost_usd == pytest.approx(0.01)
    assert second.cost_usd == pytest.approx(0.02)


async def test_router_skips_provider_without_a_key(redis: RecordingRedis, keys: Keys) -> None:
    brave = FakeProvider("brave", [_one_hit("brave")])
    exa = FakeProvider("exa", [_one_hit("exa")])
    settings = _settings(
        SEARCH_PROVIDERS=["brave", "exa"],
        BRAVE_API_KEY=None,
        EXA_API_KEY="exa-key",
    )
    router = SearchRouter(settings, redis, keys, [brave, exa])

    result = await router.search("query", job_id="job-1")

    assert result.provider_used == "exa"
    assert brave.calls == []
    assert len(exa.calls) == 1


async def test_router_returns_empty_without_budget_when_no_provider_is_usable(
    redis: RecordingRedis, keys: Keys
) -> None:
    router = SearchRouter(_settings(SEARCH_PROVIDERS=[]), redis, keys, [])

    result = await router.search("query", job_id="job-1")

    assert result.hits == []
    assert result.provider_used is None
    assert await redis.get(keys.search_budget("job-1")) is None


async def test_router_token_bucket_waits_for_brave_rate_limit(redis: RecordingRedis, keys: Keys) -> None:
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    provider = FakeProvider("brave", [_one_hit("brave")] * 6)
    settings = _settings(
        SEARCH_PROVIDERS=["brave"],
        BRAVE_API_KEY="brave-key",
        SEARCH_MAX_QUERIES_PER_JOB=10,
    )
    router = SearchRouter(settings, redis, keys, [provider], clock=clock, sleep=sleep)

    for index in range(6):
        result = await router.search(f"query number {index}", job_id="job-1")
        assert result.provider_used == "brave"

    assert sleep.delays == [pytest.approx(0.2)]


async def test_router_accepts_uuid_job_ids(redis: RecordingRedis, keys: Keys) -> None:
    provider = FakeProvider("first", [_one_hit("first")])
    router = SearchRouter(_settings(), redis, keys, [provider])
    job_id = uuid.uuid4()

    result = await router.search("query", job_id=job_id)

    assert result.provider_used == "first"
    assert await redis.get(keys.search_budget(job_id)) == b"1"


def test_search_budget_key_format(keys: Keys) -> None:
    assert keys.search_budget("job-1") == f"{PREFIX}search:budget:job-1"


async def test_router_caps_long_queries_at_400_chars(redis: RecordingRedis, keys: Keys) -> None:
    provider = FakeProvider("first", [_one_hit("first")])
    router = SearchRouter(_settings(), redis, keys, [provider])

    await router.search("x " * 500, job_id="job-1")

    assert len(provider.calls[0][0]) <= 400
    assert provider.calls[0][0] == " ".join(["x"] * 400)[:400].strip()


async def test_router_caches_the_provider_that_actually_answered(redis: RecordingRedis, keys: Keys) -> None:
    first = FakeProvider("first", [[]])
    second = FakeProvider("second", [_one_hit("second"), _one_hit("second")])
    router = SearchRouter(_settings(), redis, keys, [first, second])

    await router.search("query", job_id="job-1")
    cached = await router.search("query", job_id="job-1")

    assert cached.cached is True
    assert cached.provider_used == "second"
    assert len(first.calls) == 1
    assert len(second.calls) == 1


async def test_router_token_bucket_waits_for_exa_rate_limit(redis: RecordingRedis, keys: Keys) -> None:
    clock = FakeClock()
    sleep = RecordingSleep(clock)
    provider = FakeProvider("exa", [_one_hit("exa")] * 3)
    settings = _settings(SEARCH_PROVIDERS=["exa"], EXA_API_KEY="exa-key")
    router = SearchRouter(settings, redis, keys, [provider], clock=clock, sleep=sleep)

    for index in range(3):
        assert (await router.search(f"query {index}", job_id="job-1")).provider_used == "exa"

    assert sleep.delays == [pytest.approx(0.5)]


async def test_router_empty_query_is_a_no_op(redis: RecordingRedis, keys: Keys) -> None:
    provider = FakeProvider("first", [_one_hit("first")])
    router = SearchRouter(_settings(), redis, keys, [provider])

    result = await router.search("   \t  ", job_id="job-1")

    assert result.hits == []
    assert result.provider_used is None
    assert provider.calls == []
    assert await redis.get(keys.search_budget("job-1")) is None
