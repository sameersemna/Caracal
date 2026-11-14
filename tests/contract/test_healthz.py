"""``GET /healthz``: dependency probes, status codes and the 10 s cache (docs/03 §2)."""

from __future__ import annotations

from collections.abc import Callable

from tests.contract.conftest import Harness


async def _health(h: Harness) -> tuple[int, dict[str, object]]:
    async with h.client() as client:
        response = await client.get("/healthz")
    return response.status_code, response.json()


async def test_all_dependencies_up(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    status, payload = await _health(h)
    assert status == 200
    assert payload == {
        "ok": True,
        "version": "0.1.0",
        "deps": {"postgres": "ok", "redis": "ok", "whisper": "ok", "ollama": "ok"},
    }


async def test_whisper_500_is_down_but_http_stays_200(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    h.upstreams.whisper = "down"
    status, payload = await _health(h)
    assert status == 200
    deps = payload["deps"]
    assert isinstance(deps, dict)
    assert deps["whisper"] == "down"
    assert deps["postgres"] == "ok"
    assert payload["ok"] is True


async def test_whisper_timeout_is_down(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    h.upstreams.whisper = "timeout"
    status, payload = await _health(h)
    assert status == 200
    deps = payload["deps"]
    assert isinstance(deps, dict)
    assert deps["whisper"] == "down"


async def test_postgres_down_makes_ok_false(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    h.db["down"] = True
    status, payload = await _health(h)
    assert status == 200
    deps = payload["deps"]
    assert isinstance(deps, dict)
    assert deps["postgres"] == "down"
    assert payload["ok"] is False


async def test_redis_down_makes_ok_false(make_harness: Callable[..., Harness]) -> None:
    class BrokenRedis:
        async def ping(self) -> None:
            raise ConnectionError("redis is down")

    h = make_harness()
    h.deps.redis = BrokenRedis()  # type: ignore[assignment]
    status, payload = await _health(h)
    assert status == 200
    deps = payload["deps"]
    assert isinstance(deps, dict)
    assert deps["redis"] == "down"
    assert payload["ok"] is False


async def test_results_are_cached_for_ten_seconds_and_then_refreshed(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    _status, first = await _health(h)
    deps = first["deps"]
    assert isinstance(deps, dict)
    assert deps["whisper"] == "ok"
    calls_after_first = len(h.upstreams.calls)
    assert calls_after_first == 2

    h.upstreams.whisper = "down"
    h.clock.advance(5.0)
    _status, cached = await _health(h)
    cached_deps = cached["deps"]
    assert isinstance(cached_deps, dict)
    assert cached_deps["whisper"] == "ok"
    assert len(h.upstreams.calls) == calls_after_first

    h.clock.advance(6.0)
    _status, refreshed = await _health(h)
    refreshed_deps = refreshed["deps"]
    assert isinstance(refreshed_deps, dict)
    assert refreshed_deps["whisper"] == "down"
    assert len(h.upstreams.calls) == calls_after_first + 2
