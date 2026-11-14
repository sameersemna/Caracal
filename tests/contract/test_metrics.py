"""``GET /metrics``: bearer protection and the Prometheus text contract (card T4.1)."""

from __future__ import annotations

import re
from collections.abc import Callable

from caracal.ops import MetricsRegistry
from tests.contract.conftest import AUTH, Harness

_METRIC_LINE = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?\d+(\.\d+)?")


async def test_metrics_requires_a_bearer_token(make_harness: Callable[..., Harness]) -> None:
    harness = make_harness()
    async with harness.client() as client:
        unauthorized = await client.get("/metrics")
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["code"] == "unauthorized"
        assert unauthorized.headers["www-authenticate"] == "Bearer"

        authorized = await client.get("/metrics", headers=AUTH)
        assert authorized.status_code == 200
        assert authorized.headers["content-type"].startswith("text/plain")
        assert "caracal_search_cost_usd_total 0" in authorized.text


async def test_metrics_renders_shared_counters(make_harness: Callable[..., Harness]) -> None:
    harness = make_harness()
    registry = MetricsRegistry(harness.redis, harness.keys)
    await registry.incr_job("done")
    await registry.observe_stage("transcribe", 3.5)
    await registry.incr_llm_call("cloud", "tool", False)
    await registry.incr_search_query("exa", cached=True)
    await registry.add_search_cost(0.002)
    await registry.add_whisper_seconds(11.0)

    async with harness.client() as client:
        response = await client.get("/metrics", headers=AUTH)

    assert response.status_code == 200
    lines = [line for line in response.text.splitlines() if line]
    for line in lines:
        assert _METRIC_LINE.fullmatch(line), line
    assert 'caracal_jobs_total{status="done"} 1' in lines
    assert 'caracal_stage_seconds_count{stage="transcribe"} 1' in lines
    assert 'caracal_stage_seconds_sum{stage="transcribe"} 3.5' in lines
    assert 'caracal_llm_calls_total{endpoint="cloud",mode="tool",fallback="false"} 1' in lines
    assert 'caracal_search_queries_total{provider="exa",cached="true"} 1' in lines
    assert "caracal_search_cost_usd_total 0.002" in lines
    assert "caracal_whisper_audio_seconds_total 11" in lines
