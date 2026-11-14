"""Worker loops: end-to-end processing, recovery, delivery caps, concurrency, shutdown."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from tests.fakes.redis import RecordingRedis, all_keys, make_fake_redis, text
from tests.fakes.repo import make_fake_repos
from tests.repo_contract import Repos

from caracal.config import Settings
from caracal.pipeline.stages import Stage, StageContext, StageResult, default_registry
from caracal.queue import JobQueue, Keys
from caracal.storage import JobFiles
from caracal.worker import Worker, WorkerIntervals

MakeSettings = Callable[..., Settings]
PREFIX = "t3st:"


async def _wait_for(
    predicate: Callable[[], Awaitable[bool]],
    *,
    limit_s: float = 5.0,
    interval: float = 0.01,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_s
    while loop.time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition was not met before the timeout")


def _fast_intervals(**overrides: Any) -> WorkerIntervals:
    values: dict[str, Any] = {
        "heartbeat_s": 0.05,
        "touch_s": 100.0,
        "reclaim_s": 0.05,
        "reclaim_min_idle_ms": 50,
        "block_ms": 10,
        "max_deliveries": 3,
        "shutdown_grace_s": 0.5,
    }
    values.update(overrides)
    return WorkerIntervals(**values)


async def _new_job(repos: Repos) -> uuid.UUID:
    record = await repos.jobs.create(
        source_kind="platform",
        source_url="https://example.com/talk",
        options={"depth": "standard"},
    )
    return record.id


async def _enqueue(redis: RecordingRedis, keys: Keys, job_id: uuid.UUID) -> str:
    queue = JobQueue(redis, keys)
    await queue.ensure_group()
    return await queue.enqueue(job_id, "pipeline")


def _done_stage(stage: Stage, calls: list[str]) -> Callable[[StageContext], Awaitable[StageResult]]:
    async def run(ctx: StageContext) -> StageResult:
        calls.append(stage.value)
        return StageResult(status="done")

    return run


@pytest.fixture
async def redis() -> Any:
    client = make_fake_redis()
    try:
        yield client
    finally:
        await client.aclose()


async def test_worker_processes_a_job_end_to_end_and_cleans_the_stream(
    make_settings: MakeSettings, tmp_path: Path, redis: RecordingRedis
) -> None:
    keys = Keys(PREFIX)
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    await _enqueue(redis, keys, job_id)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path, REDIS_PREFIX=PREFIX, CARACAL_WORKER_CONCURRENCY=1)
    worker = Worker(
        settings,
        redis,
        repos,
        registry=default_registry(),
        worker_id="worker-a",
        intervals=_fast_intervals(),
    )
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    try:
        await _wait_for(lambda: _status_is(repos, job_id, "done"))

        assert await redis.exists(keys.worker("worker-a")) == 1
        await _wait_for(lambda: _stream_is_empty(redis, keys))
        stages = await repos.jobs.list_stages(job_id)
        assert len(stages) == len(Stage)
        assert all(stage.status == "skipped" for stage in stages)
        assert await redis.exists(keys.worker("worker-a")) == 1
    finally:
        stop.set()
        await asyncio.wait_for(run, timeout=5)

    keys_seen = await all_keys(redis)
    assert keys_seen
    assert all(key.startswith(PREFIX) for key in keys_seen)
    assert redis.published
    assert all(channel.startswith(PREFIX) for channel in redis.published)


async def test_ownership_loss_stops_the_job_without_ack_and_keeps_the_new_owners_entry(
    make_settings: MakeSettings,
    tmp_path: Path,
    redis: RecordingRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verdict 04 fix 2: the stale worker returns "lost", writes nothing and does not ack."""
    caplog.set_level(logging.WARNING, logger="caracal.worker")
    keys = Keys(PREFIX)
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    entry_id = await _enqueue(redis, keys, job_id)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(ctx: StageContext) -> StageResult:
        started.set()
        await release.wait()
        return StageResult(status="done", output={"must": "not be written"})

    settings = make_settings(CARACAL_DATA_DIR=tmp_path, REDIS_PREFIX=PREFIX, CARACAL_WORKER_CONCURRENCY=1)
    worker = Worker(
        settings,
        redis,
        repos,
        registry={Stage.INGEST: blocked},
        worker_id="worker-a",
        intervals=_fast_intervals(),
    )
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)

        # Another worker reclaims the idle entry while worker-a is still inside the stage.
        stolen = await JobQueue(redis, keys).reclaim("worker-b", min_idle_ms=0)
        assert [item.entry_id for item in stolen] == [entry_id]
        assert stolen[0].times_delivered == 2

        release.set()
        await _wait_for(lambda: _all_tasks_finished(worker))
    finally:
        stop.set()
        await asyncio.wait_for(run, timeout=5)

    assert "job.lost" in caplog.text
    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.status == "running"
    stages = {row.name: row for row in await repos.jobs.list_stages(job_id)}
    assert stages["ingest"].status == "running"
    assert not JobFiles(tmp_path, job_id).stage(Stage.INGEST).exists()

    # The entry is still pending, still owned by worker-b: worker-a must not have acked it.
    assert await redis.xlen(keys.jobs_stream()) == 1
    pending = await redis.xpending_range(keys.jobs_stream(), "workers", min="-", max="+", count=10)
    assert [(text(row["message_id"]), text(row["consumer"])) for row in pending] == [(entry_id, "worker-b")]


async def _all_tasks_finished(worker: Worker) -> bool:
    return all(task.done() for task in worker._in_flight)


async def _status_is(repos: Repos, job_id: uuid.UUID, expected: str) -> bool:
    job = await repos.jobs.get(job_id)
    return job is not None and job.status == expected


async def _stream_is_empty(redis: RecordingRedis, keys: Keys) -> bool:
    return await redis.xlen(keys.jobs_stream()) == 0


async def test_job_enqueued_before_the_worker_starts_is_delivered(
    make_settings: MakeSettings, tmp_path: Path, redis: RecordingRedis
) -> None:
    keys = Keys(PREFIX)
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    # The API can enqueue before any worker created the consumer group.
    await JobQueue(redis, keys).enqueue(job_id, "pipeline")

    settings = make_settings(CARACAL_DATA_DIR=tmp_path, REDIS_PREFIX=PREFIX, CARACAL_WORKER_CONCURRENCY=1)
    worker = Worker(
        settings,
        redis,
        repos,
        registry=default_registry(),
        worker_id="worker-a",
        intervals=_fast_intervals(),
    )
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    try:
        await _wait_for(lambda: _status_is(repos, job_id, "done"))
        await _wait_for(lambda: _stream_is_empty(redis, keys))
    finally:
        stop.set()
        await asyncio.wait_for(run, timeout=5)


async def test_dead_consumer_entry_is_reclaimed_and_resumed(
    make_settings: MakeSettings, tmp_path: Path, redis: RecordingRedis
) -> None:
    keys = Keys(PREFIX)
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    entry_id = await _enqueue(redis, keys, job_id)
    queue = JobQueue(redis, keys)

    dead = await queue.read("worker-dead", count=1, block_ms=10)
    assert dead[0].entry_id == entry_id

    files = JobFiles(tmp_path, job_id)
    files.ensure()
    for stage in (Stage.INGEST, Stage.NORMALIZE):
        files.write_json(files.stage(stage), {"status": "done", "reason": None, "output": {}})
        await repos.jobs.upsert_stage(job_id, stage.value, status="done")
    await repos.jobs.set_status(job_id, "running")
    await asyncio.sleep(0.1)

    calls: list[str] = []
    registry = {
        Stage.INGEST: _done_stage(Stage.INGEST, calls),
        Stage.NORMALIZE: _done_stage(Stage.NORMALIZE, calls),
        Stage.TRANSCRIBE: _done_stage(Stage.TRANSCRIBE, calls),
        Stage.ASSEMBLE: _done_stage(Stage.ASSEMBLE, calls),
    }
    settings = make_settings(CARACAL_DATA_DIR=tmp_path, REDIS_PREFIX=PREFIX, CARACAL_WORKER_CONCURRENCY=1)
    worker = Worker(
        settings,
        redis,
        repos,
        registry=registry,
        worker_id="worker-b",
        intervals=_fast_intervals(),
    )
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    try:
        await _wait_for(lambda: _status_is(repos, job_id, "done"))
    finally:
        stop.set()
        await asyncio.wait_for(run, timeout=5)

    assert calls == ["transcribe", "assemble"]
    assert await redis.xlen(keys.jobs_stream()) == 0


async def test_entry_over_max_deliveries_fails_the_job(
    make_settings: MakeSettings, tmp_path: Path, redis: RecordingRedis
) -> None:
    keys = Keys(PREFIX)
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    await _enqueue(redis, keys, job_id)
    queue = JobQueue(redis, keys)
    await queue.read("worker-dead", count=1, block_ms=10)
    await repos.jobs.set_status(job_id, "running")
    await asyncio.sleep(0.1)

    settings = make_settings(CARACAL_DATA_DIR=tmp_path, REDIS_PREFIX=PREFIX, CARACAL_WORKER_CONCURRENCY=1)
    worker = Worker(
        settings,
        redis,
        repos,
        worker_id="worker-b",
        intervals=_fast_intervals(max_deliveries=1, reclaim_min_idle_ms=0),
    )
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    try:
        await _wait_for(lambda: _status_is(repos, job_id, "failed"))
    finally:
        stop.set()
        await asyncio.wait_for(run, timeout=5)

    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.error == {"code": "internal", "message": "worker crashed repeatedly"}
    assert await redis.xlen(keys.jobs_stream()) == 0


async def test_concurrency_limit_is_respected(
    make_settings: MakeSettings, tmp_path: Path, redis: RecordingRedis
) -> None:
    keys = Keys(PREFIX)
    repos = make_fake_repos()
    job_ids = [await _new_job(repos) for _ in range(4)]
    for job_id in job_ids:
        await _enqueue(redis, keys, job_id)

    lock = asyncio.Lock()
    active = 0
    peak = 0

    async def slow(ctx: StageContext) -> StageResult:
        nonlocal active, peak
        async with lock:
            active += 1
            peak = max(peak, active)
        await asyncio.sleep(0.15)
        async with lock:
            active -= 1
        return StageResult(status="done")

    settings = make_settings(CARACAL_DATA_DIR=tmp_path, REDIS_PREFIX=PREFIX, CARACAL_WORKER_CONCURRENCY=2)
    worker = Worker(
        settings,
        redis,
        repos,
        registry={Stage.INGEST: slow},
        worker_id="worker-a",
        intervals=_fast_intervals(),
    )
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    try:

        async def all_done() -> bool:
            return all([await _status_is(repos, job_id, "done") for job_id in job_ids])

        await _wait_for(all_done, limit_s=10)
    finally:
        stop.set()
        await asyncio.wait_for(run, timeout=5)

    assert peak == 2


async def test_shutdown_leaves_a_running_job_unacked(
    make_settings: MakeSettings, tmp_path: Path, redis: RecordingRedis
) -> None:
    keys = Keys(PREFIX)
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    await _enqueue(redis, keys, job_id)
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked(ctx: StageContext) -> StageResult:
        started.set()
        await release.wait()
        return StageResult(status="done")

    settings = make_settings(CARACAL_DATA_DIR=tmp_path, REDIS_PREFIX=PREFIX, CARACAL_WORKER_CONCURRENCY=1)
    worker = Worker(
        settings,
        redis,
        repos,
        registry={Stage.INGEST: blocked},
        worker_id="worker-a",
        intervals=_fast_intervals(shutdown_grace_s=0.05),
    )
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    await asyncio.wait_for(started.wait(), timeout=5)
    stop.set()
    await asyncio.wait_for(run, timeout=5)

    pending = await redis.xpending(keys.jobs_stream(), "workers")
    assert pending["pending"] == 1
    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.status == "running"
    release.set()
