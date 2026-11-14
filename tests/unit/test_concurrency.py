"""Worker concurrency: two pipelines + one translate, the whisper lock and mid-run cancel.

A worker with ``CARACAL_WORKER_CONCURRENCY=2`` runs two pipeline deliveries at once. Both
reach the transcribe stage, but only one may hold the Redis whisper lock at a time; the
second blocks. Cancelling one job while it waits settles that job ``cancelled`` at the next
stage boundary while the other pipeline and a queued translate delivery still finish.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from tests.fakes.redis import RecordingRedis, make_fake_redis
from tests.fakes.repo import make_fake_repos
from tests.repo_contract import Repos

from caracal.asr import Segment, WhisperResult, Word
from caracal.config import Settings
from caracal.pipeline.stages import JobOptions, Stage, StageContext, StageResult, stage_transcribe
from caracal.queue import JobQueue, Keys, set_cancel, whisper_lock
from caracal.storage import JobFiles
from caracal.worker import Worker, WorkerIntervals

PREFIX = "t3st:"
MakeSettings = Callable[..., Settings]


def _fast_intervals(**overrides: Any) -> WorkerIntervals:
    values: dict[str, Any] = {
        "heartbeat_s": 0.05,
        "touch_s": 100.0,
        "reclaim_s": 60.0,
        "reclaim_min_idle_ms": 60_000,
        "block_ms": 10,
        "max_deliveries": 3,
        "shutdown_grace_s": 0.5,
        "retention_s": 86_400.0,
    }
    values.update(overrides)
    return WorkerIntervals(**values)


async def _wait_for(
    predicate: Callable[[], Awaitable[bool]],
    *,
    limit_s: float = 10.0,
    interval: float = 0.01,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_s
    while loop.time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition was not met before the timeout")


async def _new_job(repos: Repos) -> uuid.UUID:
    record = await repos.jobs.create(
        source_kind="platform",
        source_url="https://example.com/talk",
        options=JobOptions().model_dump(),
    )
    return record.id


async def _enqueue(
    redis: RecordingRedis,
    keys: Keys,
    job_id: uuid.UUID,
    kind: str = "pipeline",
    payload: dict[str, Any] | None = None,
) -> None:
    queue = JobQueue(redis, keys)
    await queue.ensure_group()
    await queue.enqueue(job_id, kind, payload=payload)


def _chunk_checkpoint(files: JobFiles) -> None:
    path = files.chunk(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    files.write_json(
        files.stage(Stage.CHUNK),
        {
            "status": "done",
            "reason": None,
            "output": {"chunks": [{"index": 0, "start_s": 0.0, "end_s": 1.0, "path": "chunks/000.wav"}]},
        },
    )


def _whisper_result() -> WhisperResult:
    word = Word(text="hello", start=0.0, end=0.5, prob=0.9, space_before=True)
    return WhisperResult(
        language_code="en",
        language_prob=0.95,
        language_probs={"en": 0.95},
        duration=1.0,
        segments=(
            Segment(
                id=0,
                start=0.0,
                end=0.9,
                text="hello",
                words=(word,),
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                lang="en",
            ),
        ),
    )


class _LockedWhisper:
    """A whisper client that really takes the Redis single-slot lock and records holders."""

    def __init__(self, redis: RecordingRedis, keys: Keys, release: asyncio.Event) -> None:
        self.redis = redis
        self.keys = keys
        self.release = release
        self.attempts = 0
        self.active = 0
        self.peak = 0

    async def transcribe(
        self,
        _wav: Path,
        *,
        language: str = "auto",
        translate: bool = False,
        temperature: float = 0.0,
    ) -> WhisperResult:
        self.attempts += 1
        async with whisper_lock(self.redis, self.keys, acquire_timeout_s=10.0):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await self.release.wait()
            finally:
                self.active -= 1
        return _whisper_result()


def _done_stage(stage: Stage, calls: list[str]) -> Callable[[StageContext], Awaitable[StageResult]]:
    async def run(_ctx: StageContext) -> StageResult:
        calls.append(stage.value)
        return StageResult(status="done")

    return run


async def _translate_stage(_ctx: StageContext) -> StageResult:
    await asyncio.sleep(0.02)
    return StageResult(status="done")


@pytest.fixture
async def redis() -> Any:
    client = make_fake_redis()
    try:
        yield client
    finally:
        await client.aclose()


async def test_two_pipelines_and_a_translate_share_the_worker_lock_and_cancel_cleanly(
    make_settings: MakeSettings, tmp_path: Path, redis: RecordingRedis
) -> None:
    keys = Keys(PREFIX)
    repos = make_fake_repos()
    job_a = await _new_job(repos)
    job_b = await _new_job(repos)
    job_c = await _new_job(repos)
    await repos.jobs.set_status(job_c, "done")

    for job_id in (job_a, job_b):
        files = JobFiles(tmp_path, job_id)
        files.ensure()
        _chunk_checkpoint(files)
    await _enqueue(redis, keys, job_a)
    await _enqueue(redis, keys, job_b)
    await _enqueue(redis, keys, job_c, kind="translate", payload={"target_langs": ["de"]})

    release = asyncio.Event()
    whisper = _LockedWhisper(redis, keys, release)
    calls: list[str] = []
    registry = {
        Stage.TRANSCRIBE: stage_transcribe,
        Stage.TRANSLATE: _translate_stage,
        Stage.SUBTITLES: _done_stage(Stage.SUBTITLES, calls),
        Stage.ASSEMBLE: _done_stage(Stage.ASSEMBLE, calls),
    }
    settings = make_settings(
        CARACAL_DATA_DIR=tmp_path,
        REDIS_PREFIX=PREFIX,
        CARACAL_WORKER_CONCURRENCY=2,
    )
    worker = Worker(
        settings,
        redis,
        repos,
        registry=registry,
        worker_id="worker-concurrency",
        intervals=_fast_intervals(),
        whisper=whisper,
    )
    stop = asyncio.Event()
    run = asyncio.create_task(worker.run(stop))
    try:
        # Both pipeline deliveries reached transcribe; the lock serialises them.
        await _wait_for(lambda: _both_transcribing(whisper))
        assert whisper.active == 1

        # Cancel one job while it is waiting inside the stage (mid-run, before any result).
        await set_cancel(redis, keys, job_a)
        release.set()
        await _wait_for(lambda: _all_terminal(repos, (job_a, job_b, job_c)))
    finally:
        stop.set()
        await asyncio.wait_for(run, timeout=5)

    assert whisper.attempts == 2
    assert whisper.peak == 1
    statuses: dict[uuid.UUID, str] = {}
    for job_id in (job_a, job_b, job_c):
        job = await repos.jobs.get(job_id)
        assert job is not None
        statuses[job_id] = job.status
    assert statuses == {job_a: "cancelled", job_b: "done", job_c: "done"}
    assert calls.count("subtitles") >= 2
    assert calls.count("assemble") >= 2


async def _both_transcribing(whisper: _LockedWhisper) -> bool:
    return whisper.attempts >= 2


async def _all_terminal(repos: Repos, job_ids: tuple[uuid.UUID, ...]) -> bool:
    for job_id in job_ids:
        job = await repos.jobs.get(job_id)
        if job is None or job.status not in {"done", "cancelled", "failed"}:
            return False
    return True
