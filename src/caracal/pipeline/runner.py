"""Checkpointed, resumable pipeline runner (docs/01 §4)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError
from redis.asyncio import Redis

from caracal.asr.whisper import WhisperClient
from caracal.config import Settings
from caracal.db.tables import JOB_TERMINAL_STATUSES
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.ssrf import Resolver
from caracal.pipeline.stages import (
    CRITICAL_STAGES,
    JobOptions,
    Repos,
    Stage,
    StageContext,
    StageFn,
    StageResult,
    default_registry,
)
from caracal.queue import Keys
from caracal.speakers.diarize import Diarizer
from caracal.storage import JobFiles

_RETRY_DELAYS: tuple[float, ...] = (2.0, 8.0)
_CHECKPOINT_TERMINAL = frozenset({"done", "skipped"})

EventPublisher = Callable[[uuid.UUID, dict[str, Any]], Awaitable[None]]
CancelCheck = Callable[[uuid.UUID], Awaitable[bool]]
OwnerCheck = Callable[[], Awaitable[bool]]


async def _always_owner() -> bool:
    """Default ownership check for callers that are not queue workers."""
    return True


@dataclass
class RunnerDeps:
    """Everything :func:`run_pipeline` needs from its process, injected for testability."""

    settings: Settings
    repos: Repos
    queue_events: EventPublisher
    cancel_check: CancelCheck
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    still_owner: OwnerCheck = _always_owner
    # Process-wide stage clients, built once by the worker (docs/01 §4). They stay optional
    # so the M0 stub registry can run without any network-capable objects.
    whisper: WhisperClient | None = None
    diarizer: Diarizer | None = None
    resolver: Resolver | None = None
    redis: Redis | None = None
    keys: Keys | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _pct(finished: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(finished / total * 100, 1)


def _event(payload: dict[str, Any], *, finished: int, total: int) -> dict[str, Any]:
    """Attach the two fields every event carries: UTC timestamp and progress percentage."""
    return {"ts": _now(), "pct": _pct(finished, total), **payload}


def _failure(error: CaracalError | None, other: Exception | None) -> tuple[str, str]:
    if error is not None:
        return error.code.value, error.message
    message = str(other) if other is not None else "unknown error"
    return ErrorCode.INTERNAL.value, message


async def _run_stage(
    fn: StageFn,
    ctx: StageContext,
    deps: RunnerDeps,
) -> tuple[StageResult | None, CaracalError | None, Exception | None]:
    """Run one stage function, retrying only retryable :class:`CaracalError` failures."""
    attempt = 0
    while True:
        try:
            return await fn(ctx), None, None
        except CaracalError as exc:
            if exc.retryable and attempt < len(_RETRY_DELAYS):
                await deps.sleep(_RETRY_DELAYS[attempt])
                attempt += 1
                continue
            return None, exc, None
        except Exception as exc:
            return None, None, exc


async def run_pipeline(
    job_id: uuid.UUID,
    deps: RunnerDeps,
    registry: Mapping[Stage, StageFn] | None = None,
) -> str:
    """Execute the registry's stages in canonical order, checkpointing and resuming.

    Returns the job's final status: ``done``, ``failed``, ``cancelled``, ``lost`` when another
    worker took the queue entry over (nothing is written after that point), or the existing
    terminal status when the job was already finished.
    """
    active = dict(registry) if registry is not None else default_registry()
    scheduled = [stage for stage in Stage if stage in active]

    job = await deps.repos.jobs.get(job_id)
    if job is None:
        raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
    if job.status in JOB_TERMINAL_STATUSES:
        return job.status
    if not await deps.still_owner():
        return "lost"

    try:
        options = JobOptions.model_validate(job.options)
    except ValidationError as exc:
        raise CaracalError(
            ErrorCode.INVALID_INPUT,
            "stored job options are invalid",
            {"job_id": str(job_id)},
        ) from exc

    files = JobFiles(deps.settings.CARACAL_DATA_DIR, job_id)
    files.ensure()

    total = len(scheduled)
    finished = 0
    current: Stage | None = None
    rows = {row.name: row for row in await deps.repos.jobs.list_stages(job_id)}

    async def load(stage: Stage) -> dict[str, object] | None:
        value = files.read_json(files.stage(stage))
        if value is None:
            return None
        if not isinstance(value, dict):
            raise CaracalError(
                ErrorCode.INTERNAL,
                "stage checkpoint is not a JSON object",
                {"stage": stage.value},
            )
        return value

    async def progress(pct: float, message: str) -> None:
        stage_name = current.value if current is not None else None
        await deps.repos.jobs.set_progress(
            job_id,
            {"pct": pct, "stage": stage_name, "message": message},
        )
        await deps.queue_events(
            job_id,
            _event(
                {"type": "progress", "stage": stage_name, "message": message},
                finished=finished,
                total=total,
            ),
        )

    async def cancelled() -> bool:
        return await deps.cancel_check(job_id)

    ctx = StageContext(
        job_id=job_id,
        options=options,
        files=files,
        settings=deps.settings,
        repos=deps.repos,
        load=load,
        progress=progress,
        whisper=deps.whisper,
        diarizer=deps.diarizer,
        resolver=deps.resolver,
        redis=deps.redis,
        keys=deps.keys,
        cancelled=cancelled,
    )

    try:
        async with asyncio.timeout(float(deps.settings.CARACAL_JOB_TIMEOUT_S)):
            if job.status == "queued":
                await deps.repos.jobs.set_status(job_id, "running")
            await deps.queue_events(
                job_id,
                _event({"type": "status", "status": "running"}, finished=finished, total=total),
            )
            for stage in scheduled:
                current = stage
                if not await deps.still_owner():
                    return "lost"
                if await deps.cancel_check(job_id):
                    await deps.repos.jobs.set_status(job_id, "cancelled")
                    await deps.queue_events(
                        job_id,
                        _event({"type": "end", "status": "cancelled"}, finished=finished, total=total),
                    )
                    return "cancelled"

                row = rows.get(stage.value)
                if files.stage(stage).exists() and row is not None and row.status in _CHECKPOINT_TERMINAL:
                    finished += 1
                    continue

                await deps.repos.jobs.upsert_stage(job_id, stage.value, status="running")
                await deps.queue_events(
                    job_id,
                    _event(
                        {"type": "stage", "stage": stage.value, "status": "running"},
                        finished=finished,
                        total=total,
                    ),
                )

                result, stage_error, other = await _run_stage(active[stage], ctx, deps)
                # A stage that was already running when ownership was lost finishes, but its
                # result must not reach the store or the artifact directory (docs/05 §2).
                if not await deps.still_owner():
                    return "lost"
                if result is not None:
                    files.write_json(
                        files.stage(stage),
                        {"status": result.status, "reason": result.reason, "output": result.output},
                    )
                    await deps.repos.jobs.upsert_stage(
                        job_id,
                        stage.value,
                        status=result.status,
                        reason=result.reason,
                    )
                    finished += 1
                    await deps.queue_events(
                        job_id,
                        _event(
                            {"type": "stage", "stage": stage.value, "status": result.status},
                            finished=finished,
                            total=total,
                        ),
                    )
                    continue

                code, message = _failure(stage_error, other)
                await deps.repos.jobs.upsert_stage(
                    job_id,
                    stage.value,
                    status="failed",
                    reason=f"{code}: {message}",
                )
                await deps.queue_events(
                    job_id,
                    _event(
                        {"type": "stage", "stage": stage.value, "status": "failed"},
                        finished=finished,
                        total=total,
                    ),
                )
                if stage in CRITICAL_STAGES:
                    await deps.repos.jobs.set_status(
                        job_id,
                        "failed",
                        error={"code": code, "message": message, "stage": stage.value},
                    )
                    await deps.queue_events(
                        job_id,
                        _event({"type": "end", "status": "failed"}, finished=finished, total=total),
                    )
                    return "failed"
    except TimeoutError:
        # Another worker may own the entry by now; never overwrite its outcome.
        if not await deps.still_owner():
            return "lost"
        latest = await deps.repos.jobs.get(job_id)
        if latest is not None and latest.status in JOB_TERMINAL_STATUSES:
            return latest.status
        timeout_error: dict[str, Any] = {"code": ErrorCode.INTERNAL.value, "message": "job timeout"}
        if current is not None:
            timeout_error["stage"] = current.value
        await deps.repos.jobs.set_status(job_id, "failed", error=timeout_error)
        await deps.queue_events(
            job_id,
            _event({"type": "end", "status": "failed"}, finished=finished, total=total),
        )
        return "failed"

    if not await deps.still_owner():
        return "lost"
    await deps.repos.jobs.set_status(job_id, "done")
    await deps.queue_events(
        job_id,
        _event({"type": "end", "status": "done"}, finished=finished, total=total),
    )
    return "done"
