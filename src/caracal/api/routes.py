"""``/v1`` REST routes and the public ``/healthz`` (docs/03 §1-2, docs/06 S1/S10/S11)."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from caracal.api.auth import bearer_token, require_auth
from caracal.api.schemas import JobCreate
from caracal.db.repo import JobRecord, StageRecord
from caracal.db.tables import JOB_STATUSES, JOB_TERMINAL_STATUSES
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.router import classify_source_static
from caracal.log import get_logger
from caracal.pipeline.stages import Stage
from caracal.queue import rate_limit, set_cancel
from caracal.storage import JobFiles

if TYPE_CHECKING:
    from caracal.app import AppDeps
    from caracal.config import Settings

logger = get_logger("caracal.api")

public_router = APIRouter()
router = APIRouter(prefix="/v1", dependencies=[Depends(require_auth)])

_TERMINAL = frozenset(JOB_TERMINAL_STATUSES)


def _deps(request: Request) -> AppDeps:
    return cast("AppDeps", request.app.state.deps)


def _settings(request: Request) -> Settings:
    return cast("Settings", request.app.state.settings)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _base(settings: Settings) -> str:
    return settings.CARACAL_PUBLIC_BASE_URL.rstrip("/")


def _links(settings: Settings, job_id: uuid.UUID) -> dict[str, object]:
    base = _base(settings)
    return {
        "report": f"{base}/v1/jobs/{job_id}/report",
        "transcript": f"{base}/v1/jobs/{job_id}/transcript",
        "events": f"{base}/v1/jobs/{job_id}/events",
        "subtitles": _subtitle_links(settings, job_id),
    }


def _subtitle_links(settings: Settings, job_id: uuid.UUID) -> dict[str, str]:
    root = JobFiles(settings.CARACAL_DATA_DIR, job_id).subtitles_dir
    try:
        entries = sorted(root.iterdir())
    except FileNotFoundError:
        return {}
    base = _base(settings)
    links: dict[str, str] = {}
    for entry in entries:
        if entry.is_file() and entry.suffix in {".srt", ".vtt"}:
            links[entry.name] = f"{base}/v1/jobs/{job_id}/subtitles/{entry.name}"
    return links


def _progress(raw: dict[str, Any]) -> dict[str, object]:
    return {
        "pct": raw.get("pct", 0),
        "stage": raw.get("stage"),
        "message": raw.get("message"),
    }


def _stage(stage: Stage, row: StageRecord | None) -> dict[str, object]:
    if row is None:
        return {"name": stage.value, "status": "pending"}
    item: dict[str, object] = {"name": stage.value, "status": row.status}
    if row.reason is not None:
        item["reason"] = row.reason
    if row.started_at is not None:
        item["started_at"] = _iso(row.started_at)
    if row.ended_at is not None:
        item["ended_at"] = _iso(row.ended_at)
    return item


async def build_job_status(
    settings: Settings,
    deps: AppDeps,
    job: JobRecord,
) -> dict[str, object]:
    """Build the ``JobStatus`` object shared by ``GET /v1/jobs/{id}`` and the SSE snapshot."""
    rows = {row.name: row for row in await deps.repos.jobs.list_stages(job.id)}
    source: dict[str, object] = {"kind": job.source_kind}
    if job.source_url is not None:
        source["url"] = job.source_url
    if job.title is not None:
        source["title"] = job.title
    status: dict[str, object] = {
        "job_id": str(job.id),
        "status": job.status,
        "created_at": _iso(job.created_at),
        "updated_at": _iso(job.updated_at),
        "source": source,
        "progress": _progress(job.progress),
        "stages": [_stage(stage, rows.get(stage.value)) for stage in Stage],
        "links": _links(settings, job.id),
    }
    if job.error is not None:
        status["error"] = job.error
    return status


def _not_found(job_id: uuid.UUID) -> CaracalError:
    return CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})


async def _delete_quietly(deps: AppDeps, job_id: uuid.UUID) -> None:
    try:
        await deps.repos.jobs.delete(job_id)
    except CaracalError as exc:
        logger.warning(
            "could not roll back job after upload failure",
            extra={"job_id": str(job_id), "code": exc.code.value},
        )


def _remove_job_dir(settings: Settings, job_id: uuid.UUID) -> None:
    jobs_dir = (settings.CARACAL_DATA_DIR / "jobs").resolve()
    target = (settings.CARACAL_DATA_DIR / "jobs" / str(job_id)).resolve()
    if target.parent != jobs_dir:
        logger.warning(
            "refusing to delete a job directory outside the data dir",
            extra={"job_id": str(job_id)},
        )
        return
    shutil.rmtree(target, ignore_errors=True)


@router.post("/jobs")
async def create_job(payload: JobCreate, request: Request) -> JSONResponse:
    """Validate, deduplicate, persist and enqueue one job (docs/03 §2)."""
    deps = _deps(request)
    settings = _settings(request)

    source = payload.source
    upload_id = source.upload_id
    url = source.url
    options = payload.options.model_dump()
    sha256: str | None = None

    if upload_id is not None:
        upload = await deps.repos.uploads.get(upload_id)
        if upload is None:
            raise CaracalError(ErrorCode.NOT_FOUND, "upload not found", {"upload_id": str(upload_id)})
        if upload.consumed_by is not None:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                "upload already consumed",
                {"upload_id": str(upload_id)},
            )
        sha256 = upload.sha256

    client_ip = request.client.host if request.client else ""
    ident_source = f"{bearer_token(request)}{client_ip}".encode()
    ident = hashlib.sha1(ident_source, usedforsecurity=False).hexdigest()
    allowed, retry_after = await rate_limit(
        deps.redis,
        deps.keys,
        ident,
        limit=settings.CARACAL_RATE_LIMIT_PER_MIN,
    )
    if not allowed:
        raise CaracalError(
            ErrorCode.RATE_LIMITED,
            "rate limit exceeded",
            {"retry_after_s": retry_after},
        )

    if options["depth"] != "deep":
        existing = await deps.repos.jobs.find_recent_done(
            sha256=sha256,
            source_url=url if upload_id is None else None,
            options=options,
        )
        if existing is not None:
            return JSONResponse(
                status_code=200,
                content={
                    "job_id": str(existing.id),
                    "status": existing.status,
                    "deduplicated": True,
                    "links": _links(settings, existing.id),
                },
            )

    source_kind: str
    if url is not None:
        source_kind = classify_source_static(url)
    else:
        source_kind = "upload"
    job = await deps.repos.jobs.create(
        source_kind=source_kind,
        source_url=url,
        upload_id=upload_id,
        options=options,
        sha256=sha256,
    )

    if upload_id is not None:
        try:
            await deps.repos.uploads.consume(upload_id, job.id)
        except CaracalError as exc:
            await _delete_quietly(deps, job.id)
            if exc.code in {ErrorCode.NOT_FOUND, ErrorCode.INVALID_INPUT}:
                raise CaracalError(
                    ErrorCode.INVALID_INPUT,
                    "upload already consumed",
                    {"upload_id": str(upload_id)},
                ) from exc
            raise

    try:
        await deps.queue.enqueue(job.id, "pipeline")
    except Exception as exc:
        try:
            await deps.repos.jobs.set_status(
                job.id,
                "failed",
                error={"code": ErrorCode.DEPENDENCY_DOWN.value, "message": "queue unavailable"},
            )
        except CaracalError:
            logger.warning("could not mark job failed after enqueue error", extra={"job_id": str(job.id)})
        raise CaracalError(
            ErrorCode.DEPENDENCY_DOWN,
            "queue unavailable",
            {"job_id": str(job.id)},
            retryable=True,
        ) from exc

    return JSONResponse(
        status_code=202,
        content={
            "job_id": str(job.id),
            "status": job.status,
            "deduplicated": False,
            "links": _links(settings, job.id),
        },
    )


@router.get("/jobs")
async def list_jobs(
    request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> dict[str, object]:
    deps = _deps(request)
    status = status or None
    cursor = cursor or None
    if status is not None and status not in JOB_STATUSES:
        raise CaracalError(ErrorCode.INVALID_INPUT, "invalid job status", {"allowed": list(JOB_STATUSES)})
    jobs, next_cursor = await deps.repos.jobs.list(status=status, limit=limit, cursor=cursor)
    items = [
        {
            "job_id": str(job.id),
            "status": job.status,
            "created_at": _iso(job.created_at),
            "title": job.title,
            "source_kind": job.source_kind,
        }
        for job in jobs
    ]
    return {"items": items, "next_cursor": next_cursor}


@router.get("/jobs/{job_id}")
async def get_job(job_id: uuid.UUID, request: Request) -> dict[str, object]:
    deps = _deps(request)
    job = await deps.repos.jobs.get(job_id)
    if job is None:
        raise _not_found(job_id)
    return await build_job_status(_settings(request), deps, job)


def _sse(event: str, data: object) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n"


async def job_event_stream(
    settings: Settings,
    deps: AppDeps,
    job_id: uuid.UUID,
    *,
    ping_interval_s: float = 15.0,
    after_subscribe: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[str]:
    """Subscribe first, then emit a snapshot and forward channel events (docs/03 §2).

    ``after_subscribe`` exists so tests can publish between ``SUBSCRIBE`` and the
    snapshot and prove the event is not lost.
    """
    channel = deps.keys.events(job_id)
    async with deps.redis.pubsub() as pubsub:
        await pubsub.subscribe(channel)
        if after_subscribe is not None:
            await after_subscribe()
        job = await deps.repos.jobs.get(job_id)
        if job is None:
            yield _sse("end", {"job_id": str(job_id)})
            return
        yield _sse("status", await build_job_status(settings, deps, job))
        if job.status in _TERMINAL:
            yield _sse("end", {"job_id": str(job_id)})
            return
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=ping_interval_s)
            if message is None:
                yield ": ping\n\n"
                continue
            data = message.get("data")
            if data is None:
                continue
            raw = data.decode("utf-8") if isinstance(data, bytes) else str(data)
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("ignoring malformed event payload", extra={"job_id": str(job_id)})
                continue
            if not isinstance(payload, dict):
                continue
            if payload.get("type") == "end":
                yield _sse("end", payload)
                return
            event_type = payload.get("type")
            yield _sse(str(event_type) if isinstance(event_type, str) and event_type else "stage", payload)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: uuid.UUID, request: Request) -> StreamingResponse:
    deps = _deps(request)
    job = await deps.repos.jobs.get(job_id)
    if job is None:
        raise _not_found(job_id)
    return StreamingResponse(
        job_event_stream(_settings(request), deps, job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: uuid.UUID, request: Request) -> Response:
    deps = _deps(request)
    job = await deps.repos.jobs.get(job_id)
    if job is None:
        raise _not_found(job_id)
    if job.status not in _TERMINAL:
        await set_cancel(deps.redis, deps.keys, job_id)
        return JSONResponse(
            status_code=202,
            content={"job_id": str(job_id), "status": "cancelling"},
        )
    await deps.repos.jobs.delete(job_id)
    _remove_job_dir(_settings(request), job_id)
    return Response(status_code=204)


@public_router.get("/healthz")
async def healthz(request: Request) -> dict[str, object]:
    from caracal.app import health_payload

    return await health_payload(_settings(request), _deps(request))
