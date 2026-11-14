"""``/v1`` REST routes and the public ``/healthz`` (docs/03 §1-2, docs/06 S1/S10/S11).

Every route is a thin adapter: HTTP parsing and status codes live here, behaviour lives in
:class:`caracal.services.JobService` so the MCP tools return exactly the same models.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Literal, cast

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from python_multipart.multipart import parse_options_header

from caracal.api.auth import bearer_token, require_auth
from caracal.api.schemas import EnrollSpeakerIn, JobCreate
from caracal.db.tables import JOB_TERMINAL_STATUSES
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.upload import MultipartFileStream
from caracal.log import get_logger
from caracal.services import JobService

if TYPE_CHECKING:
    from caracal.app import AppDeps
    from caracal.config import Settings

logger = get_logger("caracal.api")

public_router = APIRouter()
router = APIRouter(prefix="/v1", dependencies=[Depends(require_auth)])

_TERMINAL = frozenset(JOB_TERMINAL_STATUSES)
# docs/06 S10: ISO 639-1/3 (+ optional region) or the literal "source".
_LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$|^source$")


def _service(request: Request) -> JobService:
    service = getattr(request.app.state, "service", None)
    if not isinstance(service, JobService):
        raise CaracalError(ErrorCode.INTERNAL, "job service is not configured")
    return service


def _settings(request: Request) -> Settings:
    return cast("Settings", request.app.state.settings)


def _sse(event: str, data: object) -> str:
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {encoded}\n\n"


async def _file_part(request: Request) -> tuple[str, AsyncIterator[bytes]]:
    """Parse ``multipart/form-data`` and return the single ``file`` part's stream.

    The body is consumed incrementally: only the first part named ``file`` is streamed, and
    the iterator raises ``invalid_input`` when the body does not contain exactly one such
    part (nothing has been persisted by then).
    """
    from python_multipart.exceptions import MultipartParseError

    content_type, options = parse_options_header(request.headers.get("content-type"))
    if content_type != b"multipart/form-data":
        raise CaracalError(
            ErrorCode.UNSUPPORTED_MEDIA,
            "POST /v1/uploads requires multipart/form-data",
        )
    boundary = options.get(b"boundary")
    if not boundary:
        raise CaracalError(ErrorCode.INVALID_INPUT, "multipart boundary is required")

    stream = MultipartFileStream(boundary)
    body = request.stream()
    pending: list[bytes] = []
    try:
        async for chunk in body:
            pending.extend(stream.feed(chunk))
            if stream.headers_seen:
                break
    except MultipartParseError as exc:
        raise CaracalError(ErrorCode.INVALID_INPUT, "malformed multipart body") from exc
    if not stream.headers_seen:
        raise CaracalError(
            ErrorCode.INVALID_INPUT,
            "multipart body must contain exactly one file part named 'file'",
        )
    filename = stream.target_filename or ""

    async def chunks() -> AsyncIterator[bytes]:
        try:
            for buffered in pending:
                yield buffered
            async for chunk in body:
                for data in stream.feed(chunk):
                    yield data
            stream.finalize()
        except MultipartParseError as exc:
            raise CaracalError(ErrorCode.INVALID_INPUT, "malformed multipart body") from exc
        if len(stream.file_parts) != 1 or stream.file_parts[0][0] != "file" or not stream.finished:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                "multipart body must contain exactly one file part named 'file'",
            )

    return filename, chunks()


async def create_upload(request: Request) -> JSONResponse:
    """Stream one multipart ``file`` part to disk (docs/03 §2, docs/06 S6)."""
    filename, chunks = await _file_part(request)
    created = await _service(request).create_upload_from_chunks(chunks, filename)
    return JSONResponse(status_code=201, content=created.model_dump(mode="json"))


router.add_api_route("/uploads", create_upload, methods=["POST"])


@router.post("/jobs")
async def create_job(payload: JobCreate, request: Request) -> JSONResponse:
    """Validate, deduplicate, persist and enqueue one job (docs/03 §2)."""
    client_ip = request.client.host if request.client else ""
    ident_source = f"{bearer_token(request)}{client_ip}".encode()
    ident = hashlib.sha1(ident_source, usedforsecurity=False).hexdigest()
    created = await _service(request).create_job(
        payload.source,
        payload.options,
        ident_hash=ident,
    )
    return JSONResponse(
        status_code=200 if created.deduplicated else 202,
        content=created.model_dump(mode="json"),
    )


@router.get("/jobs")
async def list_jobs(
    request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
) -> dict[str, object]:
    result = await _service(request).list_jobs(status, limit, cursor)
    return result.model_dump(mode="json")


@router.get("/jobs/{job_id}")
async def get_job(job_id: uuid.UUID, request: Request) -> dict[str, object]:
    status = await _service(request).get_status(job_id)
    return status.model_dump(mode="json")


@router.get("/jobs/{job_id}/transcript")
async def get_transcript(
    job_id: uuid.UUID,
    request: Request,
    format: Literal["json", "txt"] = Query(default="json"),
) -> Response:
    """Return the merged transcript as JSON or plain text (docs/03 §2)."""
    payload = await _service(request).transcript(job_id, format)
    if isinstance(payload, str):
        return Response(content=payload, media_type="text/plain")
    return JSONResponse(content=payload)


@router.get("/jobs/{job_id}/subtitles/{filename:path}")
async def get_subtitle(job_id: uuid.UUID, filename: str, request: Request) -> Response:
    """Serve one rendered subtitle file with its registered content type (docs/03 §2)."""
    if "." not in filename:
        raise CaracalError(ErrorCode.INVALID_INPUT, "invalid subtitle path", {"filename": filename})
    lang, ext = filename.rsplit(".", 1)
    if ext not in {"srt", "vtt"} or _LANGUAGE_TAG_RE.fullmatch(lang) is None:
        raise CaracalError(ErrorCode.INVALID_INPUT, "invalid subtitle path", {"filename": filename})

    content, media_type = await _service(request).subtitles(job_id, lang, ext)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{job_id}.{lang}.{ext}"'},
    )


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
    service = JobService(
        settings,
        deps.repos,
        deps.redis,
        deps.keys,
        deps.queue,
        settings.CARACAL_DATA_DIR,
    )
    channel = deps.keys.events(job_id)
    async with deps.redis.pubsub() as pubsub:
        await pubsub.subscribe(channel)
        if after_subscribe is not None:
            await after_subscribe()
        job = await deps.repos.jobs.get(job_id)
        if job is None:
            yield _sse("end", {"job_id": str(job_id)})
            return
        snapshot = await service.get_status(job_id)
        yield _sse("status", snapshot.model_dump(mode="json"))
        if snapshot.status in _TERMINAL:
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
            event_type = str(payload.get("type") or "stage")
            yield _sse(event_type, payload)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: uuid.UUID, request: Request) -> StreamingResponse:
    deps = cast("AppDeps", request.app.state.deps)
    job = await deps.repos.jobs.get(job_id)
    if job is None:
        raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
    return StreamingResponse(
        job_event_stream(_settings(request), deps, job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: uuid.UUID, request: Request) -> Response:
    result = await _service(request).cancel_or_delete(job_id, delete_artifacts=True)
    if result is None:
        return Response(status_code=204)
    return JSONResponse(status_code=202, content=result.model_dump(mode="json"))


@router.get("/speakers")
async def list_speakers(
    request: Request,
    q: str | None = Query(default=None),
) -> list[dict[str, object]]:
    """Known speakers and their sample counts, optionally filtered by name (docs/03 §2)."""
    speakers = await _service(request).list_speakers(q)
    return [speaker.model_dump(mode="json") for speaker in speakers]


@router.post("/speakers/enroll")
async def enroll_speaker(payload: EnrollSpeakerIn, request: Request) -> JSONResponse:
    """Store one diarized label's voiceprint under a name (docs/03 §2, docs/06 §2)."""
    speaker = await _service(request).enroll_speaker(
        payload.job_id,
        payload.speaker_label,
        payload.name,
        payload.notes,
    )
    return JSONResponse(status_code=201, content=speaker.model_dump(mode="json"))


@router.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: uuid.UUID, request: Request) -> Response:
    """Delete a known speaker and every voiceprint it owns (docs/03 §2, docs/06 §2)."""
    await _service(request).delete_speaker(speaker_id)
    return Response(status_code=204)


@public_router.get("/healthz")
async def healthz(request: Request) -> dict[str, object]:
    from caracal.app import health_payload

    deps = cast("AppDeps", request.app.state.deps)
    return await health_payload(_settings(request), deps)
