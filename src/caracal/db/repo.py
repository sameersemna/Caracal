"""Repository protocols and their Postgres implementation (docs/01 §3, docs/07 §3).

Every public method is ``async`` and owns its transaction. Callers depend on the ``Protocol``
classes, so unit tests run the identical contract against the in-memory fakes in
``tests/fakes/repo.py``.
"""

from __future__ import annotations

import base64
import binascii
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine

from caracal.db.engine import session_scope
from caracal.db.tables import (
    JOB_SOURCE_KINDS,
    JOB_STATUSES,
    JOB_TERMINAL_STATUSES,
    STAGE_STATUSES,
    job_stages,
    jobs,
    reports,
    uploads,
)
from caracal.db.tables import (
    segments as segments_table,
)
from caracal.errors import CaracalError, ErrorCode

_ENDED_STAGE_STATUSES = frozenset({"done", "skipped", "failed"})

_MAX_PAGE = 100


@dataclass(frozen=True)
class JobRecord:
    """One ``jobs`` row."""

    id: uuid.UUID
    status: str
    source_kind: str
    source_url: str | None
    upload_id: uuid.UUID | None
    title: str | None
    options: dict[str, Any]
    progress: dict[str, Any]
    error: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None
    sha256: str | None


@dataclass(frozen=True)
class StageRecord:
    """One ``job_stages`` row."""

    job_id: uuid.UUID
    name: str
    status: str
    reason: str | None
    attempts: int
    started_at: datetime | None
    ended_at: datetime | None


@dataclass(frozen=True)
class UploadRecord:
    """One ``uploads`` row."""

    id: uuid.UUID
    filename: str
    size_bytes: int
    sha256: str
    path: str
    created_at: datetime
    consumed_by: uuid.UUID | None


@dataclass(frozen=True)
class SegmentRow:
    """One ``segments`` row excluding the generated ``tsv`` column."""

    job_id: uuid.UUID
    seg_id: int
    start_s: float
    end_s: float
    speaker: str | None
    lang: str | None
    text: str


@dataclass(frozen=True)
class SearchHit:
    """A full-text search hit joined with the owning job's title."""

    job_id: uuid.UUID
    title: str | None
    seg_id: int
    start_s: float
    text: str
    rank: float


JobPage = tuple[list[JobRecord], str | None]
StageList = list[StageRecord]
UploadList = list[UploadRecord]
SearchHitList = list[SearchHit]


def encode_cursor(created_at: datetime, job_id: uuid.UUID) -> str:
    """Encode a keyset position as an opaque, URL-safe base64 cursor."""
    payload = json.dumps(
        {"t": created_at.isoformat(), "i": str(job_id)},
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    """Decode a cursor produced by :func:`encode_cursor`; invalid input is ``invalid_input``."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("cursor payload must be an object")
        created_at = datetime.fromisoformat(payload["t"])
        job_id = uuid.UUID(payload["i"])
    except (binascii.Error, KeyError, TypeError, ValueError, UnicodeDecodeError) as exc:
        raise CaracalError(ErrorCode.INVALID_INPUT, "invalid cursor") from exc
    if created_at.tzinfo is None:
        raise CaracalError(ErrorCode.INVALID_INPUT, "invalid cursor")
    return created_at, job_id


def _clamp_limit(limit: int) -> int:
    return max(1, min(_MAX_PAGE, limit))


def _days_ago(days: int) -> sa.TextClause:
    return sa.text("now() - make_interval(days => :days)").bindparams(days=days)


def _hours_ago(hours: int) -> sa.TextClause:
    return sa.text("now() - make_interval(hours => :hours)").bindparams(hours=hours)


def _job_record(row: RowMapping) -> JobRecord:
    return JobRecord(
        id=row["id"],
        status=row["status"],
        source_kind=row["source_kind"],
        source_url=row["source_url"],
        upload_id=row["upload_id"],
        title=row["title"],
        options=dict(row["options"] or {}),
        progress=dict(row["progress"] or {}),
        error=dict(row["error"]) if row["error"] is not None else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
        sha256=row["sha256"],
    )


def _stage_record(row: RowMapping) -> StageRecord:
    return StageRecord(
        job_id=row["job_id"],
        name=row["name"],
        status=row["status"],
        reason=row["reason"],
        attempts=row["attempts"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


def _upload_record(row: RowMapping) -> UploadRecord:
    return UploadRecord(
        id=row["id"],
        filename=row["filename"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        path=row["path"],
        created_at=row["created_at"],
        consumed_by=row["consumed_by"],
    )


def _require_job_source_kind(source_kind: str) -> None:
    if source_kind not in JOB_SOURCE_KINDS:
        raise CaracalError(
            ErrorCode.INVALID_INPUT,
            f"invalid source_kind: {source_kind}",
            {"allowed": list(JOB_SOURCE_KINDS)},
        )


def _require_job_status(status: str) -> None:
    if status not in JOB_STATUSES:
        raise CaracalError(
            ErrorCode.INVALID_INPUT,
            f"invalid job status: {status}",
            {"allowed": list(JOB_STATUSES)},
        )


def _require_stage_status(status: str) -> None:
    if status not in STAGE_STATUSES:
        raise CaracalError(
            ErrorCode.INVALID_INPUT,
            f"invalid stage status: {status}",
            {"allowed": list(STAGE_STATUSES)},
        )


class JobRepo(Protocol):
    """Persistence for jobs and their stage rows."""

    async def create(
        self,
        *,
        source_kind: str,
        source_url: str | None = None,
        upload_id: uuid.UUID | None = None,
        title: str | None = None,
        options: dict[str, Any],
        sha256: str | None = None,
    ) -> JobRecord: ...

    async def get(self, job_id: uuid.UUID) -> JobRecord | None: ...

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> JobPage: ...

    async def set_status(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        error: dict[str, Any] | None = None,
    ) -> None: ...

    async def set_progress(self, job_id: uuid.UUID, progress: dict[str, Any]) -> None: ...

    async def set_title(self, job_id: uuid.UUID, title: str | None) -> None: ...

    async def upsert_stage(
        self,
        job_id: uuid.UUID,
        name: str,
        *,
        status: str,
        reason: str | None = None,
    ) -> None: ...

    async def list_stages(self, job_id: uuid.UUID) -> StageList: ...

    async def find_recent_done(
        self,
        *,
        sha256: str | None = None,
        source_url: str | None = None,
        options: dict[str, Any],
        within_days: int = 7,
    ) -> JobRecord | None: ...

    async def delete(self, job_id: uuid.UUID) -> None: ...


class UploadRepo(Protocol):
    """Persistence for uploads awaiting (or consumed by) a job."""

    async def create(
        self,
        *,
        filename: str,
        size_bytes: int,
        sha256: str,
        path: str,
    ) -> UploadRecord: ...

    async def get(self, upload_id: uuid.UUID) -> UploadRecord | None: ...

    async def consume(self, upload_id: uuid.UUID, job_id: uuid.UUID) -> None: ...

    async def list_unconsumed_older_than(self, hours: int) -> UploadList: ...


class ReportRepo(Protocol):
    """Persistence for the final report JSON of a job."""

    async def put(self, job_id: uuid.UUID, report: dict[str, Any], schema_version: str) -> None: ...

    async def get(self, job_id: uuid.UUID) -> dict[str, Any] | None: ...


class SegmentRepo(Protocol):
    """Persistence and full-text search over transcript segments."""

    async def replace_for_job(self, job_id: uuid.UUID, segments: Sequence[SegmentRow]) -> None: ...

    async def search(self, query: str, *, limit: int = 20) -> SearchHitList: ...


class PostgresJobRepo:
    """``JobRepo`` backed by Postgres."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create(
        self,
        *,
        source_kind: str,
        source_url: str | None = None,
        upload_id: uuid.UUID | None = None,
        title: str | None = None,
        options: dict[str, Any],
        sha256: str | None = None,
    ) -> JobRecord:
        _require_job_source_kind(source_kind)
        job_id = uuid.uuid4()
        statement = (
            sa.insert(jobs)
            .values(
                id=job_id,
                status="queued",
                source_kind=source_kind,
                source_url=source_url,
                upload_id=upload_id,
                title=title,
                options=options,
                progress={},
                sha256=sha256,
            )
            .returning(*jobs.c)
        )
        async with session_scope(self._engine) as connection:
            row = (await connection.execute(statement)).mappings().one()
        return _job_record(row)

    async def get(self, job_id: uuid.UUID) -> JobRecord | None:
        statement = sa.select(jobs).where(jobs.c.id == job_id)
        async with session_scope(self._engine) as connection:
            row = (await connection.execute(statement)).mappings().one_or_none()
        return _job_record(row) if row is not None else None

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> JobPage:
        page_size = _clamp_limit(limit)
        statement = sa.select(jobs).order_by(jobs.c.created_at.desc(), jobs.c.id.desc()).limit(page_size + 1)
        if status is not None:
            statement = statement.where(jobs.c.status == status)
        if cursor is not None:
            cut_at, cut_id = decode_cursor(cursor)
            statement = statement.where(sa.tuple_(jobs.c.created_at, jobs.c.id) < (cut_at, cut_id))
        async with session_scope(self._engine) as connection:
            rows = (await connection.execute(statement)).mappings().all()
        records = [_job_record(row) for row in rows]
        if len(records) > page_size:
            page = records[:page_size]
            last = page[-1]
            return page, encode_cursor(last.created_at, last.id)
        return records, None

    async def set_status(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        error: dict[str, Any] | None = None,
    ) -> None:
        _require_job_status(status)
        values: dict[str, Any] = {
            "status": status,
            "updated_at": sa.func.now(),
            "error": error,
        }
        if status in JOB_TERMINAL_STATUSES:
            values["finished_at"] = sa.func.now()
        statement = (
            sa.update(jobs)
            .where(jobs.c.id == job_id, jobs.c.status.not_in(tuple(JOB_TERMINAL_STATUSES)))
            .values(**values)
        )
        async with session_scope(self._engine) as connection:
            result = await connection.execute(statement)
            if result.rowcount == 0:
                exists = (
                    await connection.execute(sa.select(jobs.c.id).where(jobs.c.id == job_id))
                ).scalar_one_or_none()
                if exists is None:
                    raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
                raise CaracalError(
                    ErrorCode.INVALID_INPUT,
                    "job is in a terminal state",
                    {"job_id": str(job_id)},
                )

    async def set_progress(self, job_id: uuid.UUID, progress: dict[str, Any]) -> None:
        statement = (
            sa.update(jobs).where(jobs.c.id == job_id).values(progress=progress, updated_at=sa.func.now())
        )
        async with session_scope(self._engine) as connection:
            result = await connection.execute(statement)
        if result.rowcount == 0:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})

    async def set_title(self, job_id: uuid.UUID, title: str | None) -> None:
        statement = sa.update(jobs).where(jobs.c.id == job_id).values(title=title, updated_at=sa.func.now())
        async with session_scope(self._engine) as connection:
            result = await connection.execute(statement)
        if result.rowcount == 0:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})

    async def upsert_stage(
        self,
        job_id: uuid.UUID,
        name: str,
        *,
        status: str,
        reason: str | None = None,
    ) -> None:
        _require_stage_status(status)
        values: dict[str, Any] = {
            "job_id": job_id,
            "name": name,
            "status": status,
            "reason": reason,
            "attempts": 1 if status == "running" else 0,
            "started_at": sa.func.now() if status == "running" else None,
            "ended_at": sa.func.now() if status in _ENDED_STAGE_STATUSES else None,
        }
        if status == "running":
            conflict_values: dict[str, Any] = {
                "status": status,
                "reason": reason,
                "attempts": job_stages.c.attempts + 1,
                "started_at": sa.func.now(),
                "ended_at": None,
            }
        elif status in _ENDED_STAGE_STATUSES:
            conflict_values = {
                "status": status,
                "reason": reason,
                "ended_at": sa.func.now(),
            }
        else:
            conflict_values = {"status": status, "reason": reason}
        statement = pg_insert(job_stages).values(**values)
        statement = statement.on_conflict_do_update(
            index_elements=["job_id", "name"],
            set_=conflict_values,
        )
        async with session_scope(self._engine) as connection:
            exists = (
                await connection.execute(sa.select(jobs.c.id).where(jobs.c.id == job_id))
            ).scalar_one_or_none()
            if exists is None:
                raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
            await connection.execute(statement)

    async def list_stages(self, job_id: uuid.UUID) -> StageList:
        statement = sa.select(job_stages).where(job_stages.c.job_id == job_id).order_by(job_stages.c.name)
        async with session_scope(self._engine) as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return [_stage_record(row) for row in rows]

    async def find_recent_done(
        self,
        *,
        sha256: str | None = None,
        source_url: str | None = None,
        options: dict[str, Any],
        within_days: int = 7,
    ) -> JobRecord | None:
        if sha256 is None and source_url is None:
            return None
        filters: list[Any] = [
            jobs.c.status == "done",
            jobs.c.created_at >= _days_ago(max(0, within_days)),
            jobs.c.options == options,
        ]
        if sha256 is not None:
            filters.append(jobs.c.sha256 == sha256)
        if source_url is not None:
            filters.append(jobs.c.source_url == source_url)
        statement = (
            sa.select(jobs).where(*filters).order_by(jobs.c.created_at.desc(), jobs.c.id.desc()).limit(1)
        )
        async with session_scope(self._engine) as connection:
            row = (await connection.execute(statement)).mappings().one_or_none()
        return _job_record(row) if row is not None else None

    async def delete(self, job_id: uuid.UUID) -> None:
        statement = sa.delete(jobs).where(jobs.c.id == job_id)
        async with session_scope(self._engine) as connection:
            result = await connection.execute(statement)
        if result.rowcount == 0:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})


class PostgresUploadRepo:
    """``UploadRepo`` backed by Postgres."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create(
        self,
        *,
        filename: str,
        size_bytes: int,
        sha256: str,
        path: str,
    ) -> UploadRecord:
        statement = (
            sa.insert(uploads)
            .values(
                id=uuid.uuid4(),
                filename=filename,
                size_bytes=size_bytes,
                sha256=sha256,
                path=path,
            )
            .returning(*uploads.c)
        )
        async with session_scope(self._engine) as connection:
            row = (await connection.execute(statement)).mappings().one()
        return _upload_record(row)

    async def get(self, upload_id: uuid.UUID) -> UploadRecord | None:
        statement = sa.select(uploads).where(uploads.c.id == upload_id)
        async with session_scope(self._engine) as connection:
            row = (await connection.execute(statement)).mappings().one_or_none()
        return _upload_record(row) if row is not None else None

    async def consume(self, upload_id: uuid.UUID, job_id: uuid.UUID) -> None:
        statement = (
            sa.update(uploads)
            .where(uploads.c.id == upload_id, uploads.c.consumed_by.is_(None))
            .values(consumed_by=job_id)
        )
        async with session_scope(self._engine) as connection:
            result = await connection.execute(statement)
            if result.rowcount == 0:
                exists = (
                    await connection.execute(sa.select(uploads.c.id).where(uploads.c.id == upload_id))
                ).scalar_one_or_none()
                if exists is None:
                    raise CaracalError(ErrorCode.NOT_FOUND, "upload not found", {"upload_id": str(upload_id)})
                raise CaracalError(
                    ErrorCode.INVALID_INPUT,
                    "upload already consumed",
                    {"upload_id": str(upload_id)},
                )

    async def list_unconsumed_older_than(self, hours: int) -> UploadList:
        statement = (
            sa.select(uploads)
            .where(
                uploads.c.consumed_by.is_(None),
                uploads.c.created_at < _hours_ago(max(0, hours)),
            )
            .order_by(uploads.c.created_at.asc(), uploads.c.id.asc())
        )
        async with session_scope(self._engine) as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return [_upload_record(row) for row in rows]


class PostgresReportRepo:
    """``ReportRepo`` backed by Postgres."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def put(self, job_id: uuid.UUID, report: dict[str, Any], schema_version: str) -> None:
        statement = pg_insert(reports).values(
            job_id=job_id,
            schema_version=schema_version,
            report=report,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["job_id"],
            set_={"schema_version": schema_version, "report": report},
        )
        async with session_scope(self._engine) as connection:
            exists = (
                await connection.execute(sa.select(jobs.c.id).where(jobs.c.id == job_id))
            ).scalar_one_or_none()
            if exists is None:
                raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
            await connection.execute(statement)

    async def get(self, job_id: uuid.UUID) -> dict[str, Any] | None:
        statement = sa.select(reports.c.report).where(reports.c.job_id == job_id)
        async with session_scope(self._engine) as connection:
            value = (await connection.execute(statement)).scalar_one_or_none()
        if value is None:
            return None
        if not isinstance(value, dict):
            raise CaracalError(ErrorCode.INTERNAL, "stored report is not a JSON object")
        return cast("dict[str, Any]", value)


class PostgresSegmentRepo:
    """``SegmentRepo`` backed by Postgres full-text search."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def replace_for_job(self, job_id: uuid.UUID, segments: Sequence[SegmentRow]) -> None:
        for segment in segments:
            if segment.job_id != job_id:
                raise CaracalError(
                    ErrorCode.INVALID_INPUT,
                    "segment job_id does not match the target job",
                    {"job_id": str(job_id), "segment_job_id": str(segment.job_id)},
                )
        statement = sa.delete(segments_table).where(segments_table.c.job_id == job_id)
        rows = [
            {
                "job_id": job_id,
                "seg_id": segment.seg_id,
                "start_s": segment.start_s,
                "end_s": segment.end_s,
                "speaker": segment.speaker,
                "lang": segment.lang,
                "text": segment.text,
            }
            for segment in segments
        ]
        async with session_scope(self._engine) as connection:
            exists = (
                await connection.execute(sa.select(jobs.c.id).where(jobs.c.id == job_id))
            ).scalar_one_or_none()
            if exists is None:
                raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
            await connection.execute(statement)
            if rows:
                await connection.execute(sa.insert(segments_table), rows)

    async def search(self, query: str, *, limit: int = 20) -> SearchHitList:
        page_size = _clamp_limit(limit)
        tsquery = sa.func.websearch_to_tsquery("simple", query)
        rank = sa.func.ts_rank(segments_table.c.tsv, tsquery).label("rank")
        statement = (
            sa.select(
                segments_table.c.job_id,
                jobs.c.title,
                segments_table.c.seg_id,
                segments_table.c.start_s,
                segments_table.c.text,
                rank,
            )
            .select_from(segments_table.join(jobs, jobs.c.id == segments_table.c.job_id))
            .where(segments_table.c.tsv.op("@@")(tsquery))
            .order_by(rank.desc(), segments_table.c.job_id, segments_table.c.seg_id)
            .limit(page_size)
        )
        async with session_scope(self._engine) as connection:
            rows = (await connection.execute(statement)).mappings().all()
        return [
            SearchHit(
                job_id=row["job_id"],
                title=row["title"],
                seg_id=row["seg_id"],
                start_s=row["start_s"],
                text=row["text"],
                rank=float(row["rank"]),
            )
            for row in rows
        ]
