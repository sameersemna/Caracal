"""In-memory implementations of the repository protocols (docs/07 §4).

These fakes are exercised by the *same* contract suite as the Postgres implementations in
``tests/repo_contract.py``. Search results match on all query terms (the ``simple`` text-search
configuration) and the limit is applied identically; only the *ranking order* is approximated,
by counting matching terms, because reproducing ``ts_rank`` would duplicate Postgres internals.
"""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from caracal.db.repo import (
    JobRecord,
    SearchHit,
    SegmentRow,
    StageRecord,
    UploadRecord,
    decode_cursor,
    encode_cursor,
)
from caracal.db.tables import (
    JOB_SOURCE_KINDS,
    JOB_STATUSES,
    JOB_TERMINAL_STATUSES,
    STAGE_STATUSES,
)
from caracal.errors import CaracalError, ErrorCode
from tests.repo_contract import Repos

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_ENDED_STAGE_STATUSES = frozenset({"done", "skipped", "failed"})
_MAX_PAGE = 100


def _now() -> datetime:
    return datetime.now(UTC)


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.lower())


@dataclass
class _Store:
    jobs: dict[uuid.UUID, JobRecord] = field(default_factory=dict)
    stages: dict[tuple[uuid.UUID, str], StageRecord] = field(default_factory=dict)
    uploads: dict[uuid.UUID, UploadRecord] = field(default_factory=dict)
    reports: dict[uuid.UUID, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    segments: dict[uuid.UUID, list[SegmentRow]] = field(default_factory=dict)

    def clear(self) -> None:
        self.jobs.clear()
        self.stages.clear()
        self.uploads.clear()
        self.reports.clear()
        self.segments.clear()


class FakeJobRepo:
    """In-memory :class:`caracal.db.repo.JobRepo`."""

    def __init__(self, store: _Store) -> None:
        self._store = store

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
        if source_kind not in JOB_SOURCE_KINDS:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                f"invalid source_kind: {source_kind}",
                {"allowed": list(JOB_SOURCE_KINDS)},
            )
        now = _now()
        record = JobRecord(
            id=uuid.uuid4(),
            status="queued",
            source_kind=source_kind,
            source_url=source_url,
            upload_id=upload_id,
            title=title,
            options=dict(options),
            progress={},
            error=None,
            created_at=now,
            updated_at=now,
            finished_at=None,
            sha256=sha256,
        )
        self._store.jobs[record.id] = record
        return record

    async def get(self, job_id: uuid.UUID) -> JobRecord | None:
        return self._store.jobs.get(job_id)

    async def list(
        self,
        *,
        status: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> tuple[list[JobRecord], str | None]:
        page_size = max(1, min(_MAX_PAGE, limit))
        records = list(self._store.jobs.values())
        if status is not None:
            records = [record for record in records if record.status == status]
        records.sort(key=lambda record: (record.created_at, record.id), reverse=True)
        if cursor is not None:
            cut_at, cut_id = decode_cursor(cursor)
            records = [record for record in records if (record.created_at, record.id) < (cut_at, cut_id)]
        page = records[:page_size]
        next_cursor: str | None = None
        if len(records) > page_size:
            last = page[-1]
            next_cursor = encode_cursor(last.created_at, last.id)
        return page, next_cursor

    async def set_status(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        error: dict[str, Any] | None = None,
    ) -> None:
        if status not in JOB_STATUSES:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                f"invalid job status: {status}",
                {"allowed": list(JOB_STATUSES)},
            )
        record = self._store.jobs.get(job_id)
        if record is None:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
        if record.status in JOB_TERMINAL_STATUSES:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                "job is in a terminal state",
                {"job_id": str(job_id)},
            )
        now = _now()
        self._store.jobs[job_id] = replace(
            record,
            status=status,
            updated_at=now,
            error=error,
            finished_at=now if status in JOB_TERMINAL_STATUSES else record.finished_at,
        )

    async def set_progress(self, job_id: uuid.UUID, progress: dict[str, Any]) -> None:
        record = self._store.jobs.get(job_id)
        if record is None:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
        self._store.jobs[job_id] = replace(record, progress=dict(progress), updated_at=_now())

    async def set_title(self, job_id: uuid.UUID, title: str | None) -> None:
        record = self._store.jobs.get(job_id)
        if record is None:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
        self._store.jobs[job_id] = replace(record, title=title, updated_at=_now())

    async def upsert_stage(
        self,
        job_id: uuid.UUID,
        name: str,
        *,
        status: str,
        reason: str | None = None,
    ) -> None:
        if status not in STAGE_STATUSES:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                f"invalid stage status: {status}",
                {"allowed": list(STAGE_STATUSES)},
            )
        if job_id not in self._store.jobs:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
        now = _now()
        key = (job_id, name)
        existing = self._store.stages.get(key)
        if existing is None:
            record = StageRecord(
                job_id=job_id,
                name=name,
                status=status,
                reason=reason,
                attempts=1 if status == "running" else 0,
                started_at=now if status == "running" else None,
                ended_at=now if status in _ENDED_STAGE_STATUSES else None,
            )
        else:
            if status == "running":
                started_at, ended_at = now, None
            elif status in _ENDED_STAGE_STATUSES:
                started_at, ended_at = existing.started_at, now
            else:
                started_at, ended_at = existing.started_at, existing.ended_at
            record = replace(
                existing,
                status=status,
                reason=reason,
                attempts=existing.attempts + 1 if status == "running" else existing.attempts,
                started_at=started_at,
                ended_at=ended_at,
            )
        self._store.stages[key] = record

    async def list_stages(self, job_id: uuid.UUID) -> list[StageRecord]:
        return sorted(
            (record for (owner, _), record in self._store.stages.items() if owner == job_id),
            key=lambda record: record.name,
        )

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
        cutoff = _now() - timedelta(days=max(0, within_days))
        candidates = [
            record
            for record in self._store.jobs.values()
            if record.status == "done"
            and record.created_at >= cutoff
            and record.options == options
            and (
                (sha256 is not None and record.sha256 == sha256)
                or (source_url is not None and record.source_url == source_url)
            )
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda record: (record.created_at, record.id))

    async def delete(self, job_id: uuid.UUID) -> None:
        if job_id not in self._store.jobs:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
        del self._store.jobs[job_id]
        for key in [key for key in self._store.stages if key[0] == job_id]:
            del self._store.stages[key]
        self._store.reports.pop(job_id, None)
        self._store.segments.pop(job_id, None)


class FakeUploadRepo:
    """In-memory :class:`caracal.db.repo.UploadRepo`."""

    def __init__(self, store: _Store) -> None:
        self._store = store

    async def create(
        self,
        *,
        filename: str,
        size_bytes: int,
        sha256: str,
        path: str,
    ) -> UploadRecord:
        record = UploadRecord(
            id=uuid.uuid4(),
            filename=filename,
            size_bytes=size_bytes,
            sha256=sha256,
            path=path,
            created_at=_now(),
            consumed_by=None,
        )
        self._store.uploads[record.id] = record
        return record

    async def get(self, upload_id: uuid.UUID) -> UploadRecord | None:
        return self._store.uploads.get(upload_id)

    async def consume(self, upload_id: uuid.UUID, job_id: uuid.UUID) -> None:
        record = self._store.uploads.get(upload_id)
        if record is None:
            raise CaracalError(ErrorCode.NOT_FOUND, "upload not found", {"upload_id": str(upload_id)})
        if record.consumed_by is not None:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                "upload already consumed",
                {"upload_id": str(upload_id)},
            )
        self._store.uploads[upload_id] = replace(record, consumed_by=job_id)

    async def list_unconsumed_older_than(self, hours: int) -> list[UploadRecord]:
        cutoff = _now() - timedelta(hours=max(0, hours))
        records = [
            record
            for record in self._store.uploads.values()
            if record.consumed_by is None and record.created_at < cutoff
        ]
        return sorted(records, key=lambda record: (record.created_at, record.id))


class FakeReportRepo:
    """In-memory :class:`caracal.db.repo.ReportRepo`."""

    def __init__(self, store: _Store) -> None:
        self._store = store

    async def put(self, job_id: uuid.UUID, report: dict[str, Any], schema_version: str) -> None:
        if job_id not in self._store.jobs:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
        self._store.reports[job_id] = (schema_version, copy.deepcopy(report))

    async def get(self, job_id: uuid.UUID) -> dict[str, Any] | None:
        stored = self._store.reports.get(job_id)
        return None if stored is None else copy.deepcopy(stored[1])


class FakeSegmentRepo:
    """In-memory :class:`caracal.db.repo.SegmentRepo` with approximate ranking."""

    def __init__(self, store: _Store) -> None:
        self._store = store

    async def replace_for_job(self, job_id: uuid.UUID, segments: Sequence[SegmentRow]) -> None:
        for segment in segments:
            if segment.job_id != job_id:
                raise CaracalError(
                    ErrorCode.INVALID_INPUT,
                    "segment job_id does not match the target job",
                    {"job_id": str(job_id), "segment_job_id": str(segment.job_id)},
                )
        if job_id not in self._store.jobs:
            raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
        self._store.segments[job_id] = list(segments)

    async def search(self, query: str, *, limit: int = 20) -> list[SearchHit]:
        terms = _tokens(query)
        if not terms:
            return []
        page_size = max(1, min(_MAX_PAGE, limit))
        hits: list[SearchHit] = []
        for job_id, rows in self._store.segments.items():
            job = self._store.jobs.get(job_id)
            if job is None:
                continue
            for row in rows:
                words = set(_tokens(row.text))
                if not all(term in words for term in terms):
                    continue
                lowered = row.text.lower()
                rank = float(sum(lowered.count(term) for term in terms))
                hits.append(
                    SearchHit(
                        job_id=job_id,
                        title=job.title,
                        seg_id=row.seg_id,
                        start_s=row.start_s,
                        text=row.text,
                        rank=rank,
                    )
                )
        hits.sort(key=lambda hit: (str(hit.job_id), hit.seg_id))
        return hits[:page_size]


def make_fake_repos() -> Repos:
    """Build the four fakes over one shared store, ready for the contract suite."""
    store = _Store()

    async def cleanup() -> None:
        store.clear()

    return Repos(
        jobs=FakeJobRepo(store),
        uploads=FakeUploadRepo(store),
        reports=FakeReportRepo(store),
        segments=FakeSegmentRepo(store),
        cleanup=cleanup,
    )
