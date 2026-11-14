"""Request and response models for the REST contract (docs/03 §2)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from caracal.pipeline.stages import JobOptions

JobStatusValue = Literal["queued", "running", "done", "failed", "cancelled"]
SourceKindValue = Literal["upload", "direct", "platform"]
StageStatusValue = Literal["pending", "running", "done", "skipped", "failed"]

_MAX_URL_LENGTH = 2048


class JobSourceIn(BaseModel):
    """Exactly one of ``url`` or ``upload_id`` (docs/03 §2 ``JobCreate.source``)."""

    model_config = ConfigDict(extra="forbid")

    url: str | None = None
    upload_id: uuid.UUID | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> JobSourceIn:
        if (self.url is None) == (self.upload_id is None):
            raise ValueError("source must contain exactly one of url or upload_id")
        if self.url is not None:
            if len(self.url) > _MAX_URL_LENGTH:
                raise ValueError("url must be at most 2048 characters")
            try:
                parts = urlsplit(self.url)
            except ValueError as exc:
                raise ValueError("url must be an http(s) URL with a host") from exc
            if parts.scheme not in {"http", "https"} or not parts.netloc or parts.hostname is None:
                raise ValueError("url must be an http(s) URL with a host")
        return self


class JobCreate(BaseModel):
    """The ``POST /v1/jobs`` body."""

    model_config = ConfigDict(extra="forbid")

    source: JobSourceIn
    options: JobOptions = Field(default_factory=JobOptions)


class JobSummary(BaseModel):
    """One row of ``GET /v1/jobs``."""

    job_id: uuid.UUID
    status: str
    created_at: datetime
    title: str | None
    source_kind: str


class JobList(BaseModel):
    """The ``GET /v1/jobs`` response."""

    items: list[JobSummary]
    next_cursor: str | None


class JobSourceStatus(BaseModel):
    """``JobStatus.source``."""

    kind: str
    url: str | None = None
    title: str | None = None


class JobProgress(BaseModel):
    """``JobStatus.progress``; keys are always present in the JSON."""

    pct: float = 0
    stage: str | None = None
    message: str | None = None


class StageStatus(BaseModel):
    """One ``JobStatus.stages`` item."""

    name: str
    status: str
    reason: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class JobLinks(BaseModel):
    """Download links built from ``CARACAL_PUBLIC_BASE_URL``."""

    report: str
    transcript: str
    events: str
    subtitles: dict[str, str] = Field(default_factory=dict)


class JobStatus(BaseModel):
    """The ``GET /v1/jobs/{id}`` response (docs/03 §2)."""

    job_id: uuid.UUID
    status: str
    created_at: datetime
    updated_at: datetime
    source: JobSourceStatus
    progress: JobProgress
    stages: list[StageStatus]
    error: dict[str, object] | None = None
    links: JobLinks
