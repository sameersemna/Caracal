"""Request and response models for the REST contract (docs/03 §2)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

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


class UploadCreated(BaseModel):
    """The ``POST /v1/uploads`` response (docs/03 §2)."""

    upload_id: uuid.UUID
    filename: str
    size_bytes: int
    sha256: str


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
    """``JobStatus.source``; absent optional keys are omitted from the JSON (docs/03 §2)."""

    kind: str
    url: str | None = None
    title: str | None = None

    @model_serializer(mode="wrap")
    def _serialise(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return {key: value for key, value in handler(self).items() if value is not None}


class JobProgress(BaseModel):
    """``JobStatus.progress``; keys are always present in the JSON."""

    pct: float = 0
    stage: str | None = None
    message: str | None = None


class StageStatus(BaseModel):
    """One ``JobStatus.stages`` item; absent timestamp keys are omitted (docs/03 §2)."""

    name: str
    status: str
    reason: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    @model_serializer(mode="wrap")
    def _serialise(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return {key: value for key, value in handler(self).items() if value is not None}


class JobLinks(BaseModel):
    """Download links built from ``CARACAL_PUBLIC_BASE_URL``."""

    report: str
    transcript: str
    events: str
    subtitles: dict[str, str] = Field(default_factory=dict)


class JobStatus(BaseModel):
    """The ``GET /v1/jobs/{id}`` response (docs/03 §2).

    ``error`` is omitted entirely while a job is healthy, exactly as the REST contract
    specifies; the progress sub-object always keeps its three keys.
    """

    job_id: uuid.UUID
    status: str
    created_at: datetime
    updated_at: datetime
    source: JobSourceStatus
    progress: JobProgress
    stages: list[StageStatus]
    error: dict[str, object] | None = None
    links: JobLinks

    @model_serializer(mode="wrap")
    def _serialise(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        if data.get("error") is None:
            data.pop("error", None)
        return data


class JobCreated(BaseModel):
    """The ``POST /v1/jobs`` response; ``deduplicated`` selects 200 vs 202 in the route."""

    job_id: uuid.UUID
    status: str
    deduplicated: bool
    links: JobLinks


class CancelResult(BaseModel):
    """``DELETE /v1/jobs/{id}`` / MCP ``cancel_job`` when the job still exists."""

    job_id: uuid.UUID
    status: str


class AnalyzeSummary(BaseModel):
    """The digest MCP ``analyze_audio`` returns once a job is done (card 15)."""

    title: str | None
    duration_s: float
    languages: list[dict[str, object]]
    transcript_preview: str


class KnownSpeaker(BaseModel):
    """One enrolled speaker (docs/03 §5)."""

    id: uuid.UUID
    name: str
    notes: str | None = None
    samples: int
    created_at: datetime
    updated_at: datetime

    @model_serializer(mode="wrap")
    def _serialise(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        return {key: value for key, value in handler(self).items() if value is not None}


class SpeakerList(BaseModel):
    """The ``GET /v1/speakers`` response and MCP ``list_speakers`` output (docs/03 §2-§3).

    A wrapper object, not a bare array: MCP ``structuredContent`` must be a JSON object, so
    the REST and MCP shapes are identical and a real MCP client can accept both.
    """

    items: list[KnownSpeaker]


class EnrollSpeakerIn(BaseModel):
    """The ``POST /v1/speakers/enroll`` body (docs/03 §2).

    Enrollment stores biometric voice data, so it only ever happens through this explicit
    contract and only for a finished job (docs/06 §2).
    """

    model_config = ConfigDict(extra="forbid")

    job_id: uuid.UUID
    speaker_label: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=200)
    notes: str | None = Field(default=None, max_length=2000)


class SpeakerDeleted(BaseModel):
    """Outcome of ``DELETE /v1/speakers/{id}`` for MCP ``delete_speaker`` (docs/03 §3)."""

    deleted: bool = True
    voiceprints_removed: int
