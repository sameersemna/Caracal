"""Stage names, job options, the stage context and the M0 stub registry (docs/01 §4)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from caracal.config import Settings
from caracal.db.repo import JobRepo, ReportRepo, SegmentRepo, UploadRepo
from caracal.storage import JobFiles


class Stage(StrEnum):
    """Pipeline stages in canonical execution order (docs/01 §4)."""

    INGEST = "ingest"
    NORMALIZE = "normalize"
    CHUNK = "chunk"
    TRANSCRIBE = "transcribe"
    LANGID = "langid"
    DIARIZE = "diarize"
    ALIGN = "align"
    VOICEPRINT = "voiceprint"
    PLAN = "plan"
    CONTENT = "content"
    IDENTITY = "identity"
    VERIFY = "verify"
    TRANSLATE = "translate"
    SUBTITLES = "subtitles"
    REVIEW = "review"
    ASSEMBLE = "assemble"


CRITICAL_STAGES: frozenset[Stage] = frozenset({Stage.INGEST, Stage.NORMALIZE, Stage.TRANSCRIBE})

SpeakerHint = Annotated[str, StringConstraints(max_length=200)]


class JobOptions(BaseModel):
    """The validated ``JobCreate.options`` object (docs/03 §2)."""

    model_config = ConfigDict(extra="forbid")

    language: str = "auto"
    target_langs: list[str] | None = None
    depth: Literal["quick", "standard", "deep"] = "standard"
    diarize: Literal["auto", "on", "off"] = "auto"
    verify: Literal["auto", "on", "off"] = "auto"
    speaker_hints: list[SpeakerHint] = Field(default_factory=list, max_length=20)
    context_hint: str | None = Field(default=None, max_length=2000)


class Repos(Protocol):
    """The repository bundle a stage and the runner depend on (docs/01 §3)."""

    @property
    def jobs(self) -> JobRepo: ...

    @property
    def uploads(self) -> UploadRepo: ...

    @property
    def reports(self) -> ReportRepo: ...

    @property
    def segments(self) -> SegmentRepo: ...


@dataclass
class StageContext:
    """Everything a stage function receives."""

    job_id: UUID
    options: JobOptions
    files: JobFiles
    settings: Settings
    repos: Repos
    load: Callable[[Stage], Awaitable[dict[str, object] | None]]
    progress: Callable[[float, str], Awaitable[None]]


@dataclass(frozen=True)
class StageResult:
    """The outcome a stage reports; the runner persists ``output`` as its checkpoint."""

    status: Literal["done", "skipped"]
    reason: str | None = None
    output: dict[str, object] = field(default_factory=dict)


StageFn = Callable[[StageContext], Awaitable[StageResult]]


async def _stub_stage(ctx: StageContext) -> StageResult:
    return StageResult(status="skipped", reason="not implemented in M0")


def default_registry() -> dict[Stage, StageFn]:
    """The M0 registry: every stage is a clean no-op checkpoint."""
    return {stage: _stub_stage for stage in Stage}
