"""The one implementation of every Caracal capability (docs/01 §3, docs/03 §0).

REST routes (:mod:`caracal.api.routes`) and MCP tools (:mod:`caracal.mcp.server`) are thin
adapters: both call the same :class:`JobService` methods and receive the same pydantic
models, so the two front doors can never diverge. Only a transport-specific concern (HTTP
status codes, multipart parsing, SSE framing, MCP errors) stays in an adapter.

Every failure leaves here as :class:`caracal.errors.CaracalError` carrying an
:class:`~caracal.errors.ErrorCode`.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
import shutil
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from pydantic import ValidationError
from redis.asyncio import Redis

from caracal.analysis.qa import QaSegment, qa_report, validate_question
from caracal.analysis.qa import answer as answer_question
from caracal.analysis.translate import is_valid_target_lang
from caracal.api.schemas import (
    AnalyzeSummary,
    CancelResult,
    JobCreated,
    JobLinks,
    JobList,
    JobProgress,
    JobSourceIn,
    JobSourceStatus,
    JobStatus,
    JobSummary,
    KnownSpeaker,
    SearchItem,
    SearchResults,
    SpeakerDeleted,
    SpeakerList,
    StageStatus,
    TranslateQueued,
    UploadCreated,
    VoiceprintExport,
    VoiceprintItem,
)
from caracal.config import Settings
from caracal.db.repo import JobRecord, KnownSpeakerRecord, StageRecord
from caracal.db.tables import EMB_DIM, JOB_STATUSES, JOB_TERMINAL_STATUSES
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.router import classify_source_static
from caracal.ingest.upload import UploadInfo, save_upload
from caracal.llm.client import LLMClient
from caracal.log import get_logger
from caracal.ops import MetricsRegistry
from caracal.pipeline.stages import JobOptions, Repos, Stage
from caracal.queue import JobQueue, Keys, rate_limit, set_cancel
from caracal.report.model import Answer, Report, ReportSummary
from caracal.report.render_md import render_markdown
from caracal.speakers.voiceprint import EnrollmentDeps
from caracal.speakers.voiceprint import enroll as enroll_voiceprint
from caracal.storage import JobFiles, UploadFiles

logger = get_logger("caracal.services")

MIB = 1024 * 1024
MAX_BASE64_DECODED_BYTES = 25 * MIB
"""S7: the largest inline ``audio_base64`` payload, measured before decoding."""

_TERMINAL = frozenset(JOB_TERMINAL_STATUSES)
_ACTIVE_STATUSES = frozenset({"queued", "running"})
_SUBTITLE_EXTENSIONS = frozenset({"srt", "vtt"})
_TRANSCRIPT_PREVIEW_CHARS = 1000
_WAIT_POLL_S = 1.0
_SEARCH_POOL = 50
"""The largest page the REST contract allows; the service filters it down to ``done`` jobs."""

TranscriptFormat = Literal["json", "txt", "srt", "vtt"]
ProgressCallback = Callable[[JobProgress], Awaitable[None]]


def _not_found(job_id: uuid.UUID) -> CaracalError:
    return CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})


def _known_speaker(record: KnownSpeakerRecord) -> KnownSpeaker:
    return KnownSpeaker(
        id=record.id,
        name=record.name,
        notes=record.notes,
        samples=record.samples,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


class SpeakerService:
    """Known speakers and their voiceprints (docs/03 §2-§3, docs/04 §5).

    Enrollment is biometric: it exists only as an explicit request, never as a pipeline side
    effect (docs/06 §2). Both front doors call these methods so their answers cannot diverge.
    """

    def __init__(self, repos: Repos, data_dir: Path) -> None:
        self.repos = repos
        self.data_dir = data_dir

    async def list(self, query: str | None = None) -> list[KnownSpeaker]:
        """Known speakers, optionally filtered by name containment (docs/03 §2)."""
        query = query.strip() if query is not None else None
        records = await self.repos.speakers.list(query or None)
        return [_known_speaker(record) for record in records]

    async def enroll(
        self,
        job_id: uuid.UUID,
        speaker_label: str,
        name: str,
        notes: str | None = None,
    ) -> KnownSpeaker:
        """Store one label's embedding under a name for a finished job (docs/04 §5)."""
        deps = EnrollmentDeps(jobs=self.repos.jobs, speakers=self.repos.speakers, data_dir=self.data_dir)
        record = await enroll_voiceprint(deps, job_id, speaker_label, name, notes)
        return _known_speaker(record)

    async def delete(self, speaker_id: uuid.UUID) -> SpeakerDeleted:
        """Delete a known speaker and every embedding it owns (docs/06 §2)."""
        removed = await self.repos.speakers.delete(speaker_id)
        return SpeakerDeleted(deleted=True, voiceprints_removed=removed)


class JobService:
    """Every capability, independent of REST or MCP (docs/03 §0)."""

    def __init__(
        self,
        settings: Settings,
        repos: Repos,
        redis: Redis,
        keys: Keys,
        queue: JobQueue,
        data_dir: Path,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.repos = repos
        self.redis = redis
        self.keys = keys
        self.queue = queue
        self.data_dir = data_dir
        self.speaker_service = SpeakerService(repos, data_dir)
        # Q&A is the only read-side capability that calls the LLM. Production builds a client
        # on first use; tests inject a scripted one so no request leaves the process.
        self.llm = llm

    # -- links and status -------------------------------------------------

    def links(self, job_id: uuid.UUID) -> JobLinks:
        """Download links built from ``CARACAL_PUBLIC_BASE_URL`` (docs/03 §2)."""
        base = self.settings.CARACAL_PUBLIC_BASE_URL.rstrip("/")
        return JobLinks(
            report=f"{base}/v1/jobs/{job_id}/report",
            transcript=f"{base}/v1/jobs/{job_id}/transcript",
            events=f"{base}/v1/jobs/{job_id}/events",
            subtitles=self._subtitle_links(job_id),
        )

    def _subtitle_links(self, job_id: uuid.UUID) -> dict[str, str]:
        root = JobFiles(self.data_dir, job_id).subtitles_dir
        try:
            entries = sorted(root.iterdir())
        except FileNotFoundError:
            return {}
        base = self.settings.CARACAL_PUBLIC_BASE_URL.rstrip("/")
        links: dict[str, str] = {}
        for entry in entries:
            if entry.is_file() and entry.suffix in {".srt", ".vtt"}:
                links[entry.name] = f"{base}/v1/jobs/{job_id}/subtitles/{entry.name}"
        return links

    def _status_from_record(self, job: JobRecord, stages: list[StageRecord]) -> JobStatus:
        rows = {row.name: row for row in stages}
        return JobStatus(
            job_id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            source=JobSourceStatus(kind=job.source_kind, url=job.source_url, title=job.title),
            progress=JobProgress(
                pct=job.progress.get("pct", 0),
                stage=cast("str | None", job.progress.get("stage")),
                message=cast("str | None", job.progress.get("message")),
            ),
            stages=[self._stage_status(stage, rows.get(stage.value)) for stage in Stage],
            error=cast("dict[str, object] | None", job.error),
            links=self.links(job.id),
        )

    def _stage_status(self, stage: Stage, row: StageRecord | None) -> StageStatus:
        if row is None:
            return StageStatus(name=stage.value, status="pending")
        return StageStatus(
            name=stage.value,
            status=row.status,
            reason=row.reason,
            started_at=row.started_at,
            ended_at=row.ended_at,
        )

    async def get_status(self, job_id: uuid.UUID) -> JobStatus:
        """``JobStatus`` for one job (docs/03 §2)."""
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        stages = await self.repos.jobs.list_stages(job_id)
        return self._status_from_record(job, stages)

    async def list_jobs(self, status: str | None, limit: int, cursor: str | None) -> JobList:
        """One page of jobs, newest first (docs/03 §2)."""
        status = status or None
        cursor = cursor or None
        if status is not None and status not in JOB_STATUSES:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                "invalid job status",
                {"allowed": list(JOB_STATUSES)},
            )
        jobs, next_cursor = await self.repos.jobs.list(status=status, limit=limit, cursor=cursor)
        return JobList(
            items=[
                JobSummary(
                    job_id=job.id,
                    status=job.status,
                    created_at=job.created_at,
                    title=job.title,
                    source_kind=job.source_kind,
                )
                for job in jobs
            ],
            next_cursor=next_cursor,
        )

    # -- job creation -----------------------------------------------------

    async def create_job(
        self,
        source: JobSourceIn,
        options: JobOptions,
        *,
        ident_hash: str,
    ) -> JobCreated:
        """Validate, deduplicate, persist and enqueue one job (docs/03 §2, docs/06 S11)."""
        upload_id = source.upload_id
        url = source.url
        options_dict = options.model_dump()
        sha256: str | None = None

        if upload_id is not None:
            upload = await self.repos.uploads.get(upload_id)
            if upload is None:
                raise CaracalError(
                    ErrorCode.NOT_FOUND,
                    "upload not found",
                    {"upload_id": str(upload_id)},
                )
            if upload.consumed_by is not None:
                raise CaracalError(
                    ErrorCode.INVALID_INPUT,
                    "upload already consumed",
                    {"upload_id": str(upload_id)},
                )
            sha256 = upload.sha256

        allowed, retry_after = await rate_limit(
            self.redis,
            self.keys,
            ident_hash,
            limit=self.settings.CARACAL_RATE_LIMIT_PER_MIN,
        )
        if not allowed:
            raise CaracalError(
                ErrorCode.RATE_LIMITED,
                "rate limit exceeded",
                {"retry_after_s": retry_after},
            )

        if options_dict["depth"] != "deep":
            existing = await self.repos.jobs.find_recent_done(
                sha256=sha256,
                source_url=url if upload_id is None else None,
                options=options_dict,
            )
            if existing is not None:
                return JobCreated(
                    job_id=existing.id,
                    status=existing.status,
                    deduplicated=True,
                    links=self.links(existing.id),
                )

        source_kind = classify_source_static(url) if url is not None else "upload"
        job = await self.repos.jobs.create(
            source_kind=source_kind,
            source_url=url,
            upload_id=upload_id,
            options=options_dict,
            sha256=sha256,
        )

        if upload_id is not None:
            try:
                await self.repos.uploads.consume(upload_id, job.id)
            except CaracalError as exc:
                await self._delete_quietly(job.id)
                if exc.code in {ErrorCode.NOT_FOUND, ErrorCode.INVALID_INPUT}:
                    raise CaracalError(
                        ErrorCode.INVALID_INPUT,
                        "upload already consumed",
                        {"upload_id": str(upload_id)},
                    ) from exc
                raise

        try:
            await self.queue.enqueue(job.id, "pipeline")
        except Exception as exc:
            try:
                await self.repos.jobs.set_status(
                    job.id,
                    "failed",
                    error={"code": ErrorCode.DEPENDENCY_DOWN.value, "message": "queue unavailable"},
                )
            except CaracalError:
                logger.warning(
                    "could not mark job failed after enqueue error",
                    extra={"job_id": str(job.id)},
                )
            raise CaracalError(
                ErrorCode.DEPENDENCY_DOWN,
                "queue unavailable",
                {"job_id": str(job.id)},
                retryable=True,
            ) from exc

        return JobCreated(
            job_id=job.id,
            status=job.status,
            deduplicated=False,
            links=self.links(job.id),
        )

    async def _delete_quietly(self, job_id: uuid.UUID) -> None:
        try:
            await self.repos.jobs.delete(job_id)
        except CaracalError as exc:
            logger.warning(
                "could not roll back job after upload failure",
                extra={"job_id": str(job_id), "code": exc.code.value},
            )

    # -- uploads ----------------------------------------------------------

    async def create_upload_from_chunks(
        self,
        chunks: AsyncIterator[bytes],
        filename: str,
    ) -> UploadCreated:
        """Stream one already-parsed file part to disk and record it (docs/06 S6)."""
        upload_id = uuid.uuid4()
        files = UploadFiles(self.data_dir, upload_id)
        info = await save_upload(
            chunks,
            filename,
            files,
            max_bytes=self.settings.CARACAL_MAX_UPLOAD_MB * MIB,
        )
        return await self._record_upload(info)

    async def create_upload_from_base64(self, b64: str, filename: str) -> UploadCreated:
        """Store an inline base64 audio payload (S7: size checked before decoding)."""
        # S7: ``len(b64) * 3 // 4`` is an upper bound on the decoded size; checking it first
        # means an oversize payload is rejected without ever allocating the decoded bytes.
        if len(b64) * 3 // 4 > MAX_BASE64_DECODED_BYTES:
            raise CaracalError(
                ErrorCode.TOO_LARGE,
                "audio_base64 exceeds the maximum decoded size",
                {"max_decoded_bytes": MAX_BASE64_DECODED_BYTES},
            )
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                "audio_base64 is not valid base64",
            ) from exc
        if len(data) > MAX_BASE64_DECODED_BYTES:
            raise CaracalError(
                ErrorCode.TOO_LARGE,
                "audio_base64 exceeds the maximum decoded size",
                {"max_decoded_bytes": MAX_BASE64_DECODED_BYTES},
            )

        async def chunks() -> AsyncIterator[bytes]:
            yield data

        return await self.create_upload_from_chunks(
            chunks(),
            filename or "audio.bin",
        )

    async def _record_upload(self, info: UploadInfo) -> UploadCreated:
        try:
            record = await self.repos.uploads.create(
                upload_id=info.upload_id,
                filename=info.filename,
                size_bytes=info.size_bytes,
                sha256=info.sha256,
                path=str(info.path),
            )
        except BaseException:
            # The row is the only reference the API retains; without it the file is junk.
            try:
                info.path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "could not remove an orphaned upload file",
                    extra={"upload_id": str(info.upload_id)},
                )
            raise
        return UploadCreated(
            upload_id=record.id,
            filename=record.filename,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
        )

    # -- artifacts --------------------------------------------------------

    def _missing_artifact(self, job: JobRecord) -> CaracalError:
        """404 for a failed job, 409 while the artifact does not exist yet (docs/03 §2)."""
        if job.status == "failed":
            return CaracalError(
                ErrorCode.NOT_FOUND,
                "artifact not found",
                {"job_id": str(job.id), "status": job.status},
            )
        return CaracalError(
            ErrorCode.JOB_NOT_READY,
            "job is not ready",
            {"job_id": str(job.id), "status": job.status},
        )

    def _transcript_languages(self, files: JobFiles) -> list[dict[str, object]]:
        checkpoint = files.read_json(files.stage(Stage.LANGID))
        if not isinstance(checkpoint, dict):
            return []
        output = checkpoint.get("output")
        if not isinstance(output, dict):
            return []
        languages = output.get("languages")
        if not isinstance(languages, list):
            return []
        return [cast("dict[str, object]", item) for item in languages if isinstance(item, dict)]

    def _artifact_json(self, path: Path) -> dict[str, object]:
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CaracalError(ErrorCode.INTERNAL, "stored transcript is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise CaracalError(ErrorCode.INTERNAL, "stored transcript is not a JSON object")
        return cast("dict[str, object]", payload)

    async def transcript(self, job_id: uuid.UUID, fmt: TranscriptFormat) -> dict[str, object] | str:
        """The merged transcript as JSON, plain text, SRT or WebVTT (docs/03 §2)."""
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        files = JobFiles(self.data_dir, job_id)

        if fmt == "txt":
            if not files.transcript_txt.is_file():
                raise self._missing_artifact(job)
            return files.transcript_txt.read_text(encoding="utf-8")
        if fmt in _SUBTITLE_EXTENSIONS:
            data, _media_type = await self.subtitles(job_id, "source", fmt)
            return data.decode("utf-8")
        if fmt != "json":
            raise CaracalError(ErrorCode.INVALID_INPUT, "unsupported transcript format", {"format": fmt})
        if not files.transcript_json.is_file():
            raise self._missing_artifact(job)
        payload = self._artifact_json(files.transcript_json)
        payload["job_id"] = str(job_id)
        payload["languages"] = self._transcript_languages(files)
        return payload

    async def subtitles(self, job_id: uuid.UUID, lang: str, ext: str) -> tuple[bytes, str]:
        """One rendered subtitle file plus its media type (docs/03 §2, docs/06 S10)."""
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        files = JobFiles(self.data_dir, job_id)
        path = files.subtitle(lang, ext)
        if not path.is_file():
            if job.status == "failed" or (job.status not in _ACTIVE_STATUSES and lang != "source"):
                # An unknown language never becomes available; a failed job has no artifacts.
                raise CaracalError(
                    ErrorCode.NOT_FOUND,
                    "subtitle not found",
                    {"job_id": str(job_id), "lang": lang, "ext": ext},
                )
            raise self._missing_artifact(job)
        media_type = "application/x-subrip" if ext == "srt" else "text/vtt; charset=utf-8"
        return path.read_bytes(), media_type

    # -- report -----------------------------------------------------------

    async def translate_job(self, job_id: uuid.UUID, target_langs: list[str]) -> TranslateQueued:
        """Queue an on-demand translation for a finished job (docs/03 §2-§3).

        The job keeps its ``done`` status: only translate, subtitles and assemble re-run,
        and existing translations are kept. The requested languages travel inside the queue
        entry, so back-to-back requests for one job each keep their own languages.
        """
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        if job.status != "done":
            raise CaracalError(
                ErrorCode.JOB_NOT_READY,
                "job is not done",
                {"job_id": str(job_id), "status": job.status},
            )
        languages: list[str] = []
        for code in target_langs:
            if not is_valid_target_lang(code):
                raise CaracalError(
                    ErrorCode.INVALID_INPUT,
                    "invalid target language code",
                    {"code": code},
                )
            if code not in languages:
                languages.append(code)
        if not languages:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                "target_langs must not be empty",
                {"job_id": str(job_id)},
            )

        try:
            await self.queue.enqueue(job_id, "translate", payload={"target_langs": languages})
        except Exception as exc:
            raise CaracalError(
                ErrorCode.DEPENDENCY_DOWN,
                "queue unavailable",
                {"job_id": str(job_id)},
                retryable=True,
            ) from exc
        logger.info(
            "translate queued",
            extra={"job_id": str(job_id), "langs": ",".join(languages)},
        )
        return TranslateQueued(job_id=job_id, status="queued", links=self.links(job_id))

    async def report(self, job_id: uuid.UUID) -> Report:
        """The finished report from the shared store (docs/03 §2-§3).

        Unknown jobs are ``not_found``; a job whose report does not exist yet (still running,
        or never assembled) is ``job_not_ready`` (409); a failed job without a report is
        ``not_found`` (404), exactly as the contract specifies.
        """
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        stored = await self.repos.reports.get(job_id)
        if stored is None:
            raise self._missing_artifact(job)
        try:
            return Report.model_validate(stored)
        except ValidationError as exc:
            raise CaracalError(
                ErrorCode.INTERNAL,
                "stored report is invalid",
                {"job_id": str(job_id)},
            ) from exc

    async def report_markdown(self, job_id: uuid.UUID) -> str:
        """The report rendered as escaped Markdown (docs/03 §2 ``format=md``)."""
        return render_markdown(await self.report(job_id))

    async def voiceprints(self, job_id: uuid.UUID) -> VoiceprintExport:
        """Raw diarization embeddings for calibration (card T4.3, REST only).

        The embeddings come straight from the diarize checkpoint; nothing is enrolled or
        matched. Only an explicit, bearer-protected REST call can read them because they
        are biometric data (docs/06 §2).
        """
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        files = JobFiles(self.data_dir, job_id)
        checkpoint = files.read_json(files.stage(Stage.DIARIZE))
        if not isinstance(checkpoint, dict):
            raise self._missing_artifact(job)
        output = checkpoint.get("output")
        raw = output.get("embeddings") if isinstance(output, dict) else None
        if raw is None:
            return VoiceprintExport(items=[])
        if not isinstance(raw, dict) or not all(isinstance(label, str) and label for label in raw):
            raise CaracalError(ErrorCode.INTERNAL, "diarize checkpoint has invalid embeddings")
        embeddings = cast("dict[str, object]", raw)
        items: list[VoiceprintItem] = []
        for label in sorted(embeddings):
            values = embeddings[label]
            if not isinstance(values, list) or len(values) != EMB_DIM:
                raise CaracalError(
                    ErrorCode.INTERNAL,
                    "diarize checkpoint has invalid embeddings",
                    {"label": label},
                )
            embedding: list[float] = []
            for value in values:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise CaracalError(ErrorCode.INTERNAL, "diarize checkpoint has invalid embeddings")
                number = float(value)
                if not math.isfinite(number):
                    raise CaracalError(ErrorCode.INTERNAL, "diarize checkpoint has invalid embeddings")
                embedding.append(number)
            items.append(VoiceprintItem(label=label, embedding=embedding))
        return VoiceprintExport(items=items)

    async def report_summary(self, job_id: uuid.UUID) -> ReportSummary | None:
        """The compact report view for MCP ``analyze_audio``, or ``None`` without a report."""
        try:
            report = await self.report(job_id)
        except CaracalError as exc:
            if exc.code in {ErrorCode.NOT_FOUND, ErrorCode.JOB_NOT_READY}:
                return None
            raise
        return report.summary_view()

    # -- question answering / search --------------------------------------

    def _qa_segments(self, job_id: uuid.UUID) -> tuple[QaSegment, ...]:
        """The stored transcript as Q&A segments; an absent file means hits-only context."""
        files = JobFiles(self.data_dir, job_id)
        if not files.transcript_json.is_file():
            return ()
        payload = self._artifact_json(files.transcript_json)
        raw = payload.get("segments")
        if not isinstance(raw, list):
            return ()
        segments: list[QaSegment] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            seg_id = item.get("id")
            start = item.get("start")
            end = item.get("end")
            text = item.get("text")
            if not isinstance(seg_id, int) or isinstance(seg_id, bool):
                continue
            if not isinstance(start, (int, float)) or isinstance(start, bool):
                continue
            if not isinstance(text, str):
                continue
            end_s = (
                float(end) if isinstance(end, (int, float)) and not isinstance(end, bool) else float(start)
            )
            speaker = item.get("speaker")
            segments.append(
                QaSegment(
                    seg_id=seg_id,
                    start=float(start),
                    end=end_s,
                    speaker=speaker if isinstance(speaker, str) else None,
                    text=text,
                )
            )
        return tuple(segments)

    async def ask_about_audio(self, job_id: uuid.UUID, question: str) -> Answer:
        """Answer a question about a finished job (docs/03 §2-§3, docs/04 §12)."""
        question = validate_question(question)
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        if job.status != "done":
            raise CaracalError(
                ErrorCode.JOB_NOT_READY,
                "job is not done",
                {"job_id": str(job_id), "status": job.status},
            )
        report = await self.report(job_id)
        context = qa_report(report)
        transcript = self._qa_segments(job_id)
        client = self.llm
        if client is not None:
            return await answer_question(
                job_id,
                question,
                repos=self.repos.segments,
                report=context,
                transcript=transcript,
                settings=self.settings,
                llm=client,
            )
        async with httpx.AsyncClient() as http:
            return await answer_question(
                job_id,
                question,
                repos=self.repos.segments,
                report=context,
                transcript=transcript,
                settings=self.settings,
                llm=LLMClient(
                    self.settings,
                    http,
                    metrics=MetricsRegistry(self.redis, self.keys),
                ),
            )

    async def search_transcripts(self, query: str, limit: int = 20) -> SearchResults:
        """Full-text search over finished jobs' transcripts (docs/03 §2-§3, docs/05 §1)."""
        needle = query.strip()
        if not needle:
            raise CaracalError(ErrorCode.INVALID_INPUT, "q must not be empty")
        if not 1 <= limit <= _SEARCH_POOL:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                "limit must be between 1 and 50",
                {"limit": limit},
            )
        hits = await self.repos.segments.search(needle, limit=_SEARCH_POOL)
        finished: dict[uuid.UUID, bool] = {}
        items: list[SearchItem] = []
        for hit in hits:
            done = finished.get(hit.job_id)
            if done is None:
                job = await self.repos.jobs.get(hit.job_id)
                done = job is not None and job.status == "done"
                finished[hit.job_id] = done
            if not done:
                continue
            items.append(
                SearchItem(
                    job_id=hit.job_id,
                    title=hit.title,
                    seg_id=hit.seg_id,
                    start=hit.start_s,
                    speaker=hit.speaker,
                    text=hit.text,
                    rank=hit.rank,
                )
            )
            if len(items) >= limit:
                break
        return SearchResults(items=items)

    # -- speakers ---------------------------------------------------------

    async def list_speakers(self, query: str | None = None) -> SpeakerList:
        """``GET /v1/speakers`` and MCP ``list_speakers`` (docs/03 §2-§3)."""
        return SpeakerList(items=await self.speaker_service.list(query))

    async def enroll_speaker(
        self,
        job_id: uuid.UUID,
        speaker_label: str,
        name: str,
        notes: str | None = None,
    ) -> KnownSpeaker:
        """``POST /v1/speakers/enroll`` and MCP ``enroll_speaker`` (docs/03 §2-§3)."""
        return await self.speaker_service.enroll(job_id, speaker_label, name, notes)

    async def delete_speaker(self, speaker_id: uuid.UUID) -> SpeakerDeleted:
        """``DELETE /v1/speakers/{id}`` and MCP ``delete_speaker`` (docs/03 §2-§3)."""
        return await self.speaker_service.delete(speaker_id)

    # -- cancel / delete --------------------------------------------------

    async def cancel_or_delete(
        self,
        job_id: uuid.UUID,
        *,
        delete_artifacts: bool,
    ) -> CancelResult | None:
        """Cancel an active job, or delete a terminal one (docs/03 §2).

        Returns ``None`` when the job was deleted, otherwise the job's current/cancelling
        status.
        """
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        if job.status not in _TERMINAL:
            await set_cancel(self.redis, self.keys, job_id)
            return CancelResult(job_id=job_id, status="cancelling")
        if not delete_artifacts:
            return CancelResult(job_id=job_id, status=job.status)
        await self.repos.jobs.delete(job_id)
        self._remove_job_dir(job_id)
        return None

    def _remove_job_dir(self, job_id: uuid.UUID) -> None:
        jobs_dir = (self.data_dir / "jobs").resolve()
        target = (self.data_dir / "jobs" / str(job_id)).resolve()
        if target.parent != jobs_dir:
            logger.warning(
                "refusing to delete a job directory outside the data dir",
                extra={"job_id": str(job_id)},
            )
            return
        shutil.rmtree(target, ignore_errors=True)

    # -- waiting ----------------------------------------------------------

    async def wait_for_job(
        self,
        job_id: uuid.UUID,
        timeout_s: float,
        on_progress: ProgressCallback | None = None,
    ) -> JobStatus:
        """Block until the job reaches a terminal state or ``timeout_s`` elapses.

        Subscription happens *before* the first snapshot, exactly like the SSE stream, so an
        event published between the two is never lost.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout_s)
        channel = self.keys.events(job_id)
        async with self.redis.pubsub() as pubsub:
            await pubsub.subscribe(channel)
            status = await self.get_status(job_id)
            if status.status in _TERMINAL or timeout_s <= 0:
                return status
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return await self.get_status(job_id)
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=min(_WAIT_POLL_S, remaining),
                )
                if message is None:
                    continue
                payload = self._event_payload(message.get("data"))
                if payload is None:
                    continue
                if payload.get("type") == "end":
                    return await self.get_status(job_id)
                if on_progress is not None:
                    raw_pct = payload.get("pct", 0)
                    pct = float(raw_pct) if isinstance(raw_pct, (int, float)) else 0.0
                    await on_progress(
                        JobProgress(
                            pct=pct,
                            stage=cast("str | None", payload.get("stage")),
                            message=cast("str | None", payload.get("message")),
                        )
                    )
                current = await self.get_status(job_id)
                if current.status in _TERMINAL:
                    return current

    def _event_payload(self, data: object) -> dict[str, Any] | None:
        if data is None:
            return None
        raw = data.decode("utf-8") if isinstance(data, bytes) else str(data)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("ignoring malformed event payload")
            return None
        return cast("dict[str, Any]", payload) if isinstance(payload, dict) else None

    # -- MCP summary ------------------------------------------------------

    async def analyze_summary(self, job_id: uuid.UUID) -> AnalyzeSummary:
        """The compact digest ``analyze_audio`` returns when a job is done."""
        job = await self.repos.jobs.get(job_id)
        if job is None:
            raise _not_found(job_id)
        files = JobFiles(self.data_dir, job_id)
        preview = ""
        if files.transcript_txt.is_file():
            preview = files.transcript_txt.read_text(encoding="utf-8")[:_TRANSCRIPT_PREVIEW_CHARS]
        return AnalyzeSummary(
            title=job.title,
            duration_s=self._duration_s(files),
            languages=self._transcript_languages(files),
            transcript_preview=preview,
        )

    def _duration_s(self, files: JobFiles) -> float:
        checkpoint = files.read_json(files.stage(Stage.NORMALIZE))
        if isinstance(checkpoint, dict):
            output = checkpoint.get("output")
            if isinstance(output, dict):
                stats = output.get("stats")
                if isinstance(stats, dict):
                    duration = stats.get("duration_s")
                    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
                        return float(duration)
        payload = files.read_json(files.transcript_json)
        if isinstance(payload, dict):
            segments = payload.get("segments")
            if isinstance(segments, list) and segments:
                last = segments[-1]
                if isinstance(last, dict):
                    end = last.get("end")
                    if isinstance(end, (int, float)) and not isinstance(end, bool):
                        return float(end)
        return 0.0


def ident_hash_for_mcp(settings: Settings) -> str:
    """Rate-limit identity for MCP callers (stdio has no client IP; the token is the actor)."""
    token = settings.CARACAL_API_TOKEN.get_secret_value()
    return hashlib.sha1(f"{token}:mcp".encode(), usedforsecurity=False).hexdigest()


__all__ = ["MAX_BASE64_DECODED_BYTES", "JobService", "SpeakerService", "ident_hash_for_mcp"]
