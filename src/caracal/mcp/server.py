"""MCP tools over the shared service layer (docs/03 §3, docs/01 §3).

The tools exist on two transports — streamable HTTP mounted at ``/mcp`` and stdio — and are
thin adapters: every call delegates to :class:`caracal.services.JobService` and returns the
same pydantic models the REST routes serialise. Tool descriptions are written for an LLM
caller (what it does, when to use it, that analysis is asynchronous, what to call next).
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, RootModel, SerializerFunctionWrapHandler, ValidationError, model_serializer

from caracal.api.schemas import (
    AnalyzeSummary,
    CancelResult,
    JobLinks,
    JobList,
    JobProgress,
    JobSourceIn,
    JobStatus,
    KnownSpeaker,
    SpeakerDeleted,
)
from caracal.config import Settings
from caracal.errors import CaracalError
from caracal.pipeline.stages import JobOptions
from caracal.services import MAX_BASE64_DECODED_BYTES, JobService, ident_hash_for_mcp

HTTP_INLINE_BASE64_BYTES = 750 * 1024
"""Largest inline ``audio_base64`` payload accepted over HTTP (the 1 MiB body cap)."""

_PENDING_HINT = "call get_job / get_transcript later"
_BIOMETRIC_NOTE = (
    "Voiceprints are biometric data: only enroll with the person's explicit consent, and "
    "delete them as soon as they are no longer needed."
)
_SERVER_INSTRUCTIONS = (
    "Caracal turns an audio or video file or link into a timestamped transcript and "
    "subtitles. Analysis is asynchronous: analyze_audio creates a job, then get_job polls "
    "it and get_transcript reads the result."
)


class AnalyzeResult(BaseModel):
    """Outcome of ``analyze_audio``: a finished summary, or the current async progress."""

    job_id: uuid.UUID
    status: str
    summary: AnalyzeSummary | None = None
    links: JobLinks | None = None
    progress: JobProgress | None = None
    hint: str | None = None

    @model_serializer(mode="wrap")
    def _serialise(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        data: dict[str, object] = handler(self)
        return {key: value for key, value in data.items() if value is not None}


class TranscriptResult(RootModel[dict[str, Any]]):
    """``get_transcript`` output.

    For ``format="json"`` the root *is* the transcript payload, identical to
    ``GET /v1/jobs/{id}/transcript``. Text/SRT/VTT formats use ``format``, ``lang`` and
    ``content`` keys.
    """


class SpeakerList(RootModel[list[KnownSpeaker]]):
    """``list_speakers`` output: the same array ``GET /v1/speakers`` returns."""


def _tool_error(exc: CaracalError) -> ToolError:
    """Caracal error -> MCP error carrying the stable code, never a traceback or input value."""
    return ToolError(f"{exc.code.value}: {exc.message}")


def _job_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ToolError("invalid_input: job_id must be a UUID") from exc


def _speaker_uuid(value: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ToolError("invalid_input: speaker_id must be a UUID") from exc


def _pending(job_id: uuid.UUID, status: JobStatus) -> AnalyzeResult:
    return AnalyzeResult(
        job_id=job_id,
        status=status.status,
        progress=status.progress,
        hint=_PENDING_HINT,
    )


def build_mcp(
    service: JobService,
    settings: Settings,
    *,
    max_inline_bytes: int = MAX_BASE64_DECODED_BYTES,
) -> MCPServer:
    """Build the MCP server; ``max_inline_bytes`` caps inline base64 for this transport."""
    server = MCPServer(name="caracal", instructions=_SERVER_INSTRUCTIONS)
    ident_hash = ident_hash_for_mcp(settings)

    @server.tool(
        description=(
            "Submit an audio or video source to Caracal for asynchronous analysis. Use "
            "exactly one source: a public URL (url), a file already uploaded with "
            "POST /v1/uploads (upload_id), or inline audio (audio_base64, plus optional "
            "filename). Inline audio is limited to 25 MB decoded over stdio and 750 KiB "
            "over HTTP — upload larger files with POST /v1/uploads and pass upload_id. "
            "Analysis runs in the background: the result carries the job_id and links, and "
            "when wait_seconds > 0 the call waits up to that many seconds while reporting "
            "progress, returning a summary once the job is done. If it is still running, "
            "call get_job or get_transcript later."
        )
    )
    async def analyze_audio(
        ctx: Context,
        url: str | None = None,
        upload_id: str | None = None,
        audio_base64: str | None = None,
        filename: str | None = None,
        options: dict[str, Any] | None = None,
        wait_seconds: int = 120,
    ) -> AnalyzeResult:
        if sum(value is not None for value in (url, upload_id, audio_base64)) != 1:
            raise ToolError("invalid_input: provide exactly one of url, upload_id or audio_base64")
        if not 0 <= wait_seconds <= 600:
            raise ToolError("invalid_input: wait_seconds must be between 0 and 600")
        try:
            job_options = JobOptions.model_validate(options or {})
        except ValidationError as exc:
            raise ToolError("invalid_input: invalid options") from exc

        source: JobSourceIn
        if url is not None:
            try:
                source = JobSourceIn(url=url)
            except ValidationError as exc:
                raise ToolError("invalid_input: url must be an http(s) URL with a host") from exc
        elif upload_id is not None:
            try:
                parsed = uuid.UUID(upload_id)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ToolError("invalid_input: upload_id must be a UUID") from exc
            source = JobSourceIn(upload_id=parsed)
        else:
            if audio_base64 is None:
                raise ToolError("invalid_input: provide exactly one of url, upload_id or audio_base64")
            # Checked before decoding, exactly like S7; this is the transport's own cap.
            if len(audio_base64) * 3 // 4 > max_inline_bytes:
                raise ToolError(
                    f"too_large: inline audio exceeds {max_inline_bytes} decoded bytes; "
                    "upload the file with POST /v1/uploads and pass upload_id"
                )
            try:
                upload = await service.create_upload_from_base64(
                    audio_base64,
                    filename or "audio.bin",
                )
            except CaracalError as exc:
                raise _tool_error(exc) from exc
            source = JobSourceIn(upload_id=upload.upload_id)

        try:
            created = await service.create_job(source, job_options, ident_hash=ident_hash)
        except CaracalError as exc:
            raise _tool_error(exc) from exc

        if wait_seconds == 0:
            status = await service.get_status(created.job_id)
            return _pending(created.job_id, status)

        async def on_progress(progress: JobProgress) -> None:
            await ctx.report_progress(progress.pct, 100, progress.message)

        try:
            status = await service.wait_for_job(created.job_id, float(wait_seconds), on_progress)
        except CaracalError as exc:
            raise _tool_error(exc) from exc
        if status.status == "done":
            try:
                summary = await service.analyze_summary(created.job_id)
            except CaracalError as exc:
                raise _tool_error(exc) from exc
            return AnalyzeResult(
                job_id=created.job_id,
                status="done",
                summary=summary,
                links=service.links(created.job_id),
            )
        return _pending(created.job_id, status)

    @server.tool(
        description=(
            "Get one analysis job's status, per-stage progress, source and links. Use it "
            "after analyze_audio returns queued or running, polling until the status is "
            "done, failed or cancelled. When it is done, call get_transcript for the text; "
            "call cancel_job to stop a job you no longer need."
        )
    )
    async def get_job(job_id: str) -> JobStatus:
        try:
            return await service.get_status(_job_uuid(job_id))
        except CaracalError as exc:
            raise _tool_error(exc) from exc

    @server.tool(
        description=(
            "List recent analysis jobs, newest first, optionally filtered by status "
            "(queued, running, done, failed or cancelled). Use it to recover a job_id; then "
            "call get_job for detail."
        )
    )
    async def list_jobs(
        status: str | None = None,
        limit: int = 20,
        cursor: str | None = None,
    ) -> JobList:
        if not 1 <= limit <= 50:
            raise ToolError("invalid_input: limit must be between 1 and 50")
        try:
            return await service.list_jobs(status, limit, cursor)
        except CaracalError as exc:
            raise _tool_error(exc) from exc

    @server.tool(
        description=(
            "Read a finished job's transcript. format=json returns timestamped segments, "
            "words and detected languages (the same payload as the REST transcript "
            "endpoint); format=text returns a plain-text transcript; format=srt or vtt "
            "returns subtitle text for lang (source or an ISO 639 code). Use it after "
            "get_job reports done; while the job is still running it fails with "
            "job_not_ready — poll get_job first."
        )
    )
    async def get_transcript(
        job_id: str,
        format: Literal["text", "json", "srt", "vtt"] = "text",
        lang: str = "source",
    ) -> TranscriptResult:
        parsed = _job_uuid(job_id)
        try:
            if format == "json":
                payload = await service.transcript(parsed, "json")
                if not isinstance(payload, dict):
                    raise ToolError("internal: transcript is not a JSON object")
                return TranscriptResult(root=payload)
            if format == "text":
                text = await service.transcript(parsed, "txt")
                if not isinstance(text, str):
                    raise ToolError("internal: transcript text is not text")
                return TranscriptResult(root={"format": "text", "lang": "source", "content": text})
            content, _media_type = await service.subtitles(parsed, lang, format)
        except CaracalError as exc:
            raise _tool_error(exc) from exc
        return TranscriptResult(root={"format": format, "lang": lang, "content": content.decode("utf-8")})

    @server.tool(
        description=(
            "List enrolled known speakers (people whose voiceprints are stored), each with "
            "name, notes, sample count and timestamps; query filters by name. Use it to find "
            "a speaker_id or to check whether a voice is already enrolled. " + _BIOMETRIC_NOTE
        )
    )
    async def list_speakers(query: str | None = None) -> SpeakerList:
        try:
            return SpeakerList(root=await service.list_speakers(query))
        except CaracalError as exc:
            raise _tool_error(exc) from exc

    @server.tool(
        description=(
            "Enroll one speaker from a finished analysis job: give its job_id, the diarized "
            "speaker_label (S1, S2, ...), and the person's name (notes optional). The stored "
            "embedding lets later jobs recognize the same voice; the same normalised name "
            "adds another sample to the existing speaker. Fails with job_not_ready until the "
            "job is done and with not_found when the label is not in the diarization. " + _BIOMETRIC_NOTE
        )
    )
    async def enroll_speaker(
        job_id: str,
        speaker_label: str,
        name: str,
        notes: str | None = None,
    ) -> KnownSpeaker:
        parsed = _job_uuid(job_id)
        try:
            return await service.enroll_speaker(parsed, speaker_label, name, notes)
        except CaracalError as exc:
            raise _tool_error(exc) from exc

    @server.tool(
        description=(
            "Delete a known speaker by speaker_id, together with every voiceprint stored for "
            "them, and report how many embeddings were removed. Irreversible: use it to "
            "honour a deletion request. " + _BIOMETRIC_NOTE
        )
    )
    async def delete_speaker(speaker_id: str) -> SpeakerDeleted:
        parsed = _speaker_uuid(speaker_id)
        try:
            return await service.delete_speaker(parsed)
        except CaracalError as exc:
            raise _tool_error(exc) from exc

    @server.tool(
        description=(
            "Cancel a queued or running job, or delete a finished job. With the default "
            "delete_artifacts=false an active job gets a cancel flag and keeps its files; "
            "pass true to also delete a finished job's database rows and artifacts. Use it "
            "to stop work you no longer need or to remove a job and its evidence."
        )
    )
    async def cancel_job(job_id: str, delete_artifacts: bool = False) -> CancelResult:
        parsed = _job_uuid(job_id)
        try:
            result = await service.cancel_or_delete(parsed, delete_artifacts=delete_artifacts)
        except CaracalError as exc:
            raise _tool_error(exc) from exc
        if result is None:
            return CancelResult(job_id=parsed, status="deleted")
        return result

    return server


__all__ = [
    "HTTP_INLINE_BASE64_BYTES",
    "AnalyzeResult",
    "SpeakerList",
    "TranscriptResult",
    "build_mcp",
]
