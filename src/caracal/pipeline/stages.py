"""Stage names, job options, the stage context and the M1 stage implementations (docs/01 §4).

The M1 pipeline is the *hear* slice: ingest → normalize → chunk → transcribe → langid →
subtitles (docs/04 §3-§4, §10). Later stages stay clean ``skipped`` checkpoints until their
milestones land. The runner writes every output as JSON and stores artifact paths relative
to the job root so a resumed job is portable.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from caracal.analysis.content import analyse_content
from caracal.analysis.identity import VoiceprintMatch, resolve_identities
from caracal.analysis.planner import PlanInputs
from caracal.analysis.planner import plan as build_plan
from caracal.asr import (
    Segment,
    Transcript,
    WhisperResult,
    Word,
)
from caracal.asr.captions import caption_agreement, parse_vtt
from caracal.asr.langid import languages_summary, refine_spans
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import WhisperClient
from caracal.audio import ffmpeg
from caracal.audio.chunking import Chunk, plan_chunks
from caracal.config import Settings
from caracal.db.repo import JobRepo, ReportRepo, SegmentRepo, SegmentRow, SpeakerRepo, UploadRepo
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest import ytdlp
from caracal.ingest.direct import download_direct
from caracal.ingest.ssrf import Resolver
from caracal.llm.client import LLMClient
from caracal.log import get_logger
from caracal.report.model import Entity
from caracal.speakers.align import assign_speakers, retime_outside_speech, speaker_stats
from caracal.speakers.diarize import Diarizer, Turn
from caracal.speakers.voiceprint import classify
from caracal.storage import JobFiles
from caracal.subtitles import cues_from_transcript, to_srt, to_vtt

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from caracal.queue import Keys

logger = get_logger("caracal.pipeline.stages")

_MIB = 1024 * 1024
_M1_NOT_IMPLEMENTED = "not implemented in M1"
_PENDING_STAGE = "not implemented yet"


class Stage(StrEnum):
    """Pipeline stages in canonical execution order (docs/01 §4)."""

    INGEST = "ingest"
    NORMALIZE = "normalize"
    CHUNK = "chunk"
    TRANSCRIBE = "transcribe"
    LANGID = "langid"
    DIARIZE = "diarize"
    PLAN = "plan"
    ALIGN = "align"
    VOICEPRINT = "voiceprint"
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

    @property
    def speakers(self) -> SpeakerRepo: ...


async def _never_cancelled() -> bool:
    """Default cancel probe for contexts built outside the runner."""
    return False


@dataclass
class StageContext:
    """Everything a stage function receives.

    The client fields are populated by the worker (one ``httpx.AsyncClient`` per process,
    docs/01 §4). They are optional only so the M0 stub registry can run in isolation;
    every M1 stage that needs one raises ``internal`` when it is missing.
    """

    job_id: UUID
    options: JobOptions
    files: JobFiles
    settings: Settings
    repos: Repos
    load: Callable[[Stage], Awaitable[dict[str, object] | None]]
    progress: Callable[[float, str], Awaitable[None]]
    whisper: WhisperClient | None = None
    llm: LLMClient | None = None
    diarizer: Diarizer | None = None
    resolver: Resolver | None = None
    redis: Redis | None = None
    keys: Keys | None = None
    cancelled: Callable[[], Awaitable[bool]] = _never_cancelled


@dataclass(frozen=True)
class StageResult:
    """The outcome a stage reports; the runner persists ``output`` as its checkpoint."""

    status: Literal["done", "skipped"]
    reason: str | None = None
    output: dict[str, object] = field(default_factory=dict)


StageFn = Callable[[StageContext], Awaitable[StageResult]]


def _malformed(what: str) -> CaracalError:
    return CaracalError(ErrorCode.INTERNAL, f"{what} checkpoint is malformed")


def _require_whisper(ctx: StageContext) -> WhisperClient:
    if ctx.whisper is None:
        raise CaracalError(ErrorCode.INTERNAL, "whisper client is not configured")
    return ctx.whisper


def _require_resolver(ctx: StageContext) -> Resolver:
    if ctx.resolver is None:
        raise CaracalError(ErrorCode.INTERNAL, "url resolver is not configured")
    return ctx.resolver


def _as_dict(value: object, what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _malformed(what)
    return cast("dict[str, object]", value)


def _as_list(value: object, what: str) -> list[object]:
    if not isinstance(value, list):
        raise _malformed(what)
    return value


def _as_str(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise _malformed(what)
    return value


def _as_number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _malformed(what)
    return float(value)


def _as_bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _relative(ctx: StageContext, path: Path) -> str:
    return path.relative_to(ctx.files.root).as_posix()


def _job_path(ctx: StageContext, relative: str) -> Path:
    """Resolve a checkpoint-relative artifact path, refusing anything outside the job root."""
    root = ctx.files.root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise CaracalError(ErrorCode.INTERNAL, "artifact path escapes the job directory")
    return candidate


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link an upload into the job directory, falling back to a copy across devices."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


async def _stage_output(ctx: StageContext, stage: Stage) -> dict[str, object] | None:
    """Return the ``output`` object of a previous checkpoint (the runner stores a wrapper)."""
    checkpoint = await ctx.load(stage)
    if checkpoint is None:
        return None
    return _as_dict(checkpoint.get("output"), f"{stage.value} checkpoint")


def _source_file(ctx: StageContext, ingest: dict[str, object] | None) -> Path:
    if ingest is None:
        raise CaracalError(ErrorCode.INTERNAL, "ingest checkpoint is missing")
    filename = _as_str(ingest.get("filename"), "ingest")
    return ctx.files.source_dir / filename


async def stage_ingest(ctx: StageContext) -> StageResult:
    """Bring the source into the job directory and record its provenance (docs/04 §3)."""
    job = await ctx.repos.jobs.get(ctx.job_id)
    if job is None:
        raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(ctx.job_id)})
    ctx.files.ensure()
    settings = ctx.settings

    output: dict[str, object]
    if job.source_kind == "upload":
        if job.upload_id is None:
            raise CaracalError(ErrorCode.INTERNAL, "upload job has no upload id")
        upload = await ctx.repos.uploads.get(job.upload_id)
        if upload is None:
            raise CaracalError(
                ErrorCode.NOT_FOUND,
                "upload not found",
                {"upload_id": str(job.upload_id)},
            )
        destination = ctx.files.source_dir / upload.filename
        await asyncio.to_thread(_link_or_copy, Path(upload.path), destination)
        await ctx.repos.jobs.set_title(ctx.job_id, upload.filename)
        output = {
            "source_kind": "upload",
            "url": None,
            "final_url": None,
            "filename": upload.filename,
            "size_bytes": upload.size_bytes,
            "sha256": upload.sha256,
            "platform": None,
            "captions": [],
        }
        logger.info(
            "ingest complete",
            extra={"job_id": str(ctx.job_id), "source_kind": "upload", "size_bytes": upload.size_bytes},
        )
        return StageResult(status="done", output=output)

    url = job.source_url
    if url is None:
        raise CaracalError(ErrorCode.INTERNAL, "source job has no URL")
    resolver = _require_resolver(ctx)

    if job.source_kind == "direct":
        destination = ctx.files.source_dir / "download"
        info = await download_direct(
            url,
            destination,
            resolver=resolver,
            max_bytes=settings.CARACAL_MAX_UPLOAD_MB * _MIB,
        )
        await ctx.repos.jobs.set_title(ctx.job_id, info.filename)
        output = {
            "source_kind": "direct",
            "url": url,
            "final_url": info.final_url,
            "filename": destination.name,
            "size_bytes": info.size_bytes,
            "sha256": info.sha256,
            "platform": None,
            "captions": [],
        }
        logger.info(
            "ingest complete",
            extra={"job_id": str(ctx.job_id), "source_kind": "direct", "size_bytes": info.size_bytes},
        )
        return StageResult(status="done", output=output)

    if job.source_kind != "platform":
        raise CaracalError(
            ErrorCode.INTERNAL,
            "unknown source kind",
            {"source_kind": job.source_kind},
        )

    meta = await ytdlp.fetch_metadata(url, settings, resolver=resolver)
    audio = await ytdlp.download_audio(
        url,
        ctx.files,
        settings,
        resolver=resolver,
        cancel=threading.Event(),
    )
    captions: list[dict[str, object]] = []
    try:
        tracks = await ytdlp.fetch_platform_captions(url, meta, ctx.files, settings, resolver=resolver)
    except CaracalError as exc:
        # Captions are a bonus cross-check: a platform that refuses them never fails ingest.
        logger.warning(
            "platform captions unavailable",
            extra={"job_id": str(ctx.job_id), "code": exc.code.value},
        )
        tracks = []
    for track in tracks:
        captions.append(
            {
                "lang": track.lang,
                "kind": track.kind,
                "path": _relative(ctx, track.path),
            }
        )
    await ctx.repos.jobs.set_title(ctx.job_id, meta.title or audio.path.name)
    output = {
        "source_kind": "platform",
        "url": url,
        "final_url": meta.webpage_url or url,
        "filename": audio.path.name,
        "size_bytes": audio.size_bytes,
        "sha256": audio.sha256,
        "platform": asdict(meta),
        "captions": captions,
    }
    logger.info(
        "ingest complete",
        extra={
            "job_id": str(ctx.job_id),
            "source_kind": "platform",
            "size_bytes": audio.size_bytes,
            "captions": len(captions),
        },
    )
    return StageResult(status="done", output=output)


async def stage_normalize(ctx: StageContext) -> StageResult:
    """Decode to 16 kHz mono PCM, measure speech and drop the raw source (docs/04 §3)."""
    ingest = await _stage_output(ctx, Stage.INGEST)
    source = _source_file(ctx, ingest)
    probe = await ffmpeg.normalize(
        source,
        ctx.files.audio_wav,
        max_duration_s=float(ctx.settings.CARACAL_MAX_DURATION_S),
    )
    stats = await ffmpeg.analyse(ctx.files.audio_wav)
    if not ctx.settings.CARACAL_KEEP_MEDIA:
        shutil.rmtree(ctx.files.source_dir, ignore_errors=True)
    logger.info(
        "normalize complete",
        extra={
            "job_id": str(ctx.job_id),
            "duration_s": stats.duration_s,
            "speech_ratio": stats.speech_ratio,
            "quality": stats.quality,
        },
    )
    return StageResult(
        status="done",
        output={
            "probe": asdict(probe),
            "stats": {
                "duration_s": stats.duration_s,
                "speech_ratio": stats.speech_ratio,
                "loudness_lufs": stats.loudness_lufs,
                "quality": stats.quality,
                "silences": [[start, end] for start, end in stats.silences],
            },
        },
    )


async def stage_chunk(ctx: StageContext) -> StageResult:
    """Cut the normalized audio at planned boundaries (docs/04 §3)."""
    normalize = await _stage_output(ctx, Stage.NORMALIZE)
    if normalize is None:
        raise CaracalError(ErrorCode.INTERNAL, "normalize checkpoint is missing")
    stats = _as_dict(normalize.get("stats"), "normalize")
    duration_s = _as_number(stats.get("duration_s"), "normalize")
    raw_silences = _as_list(stats.get("silences", []), "normalize")
    silences: list[tuple[float, float]] = []
    for item in raw_silences:
        pair = _as_list(item, "normalize")
        if len(pair) != 2:
            raise _malformed("normalize")
        silences.append(
            (
                _as_number(pair[0], "normalize"),
                _as_number(pair[1], "normalize"),
            )
        )

    chunks = plan_chunks(duration_s, silences)
    for chunk in chunks:
        await ffmpeg.cut(ctx.files.audio_wav, ctx.files.chunk(chunk.index), chunk.start_s, chunk.end_s)
    return StageResult(
        status="done",
        output={
            "chunks": [
                {
                    "index": chunk.index,
                    "start_s": chunk.start_s,
                    "end_s": chunk.end_s,
                    "path": _relative(ctx, ctx.files.chunk(chunk.index)),
                }
                for chunk in chunks
            ]
        },
    )


def _chunk_specs(chunk_checkpoint: dict[str, object] | None) -> list[dict[str, object]]:
    if chunk_checkpoint is None:
        raise CaracalError(ErrorCode.INTERNAL, "chunk checkpoint is missing")
    specs: list[dict[str, object]] = []
    for item in _as_list(chunk_checkpoint.get("chunks"), "chunk"):
        specs.append(_as_dict(item, "chunk"))
    return specs


def _spec_chunk(spec: dict[str, object]) -> Chunk:
    return Chunk(
        index=int(_as_number(spec.get("index"), "chunk")),
        start_s=_as_number(spec.get("start_s"), "chunk"),
        end_s=_as_number(spec.get("end_s"), "chunk"),
    )


def _transcript_text(transcript: Transcript) -> str:
    lines: list[str] = []
    for segment in transcript.segments:
        stamp = f"{int(segment.start // 60):02d}:{int(segment.start % 60):02d}"
        if segment.speaker is not None:
            lines.append(f"[{stamp}] {segment.speaker}: {segment.text}")
        else:
            lines.append(f"[{stamp}] {segment.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _write_transcript(ctx: StageContext, transcript: Transcript) -> None:
    """Persist the canonical transcript JSON and its human-readable text form."""
    ctx.files.write_json(ctx.files.transcript_json, transcript.to_dict())
    ctx.files.write_text(ctx.files.transcript_txt, _transcript_text(transcript))


def _segment_rows(ctx: StageContext, transcript: Transcript) -> list[SegmentRow]:
    return [
        SegmentRow(
            job_id=ctx.job_id,
            seg_id=segment.id,
            start_s=segment.start,
            end_s=segment.end,
            speaker=segment.speaker,
            lang=segment.lang,
            text=segment.text,
        )
        for segment in transcript.segments
    ]


async def stage_transcribe(ctx: StageContext) -> StageResult:
    """Transcribe every chunk under the whisper lock and persist the merged transcript."""
    specs = _chunk_specs(await _stage_output(ctx, Stage.CHUNK))
    chunks = [_spec_chunk(spec) for spec in specs]
    whisper = _require_whisper(ctx)

    results: list[WhisperResult] = []
    for position, (spec, _chunk) in enumerate(zip(specs, chunks, strict=True), start=1):
        if await ctx.cancelled():
            # The runner sees the cancel flag at the next stage boundary and settles the job
            # as ``cancelled``; no partial transcript is written (docs/05 §2).
            return StageResult(status="skipped", reason="cancelled")
        chunk_path = _job_path(ctx, _as_str(spec.get("path"), "chunk"))
        results.append(await whisper.transcribe(chunk_path, language=ctx.options.language))
        await ctx.progress(
            round(position / len(chunks) * 100, 1),
            f"transcribed chunk {position}/{len(chunks)}",
        )

    transcript = merge_chunks(results, chunks)
    _write_transcript(ctx, transcript)
    await ctx.repos.segments.replace_for_job(ctx.job_id, _segment_rows(ctx, transcript))
    shutil.rmtree(ctx.files.chunks_dir, ignore_errors=True)
    logger.info(
        "transcribe complete",
        extra={
            "job_id": str(ctx.job_id),
            "segments": len(transcript.segments),
            "language_code": transcript.language_code,
            "chunks": len(chunks),
        },
    )
    return StageResult(
        status="done",
        output={
            "segments": len(transcript.segments),
            "words": sum(len(segment.words) for segment in transcript.segments),
            "language_code": transcript.language_code,
            "dropped": dict(transcript.dropped),
            "chunk_langs": [
                {
                    "start_s": chunk.start_s,
                    "code": result.language_code,
                    "prob": result.language_prob,
                }
                for chunk, result in zip(chunks, results, strict=True)
            ],
        },
    )


def _load_transcript(ctx: StageContext) -> Transcript:
    payload = ctx.files.read_json(ctx.files.transcript_json)
    if not isinstance(payload, dict):
        raise CaracalError(ErrorCode.INTERNAL, "transcript is missing")
    body = cast("dict[str, object]", payload)

    segments: list[Segment] = []
    for raw_segment in _as_list(body.get("segments", []), "transcript"):
        item = _as_dict(raw_segment, "transcript")
        words: list[Word] = []
        for raw_word in _as_list(item.get("words", []), "transcript"):
            word = _as_dict(raw_word, "transcript")
            word_speaker = word.get("speaker")
            words.append(
                Word(
                    text=_as_str(word.get("text"), "transcript"),
                    start=_as_number(word.get("start"), "transcript"),
                    end=_as_number(word.get("end"), "transcript"),
                    prob=_as_number(word.get("prob"), "transcript"),
                    speaker=word_speaker if isinstance(word_speaker, str) else None,
                    space_before=_as_bool(word.get("space_before")),
                )
            )
        speaker = item.get("speaker")
        lang = item.get("lang")
        segments.append(
            Segment(
                id=int(_as_number(item.get("id"), "transcript")),
                start=_as_number(item.get("start"), "transcript"),
                end=_as_number(item.get("end"), "transcript"),
                text=str(item.get("text", "")),
                words=tuple(words),
                avg_logprob=_as_number(item.get("avg_logprob", 0.0), "transcript"),
                no_speech_prob=_as_number(item.get("no_speech_prob", 0.0), "transcript"),
                lang=lang if isinstance(lang, str) else None,
                speaker=speaker if isinstance(speaker, str) else None,
                lang_prob=_as_number(item.get("lang_prob", 1.0), "transcript"),
            )
        )
    language_code = body.get("language_code")
    dropped = body.get("dropped")
    return Transcript(
        segments=tuple(segments),
        language_code=language_code if isinstance(language_code, str) else None,
        dropped=cast("dict[str, int]", dropped) if isinstance(dropped, dict) else {},
    )


async def stage_langid(ctx: StageContext) -> StageResult:
    """Refine language spans, summarise them and cross-check platform captions (docs/04 §4)."""
    transcript = _load_transcript(ctx)
    specs = _chunk_specs(await _stage_output(ctx, Stage.CHUNK))
    transcribe_output = await _stage_output(ctx, Stage.TRANSCRIBE)
    if transcribe_output is None:
        raise CaracalError(ErrorCode.INTERNAL, "transcribe checkpoint is missing")
    durations = {
        _as_number(spec.get("start_s"), "chunk"): _as_number(spec.get("end_s"), "chunk") for spec in specs
    }
    chunk_results: list[tuple[float, WhisperResult]] = []
    for raw in _as_list(transcribe_output.get("chunk_langs", []), "transcribe"):
        entry = _as_dict(raw, "transcribe")
        start_s = _as_number(entry.get("start_s"), "transcribe")
        code = entry.get("code")
        chunk_results.append(
            (
                start_s,
                WhisperResult(
                    language_code=code if isinstance(code, str) else None,
                    language_prob=_as_number(entry.get("prob"), "transcribe"),
                    language_probs={},
                    duration=durations.get(start_s, 0.0) - start_s,
                    segments=(),
                ),
            )
        )

    whisper = _require_whisper(ctx)

    async def recheck(start_s: float, end_s: float) -> WhisperResult:
        window = ctx.files.root / "langid" / f"{start_s:.3f}-{end_s:.3f}.wav"
        await ffmpeg.cut(ctx.files.audio_wav, window, start_s, end_s)
        try:
            return await whisper.transcribe(window, language=ctx.options.language)
        finally:
            window.unlink(missing_ok=True)

    refined = await refine_spans(transcript, chunk_results, recheck=recheck)
    if refined.to_dict() != transcript.to_dict():
        ctx.files.write_json(ctx.files.transcript_json, refined.to_dict())

    agreement: float | None = None
    caption_lang: str | None = None
    ingest = await _stage_output(ctx, Stage.INGEST)
    if ingest is not None:
        for raw in _as_list(ingest.get("captions", []), "ingest"):
            entry = _as_dict(raw, "ingest")
            caption_path = _job_path(ctx, _as_str(entry.get("path"), "ingest"))
            if not caption_path.is_file():
                continue
            words = parse_vtt(caption_path.read_text(encoding="utf-8"))
            agreement = caption_agreement(refined, words)
            caption_lang = _as_str(entry.get("lang"), "ingest")
            break

    languages = languages_summary(refined)
    logger.info(
        "langid complete",
        extra={
            "job_id": str(ctx.job_id),
            "languages": [item.get("code") for item in languages],
            "caption_agreement": agreement,
        },
    )
    return StageResult(
        status="done",
        output={
            "languages": languages,
            "caption_agreement": agreement,
            "caption_lang": caption_lang,
        },
    )


_PLAN_METADATA_KEYS = (
    "title",
    "description",
    "channel",
    "uploader",
    "upload_date",
    "tags",
    "categories",
)


def _metadata_text(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (list, tuple)):
        joined = ", ".join(str(item).strip() for item in value if str(item).strip())
        return joined or None
    return None


def _metadata_mapping(
    platform: dict[str, object] | None,
    fallback_title: str | None,
) -> dict[str, str]:
    meta: dict[str, str] = {}
    if platform is not None:
        for key in _PLAN_METADATA_KEYS:
            text = _metadata_text(platform.get(key))
            if text is not None:
                meta[key] = text
    if "title" not in meta and fallback_title:
        meta["title"] = fallback_title
    if "description" in meta:
        meta["description"] = meta["description"][:2000]
    return meta


def _search_configured(settings: Settings) -> bool:
    for provider in settings.SEARCH_PROVIDERS:
        if provider == "brave" and settings.BRAVE_API_KEY is not None:
            return True
        if provider == "exa" and settings.EXA_API_KEY is not None:
            return True
    return False


async def _plan_skip(ctx: StageContext, stage: Stage) -> str | None:
    """The plan's skip reason for ``stage``, or ``None`` when it should run."""
    plan = await _stage_output(ctx, Stage.PLAN)
    if plan is None:
        return None
    run = plan.get("run")
    if not isinstance(run, dict):
        return None
    if run.get(stage.value) is not False:
        return None
    reasons = plan.get("reasons")
    if isinstance(reasons, dict):
        reason = reasons.get(stage.value)
        if isinstance(reason, str) and reason.strip():
            return reason
    return "skipped by plan"


async def _ingest_platform(ctx: StageContext) -> dict[str, object] | None:
    ingest = await _stage_output(ctx, Stage.INGEST)
    if ingest is None:
        return None
    platform = ingest.get("platform")
    return cast("dict[str, object]", platform) if isinstance(platform, dict) else None


async def _job_title(ctx: StageContext) -> str | None:
    job = await ctx.repos.jobs.get(ctx.job_id)
    return job.title if job is not None else None


def _speech_share(diarize_output: dict[str, object] | None, duration: float) -> float | None:
    """Σ exclusive diarization turn durations / audio duration, or ``None`` without a run.

    The exclusive turns do not overlap, so their sum is a direct measure of how much of the
    file is speech (verdict 25: music 2 %, talk 87 %).
    """
    if diarize_output is None or duration <= 0.0:
        return None
    raw = diarize_output.get("exclusive_turns")
    if not isinstance(raw, list):
        return None
    spoken = 0.0
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        end = item.get("end")
        if isinstance(start, bool) or isinstance(end, bool):
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        spoken += max(0.0, float(end) - float(start))
    return min(1.0, spoken / duration)


async def stage_plan(ctx: StageContext) -> StageResult:
    """Build the deterministic stage plan right after diarization (docs/04 §2)."""
    transcript = _load_transcript(ctx)
    normalize = await _stage_output(ctx, Stage.NORMALIZE)
    if normalize is None:
        raise CaracalError(ErrorCode.INTERNAL, "normalize checkpoint is missing")
    stats = _as_dict(normalize.get("stats"), "normalize")
    duration_s = _as_number(stats.get("duration_s"), "normalize")
    speech_ratio = _as_number(stats.get("speech_ratio"), "normalize")

    languages: list[dict[str, object]] = []
    langid = await _stage_output(ctx, Stage.LANGID)
    if langid is not None:
        for raw in _as_list(langid.get("languages", []), "langid"):
            languages.append(_as_dict(raw, "langid"))

    meta = _metadata_mapping(await _ingest_platform(ctx), await _job_title(ctx))
    diarizer_available = ctx.diarizer is not None and ctx.diarizer.available()
    diarize_output = await _done_output(ctx, Stage.DIARIZE)
    speech_share = _speech_share(diarize_output, duration_s)
    inputs = PlanInputs(
        duration=duration_s,
        speech_ratio=speech_ratio,
        options=ctx.options,
        speech_share=speech_share,
        transcript=transcript,
        languages=languages,
        metadata=meta,
        diarizer_available=diarizer_available,
        search_configured=_search_configured(ctx.settings),
        default_target_langs=ctx.settings.CARACAL_DEFAULT_TARGET_LANGS,
        search_max_queries=ctx.settings.SEARCH_MAX_QUERIES_PER_JOB,
        report_language=ctx.settings.CARACAL_REPORT_LANGUAGE,
    )
    stage_plan_value = await build_plan(inputs, llm=ctx.llm)
    logger.info(
        "plan complete",
        extra={
            "job_id": str(ctx.job_id),
            "music_detected": stage_plan_value.music_detected,
            "speech_share": speech_share,
            "skipped": sorted(stage for stage, enabled in stage_plan_value.run.items() if not enabled),
            "genre": stage_plan_value.content_profile.likely_genre,
        },
    )
    return StageResult(status="done", output=stage_plan_value.model_dump(mode="json"))


def _content_hints(options: JobOptions) -> list[str]:
    hints: list[str] = []
    if options.context_hint is not None and options.context_hint.strip():
        hints.append(options.context_hint.strip())
    hints.extend(hint.strip() for hint in options.speaker_hints if hint.strip())
    return hints


async def stage_content(ctx: StageContext) -> StageResult:
    """Run the evidence-cited content analysis (docs/04 §6)."""
    reason = await _plan_skip(ctx, Stage.CONTENT)
    if reason is not None:
        return StageResult(status="skipped", reason=reason)
    if ctx.llm is None:
        return StageResult(status="skipped", reason="llm unavailable")
    transcript = _load_transcript(ctx)
    meta = _metadata_mapping(await _ingest_platform(ctx), await _job_title(ctx))
    plan = await _stage_output(ctx, Stage.PLAN)
    music = isinstance(plan, dict) and plan.get("music_detected") is True
    analysis = await analyse_content(
        transcript,
        meta,
        _content_hints(ctx.options),
        ctx.options,
        ctx.settings,
        llm=ctx.llm,
        music=music,
    )
    logger.info(
        "content complete",
        extra={
            "job_id": str(ctx.job_id),
            "chapters": len(analysis.chapters),
            "entities": len(analysis.entities),
            "claims": len(analysis.claims),
            "llm_calls": analysis.llm_meta.calls,
        },
    )
    return StageResult(status="done", output=analysis.model_dump(mode="json"))


def _speaker_stats_output(value: object) -> dict[str, dict[str, float | int]]:
    """The align checkpoint's per-label statistics, re-validated for the identity stage."""
    raw = _as_dict(value, "align")
    stats: dict[str, dict[str, float | int]] = {}
    for label, item in raw.items():
        if not isinstance(label, str) or not label:
            raise _malformed("align")
        entry = _as_dict(item, "align")
        stats[label] = {
            "talk_time_s": _as_number(entry.get("talk_time_s"), "align"),
            "share": _as_number(entry.get("share"), "align"),
            "first_seen_s": _as_number(entry.get("first_seen_s"), "align"),
        }
    return stats


def _voiceprint_matches(value: dict[str, object] | None) -> dict[str, list[VoiceprintMatch]]:
    """The voiceprint checkpoint's matches, parsed into the identity stage's value type."""
    if value is None:
        return {}
    raw = value.get("matches")
    if raw is None:
        return {}
    matches = _as_dict(raw, "voiceprint")
    parsed: dict[str, list[VoiceprintMatch]] = {}
    for label, items in matches.items():
        if not isinstance(label, str) or not label:
            raise _malformed("voiceprint")
        kept: list[VoiceprintMatch] = []
        for item in _as_list(items, "voiceprint"):
            entry = _as_dict(item, "voiceprint")
            match_class = entry.get("class")
            if match_class not in ("match", "candidate"):
                continue
            kept.append(
                VoiceprintMatch(
                    known_speaker_id=UUID(_as_str(entry.get("known_speaker_id"), "voiceprint")),
                    name=_as_str(entry.get("name"), "voiceprint"),
                    similarity=_as_number(entry.get("similarity"), "voiceprint"),
                    match_class=match_class,
                )
            )
        if kept:
            parsed[label] = kept
    return parsed


def _content_entities(value: dict[str, object] | None) -> list[Entity]:
    """The person (and other) entities the content stage found, for the metadata source."""
    if value is None:
        return []
    entities: list[Entity] = []
    for item in _as_list(value.get("entities", []), "content"):
        try:
            entities.append(Entity.model_validate(item))
        except ValidationError as exc:
            raise _malformed("content") from exc
    return entities


def _profile_genre(value: dict[str, object] | None) -> str | None:
    """The plan's ``content_profile.likely_genre`` when the checkpoint carries one."""
    if value is None:
        return None
    profile = value.get("content_profile")
    if not isinstance(profile, dict):
        return None
    genre = profile.get("likely_genre")
    if not isinstance(genre, str):
        return None
    stripped = genre.strip()
    return stripped or None


def _content_context(value: dict[str, object] | None) -> tuple[str | None, list[str]]:
    """The content stage's ``context.genre`` and ``context.tone``, for the performance rule."""
    if value is None:
        return None, []
    context = value.get("context")
    if not isinstance(context, dict):
        return None, []
    genre_value = context.get("genre")
    genre = genre_value.strip() if isinstance(genre_value, str) and genre_value.strip() else None
    tone_value = context.get("tone")
    tone: list[str] = []
    if isinstance(tone_value, list):
        tone = [item.strip() for item in tone_value if isinstance(item, str) and item.strip()]
    return genre, tone


async def stage_identity(ctx: StageContext) -> StageResult:
    """Resolve speaker names from mentions, metadata, hints and voiceprints (docs/04 §7).

    The plan gates the stage first; the aligned transcript supplies the labels and their talk
    statistics, the content checkpoint the person entities the metadata source needs, and the
    voiceprint checkpoint the strong/weak matches. The LLM only extracts mentions and writes
    role descriptions — every candidate, score and promotion is decided deterministically.
    """
    reason = await _plan_skip(ctx, Stage.IDENTITY)
    if reason is not None:
        return StageResult(status="skipped", reason=reason)
    if ctx.llm is None:
        return StageResult(status="skipped", reason="llm unavailable")
    align = await _done_output(ctx, Stage.ALIGN)
    if align is None:
        return StageResult(status="skipped", reason="unavailable")
    stats = _speaker_stats_output(align.get("speakers"))
    if not stats:
        return StageResult(status="skipped", reason="unavailable")

    transcript = _load_transcript(ctx)
    voiceprint = await _done_output(ctx, Stage.VOICEPRINT)
    content = await _done_output(ctx, Stage.CONTENT)
    plan = await _stage_output(ctx, Stage.PLAN)
    content_genre, content_tone = _content_context(content)
    metadata = _metadata_mapping(await _ingest_platform(ctx), await _job_title(ctx))
    result = await resolve_identities(
        transcript,
        metadata,
        ctx.options.speaker_hints,
        _content_entities(content),
        _voiceprint_matches(voiceprint),
        stats,
        ctx.settings,
        llm=ctx.llm,
        profile_genre=_profile_genre(plan),
        content_genre=content_genre,
        content_tone=content_tone,
    )
    logger.info(
        "identity complete",
        extra={
            "job_id": str(ctx.job_id),
            "speakers": len(result.speakers),
            "named": sum(1 for identity in result.speakers if identity.name is not None),
            "flags": len(result.flags),
        },
    )
    return StageResult(
        status="done",
        output={
            "speakers": [identity.model_dump(mode="json") for identity in result.speakers],
            "flags": [flag.model_dump(mode="json") for flag in result.flags],
        },
    )


async def stage_verify(ctx: StageContext) -> StageResult:
    """Cross-verification is gated by the plan; the verifier itself lands in a later card."""
    reason = await _plan_skip(ctx, Stage.VERIFY)
    if reason is not None:
        return StageResult(status="skipped", reason=reason)
    return StageResult(status="skipped", reason=_PENDING_STAGE)


async def stage_translate(ctx: StageContext) -> StageResult:
    """Translation is gated by the plan; the translator itself lands in a later card."""
    reason = await _plan_skip(ctx, Stage.TRANSLATE)
    if reason is not None:
        return StageResult(status="skipped", reason=reason)
    return StageResult(status="skipped", reason=_PENDING_STAGE)


def _turns(value: object, what: str) -> list[Turn]:
    turns: list[Turn] = []
    for raw in _as_list(value, what):
        item = _as_dict(raw, what)
        turns.append(
            Turn(
                start=_as_number(item.get("start"), what),
                end=_as_number(item.get("end"), what),
                label=_as_str(item.get("label"), what),
            )
        )
    return turns


async def _done_output(ctx: StageContext, stage: Stage) -> dict[str, object] | None:
    """The ``output`` object of a checkpoint whose status is ``done``, else ``None``."""
    checkpoint = await ctx.load(stage)
    if checkpoint is None or checkpoint.get("status") != "done":
        return None
    return _as_dict(checkpoint.get("output"), f"{stage.value} checkpoint")


async def stage_diarize(ctx: StageContext) -> StageResult:
    """Separate speakers with pyannote and checkpoint the turns and embeddings (docs/04 §5).

    Diarization runs before the plan so its speech share can drive the music rule, so it
    is not gated by a plan checkpoint; only the caller's option and model availability
    can skip it.
    """
    diarizer = ctx.diarizer
    if ctx.options.diarize == "off" or diarizer is None or not diarizer.available():
        return StageResult(status="skipped", reason="unavailable")

    result = await diarizer.run(ctx.files.audio_wav)
    await ctx.progress(100.0, f"diarized {result.num_speakers} speakers")
    logger.info(
        "diarize complete",
        extra={
            "job_id": str(ctx.job_id),
            "num_speakers": result.num_speakers,
            "turns": len(result.turns),
            "exclusive_turns": len(result.exclusive_turns),
        },
    )
    return StageResult(status="done", output=result.to_dict())


def _retimed_word_count(before: Transcript, after: Transcript) -> int:
    count = 0
    for old_segment, new_segment in zip(before.segments, after.segments, strict=False):
        for old_word, new_word in zip(old_segment.words, new_segment.words, strict=False):
            if (old_word.start, old_word.end) != (new_word.start, new_word.end):
                count += 1
    return count


def _render_subtitles(ctx: StageContext, transcript: Transcript) -> dict[str, object]:
    """Write the source SRT/VTT for a transcript; VTT carries ``<v S1>`` voice tags."""
    cues = cues_from_transcript(transcript)
    srt = ctx.files.subtitle("source", "srt")
    vtt = ctx.files.subtitle("source", "vtt")
    ctx.files.write_text(srt, to_srt(cues))
    ctx.files.write_text(vtt, to_vtt(cues))
    return {
        "cues": len(cues),
        "files": {
            "source.srt": _relative(ctx, srt),
            "source.vtt": _relative(ctx, vtt),
        },
    }


async def stage_align(ctx: StageContext) -> StageResult:
    """Re-time intro words, attach speakers and rewrite transcript, segments and subtitles."""
    plan_reason = await _plan_skip(ctx, Stage.ALIGN)
    if plan_reason is not None:
        return StageResult(status="skipped", reason=plan_reason)
    diarize_output = await _done_output(ctx, Stage.DIARIZE)
    if diarize_output is None:
        return StageResult(status="skipped", reason="unavailable")

    transcript = _load_transcript(ctx)
    turns = _turns(diarize_output.get("turns"), "diarize")
    exclusive_turns = _turns(diarize_output.get("exclusive_turns"), "diarize")

    retimed = retime_outside_speech(transcript, turns)
    retimed_words = _retimed_word_count(transcript, retimed)
    aligned = assign_speakers(retimed, exclusive_turns)

    _write_transcript(ctx, aligned)
    await ctx.repos.segments.replace_for_job(ctx.job_id, _segment_rows(ctx, aligned))
    _render_subtitles(ctx, aligned)
    stats = speaker_stats(aligned)
    logger.info(
        "align complete",
        extra={
            "job_id": str(ctx.job_id),
            "speakers": len(stats),
            "retimed_words": retimed_words,
        },
    )
    return StageResult(status="done", output={"speakers": stats, "retimed_words": retimed_words})


def _embedding_vector(value: object, what: str) -> list[float]:
    return [_as_number(item, what) for item in _as_list(value, what)]


async def stage_voiceprint(ctx: StageContext) -> StageResult:
    """Match every diarized label against stored voiceprints (docs/04 §5, docs/05 §1).

    Read-only by design: enrollment happens only through the explicit API (docs/06 §2), so a
    pipeline run can never attach a name to a voiceprint. A label whose best similarity is
    below ``VOICEPRINT_CANDIDATE_THRESHOLD`` is absent from the output.
    """
    plan_reason = await _plan_skip(ctx, Stage.VOICEPRINT)
    if plan_reason is not None:
        return StageResult(status="skipped", reason=plan_reason)
    diarize_output = await _done_output(ctx, Stage.DIARIZE)
    if diarize_output is None:
        return StageResult(status="skipped", reason="unavailable")

    embeddings = _as_dict(diarize_output.get("embeddings"), "diarize")
    matches: dict[str, list[dict[str, object]]] = {}
    for label in sorted(embeddings):
        vector = _embedding_vector(embeddings[label], "diarize")
        kept: list[dict[str, object]] = []
        for match in await ctx.repos.speakers.nearest(vector, k=3):
            match_class = classify(match.similarity, ctx.settings)
            if match_class is None:
                continue
            kept.append(
                {
                    "known_speaker_id": str(match.known_speaker_id),
                    "name": match.name,
                    "similarity": round(match.similarity, 6),
                    "class": match_class,
                }
            )
        if kept:
            matches[label] = kept

    logger.info(
        "voiceprint complete",
        extra={
            "job_id": str(ctx.job_id),
            "labels": len(embeddings),
            "matched_labels": len(matches),
            "matches": sum(len(items) for items in matches.values()),
        },
    )
    return StageResult(status="done", output={"matches": matches})


async def stage_subtitles(ctx: StageContext) -> StageResult:
    """Render the source-language SRT and WebVTT cues (docs/04 §10)."""
    transcript = _load_transcript(ctx)
    output = _render_subtitles(ctx, transcript)
    cues = cast("int", output["cues"])
    logger.info("subtitles complete", extra={"job_id": str(ctx.job_id), "cues": cues})
    return StageResult(status="done", output=output)


async def _stub_stage(ctx: StageContext) -> StageResult:
    return StageResult(status="skipped", reason="not implemented in M0")


async def _m1_stub_stage(ctx: StageContext) -> StageResult:
    return StageResult(status="skipped", reason=_M1_NOT_IMPLEMENTED)


_M1_STAGES: dict[Stage, StageFn] = {
    Stage.INGEST: stage_ingest,
    Stage.NORMALIZE: stage_normalize,
    Stage.CHUNK: stage_chunk,
    Stage.TRANSCRIBE: stage_transcribe,
    Stage.LANGID: stage_langid,
    Stage.SUBTITLES: stage_subtitles,
}


def m1_registry() -> dict[Stage, StageFn]:
    """The production M1 registry: real hear stages, clean skips for later milestones."""
    return {stage: _M1_STAGES.get(stage, _m1_stub_stage) for stage in Stage}


_M2_STAGES: dict[Stage, StageFn] = {
    **_M1_STAGES,
    Stage.PLAN: stage_plan,
    Stage.DIARIZE: stage_diarize,
    Stage.ALIGN: stage_align,
    Stage.VOICEPRINT: stage_voiceprint,
    Stage.CONTENT: stage_content,
    Stage.IDENTITY: stage_identity,
    Stage.VERIFY: stage_verify,
    Stage.TRANSLATE: stage_translate,
}


def m2_registry() -> dict[Stage, StageFn]:
    """The production M2 registry: M1 plus diarization, alignment and voiceprint matching."""
    return {stage: _M2_STAGES.get(stage, _m1_stub_stage) for stage in Stage}


def default_registry() -> dict[Stage, StageFn]:
    """The M0 registry used by runner tests: every stage is a clean no-op checkpoint."""
    return {stage: _stub_stage for stage in Stage}


__all__ = [
    "CRITICAL_STAGES",
    "JobOptions",
    "Repos",
    "Stage",
    "StageContext",
    "StageFn",
    "StageResult",
    "default_registry",
    "m1_registry",
    "m2_registry",
    "stage_align",
    "stage_chunk",
    "stage_content",
    "stage_diarize",
    "stage_identity",
    "stage_ingest",
    "stage_langid",
    "stage_normalize",
    "stage_plan",
    "stage_subtitles",
    "stage_transcribe",
    "stage_translate",
    "stage_verify",
    "stage_voiceprint",
]
