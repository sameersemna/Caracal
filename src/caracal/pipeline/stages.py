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

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

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
from caracal.log import get_logger
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
    """Separate speakers with pyannote and checkpoint the turns and embeddings (docs/04 §5)."""
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
    Stage.DIARIZE: stage_diarize,
    Stage.ALIGN: stage_align,
    Stage.VOICEPRINT: stage_voiceprint,
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
    "stage_diarize",
    "stage_ingest",
    "stage_langid",
    "stage_normalize",
    "stage_subtitles",
    "stage_transcribe",
    "stage_voiceprint",
]
