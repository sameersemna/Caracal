"""M1 pipeline stages against fakes and a real local ffmpeg (docs/04 §3-§4, §10).

Media is generated with ffmpeg's ``sine`` lavfi source, so normalization, analysis, chunk
cutting and subtitle rendering all run for real. Whisper is the recorded-fixture fake, the
resolver is deterministic, and Redis is fakeredis: no test touches a network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.fakes.diarizer import FakeDiarizer
from tests.fakes.ollama import FakeOllama, json_response
from tests.fakes.redis import RecordingRedis
from tests.fakes.repo import make_fake_repos
from tests.fakes.resolver import FakeResolver
from tests.fakes.whisper import FakeWhisper
from tests.repo_contract import Repos

from caracal.asr import Segment, Transcript, WhisperResult, Word
from caracal.asr.whisper import WhisperClient
from caracal.config import Settings
from caracal.db.repo import JobRecord
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.direct import DirectInfo
from caracal.ingest.ytdlp import CaptionTrack, DownloadedAudio, PlatformMetadata
from caracal.llm.client import LLMClient
from caracal.pipeline.runner import RunnerDeps, run_pipeline
from caracal.pipeline.stages import (
    JobOptions,
    Stage,
    StageContext,
    StageResult,
    m1_registry,
    m2_registry,
    stage_align,
    stage_chunk,
    stage_content,
    stage_diarize,
    stage_identity,
    stage_ingest,
    stage_langid,
    stage_normalize,
    stage_plan,
    stage_subtitles,
    stage_transcribe,
    stage_translate,
    stage_verify,
    stage_voiceprint,
)
from caracal.queue import Keys, whisper_lock
from caracal.speakers.diarize import DiarizationResult, Diarizer, Turn
from caracal.storage import JobFiles, UploadFiles

MakeSettings = Callable[..., Settings]
PREFIX = "t3st:"


async def _make_wav(path: Path, seconds: float) -> None:
    """Generate a local tone with the real ffmpeg (lavfi, no network)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-y",
        str(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode("utf-8", "replace")


async def _create_job(
    repos: Repos,
    *,
    source_kind: str = "upload",
    source_url: str | None = None,
    upload_id: uuid.UUID | None = None,
) -> JobRecord:
    return await repos.jobs.create(
        source_kind=source_kind,
        source_url=source_url,
        upload_id=upload_id,
        options=JobOptions().model_dump(),
    )


def _context(
    settings: Settings,
    repos: Repos,
    files: JobFiles,
    *,
    whisper: WhisperClient | None = None,
    redis: RecordingRedis | None = None,
    keys: Keys | None = None,
    cancelled: Callable[[], Awaitable[bool]] | None = None,
    job_id: uuid.UUID | None = None,
    llm: LLMClient | None = None,
) -> StageContext:
    async def load(stage: Stage) -> dict[str, object] | None:
        value = files.read_json(files.stage(stage))
        return value if isinstance(value, dict) else None

    async def progress(_pct: float, _message: str) -> None:
        return None

    return StageContext(
        job_id=job_id if job_id is not None else uuid.uuid4(),
        options=JobOptions(),
        files=files,
        settings=settings,
        repos=repos,
        load=load,
        progress=progress,
        whisper=whisper,
        llm=llm,
        resolver=FakeResolver(),
        redis=redis,
        keys=keys,
        cancelled=cancelled if cancelled is not None else _never,
    )


async def _never() -> bool:
    return False


@dataclass
class _Services:
    """The process-wide fakes a stage context needs."""

    redis: RecordingRedis
    keys: Keys
    fake: FakeWhisper
    client: WhisperClient
    http: httpx.AsyncClient


def _services(settings: Settings, fixture: str = "jfk") -> _Services:
    redis = RecordingRedis()
    keys = Keys(PREFIX)
    fake = FakeWhisper.from_fixtures(fixture)
    http = httpx.AsyncClient(transport=fake.transport())
    client = WhisperClient(
        settings.WHISPER_URL,
        http,
        timeout_s=5.0,
        lock=lambda: whisper_lock(redis, keys),
    )
    return _Services(redis=redis, keys=keys, fake=fake, client=client, http=http)


def _write_output(files: JobFiles, stage: Stage, output: dict[str, object]) -> None:
    files.write_json(files.stage(stage), {"status": "done", "reason": None, "output": output})


# --- ingest ---------------------------------------------------------------------------


async def test_ingest_upload_links_the_file_and_sets_the_title(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    upload_id = uuid.uuid4()
    source = UploadFiles(tmp_path, upload_id)
    source.ensure()
    payload_file = source.file("tone.wav")
    payload_file.write_bytes(b"uploaded-audio")
    upload = await repos.uploads.create(
        upload_id=upload_id,
        filename="tone.wav",
        size_bytes=14,
        sha256="upload-sha",
        path=str(payload_file),
    )
    job = await _create_job(repos, upload_id=upload.id)
    files = JobFiles(tmp_path, job.id)
    files.ensure()

    result = await stage_ingest(_context(settings, repos, files, job_id=job.id))

    assert result.status == "done"
    assert result.output["source_kind"] == "upload"
    assert result.output["filename"] == "tone.wav"
    assert result.output["size_bytes"] == 14
    assert result.output["sha256"] == "upload-sha"
    assert result.output["platform"] is None
    assert result.output["captions"] == []
    assert (files.source_dir / "tone.wav").read_bytes() == b"uploaded-audio"
    stored = await repos.jobs.get(job.id)
    assert stored is not None
    assert stored.title == "tone.wav"


async def test_ingest_direct_downloads_through_the_pinned_client(
    make_settings: MakeSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos, source_kind="direct", source_url="https://example.com/a.mp3")
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    calls: list[Path] = []

    async def fake_download(
        url: str,
        dest: Path,
        *,
        resolver: object,
        max_bytes: int,
    ) -> DirectInfo:
        calls.append(dest)
        dest.write_bytes(b"downloaded")  # noqa: ASYNC240 - test-local file
        return DirectInfo(
            final_url="https://cdn.example.com/a.mp3",
            content_type="audio/mpeg",
            size_bytes=10,
            sha256="direct-sha",
            filename="a.mp3",
        )

    monkeypatch.setattr("caracal.pipeline.stages.download_direct", fake_download)

    result = await stage_ingest(_context(settings, repos, files, job_id=job.id))

    assert result.status == "done"
    assert result.output["source_kind"] == "direct"
    assert result.output["url"] == "https://example.com/a.mp3"
    assert result.output["final_url"] == "https://cdn.example.com/a.mp3"
    assert result.output["sha256"] == "direct-sha"
    assert calls == [files.source_dir / "download"]
    assert (files.source_dir / "download").read_bytes() == b"downloaded"
    stored = await repos.jobs.get(job.id)
    assert stored is not None
    assert stored.title == "a.mp3"


async def test_ingest_platform_stores_metadata_and_ignores_caption_failure(
    make_settings: MakeSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos, source_kind="platform", source_url="https://youtube.com/watch?v=abc")
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    meta = PlatformMetadata(
        id="abc",
        title="A talk",
        uploader="someone",
        channel="someone",
        channel_url="https://youtube.com/@someone",
        upload_date="20260101",
        duration=30.0,
        description=None,
        tags=["x"],
        categories=[],
        chapters=[],
        language="en",
        view_count=5,
        webpage_url="https://youtube.com/watch?v=abc",
        extractor="youtube",
    )

    async def fake_metadata(url: str, settings_arg: Settings, *, resolver: object) -> PlatformMetadata:
        return meta

    async def fake_download(
        url: str,
        files_arg: JobFiles,
        settings_arg: Settings,
        *,
        resolver: object,
        cancel: object,
    ) -> DownloadedAudio:
        path = files_arg.source_dir / "source.webm"
        path.write_bytes(b"platform-audio")
        return DownloadedAudio(path=path, ext="webm", size_bytes=14, sha256="platform-sha")

    async def fake_captions(*args: object, **kwargs: object) -> list[CaptionTrack]:
        raise CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "captions refused")

    monkeypatch.setattr("caracal.pipeline.stages.ytdlp.fetch_metadata", fake_metadata)
    monkeypatch.setattr("caracal.pipeline.stages.ytdlp.download_audio", fake_download)
    monkeypatch.setattr("caracal.pipeline.stages.ytdlp.fetch_platform_captions", fake_captions)

    result = await stage_ingest(_context(settings, repos, files, job_id=job.id))

    assert result.status == "done"
    assert result.output["source_kind"] == "platform"
    assert result.output["filename"] == "source.webm"
    assert result.output["sha256"] == "platform-sha"
    assert result.output["final_url"] == "https://youtube.com/watch?v=abc"
    assert result.output["captions"] == []
    platform = result.output["platform"]
    assert isinstance(platform, dict)
    assert platform["title"] == "A talk"
    assert platform["extractor"] == "youtube"
    stored = await repos.jobs.get(job.id)
    assert stored is not None
    assert stored.title == "A talk"


# --- normalize and chunk ---------------------------------------------------------------


async def test_normalize_and_chunk_run_real_ffmpeg_and_delete_the_source(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos, source_kind="direct", source_url="https://example.com/a.wav")
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    await _make_wav(files.source_dir / "a.wav", 3.0)
    _write_output(
        files,
        Stage.INGEST,
        {
            "source_kind": "direct",
            "url": "https://example.com/a.wav",
            "final_url": "https://example.com/a.wav",
            "filename": "a.wav",
            "size_bytes": 1,
            "sha256": "sha",
            "platform": None,
            "captions": [],
        },
    )
    ctx = _context(settings, repos, files, job_id=job.id)

    normalized = await stage_normalize(ctx)
    _write_output(files, Stage.NORMALIZE, normalized.output)
    chunked = await stage_chunk(ctx)

    assert normalized.status == "done"
    probe = normalized.output["probe"]
    assert isinstance(probe, dict)
    assert probe["audio_streams"] == 1
    stats = normalized.output["stats"]
    assert isinstance(stats, dict)
    assert 2.5 <= float(stats["duration_s"]) <= 3.5  # type: ignore[arg-type]
    assert stats["quality"] == "good"
    assert float(stats["speech_ratio"]) > 0.5  # type: ignore[arg-type]
    assert files.audio_wav.is_file()
    assert not files.source_dir.exists()

    assert chunked.status == "done"
    chunks = chunked.output["chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    only = chunks[0]
    assert isinstance(only, dict)
    assert only["path"] == "chunks/000.wav"
    assert files.chunk(0).is_file()
    assert 2.5 <= float(only["end_s"]) <= 3.5  # type: ignore[arg-type]


# --- transcribe ------------------------------------------------------------------------


async def _chunk_checkpoint(files: JobFiles, count: int) -> None:
    specs: list[dict[str, object]] = []
    for index in range(count):
        path = files.chunk(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFF")
        specs.append(
            {
                "index": index,
                "start_s": float(index),
                "end_s": float(index + 1),
                "path": f"chunks/{index:03d}.wav",
            }
        )
    _write_output(files, Stage.CHUNK, {"chunks": specs})


async def test_transcribe_writes_transcript_segments_and_removes_chunks(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    await _chunk_checkpoint(files, 1)
    services = _services(settings)
    try:
        result = await stage_transcribe(
            _context(
                settings,
                repos,
                files,
                whisper=services.client,
                redis=services.redis,
                keys=services.keys,
                job_id=job.id,
            )
        )

        assert result.status == "done"
        assert result.output["segments"] == 3
        assert result.output["language_code"] == "en"
        assert result.output["dropped"] == {"no_speech": 0, "repetition": 0, "empty": 0}
        chunk_langs = result.output["chunk_langs"]
        assert isinstance(chunk_langs, list)
        assert len(chunk_langs) == 1
        assert chunk_langs[0]["start_s"] == 0.0
        assert chunk_langs[0]["code"] == "en"
        assert float(chunk_langs[0]["prob"]) == pytest.approx(0.9352319240570068)
        assert len(services.fake.requests) == 1

        payload = json.loads(files.transcript_json.read_text(encoding="utf-8"))
        assert payload["language_code"] == "en"
        assert len(payload["segments"]) == 3
        text = files.transcript_txt.read_text(encoding="utf-8")
        assert text.startswith("[00:00] And so my fellow Americans,")
        assert text.count("\n") == 3

        hits = await repos.segments.search("country")
        assert len(hits) == 2
        assert {hit.seg_id for hit in hits} == {1, 2}
        assert not files.chunks_dir.exists()

        lock_key = services.keys.lock_whisper()
        acquisitions = [
            args for name, args in services.redis.commands if name == "SET" and args and args[0] == lock_key
        ]
        assert len(acquisitions) == 1
    finally:
        await services.http.aclose()


async def test_transcribe_stops_between_chunks_without_writing_a_transcript(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    await _chunk_checkpoint(files, 2)
    services = _services(settings)
    calls = 0

    async def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > 1

    try:
        result = await stage_transcribe(
            _context(
                settings,
                repos,
                files,
                whisper=services.client,
                redis=services.redis,
                keys=services.keys,
                cancelled=cancelled,
                job_id=job.id,
            )
        )

        assert result.status == "skipped"
        assert result.reason == "cancelled"
        assert not files.transcript_json.exists()
        assert not files.transcript_txt.exists()
        assert len(services.fake.requests) == 1
        assert files.chunk(0).is_file()
    finally:
        await services.http.aclose()


async def test_cancel_between_chunks_settles_the_job_cancelled(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    await _chunk_checkpoint(files, 2)
    services = _services(settings)
    checks = 0

    async def cancel_check(_job_id: uuid.UUID) -> bool:
        nonlocal checks
        checks += 1
        return checks > 2

    async def publish(_job_id: uuid.UUID, _event: dict[str, Any]) -> None:
        return None

    async def done_stage(ctx: StageContext) -> StageResult:
        return StageResult(status="done")

    deps = RunnerDeps(
        settings=settings,
        repos=repos,
        queue_events=publish,
        cancel_check=cancel_check,
        whisper=services.client,
        resolver=FakeResolver(),
        redis=services.redis,
        keys=services.keys,
    )
    try:
        status = await run_pipeline(
            job.id,
            deps,
            {Stage.TRANSCRIBE: stage_transcribe, Stage.LANGID: done_stage},
        )
    finally:
        await services.http.aclose()

    assert status == "cancelled"
    assert not files.transcript_json.exists()
    stored = await repos.jobs.get(job.id)
    assert stored is not None
    assert stored.status == "cancelled"


# --- langid and subtitles --------------------------------------------------------------


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, prob=0.9, space_before=True)


def _transcript(texts: list[str]) -> Transcript:
    segments: list[Segment] = []
    for index, text in enumerate(texts):
        start = float(index) * 2.0
        words = tuple(
            _word(token, start + position / 10, start + (position + 1) / 10)
            for position, token in enumerate(text.split())
        )
        segments.append(
            Segment(
                id=index,
                start=start,
                end=start + 0.9,
                text=text,
                words=words,
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                lang="en",
            )
        )
    return Transcript(segments=tuple(segments), language_code="en", dropped={})


async def test_langid_summarises_languages_and_caption_agreement(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    texts = ["the quick brown fox jumps over the lazy dog"] * 3
    transcript = _transcript(texts)
    files.write_json(files.transcript_json, transcript.to_dict())
    _write_output(
        files,
        Stage.CHUNK,
        {"chunks": [{"index": 0, "start_s": 0.0, "end_s": 30.0, "path": "chunks/000.wav"}]},
    )
    _write_output(
        files,
        Stage.TRANSCRIBE,
        {
            "segments": 3,
            "words": 27,
            "language_code": "en",
            "dropped": {},
            "chunk_langs": [{"start_s": 0.0, "code": "en", "prob": 0.93}],
        },
    )
    caption = files.captions("en")
    caption.parent.mkdir(parents=True, exist_ok=True)
    caption.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\n"
        + " ".join(" ".join(text.split()) for text in texts)
        + "\n",
        encoding="utf-8",
    )
    _write_output(
        files,
        Stage.INGEST,
        {"captions": [{"lang": "en", "kind": "manual", "path": "captions/en.vtt"}]},
    )
    services = _services(settings)
    try:
        result = await stage_langid(
            _context(
                settings,
                repos,
                files,
                whisper=services.client,
                redis=services.redis,
                keys=services.keys,
                job_id=job.id,
            )
        )
    finally:
        await services.http.aclose()

    assert result.status == "done"
    languages = result.output["languages"]
    assert isinstance(languages, list)
    assert len(languages) == 1
    assert languages[0]["code"] == "en"
    assert languages[0]["name"] == "English"
    assert result.output["caption_lang"] == "en"
    agreement = result.output["caption_agreement"]
    assert isinstance(agreement, float)
    assert agreement > 0.9
    assert services.fake.requests == []


async def test_langid_rewrites_the_transcript_when_a_recheck_relabels(
    make_settings: MakeSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    transcript = _transcript(["hello there friend"])
    files.write_json(files.transcript_json, transcript.to_dict())
    _write_output(
        files,
        Stage.CHUNK,
        {"chunks": [{"index": 0, "start_s": 0.0, "end_s": 120.0, "path": "chunks/000.wav"}]},
    )
    _write_output(
        files,
        Stage.TRANSCRIBE,
        {
            "segments": 1,
            "words": 3,
            "language_code": "en",
            "dropped": {},
            "chunk_langs": [{"start_s": 0.0, "code": "en", "prob": 0.3}],
        },
    )
    relabelled = _transcript(["hallo zusammen freund"])

    async def fake_refine(
        original: Transcript,
        chunk_results: list[tuple[float, WhisperResult]],
        *,
        recheck: Callable[[float, float], Awaitable[WhisperResult]],
        max_calls: int = 6,
    ) -> Transcript:
        assert chunk_results[0][1].duration == 120.0
        assert chunk_results[0][1].language_prob == 0.3
        return relabelled

    monkeypatch.setattr("caracal.pipeline.stages.refine_spans", fake_refine)
    services = _services(settings)
    try:
        result = await stage_langid(
            _context(
                settings,
                repos,
                files,
                whisper=services.client,
                redis=services.redis,
                keys=services.keys,
                job_id=job.id,
            )
        )
    finally:
        await services.http.aclose()

    assert result.status == "done"
    rewritten = json.loads(files.transcript_json.read_text(encoding="utf-8"))
    assert rewritten["language_code"] == "en"
    assert rewritten["segments"][0]["text"] == "hallo zusammen freund"


async def test_subtitles_writes_srt_and_vtt(make_settings: MakeSettings, tmp_path: Path) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    files.write_json(files.transcript_json, _transcript(["hello world", "goodbye world"]).to_dict())

    result = await stage_subtitles(_context(settings, repos, files, job_id=job.id))

    assert result.status == "done"
    assert result.output["cues"] == 2
    assert result.output["files"] == {
        "source.srt": "subtitles/source.srt",
        "source.vtt": "subtitles/source.vtt",
    }
    srt = files.subtitle("source", "srt").read_text(encoding="utf-8")
    vtt = files.subtitle("source", "vtt").read_text(encoding="utf-8")
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert "hello world" in srt
    assert vtt.startswith("WEBVTT\n\n")
    assert "goodbye world" in vtt


# --- whole M1 pipeline -----------------------------------------------------------------


async def test_m1_pipeline_runs_end_to_end_on_an_uploaded_tone(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    tone = tmp_path / "tone.wav"
    await _make_wav(tone, 30.0)
    data = tone.read_bytes()

    upload_id = uuid.uuid4()
    upload_dir = UploadFiles(tmp_path, upload_id)
    upload_dir.ensure()
    uploaded = upload_dir.file("tone.wav")
    shutil.copy2(tone, uploaded)
    await repos.uploads.create(
        upload_id=upload_id,
        filename="tone.wav",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        path=str(uploaded),
    )
    job = await _create_job(repos, upload_id=upload_id)
    await repos.uploads.consume(upload_id, job.id)
    services = _services(settings)
    events: list[dict[str, Any]] = []

    async def publish(_job_id: uuid.UUID, event: dict[str, Any]) -> None:
        events.append(dict(event))

    async def not_cancelled(_job_id: uuid.UUID) -> bool:
        return False

    deps = RunnerDeps(
        settings=settings,
        repos=repos,
        queue_events=publish,
        cancel_check=not_cancelled,
        whisper=services.client,
        resolver=FakeResolver(),
        redis=services.redis,
        keys=services.keys,
    )
    try:
        status = await run_pipeline(job.id, deps, m1_registry())
    finally:
        await services.http.aclose()

    assert status == "done"
    stored = await repos.jobs.get(job.id)
    assert stored is not None
    assert stored.status == "done"
    assert stored.title == "tone.wav"

    files = JobFiles(tmp_path, job.id)
    assert files.transcript_json.is_file()
    assert files.transcript_txt.is_file()
    assert files.subtitle("source", "srt").is_file()
    assert files.subtitle("source", "vtt").is_file()
    assert not files.chunks_dir.exists()
    assert not files.source_dir.exists()

    stages = {row.name: row for row in await repos.jobs.list_stages(job.id)}
    for name in ("ingest", "normalize", "chunk", "transcribe", "langid", "subtitles"):
        assert stages[name].status == "done"
    assert stages["diarize"].status == "skipped"
    assert stages["diarize"].reason == "not implemented in M1"

    payload = json.loads(files.transcript_json.read_text(encoding="utf-8"))
    assert payload["language_code"] == "en"
    assert len(payload["segments"]) == 3
    assert files.transcript_txt.read_text(encoding="utf-8").startswith("[00:00] And so")
    assert files.subtitle("source", "vtt").read_text(encoding="utf-8").startswith("WEBVTT")

    langid = json.loads(files.stage(Stage.LANGID).read_text(encoding="utf-8"))
    assert langid["output"]["languages"][0]["code"] == "en"

    hits = await repos.segments.search("country")
    assert len(hits) == 2
    assert len(services.fake.requests) == 1
    lock_key = services.keys.lock_whisper()
    acquisitions = [
        args for name, args in services.redis.commands if name == "SET" and args and args[0] == lock_key
    ]
    assert len(acquisitions) == 1
    assert [event["type"] for event in events if event["type"] == "end"] == ["end"]
    assert events[-1]["status"] == "done"


# --- diarize and align (M2) ------------------------------------------------------------


def _diarization_result() -> DiarizationResult:
    return DiarizationResult(
        turns=[Turn(0.0, 2.0, "S1"), Turn(2.0, 4.0, "S2")],
        exclusive_turns=[Turn(0.0, 2.0, "S1"), Turn(2.0, 4.0, "S2")],
        embeddings={"S1": [1.0, 0.0], "S2": [0.0, 1.0]},
        num_speakers=2,
    )


def _word_speakers(transcript: Transcript) -> list[str | None]:
    return [word.speaker for segment in transcript.segments for word in segment.words]


async def test_diarize_skips_when_the_diarizer_is_unavailable(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    fake = FakeDiarizer(result=_diarization_result(), available_flag=False)
    ctx = replace(_context(settings, repos, files, job_id=job.id), diarizer=fake)

    result = await stage_diarize(ctx)

    assert result.status == "skipped"
    assert result.reason == "unavailable"
    assert fake.calls == []


async def test_diarize_skips_without_a_hf_token(make_settings: MakeSettings, tmp_path: Path) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    diarizer = Diarizer(settings)
    assert diarizer.available() is False
    ctx = replace(_context(settings, repos, files, job_id=job.id), diarizer=diarizer)

    result = await stage_diarize(ctx)

    assert result.status == "skipped"
    assert result.reason == "unavailable"


async def test_diarize_skips_when_the_option_is_off(make_settings: MakeSettings, tmp_path: Path) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    fake = FakeDiarizer(result=_diarization_result())
    context = _context(settings, repos, files, job_id=job.id)
    ctx = replace(context, diarizer=fake, options=JobOptions(diarize="off"))

    result = await stage_diarize(ctx)

    assert result.status == "skipped"
    assert result.reason == "unavailable"
    assert fake.calls == []


async def test_diarize_checkpoints_turns_exclusive_turns_and_embeddings(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    fake = FakeDiarizer(result=_diarization_result())
    ctx = replace(_context(settings, repos, files, job_id=job.id), diarizer=fake)

    result = await stage_diarize(ctx)

    assert result.status == "done"
    assert result.output == {
        "num_speakers": 2,
        "turns": [
            {"start": 0.0, "end": 2.0, "label": "S1"},
            {"start": 2.0, "end": 4.0, "label": "S2"},
        ],
        "exclusive_turns": [
            {"start": 0.0, "end": 2.0, "label": "S1"},
            {"start": 2.0, "end": 4.0, "label": "S2"},
        ],
        "embeddings": {"S1": [1.0, 0.0], "S2": [0.0, 1.0]},
    }
    assert fake.calls == [(files.audio_wav, None)]


def _speaker_transcript() -> Transcript:
    def word(text: str, start: float, end: float, *, space_before: bool = True) -> Word:
        return Word(text=text, start=start, end=end, prob=0.9, space_before=space_before)

    intro = [
        word("and", 0.18, 0.48, space_before=False),
        word("so", 0.6, 0.9),
        word("my", 1.0, 1.3),
        word("fellow", 3.8, 4.2),
    ]
    first = [word("americans", 13.4, 13.9, space_before=False), word("today", 14.0, 14.5)]
    second = [word("friends", 15.4, 15.9, space_before=False), word("welcome", 16.0, 16.5)]
    segments = tuple(
        Segment(
            id=index,
            start=words[0].start,
            end=words[-1].end,
            text=" ".join(item.text for item in words),
            words=tuple(words),
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            lang="en",
        )
        for index, words in enumerate((intro, first, second))
    )
    return Transcript(segments=segments, language_code="en", dropped={})


def _diarize_checkpoint(files: JobFiles, status: str = "done") -> None:
    files.write_json(
        files.stage(Stage.DIARIZE),
        {
            "status": status,
            "reason": None if status == "done" else "unavailable",
            "output": {
                "num_speakers": 2,
                "turns": [
                    {"start": 13.2, "end": 15.0, "label": "S1"},
                    {"start": 15.2, "end": 20.0, "label": "S2"},
                ],
                "exclusive_turns": [
                    {"start": 13.2, "end": 15.0, "label": "S1"},
                    {"start": 15.2, "end": 20.0, "label": "S2"},
                ],
                "embeddings": {"S1": [1.0, 0.0], "S2": [0.0, 1.0]},
            },
        },
    )


async def test_align_retimes_intro_words_assigns_speakers_and_rewrites_artifacts(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    files.write_json(files.transcript_json, _speaker_transcript().to_dict())
    _diarize_checkpoint(files)

    result = await stage_align(_context(settings, repos, files, job_id=job.id))

    assert result.status == "done"
    assert result.output["retimed_words"] == 4
    stats = result.output["speakers"]
    assert isinstance(stats, dict)
    assert set(stats) == {"S1", "S2"}
    assert sum(float(stats[label]["share"]) for label in stats) == pytest.approx(1.0)

    payload = json.loads(files.transcript_json.read_text(encoding="utf-8"))
    speakers = [segment["speaker"] for segment in payload["segments"]]
    assert speakers == ["S1", "S1", "S2"]
    first_segment = payload["segments"][0]
    assert first_segment["start"] == pytest.approx(13.2)
    assert first_segment["end"] == pytest.approx(13.75)
    assert [word["speaker"] for word in first_segment["words"]] == ["S1"] * 4
    text = files.transcript_txt.read_text(encoding="utf-8")
    assert text.startswith("[00:13] S1: and so my fellow")

    vtt = files.subtitle("source", "vtt").read_text(encoding="utf-8")
    assert "<v S1>" in vtt
    assert "<v S2>" in vtt
    for block in vtt.split("\n\n")[1:]:
        _timing, body = block.split("\n", 1)
        assert not ("<v S1>" in body and "<v S2>" in body)

    assert len(await repos.segments.search("friends")) == 1
    assert len(await repos.segments.search("fellow")) == 1


async def test_align_skips_and_keeps_the_transcript_when_diarize_was_skipped(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    transcript = _speaker_transcript()
    files.write_json(files.transcript_json, transcript.to_dict())
    before = files.transcript_json.read_bytes()
    _diarize_checkpoint(files, status="skipped")

    result = await stage_align(_context(settings, repos, files, job_id=job.id))

    assert result.status == "skipped"
    assert result.reason == "unavailable"
    assert files.transcript_json.read_bytes() == before
    assert _word_speakers(transcript) == [None] * 8


def test_m2_registry_wires_diarize_and_align_and_keeps_m1_stages() -> None:
    registry = m2_registry()

    assert registry[Stage.DIARIZE] is stage_diarize
    assert registry[Stage.ALIGN] is stage_align
    assert registry[Stage.TRANSCRIBE] is stage_transcribe
    assert registry[Stage.SUBTITLES] is stage_subtitles


# --- T2.5 planner and content ----------------------------------------------------------


def test_stage_order_puts_plan_right_after_diarize() -> None:
    assert [stage.value for stage in Stage] == [
        "ingest",
        "normalize",
        "chunk",
        "transcribe",
        "langid",
        "diarize",
        "plan",
        "align",
        "voiceprint",
        "content",
        "identity",
        "verify",
        "translate",
        "subtitles",
        "review",
        "assemble",
    ]


@pytest.mark.parametrize(
    ("stage", "stage_fn"),
    [
        (Stage.ALIGN, stage_align),
        (Stage.VOICEPRINT, stage_voiceprint),
        (Stage.CONTENT, stage_content),
        (Stage.IDENTITY, stage_identity),
        (Stage.VERIFY, stage_verify),
        (Stage.TRANSLATE, stage_translate),
    ],
)
async def test_stages_honour_a_plan_skip(
    stage: Stage,
    stage_fn: Callable[[StageContext], Awaitable[StageResult]],
    make_settings: MakeSettings,
    tmp_path: Path,
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    _write_output(
        files,
        Stage.PLAN,
        {
            "run": {stage.value: False},
            "reasons": {stage.value: "plan says no"},
            "content_profile": {
                "likely_genre": "talk",
                "expected_speakers": None,
                "needs_identity": True,
                "checkworthy_domain": False,
            },
            "budgets": {"search_queries": 0, "llm_calls": 0, "claims": 0},
            "music_detected": False,
            "languages": None,
        },
    )

    result = await stage_fn(_context(settings, repos, files, job_id=job.id))

    assert result.status == "skipped"
    assert result.reason == "plan says no"


async def test_stage_plan_detects_music_and_overrides_the_language(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    transcript = _transcript(["🎵🎵🎵"] * 4)
    files.write_json(files.transcript_json, transcript.to_dict())
    _write_output(
        files,
        Stage.NORMALIZE,
        {
            "stats": {
                "duration_s": 8.0,
                "speech_ratio": 1.0,
                "loudness_lufs": -20.0,
                "quality": "good",
                "silences": [],
            }
        },
    )
    _write_output(
        files,
        Stage.LANGID,
        {
            "languages": [
                {
                    "code": "nn",
                    "name": "Nynorsk",
                    "share": 1.0,
                    "spans": [{"start": 0.0, "end": 8.0}],
                    "confidence": 0.4,
                }
            ],
            "caption_agreement": None,
            "caption_lang": None,
        },
    )

    ctx = replace(
        _context(settings, repos, files, job_id=job.id),
        diarizer=FakeDiarizer(result=_diarization_result()),
    )

    result = await stage_plan(ctx)

    assert result.status == "done"
    assert result.output["music_detected"] is True
    assert result.output["run"]["diarize"] is True
    assert result.output["reasons"].get("diarize") != "music detected"
    assert result.output["reasons"]["align"] == "music detected"
    languages = result.output["languages"]
    assert isinstance(languages, list)
    assert languages[0]["code"] == "und"
    assert languages[0]["spans"] == [{"start": 0.0, "end": 8.0}]


@pytest.mark.parametrize(
    ("speech_seconds", "expected_music"),
    [(0.16, True), (7.0, False)],
)
async def test_stage_plan_uses_the_diarization_speech_share(
    make_settings: MakeSettings,
    tmp_path: Path,
    speech_seconds: float,
    expected_music: bool,
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    files.write_json(files.transcript_json, _transcript(["hello there"] * 4).to_dict())
    _write_output(
        files,
        Stage.NORMALIZE,
        {
            "stats": {
                "duration_s": 8.0,
                "speech_ratio": 1.0,
                "loudness_lufs": -20.0,
                "quality": "good",
                "silences": [],
            }
        },
    )
    _write_output(
        files,
        Stage.LANGID,
        {
            "languages": [
                {
                    "code": "en",
                    "name": "English",
                    "share": 1.0,
                    "spans": [{"start": 0.0, "end": 8.0}],
                    "confidence": 0.9,
                }
            ],
            "caption_agreement": None,
            "caption_lang": None,
        },
    )
    _write_output(
        files,
        Stage.DIARIZE,
        {
            "num_speakers": 1,
            "turns": [{"start": 0.0, "end": speech_seconds, "label": "S1"}],
            "exclusive_turns": [{"start": 0.0, "end": speech_seconds, "label": "S1"}],
            "embeddings": {},
        },
    )

    result = await stage_plan(_context(settings, repos, files, job_id=job.id))

    assert result.status == "done"
    assert result.output["music_detected"] is expected_music
    if expected_music:
        assert result.output["languages"][0]["code"] == "und"
    else:
        assert result.output["languages"] is None


async def test_stage_content_writes_a_content_checkpoint_with_llm_meta(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    transcript = _transcript(["alpha beta gamma", "delta epsilon zeta"])
    files.write_json(files.transcript_json, transcript.to_dict())
    _write_output(
        files,
        Stage.PLAN,
        {
            "run": {"content": True},
            "reasons": {"content": "content analysis"},
            "content_profile": {
                "likely_genre": "talk",
                "expected_speakers": 1,
                "needs_identity": False,
                "checkworthy_domain": False,
            },
            "budgets": {"search_queries": 0, "llm_calls": 1, "claims": 8},
            "music_detected": False,
            "languages": None,
        },
    )
    arguments = {
        "summary": {
            "one_line": "Two short lines.",
            "short": "Two short lines about letters.",
            "detailed": "Two short lines about Greek letters.",
            "evidence": ["seg:0"],
        },
        "context": {
            "genre": "talk",
            "tone": ["calm"],
            "evidence": ["seg:0"],
            "confidence": 0.7,
        },
    }
    fake = FakeOllama([json_response(arguments)])
    http = httpx.AsyncClient(transport=fake.transport())
    client = LLMClient(settings, http)
    try:
        result = await stage_content(_context(settings, repos, files, job_id=job.id, llm=client))
    finally:
        await http.aclose()

    assert result.status == "done"
    assert fake.body()["format"] == "json"
    assert "tools" not in fake.body()
    summary = result.output["summary"]
    assert isinstance(summary, dict)
    assert summary["one_line"] == "Two short lines."
    llm_meta = result.output["llm_meta"]
    assert isinstance(llm_meta, dict)
    assert llm_meta["calls"] == 1
    assert result.output["claims"] == []


async def test_stage_content_music_mode_returns_no_quotes_or_claims(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    transcript = _transcript(["Utak lyder", "Norske musik"])
    files.write_json(files.transcript_json, transcript.to_dict())
    _write_output(
        files,
        Stage.PLAN,
        {
            "run": {"content": True},
            "reasons": {"content": "music: describe the track, not speech"},
            "content_profile": {
                "likely_genre": "unknown",
                "expected_speakers": None,
                "needs_identity": False,
                "checkworthy_domain": False,
            },
            "budgets": {"search_queries": 0, "llm_calls": 1, "claims": 8},
            "music_detected": True,
            "languages": None,
        },
    )
    arguments = {
        "summary": {
            "one_line": "An instrumental track.",
            "short": "An instrumental track.",
            "detailed": "An instrumental track.",
            "evidence": ["meta:title"],
        },
        "context": {"genre": "music", "tone": ["instrumental"], "evidence": ["meta:title"]},
    }
    fake = FakeOllama([json_response(arguments)])
    http = httpx.AsyncClient(transport=fake.transport())
    client = LLMClient(settings, http)
    try:
        result = await stage_content(_context(settings, repos, files, job_id=job.id, llm=client))
    finally:
        await http.aclose()

    assert result.status == "done"
    assert result.output["key_quotes"] == []
    assert result.output["claims"] == []
    assert result.output["review_flags"] == [
        {"severity": "warn", "field": "content", "message": "non-speech audio"}
    ]
    prompt = fake.body()["messages"][1]["content"]
    assert "[seg:0]" not in prompt


# --- T2.6 identity --------------------------------------------------------------------


def _identity_transcript() -> Transcript:
    rows: list[tuple[str, str, float, float]] = [
        ("S1", "Hallo, ich bin Angela Merkel und willkommen.", 0.0, 3.0),
        ("S2", "Vielen Dank fuer die Einladung.", 3.5, 6.0),
    ]
    segments = tuple(
        Segment(
            id=index,
            start=start,
            end=end,
            text=text,
            words=(),
            avg_logprob=-0.1,
            no_speech_prob=0.0,
            lang="de",
            speaker=speaker,
        )
        for index, (speaker, text, start, end) in enumerate(rows)
    )
    return Transcript(segments=segments, language_code="de", dropped={})


def _identity_arguments() -> dict[str, Any]:
    return {
        "mentions": [
            {
                "speaker_label": "S1",
                "name": "Angela Merkel",
                "relation": "self_intro",
                "seg_ids": ["seg:0"],
            }
        ],
        "speakers": [
            {"speaker_label": "S1", "role": "narrator", "description": "narrates the recording"},
            {"speaker_label": "S2", "role": "guest", "description": "appears as a guest"},
        ],
    }


def _identity_checkpoints(files: JobFiles) -> None:
    files.write_json(files.transcript_json, _identity_transcript().to_dict())
    _write_output(
        files,
        Stage.ALIGN,
        {
            "speakers": {
                "S1": {"talk_time_s": 60.0, "share": 0.6, "first_seen_s": 0.0, "words": 60},
                "S2": {"talk_time_s": 40.0, "share": 0.4, "first_seen_s": 3.5, "words": 40},
            },
            "retimed_words": 0,
        },
    )
    _write_output(files, Stage.CONTENT, {"entities": []})
    _write_output(
        files,
        Stage.VOICEPRINT,
        {
            "matches": {
                "S2": [
                    {
                        "known_speaker_id": "6f1c1d1e-2a3b-4c5d-8e9f-0a1b2c3d4e5f",
                        "name": "Jane Doe",
                        "similarity": 0.962,
                        "class": "match",
                    }
                ]
            }
        },
    )


async def test_identity_stage_resolves_mentions_and_voiceprints_from_checkpoints(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    await repos.jobs.set_title(job.id, "Ein Gespraech")
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    _identity_checkpoints(files)

    fake = FakeOllama([json_response(_identity_arguments())])
    http = httpx.AsyncClient(transport=fake.transport())
    client = LLMClient(settings, http)
    try:
        result = await stage_identity(_context(settings, repos, files, job_id=job.id, llm=client))
    finally:
        await http.aclose()

    assert result.status == "done"
    assert fake.body()["format"] == "json"
    speakers = result.output["speakers"]
    assert isinstance(speakers, list)
    by_label = {speaker["label"]: speaker for speaker in speakers}
    assert by_label["S1"]["name"] == "Angela Merkel"
    assert by_label["S1"]["confidence"] == pytest.approx(0.85)
    assert by_label["S1"]["talk_time_s"] == pytest.approx(60.0)
    assert by_label["S1"]["share"] == pytest.approx(0.6)
    assert by_label["S1"]["role"] == "narrator"
    assert by_label["S1"]["candidates"][0]["evidence"] == ["seg:0"]
    assert by_label["S1"]["evidence"] == ["seg:0"]
    assert by_label["S2"]["name"] == "Jane Doe"
    assert by_label["S2"]["confidence"] == pytest.approx(0.962)
    assert by_label["S2"]["known_speaker_id"] == "6f1c1d1e-2a3b-4c5d-8e9f-0a1b2c3d4e5f"
    assert by_label["S2"]["voiceprint_similarity"] == pytest.approx(0.962)
    assert result.output["flags"] == []


async def test_identity_stage_is_skipped_by_the_plan(make_settings: MakeSettings, tmp_path: Path) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    _write_output(
        files,
        Stage.PLAN,
        {
            "run": {"identity": False},
            "reasons": {"identity": "music detected"},
            "content_profile": {
                "likely_genre": "music",
                "expected_speakers": None,
                "needs_identity": False,
                "checkworthy_domain": False,
            },
            "budgets": {"search_queries": 0, "llm_calls": 0, "claims": 0},
            "music_detected": True,
            "languages": None,
        },
    )

    result = await stage_identity(_context(settings, repos, files, job_id=job.id))

    assert result.status == "skipped"
    assert result.reason == "music detected"


async def test_identity_stage_skips_without_an_llm(make_settings: MakeSettings, tmp_path: Path) -> None:
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _create_job(repos)
    files = JobFiles(tmp_path, job.id)
    files.ensure()
    _identity_checkpoints(files)

    result = await stage_identity(_context(settings, repos, files, job_id=job.id))

    assert result.status == "skipped"
    assert result.reason == "llm unavailable"
