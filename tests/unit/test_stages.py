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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
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
from caracal.pipeline.runner import RunnerDeps, run_pipeline
from caracal.pipeline.stages import (
    JobOptions,
    Stage,
    StageContext,
    StageResult,
    m1_registry,
    stage_chunk,
    stage_ingest,
    stage_langid,
    stage_normalize,
    stage_subtitles,
    stage_transcribe,
)
from caracal.queue import Keys, whisper_lock
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
