"""JobFiles paths, path validation and atomic artifact writes (docs/05 §3, docs/06 S10)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from caracal.errors import CaracalError, ErrorCode
from caracal.pipeline.stages import Stage
from caracal.storage import JobFiles


@pytest.fixture
def files(tmp_path: Path) -> JobFiles:
    return JobFiles(data_dir=tmp_path, job_id=uuid.UUID("11111111-2222-3333-4444-555555555555"))


def test_every_path_derives_from_the_job_uuid(files: JobFiles) -> None:
    root = files.root
    assert root == files.data_dir / "jobs" / "11111111-2222-3333-4444-555555555555"
    assert files.source_dir == root / "source"
    assert files.audio_wav == root / "audio.wav"
    assert files.chunks_dir == root / "chunks"
    assert files.chunk(0) == root / "chunks" / "000.wav"
    assert files.chunk(12) == root / "chunks" / "012.wav"
    assert files.captions("en") == root / "captions" / "en.vtt"
    assert files.captions("pt-BR") == root / "captions" / "pt-BR.vtt"
    assert files.stage(Stage.INGEST) == root / "stages" / "ingest.json"
    assert files.transcript_json == root / "transcript.json"
    assert files.transcript_txt == root / "transcript.txt"
    assert files.subtitle("source", "srt") == root / "subtitles" / "source.srt"
    assert files.subtitle("de", "vtt") == root / "subtitles" / "de.vtt"
    assert files.report_json == root / "report.json"
    assert files.report_md == root / "report.md"


@pytest.mark.parametrize("lang", ["en", "de", "pt-BR", "yue", "zh-Hant", "source"])
def test_valid_language_tags(files: JobFiles, lang: str) -> None:
    assert files.captions(lang).name == f"{lang}.vtt"
    assert files.subtitle(lang, "srt").name == f"{lang}.srt"


@pytest.mark.parametrize(
    "lang",
    ["", "EN", "english", "e", "english-US", "pt_BR", "../en", "en/../de", "en\n", " source"],
)
def test_invalid_language_tags_raise_invalid_input(files: JobFiles, lang: str) -> None:
    with pytest.raises(CaracalError) as excinfo:
        files.captions(lang)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


@pytest.mark.parametrize("ext", ["srt", "vtt"])
def test_valid_subtitle_extensions(files: JobFiles, ext: str) -> None:
    assert files.subtitle("en", ext).suffix == f".{ext}"


@pytest.mark.parametrize("ext", ["txt", "SRT", "vtt ", "", "../srt"])
def test_invalid_subtitle_extensions_raise_invalid_input(files: JobFiles, ext: str) -> None:
    with pytest.raises(CaracalError) as excinfo:
        files.subtitle("en", ext)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_stage_names_must_be_stage_values(files: JobFiles) -> None:
    assert files.stage("assemble").name == "assemble.json"
    with pytest.raises(CaracalError) as excinfo:
        files.stage("../etc/passwd")
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_negative_chunk_index_is_rejected(files: JobFiles) -> None:
    with pytest.raises(CaracalError) as excinfo:
        files.chunk(-1)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_ensure_creates_every_directory(files: JobFiles) -> None:
    files.ensure()
    for directory in (
        files.root,
        files.source_dir,
        files.chunks_dir,
        files.captions_dir,
        files.stages_dir,
        files.subtitles_dir,
    ):
        assert directory.is_dir()


def test_write_json_is_atomic_and_round_trips(files: JobFiles, monkeypatch: pytest.MonkeyPatch) -> None:
    fsynced: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(descriptor: int) -> None:
        fsynced.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    path = files.stage(Stage.INGEST)

    files.write_json(path, {"status": "done", "output": {"segments": 3}})

    assert files.read_json(path) == {"status": "done", "output": {"segments": 3}}
    assert fsynced, "write_json must fsync the file"
    assert list(path.parent.glob(".*")) == [], "temporary file must not survive a successful write"


def test_write_json_keeps_previous_content_when_serialisation_fails(files: JobFiles) -> None:
    path = files.report_json
    files.write_json(path, {"old": True})

    with pytest.raises(TypeError):
        files.write_json(path, {"bad": object()})

    assert path.read_text(encoding="utf-8") == json.dumps({"old": True}, ensure_ascii=False, indent=2)


def test_write_text_is_atomic_and_round_trips(files: JobFiles) -> None:
    path = files.transcript_txt
    files.write_text(path, "first")
    files.write_text(path, "second")
    assert path.read_text(encoding="utf-8") == "second"
    assert list(path.parent.glob(".*")) == []


def test_read_json_missing_file_returns_none(files: JobFiles) -> None:
    assert files.read_json(files.stage(Stage.ASSEMBLE)) is None
