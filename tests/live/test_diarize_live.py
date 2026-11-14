"""Live pyannote diarization and the speaker stage chain (``@pytest.mark.live``).

Run by the judge on the GB10, never by the gate: it loads the real model, runs whisper and
ffmpeg, and needs ``CARACAL_LIVE_TWO_SPEAKER_WAV`` for the two-speaker cases. The
public-domain ``jfk.wav`` fixture must yield exactly one speaker.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest

from caracal.asr.whisper import WhisperClient
from caracal.config import Settings, get_settings
from caracal.pipeline.runner import RunnerDeps, run_pipeline
from caracal.pipeline.stages import JobOptions, Stage, m2_registry
from caracal.speakers.diarize import Diarizer, Turn
from caracal.storage import JobFiles, UploadFiles
from tests.fakes.repo import make_fake_repos
from tests.repo_contract import Repos

pytestmark = pytest.mark.live

_AUDIO_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
_JFK = _AUDIO_DIR / "jfk.wav"
_TWO_SPEAKER = os.environ.get("CARACAL_LIVE_TWO_SPEAKER_WAV")

needs_two_speaker = pytest.mark.skipif(
    not _TWO_SPEAKER,
    reason="CARACAL_LIVE_TWO_SPEAKER_WAV is not set",
)


def _two_speaker_path() -> Path:
    assert _TWO_SPEAKER is not None
    return Path(_TWO_SPEAKER)


def _word_midpoint(start: float, end: float) -> float:
    return (start + end) / 2.0


def _inside_or_near(midpoint: float, turns: list[Turn]) -> bool:
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return True
    return any(min(abs(midpoint - turn.start), abs(midpoint - turn.end)) <= 1.0 for turn in turns)


async def test_live_jfk_is_one_speaker() -> None:
    settings = get_settings()
    diarizer = Diarizer(settings)

    assert diarizer.available() is True
    result = await diarizer.run(_JFK)

    assert result.num_speakers == 1
    assert len(result.turns) >= 1
    assert set(result.embeddings) == {"S1"}
    assert all(len(vector) == 256 for vector in result.embeddings.values())


@needs_two_speaker
async def test_live_two_speaker_file_is_two_speakers() -> None:
    settings = get_settings()
    diarizer = Diarizer(settings)

    result = await diarizer.run(_two_speaker_path())

    assert result.num_speakers == 2
    assert set(result.embeddings) == {"S1", "S2"}
    assert all(len(vector) == 256 for vector in result.embeddings.values())


@needs_two_speaker
async def test_live_stage_chain_assigns_words_consistent_with_the_turns(tmp_path: Path) -> None:
    settings: Settings = get_settings().model_copy(update={"CARACAL_DATA_DIR": tmp_path / "data"})
    repos: Repos = make_fake_repos()
    upload_id = uuid.uuid4()
    upload_files = UploadFiles(settings.CARACAL_DATA_DIR, upload_id)
    upload_files.ensure()
    source = upload_files.file("two-speaker.wav")
    shutil.copy2(_two_speaker_path(), source)
    data = source.read_bytes()
    await repos.uploads.create(
        upload_id=upload_id,
        filename="two-speaker.wav",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        path=str(source),
    )
    job = await repos.jobs.create(
        source_kind="upload",
        upload_id=upload_id,
        options=JobOptions(diarize="on").model_dump(),
    )
    await repos.uploads.consume(upload_id, job.id)

    async def publish(_job_id: uuid.UUID, _event: dict[str, Any]) -> None:
        return None

    async def not_cancelled(_job_id: uuid.UUID) -> bool:
        return False

    async with httpx.AsyncClient() as http:
        deps = RunnerDeps(
            settings=settings,
            repos=repos,
            queue_events=publish,
            cancel_check=not_cancelled,
            whisper=WhisperClient(
                settings.WHISPER_URL,
                http,
                timeout_s=float(settings.WHISPER_TIMEOUT_S),
            ),
            diarizer=Diarizer(settings),
        )
        status = await run_pipeline(job.id, deps, m2_registry())

    assert status == "done"
    files = JobFiles(settings.CARACAL_DATA_DIR, job.id)
    stages = {row.name: row for row in await repos.jobs.list_stages(job.id)}
    assert stages[Stage.DIARIZE.value].status == "done"
    assert stages[Stage.ALIGN.value].status == "done"

    diarize_output = files.read_json(files.stage(Stage.DIARIZE))
    assert isinstance(diarize_output, dict)
    payload = diarize_output["output"]
    assert isinstance(payload, dict)
    exclusive = [
        Turn(float(item["start"]), float(item["end"]), str(item["label"]))
        for item in payload["exclusive_turns"]
    ]
    by_label: dict[str, list[Turn]] = {}
    for turn in exclusive:
        by_label.setdefault(turn.label, []).append(turn)
    assert set(by_label) == {"S1", "S2"}

    transcript = files.read_json(files.transcript_json)
    assert isinstance(transcript, dict)
    speakers: set[str] = set()
    for segment in transcript["segments"]:
        for word in segment["words"]:
            label = word["speaker"]
            if label is None:
                continue
            speakers.add(label)
            assert label in by_label
            assert _inside_or_near(_word_midpoint(float(word["start"]), float(word["end"])), by_label[label])

    assert speakers == {"S1", "S2"}
    vtt = files.subtitle("source", "vtt").read_text(encoding="utf-8")
    assert "<v S1>" in vtt
    assert "<v S2>" in vtt
