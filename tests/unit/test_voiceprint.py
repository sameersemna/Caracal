"""Voiceprint classification, the read-only pipeline stage and explicit enrollment.

The stage and enrollment run against the in-memory repositories and a temporary artifact
directory: no model, network, Postgres or Redis is involved. Calibration (0.72 / 0.55) is
the measured one (docs/02, docs/04 §5), so the similarities below are the real gap:
same speaker 0.94-0.99, different 0.29-0.31.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from tests.fakes.repo import make_fake_repos
from tests.repo_contract import Repos

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.pipeline.stages import JobOptions, Stage, StageContext, m2_registry, stage_voiceprint
from caracal.speakers.voiceprint import EnrollmentDeps, classify, enroll
from caracal.storage import JobFiles

MakeSettings = Callable[..., Settings]
DATA_DIR = "data"


def _unit(similarity: float) -> list[float]:
    """A 256-d unit vector whose cosine similarity to ``[1, 0, ...]`` is ``similarity``."""
    rest = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return [similarity, rest, *([0.0] * 254)]


async def _job(repos: Repos, status: str = "done") -> uuid.UUID:
    job = await repos.jobs.create(
        source_kind="platform",
        source_url="https://example.com/talk",
        options=JobOptions().model_dump(),
    )
    if status != "queued":
        await repos.jobs.set_status(job.id, status)
    return job.id


def _files(tmp_path: Path, job_id: uuid.UUID) -> JobFiles:
    files = JobFiles(tmp_path / DATA_DIR, job_id)
    files.ensure()
    return files


def _write_diarize(
    files: JobFiles,
    embeddings: dict[str, list[float]],
    *,
    status: str = "done",
) -> None:
    files.write_json(
        files.stage(Stage.DIARIZE),
        {
            "status": status,
            "reason": None if status == "done" else "unavailable",
            "output": {
                "num_speakers": len(embeddings),
                "turns": [],
                "exclusive_turns": [],
                "embeddings": embeddings,
            },
        },
    )


def _context(settings: Settings, repos: Repos, files: JobFiles, job_id: uuid.UUID) -> StageContext:
    async def load(stage: Stage) -> dict[str, object] | None:
        value = files.read_json(files.stage(stage))
        return value if isinstance(value, dict) else None

    async def progress(_pct: float, _message: str) -> None:
        return None

    return StageContext(
        job_id=job_id,
        options=JobOptions(),
        files=files,
        settings=settings,
        repos=repos,
        load=load,
        progress=progress,
    )


# --- classify --------------------------------------------------------------------------


def test_classify_uses_the_calibrated_thresholds(make_settings: MakeSettings) -> None:
    settings = make_settings()

    assert classify(0.96, settings) == "match"
    assert classify(0.72, settings) == "match"
    assert classify(0.7199, settings) == "candidate"
    assert classify(0.55, settings) == "candidate"
    assert classify(0.5499, settings) is None
    assert classify(0.3, settings) is None


def test_classify_follows_configured_thresholds(make_settings: MakeSettings) -> None:
    settings = make_settings(VOICEPRINT_MATCH_THRESHOLD=0.9, VOICEPRINT_CANDIDATE_THRESHOLD=0.4)

    assert classify(0.9, settings) == "match"
    assert classify(0.85, settings) == "candidate"
    assert classify(0.39, settings) is None


# --- the voiceprint stage ---------------------------------------------------------------


async def test_voiceprint_stage_keeps_matches_and_candidates_and_drops_the_rest(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings()
    repos = make_fake_repos()
    job_id = await _job(repos)
    files = _files(tmp_path, job_id)
    _write_diarize(files, {"S1": _unit(0.96), "S2": _unit(0.6), "S3": _unit(0.3)})
    known = await repos.speakers.upsert_known("Daniel Levitin")
    await repos.speakers.add_voiceprint(known.id, _unit(1.0), None, "S1")

    result = await stage_voiceprint(_context(settings, repos, files, job_id))

    assert result.status == "done"
    matches = result.output["matches"]
    assert isinstance(matches, dict)
    assert set(matches) == {"S1", "S2"}  # 0.3 is below the candidate threshold: absent
    strong = matches["S1"]
    assert isinstance(strong, list)
    assert len(strong) == 1
    assert strong[0]["known_speaker_id"] == str(known.id)
    assert strong[0]["name"] == "Daniel Levitin"
    assert strong[0]["class"] == "match"
    assert float(strong[0]["similarity"]) == pytest.approx(0.96, abs=1e-6)
    weak = matches["S2"]
    assert isinstance(weak, list)
    assert weak[0]["class"] == "candidate"
    assert float(weak[0]["similarity"]) == pytest.approx(0.6, abs=1e-6)

    # Read-only: the pipeline never enrolls, so the only stored sample is the original one.
    record = await repos.speakers.get(known.id)
    assert record is not None
    assert record.samples == 1


async def test_voiceprint_stage_asks_for_three_matches_per_label(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings()
    repos = make_fake_repos()
    job_id = await _job(repos)
    files = _files(tmp_path, job_id)
    _write_diarize(files, {"S1": _unit(1.0)})
    for position, similarity in enumerate((0.99, 0.98, 0.97, 0.96), start=1):
        speaker = await repos.speakers.upsert_known(f"Speaker {position}")
        await repos.speakers.add_voiceprint(speaker.id, _unit(similarity), None, "S1")

    result = await stage_voiceprint(_context(settings, repos, files, job_id))

    matches = result.output["matches"]
    assert isinstance(matches, dict)
    kept = matches["S1"]
    assert isinstance(kept, list)
    assert [entry["name"] for entry in kept] == ["Speaker 1", "Speaker 2", "Speaker 3"]
    assert [float(entry["similarity"]) for entry in kept] == pytest.approx([0.99, 0.98, 0.97], abs=1e-6)


async def test_voiceprint_stage_is_skipped_when_diarize_was_skipped_or_failed(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings()
    repos = make_fake_repos()
    job_id = await _job(repos)
    files = _files(tmp_path, job_id)
    _write_diarize(files, {"S1": _unit(1.0)}, status="skipped")
    known = await repos.speakers.upsert_known("Somebody")
    await repos.speakers.add_voiceprint(known.id, _unit(1.0), None, "S1")

    skipped = await stage_voiceprint(_context(settings, repos, files, job_id))
    assert skipped.status == "skipped"
    assert skipped.reason == "unavailable"
    assert skipped.output == {}

    _write_diarize(files, {"S1": _unit(1.0)}, status="failed")
    failed = await stage_voiceprint(_context(settings, repos, files, job_id))
    assert failed.status == "skipped"


async def test_voiceprint_stage_is_skipped_without_a_diarize_checkpoint(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    settings = make_settings()
    repos = make_fake_repos()
    job_id = await _job(repos)
    files = _files(tmp_path, job_id)

    result = await stage_voiceprint(_context(settings, repos, files, job_id))

    assert result.status == "skipped"
    assert result.reason == "unavailable"


# --- enrollment -------------------------------------------------------------------------


async def test_enroll_stores_the_labels_embedding_and_appends_to_the_same_name(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _job(repos)
    files = _files(tmp_path, job_id)
    _write_diarize(files, {"S1": _unit(0.97), "S2": _unit(0.2)})
    deps = EnrollmentDeps(jobs=repos.jobs, speakers=repos.speakers, data_dir=tmp_path / DATA_DIR)

    first = await enroll(deps, job_id, "S1", "Daniel  Levitin", "Neuroscientist")
    assert first.name == "Daniel  Levitin"
    assert first.name_norm == "daniel levitin"
    assert first.notes == "Neuroscientist"
    assert first.samples == 1

    second = await enroll(deps, job_id, "S1", "daniel levitin.")
    assert second.id == first.id
    assert second.notes == "Neuroscientist"
    assert second.samples == 2

    matches = await repos.speakers.nearest(_unit(0.97), k=3)
    assert [match.known_speaker_id for match in matches] == [first.id]
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-6)


async def test_enroll_rejects_a_job_that_is_not_done(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    deps = EnrollmentDeps(jobs=repos.jobs, speakers=repos.speakers, data_dir=tmp_path / DATA_DIR)

    for status in ("queued", "running", "failed", "cancelled"):
        job_id = await _job(repos, status=status)
        _write_diarize(_files(tmp_path, job_id), {"S1": _unit(0.9)})
        with pytest.raises(CaracalError) as caught:
            await enroll(deps, job_id, "S1", "Too Early")
        assert caught.value.code is ErrorCode.JOB_NOT_READY, status
        assert caught.value.message == "job is not ready"


async def test_enroll_unknown_job_or_label_is_not_found(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    deps = EnrollmentDeps(jobs=repos.jobs, speakers=repos.speakers, data_dir=tmp_path / DATA_DIR)

    with pytest.raises(CaracalError) as unknown_job:
        await enroll(deps, uuid.uuid4(), "S1", "Nobody")
    assert unknown_job.value.code is ErrorCode.NOT_FOUND

    job_id = await _job(repos)
    files = _files(tmp_path, job_id)
    _write_diarize(files, {"S1": _unit(0.9)})
    for label in ("S2", "speaker_00", ""):
        with pytest.raises(CaracalError) as missing_label:
            await enroll(deps, job_id, label, "Wrong Label")
        assert missing_label.value.code is ErrorCode.NOT_FOUND
        assert missing_label.value.details["speaker_label"] == label
    assert await repos.speakers.list() == []


async def test_enroll_without_a_diarize_checkpoint_is_not_found(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _job(repos)
    _files(tmp_path, job_id)
    deps = EnrollmentDeps(jobs=repos.jobs, speakers=repos.speakers, data_dir=tmp_path / DATA_DIR)

    with pytest.raises(CaracalError) as caught:
        await enroll(deps, job_id, "S1", "No Checkpoint")

    assert caught.value.code is ErrorCode.NOT_FOUND


async def test_enroll_rejects_a_name_without_letters(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    job_id = await _job(repos)
    _write_diarize(_files(tmp_path, job_id), {"S1": _unit(0.9)})
    deps = EnrollmentDeps(jobs=repos.jobs, speakers=repos.speakers, data_dir=tmp_path / DATA_DIR)

    with pytest.raises(CaracalError) as caught:
        await enroll(deps, job_id, "S1", "!!!")

    assert caught.value.code is ErrorCode.INVALID_INPUT
    assert await repos.speakers.list() == []


def test_m2_registry_wires_the_voiceprint_stage() -> None:
    registry = m2_registry()

    assert registry[Stage.VOICEPRINT] is stage_voiceprint
    assert registry[Stage.ALIGN] is not stage_voiceprint
