"""Voiceprint classification and explicit enrollment (docs/04 §5).

Embeddings are biometric data: this module never runs on its own. ``enroll`` is called only
from an explicit ``POST /v1/speakers/enroll`` / MCP ``enroll_speaker`` request, never from
the pipeline (docs/06 §2). The ``voiceprint`` stage only reads: it compares the current
job's labels against stored voiceprints and reports matches with a similarity and a class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from caracal.config import Settings
from caracal.db.repo import JobRepo, KnownSpeakerRecord, SpeakerRepo
from caracal.errors import CaracalError, ErrorCode
from caracal.storage import JobFiles

MatchClass = Literal["match", "candidate"]

_DIARIZE_STAGE = "diarize"


def classify(similarity: float, settings: Settings) -> MatchClass | None:
    """``"match"`` at the match threshold, ``"candidate"`` above the candidate one, else ``None``.

    Thresholds are cosine similarities calibrated on community-1 (docs/02, docs/04 §5); the
    order matters because a candidate threshold above the match threshold would otherwise
    downgrade a strong match.
    """
    if similarity >= settings.VOICEPRINT_MATCH_THRESHOLD:
        return "match"
    if similarity >= settings.VOICEPRINT_CANDIDATE_THRESHOLD:
        return "candidate"
    return None


@dataclass(frozen=True)
class EnrollmentDeps:
    """The repositories and artifact root enrollment needs (injected for tests)."""

    jobs: JobRepo
    speakers: SpeakerRepo
    data_dir: Path


def _label_not_found(speaker_label: str) -> CaracalError:
    return CaracalError(
        ErrorCode.NOT_FOUND,
        "speaker label not found",
        {"speaker_label": speaker_label},
    )


def _checkpoint_embedding(checkpoint: object, speaker_label: str) -> list[float]:
    """The L2-normalised embedding a finished diarize checkpoint holds for one label."""
    if not isinstance(checkpoint, dict) or checkpoint.get("status") != "done":
        raise _label_not_found(speaker_label)
    output = checkpoint.get("output")
    if not isinstance(output, dict):
        raise _label_not_found(speaker_label)
    embeddings = output.get("embeddings")
    if not isinstance(embeddings, dict):
        raise _label_not_found(speaker_label)
    vector = embeddings.get(speaker_label)
    if not isinstance(vector, list) or not vector:
        raise _label_not_found(speaker_label)
    try:
        return [float(value) for value in vector]
    except (TypeError, ValueError) as exc:
        raise _label_not_found(speaker_label) from exc


async def enroll(
    deps: EnrollmentDeps,
    job_id: UUID,
    speaker_label: str,
    name: str,
    notes: str | None = None,
) -> KnownSpeakerRecord:
    """Attach a name to one diarized label and store its embedding (docs/04 §5).

    The job must be ``done`` and the label must exist in its diarize checkpoint; the same
    normalised name appends another embedding to the existing KnownSpeaker. This is the only
    path that writes a voiceprint: the pipeline never enrolls by itself.
    """
    job = await deps.jobs.get(job_id)
    if job is None:
        raise CaracalError(ErrorCode.NOT_FOUND, "job not found", {"job_id": str(job_id)})
    if job.status != "done":
        raise CaracalError(
            ErrorCode.JOB_NOT_READY,
            "job is not ready",
            {"job_id": str(job_id), "status": job.status},
        )

    files = JobFiles(deps.data_dir, job_id)
    embedding = _checkpoint_embedding(files.read_json(files.stage(_DIARIZE_STAGE)), speaker_label)

    record = await deps.speakers.upsert_known(name, notes)
    await deps.speakers.add_voiceprint(record.id, embedding, job_id, speaker_label)
    updated = await deps.speakers.get(record.id)
    return updated if updated is not None else record


__all__ = ["EnrollmentDeps", "MatchClass", "classify", "enroll"]
