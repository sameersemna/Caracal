"""Shared repository contract (docs/07 §3-§5).

Every ``case_*`` function is a plain async function taking a :class:`Repos` bundle. The unit
suite runs them against ``tests/fakes/repo.py`` and the live suite runs the very same functions
against Postgres, so both implementations must agree on records, errors and cursors.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from caracal.db.repo import (
    JobRepo,
    ReportRepo,
    SegmentRepo,
    SegmentRow,
    SpeakerRepo,
    UploadRepo,
)
from caracal.errors import CaracalError, ErrorCode


@dataclass(frozen=True)
class Repos:
    """The repository protocols plus a hook that empties all of them."""

    jobs: JobRepo
    uploads: UploadRepo
    reports: ReportRepo
    segments: SegmentRepo
    speakers: SpeakerRepo
    cleanup: Callable[[], Awaitable[None]]


async def expect_error(code: ErrorCode, awaitable: Awaitable[Any]) -> CaracalError:
    """Await ``awaitable`` expecting a :class:`CaracalError` with ``code``."""
    with pytest.raises(CaracalError) as excinfo:
        await awaitable
    error = excinfo.value
    assert error.code is code, f"expected {code}, got {error.code}: {error.message}"
    return error


def _segments_for(job_id: uuid.UUID, texts: list[str]) -> list[SegmentRow]:
    return [
        SegmentRow(
            job_id=job_id,
            seg_id=index,
            start_s=float(index),
            end_s=float(index) + 0.5,
            speaker="S1" if index % 2 == 0 else None,
            lang="en",
            text=text,
        )
        for index, text in enumerate(texts)
    ]


async def case_job_create_and_get(repos: Repos) -> None:
    job = await repos.jobs.create(
        source_kind="platform",
        source_url="https://example.com/talk",
        title="A talk",
        options={"depth": "standard"},
        sha256="deadbeef",
    )
    assert job.status == "queued"
    assert job.source_kind == "platform"
    assert job.source_url == "https://example.com/talk"
    assert job.upload_id is None
    assert job.title == "A talk"
    assert job.options == {"depth": "standard"}
    assert job.progress == {}
    assert job.error is None
    assert job.finished_at is None
    assert job.sha256 == "deadbeef"
    assert job.created_at.tzinfo is not None
    assert job.updated_at >= job.created_at

    assert await repos.jobs.get(job.id) == job
    assert await repos.jobs.get(uuid.uuid4()) is None


async def case_job_create_validates_source_kind(repos: Repos) -> None:
    error = await expect_error(
        ErrorCode.INVALID_INPUT,
        repos.jobs.create(source_kind="carrier-pigeon", options={}),
    )
    assert "source_kind" in error.message
    assert await repos.jobs.list() == ([], None)


async def case_job_list_pagination(repos: Repos) -> None:
    created = [
        await repos.jobs.create(source_kind="direct", source_url=f"https://example.com/{index}", options={})
        for index in range(5)
    ]

    first, cursor = await repos.jobs.list(limit=2)
    assert len(first) == 2
    assert cursor is not None
    second, cursor = await repos.jobs.list(limit=2, cursor=cursor)
    assert len(second) == 2
    assert cursor is not None
    third, cursor = await repos.jobs.list(limit=2, cursor=cursor)
    assert len(third) == 1
    assert cursor is None

    collected = [item.id for item in (*first, *second, *third)]
    assert len(set(collected)) == 5
    expected = [item.id for item in sorted(created, key=lambda job: (job.created_at, job.id), reverse=True)]
    assert collected == expected

    clamped_low, _ = await repos.jobs.list(limit=0)
    assert len(clamped_low) == 1
    clamped_high, high_cursor = await repos.jobs.list(limit=1000)
    assert len(clamped_high) == 5
    assert high_cursor is None


async def case_job_list_status_filter(repos: Repos) -> None:
    queued = await repos.jobs.create(source_kind="direct", options={})
    finished = await repos.jobs.create(source_kind="direct", options={})
    await repos.jobs.set_status(finished.id, "done")

    done_items, _ = await repos.jobs.list(status="done")
    assert [item.id for item in done_items] == [finished.id]
    queued_items, _ = await repos.jobs.list(status="queued")
    assert [item.id for item in queued_items] == [queued.id]


async def case_job_list_invalid_cursor(repos: Repos) -> None:
    await repos.jobs.create(source_kind="direct", options={})
    bad_cursors = [
        "",
        "not-base64!!",
        "e30",  # base64("{}")
        base64.urlsafe_b64encode(b"[]").decode("ascii").rstrip("="),  # a JSON list
        base64.urlsafe_b64encode(b'{"t": "not-a-date", "i": "not-a-uuid"}').decode("ascii"),
    ]
    for cursor in bad_cursors:
        error = await expect_error(ErrorCode.INVALID_INPUT, repos.jobs.list(cursor=cursor))
        assert error.message == "invalid cursor"


async def case_job_status_transitions(repos: Repos) -> None:
    job = await repos.jobs.create(source_kind="direct", options={})

    await repos.jobs.set_status(job.id, "running")
    running = await repos.jobs.get(job.id)
    assert running is not None
    assert running.status == "running"
    assert running.finished_at is None

    await repos.jobs.set_status(job.id, "done")
    done = await repos.jobs.get(job.id)
    assert done is not None
    assert done.status == "done"
    assert done.finished_at is not None
    assert done.updated_at >= running.updated_at

    error = await expect_error(ErrorCode.INVALID_INPUT, repos.jobs.set_status(job.id, "failed"))
    assert "terminal" in error.message
    await expect_error(ErrorCode.INVALID_INPUT, repos.jobs.set_status(job.id, "done"))
    await expect_error(ErrorCode.INVALID_INPUT, repos.jobs.set_status(job.id, "bogus"))
    await expect_error(ErrorCode.NOT_FOUND, repos.jobs.set_status(uuid.uuid4(), "running"))

    failed = await repos.jobs.create(source_kind="direct", options={})
    await repos.jobs.set_status(failed.id, "cancelled", error={"code": "internal", "message": "stopped"})
    record = await repos.jobs.get(failed.id)
    assert record is not None
    assert record.status == "cancelled"
    assert record.error == {"code": "internal", "message": "stopped"}
    assert record.finished_at is not None


async def case_job_progress_and_title(repos: Repos) -> None:
    job = await repos.jobs.create(source_kind="direct", options={})
    await repos.jobs.set_progress(job.id, {"pct": 42, "stage": "transcribe"})
    await repos.jobs.set_title(job.id, "Renamed")
    record = await repos.jobs.get(job.id)
    assert record is not None
    assert record.progress == {"pct": 42, "stage": "transcribe"}
    assert record.title == "Renamed"
    assert record.updated_at >= job.updated_at

    unknown = uuid.uuid4()
    await expect_error(ErrorCode.NOT_FOUND, repos.jobs.set_progress(unknown, {}))
    await expect_error(ErrorCode.NOT_FOUND, repos.jobs.set_title(unknown, "x"))


async def case_job_stage_lifecycle(repos: Repos) -> None:
    job = await repos.jobs.create(source_kind="direct", options={})

    await repos.jobs.upsert_stage(job.id, "ingest", status="pending")
    stages = await repos.jobs.list_stages(job.id)
    assert [stage.name for stage in stages] == ["ingest"]
    pending = stages[0]
    assert pending.status == "pending"
    assert pending.reason is None
    assert pending.attempts == 0
    assert pending.started_at is None
    assert pending.ended_at is None

    await repos.jobs.upsert_stage(job.id, "ingest", status="running")
    first_run = (await repos.jobs.list_stages(job.id))[0]
    assert first_run.status == "running"
    assert first_run.attempts == 1
    assert first_run.started_at is not None
    assert first_run.ended_at is None

    await repos.jobs.upsert_stage(job.id, "ingest", status="running")
    second_run = (await repos.jobs.list_stages(job.id))[0]
    assert second_run.attempts == 2
    assert second_run.started_at is not None
    assert second_run.started_at >= first_run.started_at

    await repos.jobs.upsert_stage(job.id, "ingest", status="failed", reason="model crashed")
    failed = (await repos.jobs.list_stages(job.id))[0]
    assert failed.status == "failed"
    assert failed.reason == "model crashed"
    assert failed.attempts == 2
    assert failed.ended_at is not None

    await repos.jobs.upsert_stage(job.id, "ingest", status="running")
    retried = (await repos.jobs.list_stages(job.id))[0]
    assert retried.attempts == 3
    assert retried.reason is None
    assert retried.ended_at is None

    await repos.jobs.upsert_stage(job.id, "transcribe", status="skipped", reason="unavailable")
    await repos.jobs.upsert_stage(job.id, "ingest", status="done")
    stages = await repos.jobs.list_stages(job.id)
    assert [stage.name for stage in stages] == ["ingest", "transcribe"]
    assert stages[0].status == "done"
    assert stages[0].attempts == 3
    assert stages[0].ended_at is not None
    assert stages[1].status == "skipped"
    assert stages[1].attempts == 0
    assert stages[1].started_at is None
    assert stages[1].ended_at is not None

    await expect_error(
        ErrorCode.INVALID_INPUT,
        repos.jobs.upsert_stage(job.id, "ingest", status="exploded"),
    )
    await expect_error(
        ErrorCode.NOT_FOUND,
        repos.jobs.upsert_stage(uuid.uuid4(), "ingest", status="running"),
    )


async def case_find_recent_done(repos: Repos) -> None:
    upload_job = await repos.jobs.create(
        source_kind="upload",
        upload_id=uuid.uuid4(),
        options={"depth": "standard"},
        sha256="abc123",
    )
    assert await repos.jobs.find_recent_done(sha256="abc123", options={"depth": "standard"}) is None

    await repos.jobs.set_status(upload_job.id, "done")
    found = await repos.jobs.find_recent_done(sha256="abc123", options={"depth": "standard"})
    assert found is not None
    assert found.id == upload_job.id
    assert await repos.jobs.find_recent_done(sha256="abc123", options={"depth": "deep"}) is None
    assert await repos.jobs.find_recent_done(sha256="other", options={"depth": "standard"}) is None
    assert (
        await repos.jobs.find_recent_done(sha256="abc123", options={"depth": "standard"}, within_days=0)
        is None
    )
    assert await repos.jobs.find_recent_done(options={"depth": "standard"}) is None

    platform_job = await repos.jobs.create(
        source_kind="platform",
        source_url="https://example.com/watch",
        options={"depth": "standard"},
    )
    await repos.jobs.set_status(platform_job.id, "done")
    by_url = await repos.jobs.find_recent_done(
        source_url="https://example.com/watch",
        options={"depth": "standard"},
    )
    assert by_url is not None
    assert by_url.id == platform_job.id


async def case_delete_cascades(repos: Repos) -> None:
    job = await repos.jobs.create(source_kind="direct", title="Delete me", options={})
    await repos.jobs.upsert_stage(job.id, "ingest", status="done")
    await repos.reports.put(job.id, {"schema_version": "1.0", "title": "Delete me"}, "1.0")
    await repos.segments.replace_for_job(job.id, _segments_for(job.id, ["hello cascading world"]))

    assert len(await repos.segments.search("cascading")) == 1

    await repos.jobs.delete(job.id)

    assert await repos.jobs.get(job.id) is None
    assert await repos.jobs.list_stages(job.id) == []
    assert await repos.reports.get(job.id) is None
    assert await repos.segments.search("cascading") == []
    await expect_error(ErrorCode.NOT_FOUND, repos.jobs.delete(job.id))


async def case_upload_lifecycle(repos: Repos) -> None:
    upload = await repos.uploads.create(
        filename="talk.wav",
        size_bytes=2048,
        sha256="upload-sha",
        path="/data/uploads/talk.wav",
    )
    assert upload.filename == "talk.wav"
    assert upload.size_bytes == 2048
    assert upload.sha256 == "upload-sha"
    assert upload.path == "/data/uploads/talk.wav"
    assert upload.consumed_by is None
    assert upload.created_at.tzinfo is not None
    assert await repos.uploads.get(upload.id) == upload
    assert await repos.uploads.get(uuid.uuid4()) is None

    job = await repos.jobs.create(source_kind="upload", upload_id=upload.id, options={})
    await repos.uploads.consume(upload.id, job.id)
    consumed = await repos.uploads.get(upload.id)
    assert consumed is not None
    assert consumed.consumed_by == job.id

    error = await expect_error(ErrorCode.INVALID_INPUT, repos.uploads.consume(upload.id, job.id))
    assert "consumed" in error.message
    await expect_error(ErrorCode.NOT_FOUND, repos.uploads.consume(uuid.uuid4(), job.id))

    fresh = await repos.uploads.create(filename="b.wav", size_bytes=1, sha256="b", path="/data/uploads/b.wav")
    outstanding = await repos.uploads.list_unconsumed_older_than(0)
    assert [record.id for record in outstanding] == [fresh.id]
    assert await repos.uploads.list_unconsumed_older_than(24) == []


async def case_upload_create_honours_the_given_id(repos: Repos) -> None:
    """Verdict 08 carry-forward: the API generates the id so the upload dir matches the row."""
    upload_id = uuid.uuid4()
    upload = await repos.uploads.create(
        upload_id=upload_id,
        filename="talk.wav",
        size_bytes=10,
        sha256="id-sha",
        path=f"/data/uploads/{upload_id}/talk.wav",
    )
    assert upload.id == upload_id
    assert await repos.uploads.get(upload_id) == upload

    generated = await repos.uploads.create(
        filename="other.wav",
        size_bytes=1,
        sha256="other",
        path="/data/uploads/other.wav",
    )
    assert generated.id != upload_id


async def case_report_upsert(repos: Repos) -> None:
    job = await repos.jobs.create(source_kind="direct", options={})
    assert await repos.reports.get(job.id) is None

    await repos.reports.put(job.id, {"schema_version": "1.0", "title": "first"}, "1.0")
    assert await repos.reports.get(job.id) == {"schema_version": "1.0", "title": "first"}

    await repos.reports.put(job.id, {"schema_version": "1.1", "title": "second"}, "1.1")
    assert await repos.reports.get(job.id) == {"schema_version": "1.1", "title": "second"}

    await expect_error(ErrorCode.NOT_FOUND, repos.reports.put(uuid.uuid4(), {}, "1.0"))
    assert await repos.reports.get(uuid.uuid4()) is None


async def case_segments_replace_and_search(repos: Repos) -> None:
    job = await repos.jobs.create(source_kind="direct", title="Audio interview", options={})
    await repos.segments.replace_for_job(
        job.id,
        _segments_for(job.id, ["Hello world of audio", "Completely unrelated"]),
    )

    hits = await repos.segments.search("world")
    assert len(hits) == 1
    hit = hits[0]
    assert hit.job_id == job.id
    assert hit.title == "Audio interview"
    assert hit.seg_id == 0
    assert hit.start_s == 0.0
    assert "world" in hit.text
    assert hit.rank > 0

    both_terms = await repos.segments.search("world audio")
    assert [entry.seg_id for entry in both_terms] == [0]
    assert await repos.segments.search("missing") == []
    assert await repos.segments.search("   ") == []

    await repos.segments.replace_for_job(
        job.id,
        _segments_for(job.id, ["replacement text", "another replacement line"]),
    )
    assert await repos.segments.search("world") == []
    assert len(await repos.segments.search("replacement")) == 2
    limited = await repos.segments.search("replacement", limit=1)
    assert len(limited) == 1

    other_job = await repos.jobs.create(source_kind="direct", options={})
    await expect_error(
        ErrorCode.INVALID_INPUT,
        repos.segments.replace_for_job(
            other_job.id,
            [SegmentRow(job_id=job.id, seg_id=99, start_s=0.0, end_s=1.0, speaker=None, lang=None, text="x")],
        ),
    )
    await expect_error(ErrorCode.NOT_FOUND, repos.segments.replace_for_job(uuid.uuid4(), []))


async def case_segments_search_job_filter_and_speaker(repos: Repos) -> None:
    first = await repos.jobs.create(source_kind="direct", title="First job", options={})
    second = await repos.jobs.create(source_kind="direct", title="Second job", options={})
    await repos.segments.replace_for_job(
        first.id,
        [
            SegmentRow(first.id, 0, 0.0, 1.0, "S1", "en", "alpha shared"),
            SegmentRow(first.id, 1, 1.0, 2.0, None, "en", "alpha other"),
        ],
    )
    await repos.segments.replace_for_job(
        second.id,
        [SegmentRow(second.id, 0, 0.0, 1.0, "S2", "en", "alpha shared elsewhere")],
    )

    everything = await repos.segments.search("alpha")
    assert {hit.job_id for hit in everything} == {first.id, second.id}

    filtered = await repos.segments.search("alpha", job_id=first.id)
    assert [hit.seg_id for hit in filtered] == [0, 1]
    assert {hit.job_id for hit in filtered} == {first.id}
    assert {hit.title for hit in filtered} == {"First job"}
    assert [hit.speaker for hit in filtered] == ["S1", None]

    limited = await repos.segments.search("alpha", job_id=first.id, limit=1)
    assert len(limited) == 1
    assert await repos.segments.search("alpha", job_id=uuid.uuid4()) == []


def _embedding(*head: float) -> list[float]:
    """A 256-d vector whose leading components are ``head`` and whose rest is zero."""
    vector = [0.0] * 256
    vector[: len(head)] = head
    return vector


async def case_speaker_name_normalisation_merges(repos: Repos) -> None:
    first = await repos.speakers.upsert_known("Daniel  Levitin", "Author of This Is Your Brain on Music")
    assert first.name == "Daniel  Levitin"
    assert first.name_norm == "daniel levitin"
    assert first.notes == "Author of This Is Your Brain on Music"
    assert first.samples == 0
    assert first.created_at.tzinfo is not None
    assert first.updated_at >= first.created_at

    merged = await repos.speakers.upsert_known("daniel levitin.")
    assert merged.id == first.id
    assert merged.name_norm == "daniel levitin"
    assert merged.notes == "Author of This Is Your Brain on Music"  # notes kept when not given

    updated = await repos.speakers.upsert_known("DANIEL LEVITIN", "Neuroscientist")
    assert updated.id == first.id
    assert updated.notes == "Neuroscientist"

    accented = await repos.speakers.upsert_known("José", None)
    decomposed = await repos.speakers.upsert_known("JOSE\u0301")
    assert decomposed.id == accented.id
    assert decomposed.name_norm == "josé"

    listed = await repos.speakers.list()
    assert [record.id for record in listed] == [first.id, accented.id]
    assert await repos.speakers.get(first.id) == updated
    assert await repos.speakers.get(uuid.uuid4()) is None

    await expect_error(ErrorCode.INVALID_INPUT, repos.speakers.upsert_known("!!!"))
    await expect_error(ErrorCode.INVALID_INPUT, repos.speakers.upsert_known("   "))


async def case_speaker_nearest_best_per_speaker(repos: Repos) -> None:
    levitin = await repos.speakers.upsert_known("Daniel Levitin")
    other = await repos.speakers.upsert_known("Someone Else")
    job = await repos.jobs.create(source_kind="direct", options={})

    await repos.speakers.add_voiceprint(levitin.id, _embedding(0.0, 1.0), job.id, "S1")
    await repos.speakers.add_voiceprint(levitin.id, _embedding(1.0), None, "S2")
    await repos.speakers.add_voiceprint(other.id, _embedding(0.6, 0.8), job.id, "S1")

    matches = await repos.speakers.nearest(_embedding(1.0), k=5)
    assert [match.known_speaker_id for match in matches] == [levitin.id, other.id]
    assert matches[0].name == "Daniel Levitin"
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-6)
    # The worse Levitin sample (similarity 0) is not a second row: one row per known speaker.
    assert len(matches) == 2
    assert matches[1].similarity == pytest.approx(0.6, abs=1e-6)

    only = await repos.speakers.nearest(_embedding(0.0, 1.0), k=1)
    assert [match.known_speaker_id for match in only] == [levitin.id]

    listed = await repos.speakers.list(q="levi")
    assert [record.id for record in listed] == [levitin.id]
    assert listed[0].samples == 2
    assert await repos.speakers.list(q="no such person") == []
    assert len(await repos.speakers.list(q="  ")) == 2


async def case_speaker_delete_cascades_and_counts(repos: Repos) -> None:
    doomed = await repos.speakers.upsert_known("Doomed Person")
    survivor = await repos.speakers.upsert_known("Survivor")
    await repos.speakers.add_voiceprint(doomed.id, _embedding(1.0), None, "S1")
    await repos.speakers.add_voiceprint(doomed.id, _embedding(0.0, 1.0), None, "S2")
    await repos.speakers.add_voiceprint(survivor.id, _embedding(0.6, 0.8), None, "S1")

    assert await repos.speakers.delete(doomed.id) == 2
    assert await repos.speakers.get(doomed.id) is None
    assert [record.id for record in await repos.speakers.list()] == [survivor.id]
    # The remaining embeddings are untouched: the survivor still matches at 0.6.
    matches = await repos.speakers.nearest(_embedding(1.0), k=5)
    assert [match.known_speaker_id for match in matches] == [survivor.id]
    assert matches[0].similarity == pytest.approx(0.6, abs=1e-6)
    survivor_record = await repos.speakers.get(survivor.id)
    assert survivor_record is not None
    assert survivor_record.samples == 1

    await expect_error(ErrorCode.NOT_FOUND, repos.speakers.delete(doomed.id))


async def case_job_delete_keeps_voiceprints(repos: Repos) -> None:
    speaker = await repos.speakers.upsert_known("Enrolled From A Job")
    job = await repos.jobs.create(source_kind="direct", options={})
    await repos.speakers.add_voiceprint(speaker.id, _embedding(1.0), job.id, "S1")

    await repos.jobs.delete(job.id)

    # Biometric data outlives the job: only the provenance link is nulled (SET NULL).
    record = await repos.speakers.get(speaker.id)
    assert record is not None
    assert record.samples == 1
    matches = await repos.speakers.nearest(_embedding(1.0), k=3)
    assert [match.name for match in matches] == ["Enrolled From A Job"]


async def case_speaker_embedding_validation(repos: Repos) -> None:
    speaker = await repos.speakers.upsert_known("Validated")

    short = await expect_error(
        ErrorCode.INVALID_INPUT,
        repos.speakers.add_voiceprint(speaker.id, [1.0, 0.0, 0.0], None, "S1"),
    )
    assert short.details["expected"] == 256
    await expect_error(
        ErrorCode.INVALID_INPUT,
        repos.speakers.add_voiceprint(speaker.id, [*_embedding(1.0)[:-1], float("nan")], None, "S1"),
    )
    await expect_error(
        ErrorCode.INVALID_INPUT,
        repos.speakers.add_voiceprint(speaker.id, [*_embedding(1.0)[:-1], float("inf")], None, "S1"),
    )
    await expect_error(
        ErrorCode.INVALID_INPUT,
        repos.speakers.nearest([0.5, 0.5]),
    )
    await expect_error(
        ErrorCode.NOT_FOUND,
        repos.speakers.add_voiceprint(uuid.uuid4(), _embedding(1.0), None, "S1"),
    )
    await expect_error(
        ErrorCode.NOT_FOUND,
        repos.speakers.add_voiceprint(speaker.id, _embedding(1.0), uuid.uuid4(), "S1"),
    )
    assert await repos.speakers.nearest(_embedding(1.0)) == []


CASES: tuple[Callable[[Repos], Awaitable[None]], ...] = (
    case_job_create_and_get,
    case_job_create_validates_source_kind,
    case_job_list_pagination,
    case_job_list_status_filter,
    case_job_list_invalid_cursor,
    case_job_status_transitions,
    case_job_progress_and_title,
    case_job_stage_lifecycle,
    case_find_recent_done,
    case_delete_cascades,
    case_upload_lifecycle,
    case_upload_create_honours_the_given_id,
    case_report_upsert,
    case_segments_replace_and_search,
    case_segments_search_job_filter_and_speaker,
    case_speaker_name_normalisation_merges,
    case_speaker_nearest_best_per_speaker,
    case_speaker_delete_cascades_and_counts,
    case_job_delete_keeps_voiceprints,
    case_speaker_embedding_validation,
)
