"""Checkpointed pipeline runner: order, retries, failure policy, resume, cancel, timeout."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.fakes.repo import FakeJobRepo, make_fake_repos
from tests.repo_contract import Repos

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.pipeline.runner import RunnerDeps, run_pipeline
from caracal.pipeline.stages import (
    JobOptions,
    Stage,
    StageContext,
    StageResult,
    default_registry,
)
from caracal.storage import JobFiles

MakeSettings = Callable[..., Settings]


def _deps(
    settings: Settings,
    repos: Repos,
    *,
    events: list[dict[str, Any]] | None = None,
    sleeps: list[float] | None = None,
    cancel: list[bool] | None = None,
    owner: list[bool] | None = None,
) -> RunnerDeps:
    recorded_events = events if events is not None else []
    recorded_sleeps = sleeps if sleeps is not None else []
    cancel_flags = cancel if cancel is not None else []
    owner_flags = owner if owner is not None else []

    async def publish(job_id: uuid.UUID, event: dict[str, Any]) -> None:
        recorded_events.append(dict(event))

    async def sleep(seconds: float) -> None:
        recorded_sleeps.append(seconds)

    async def cancelled(job_id: uuid.UUID) -> bool:
        return bool(cancel_flags.pop(0)) if cancel_flags else False

    async def still_owner() -> bool:
        return owner_flags.pop(0) if owner_flags else True

    return RunnerDeps(
        settings=settings,
        repos=repos,
        queue_events=publish,
        cancel_check=cancelled,
        sleep=sleep,
        still_owner=still_owner,
    )


async def _new_job(repos: Repos, **options: Any) -> uuid.UUID:
    record = await repos.jobs.create(
        source_kind="platform",
        source_url="https://example.com/talk",
        options=options or {"depth": "standard"},
    )
    return record.id


def _without_ts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in event.items() if key != "ts"} for event in events]


def _stage_fn(
    stage: Stage,
    calls: list[str],
    *,
    status: str = "done",
    output: dict[str, object] | None = None,
) -> Callable[[StageContext], Awaitable[StageResult]]:
    async def run(ctx: StageContext) -> StageResult:
        calls.append(stage.value)
        return StageResult(status=status, output=dict(output or {}))  # type: ignore[arg-type]

    return run


async def test_stage_order_events_and_progress_percentage(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    calls: list[str] = []
    events: list[dict[str, Any]] = []
    registry = {
        Stage.INGEST: _stage_fn(Stage.INGEST, calls, output={"a": 1}),
        Stage.NORMALIZE: _stage_fn(Stage.NORMALIZE, calls),
        Stage.TRANSCRIBE: _stage_fn(Stage.TRANSCRIBE, calls),
    }
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)

    status = await run_pipeline(job_id, _deps(settings, repos, events=events), registry)

    assert status == "done"
    assert calls == ["ingest", "normalize", "transcribe"]
    assert _without_ts(events) == [
        {"type": "status", "status": "running", "pct": 0.0},
        {"type": "stage", "stage": "ingest", "status": "running", "pct": 0.0},
        {"type": "stage", "stage": "ingest", "status": "done", "pct": 33.3},
        {"type": "stage", "stage": "normalize", "status": "running", "pct": 33.3},
        {"type": "stage", "stage": "normalize", "status": "done", "pct": 66.7},
        {"type": "stage", "stage": "transcribe", "status": "running", "pct": 66.7},
        {"type": "stage", "stage": "transcribe", "status": "done", "pct": 100.0},
        {"type": "end", "status": "done", "pct": 100.0},
    ]
    assert all(event["ts"].endswith("Z") for event in events)
    assert (await repos.jobs.get(job_id)).status == "done"


async def test_checkpoint_is_written_before_the_stage_is_marked_done(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    files = JobFiles(tmp_path, job_id)
    observed: list[tuple[str, bool, str]] = []

    class ObservingJobRepo(FakeJobRepo):
        async def upsert_stage(
            self,
            job_id: uuid.UUID,
            name: str,
            *,
            status: str,
            reason: str | None = None,
        ) -> None:
            if status in {"done", "skipped"}:
                path = files.stage(Stage(name))
                stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                observed.append((name, path.exists(), str(stored.get("status"))))
            await super().upsert_stage(job_id, name, status=status, reason=reason)

    shared = repos.jobs
    assert isinstance(shared, FakeJobRepo)
    observing = ObservingJobRepo(shared._store)
    observing_repos = replace(repos, jobs=observing)

    async def stage(ctx: StageContext) -> StageResult:
        return StageResult(status="done", output={"ok": True})

    status = await run_pipeline(
        job_id,
        _deps(settings, observing_repos),
        {Stage.INGEST: stage, Stage.NORMALIZE: stage},
    )

    assert status == "done"
    assert observed == [("ingest", True, "done"), ("normalize", True, "done")]
    checkpoint = files.read_json(files.stage(Stage.INGEST))
    assert checkpoint == {"status": "done", "reason": None, "output": {"ok": True}}


async def test_resume_skips_done_stages_without_calling_them(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    files = JobFiles(tmp_path, job_id)
    files.ensure()
    files.write_json(files.stage(Stage.INGEST), {"status": "done", "reason": None, "output": {}})
    await repos.jobs.upsert_stage(job_id, Stage.INGEST.value, status="done")

    calls: list[str] = []
    registry = {
        Stage.INGEST: _stage_fn(Stage.INGEST, calls),
        Stage.NORMALIZE: _stage_fn(Stage.NORMALIZE, calls),
        Stage.TRANSCRIBE: _stage_fn(Stage.TRANSCRIBE, calls),
    }

    status = await run_pipeline(job_id, _deps(settings, repos), registry)

    assert status == "done"
    assert calls == ["normalize", "transcribe"]


async def test_done_row_without_a_checkpoint_is_rerun(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    await repos.jobs.upsert_stage(job_id, Stage.INGEST.value, status="done")

    calls: list[str] = []
    status = await run_pipeline(
        job_id,
        _deps(settings, repos),
        {Stage.INGEST: _stage_fn(Stage.INGEST, calls)},
    )

    assert status == "done"
    assert calls == ["ingest"]


async def test_checkpoint_without_a_done_row_is_rerun(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    files = JobFiles(tmp_path, job_id)
    files.ensure()
    files.write_json(files.stage(Stage.INGEST), {"status": "done", "reason": None, "output": {}})
    await repos.jobs.upsert_stage(job_id, Stage.INGEST.value, status="failed", reason="crashed")

    calls: list[str] = []
    status = await run_pipeline(
        job_id,
        _deps(settings, repos),
        {Stage.INGEST: _stage_fn(Stage.INGEST, calls)},
    )

    assert status == "done"
    assert calls == ["ingest"]
    row = (await repos.jobs.list_stages(job_id))[0]
    assert row.status == "done"


async def test_retryable_failure_is_retried_with_backoff(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    calls: list[int] = []
    sleeps: list[float] = []

    async def flaky(ctx: StageContext) -> StageResult:
        calls.append(len(calls))
        if len(calls) < 3:
            raise CaracalError(ErrorCode.DEPENDENCY_DOWN, "whisper down", retryable=True)
        return StageResult(status="done", output={"attempts": len(calls)})

    status = await run_pipeline(
        job_id,
        _deps(settings, repos, sleeps=sleeps),
        {Stage.INGEST: flaky},
    )

    assert status == "done"
    assert len(calls) == 3
    assert sleeps == [2.0, 8.0]


async def test_critical_stage_exhausted_retries_fail_the_job(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    calls: list[str] = []
    sleeps: list[float] = []
    events: list[dict[str, Any]] = []

    async def always_retryable(ctx: StageContext) -> StageResult:
        calls.append("ingest")
        raise CaracalError(ErrorCode.DEPENDENCY_DOWN, "still down", retryable=True)

    async def later(ctx: StageContext) -> StageResult:
        calls.append("normalize")
        return StageResult(status="done")

    status = await run_pipeline(
        job_id,
        _deps(settings, repos, sleeps=sleeps, events=events),
        {Stage.INGEST: always_retryable, Stage.NORMALIZE: later},
    )

    assert status == "failed"
    assert calls == ["ingest", "ingest", "ingest"]
    assert sleeps == [2.0, 8.0]
    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error == {"code": "dependency_down", "message": "still down", "stage": "ingest"}
    assert events[-1]["type"] == "end"
    assert events[-1]["status"] == "failed"


async def test_non_retryable_failure_of_critical_stage_stops_immediately(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    calls: list[str] = []
    sleeps: list[float] = []

    async def bad_input(ctx: StageContext) -> StageResult:
        calls.append("ingest")
        raise CaracalError(ErrorCode.INVALID_INPUT, "bad media")

    async def later(ctx: StageContext) -> StageResult:
        calls.append("normalize")
        return StageResult(status="done")

    status = await run_pipeline(
        job_id,
        _deps(settings, repos, sleeps=sleeps),
        {Stage.INGEST: bad_input, Stage.NORMALIZE: later},
    )

    assert status == "failed"
    assert calls == ["ingest"]
    assert sleeps == []
    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.error == {"code": "invalid_input", "message": "bad media", "stage": "ingest"}


async def test_non_critical_failure_continues_and_job_finishes_done(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    calls: list[str] = []

    async def ok(ctx: StageContext) -> StageResult:
        calls.append("ok")
        return StageResult(status="done")

    async def broken(ctx: StageContext) -> StageResult:
        calls.append("diarize")
        raise CaracalError(ErrorCode.DEPENDENCY_DOWN, "no pyannote")

    async def plan(ctx: StageContext) -> StageResult:
        calls.append("plan")
        return StageResult(status="done")

    status = await run_pipeline(
        job_id,
        _deps(settings, repos),
        {Stage.INGEST: ok, Stage.DIARIZE: broken, Stage.PLAN: plan},
    )

    assert status == "done"
    assert calls == ["ok", "diarize", "plan"]
    stages = {row.name: row for row in await repos.jobs.list_stages(job_id)}
    assert stages["diarize"].status == "failed"
    assert stages["diarize"].reason == "dependency_down: no pyannote"
    assert stages["ingest"].status == "done"
    assert stages["plan"].status == "done"
    assert (await repos.jobs.get(job_id)).status == "done"


async def test_cancel_between_stages_settles_cancelled(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    calls: list[str] = []
    events: list[dict[str, Any]] = []
    registry = {
        Stage.INGEST: _stage_fn(Stage.INGEST, calls),
        Stage.NORMALIZE: _stage_fn(Stage.NORMALIZE, calls),
        Stage.PLAN: _stage_fn(Stage.PLAN, calls),
    }

    status = await run_pipeline(
        job_id,
        _deps(settings, repos, events=events, cancel=[False, True]),
        registry,
    )

    assert status == "cancelled"
    assert calls == ["ingest"]
    assert events[-1] == {"type": "end", "status": "cancelled", "ts": events[-1]["ts"], "pct": 33.3}
    assert (await repos.jobs.get(job_id)).status == "cancelled"


async def test_lost_ownership_before_the_first_stage_writes_nothing(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    calls: list[str] = []
    events: list[dict[str, Any]] = []
    files = JobFiles(tmp_path, job_id)
    registry = {
        Stage.INGEST: _stage_fn(Stage.INGEST, calls),
        Stage.NORMALIZE: _stage_fn(Stage.NORMALIZE, calls),
    }

    # The first check (before the job status is written) passes, the stage-loop check fails.
    status = await run_pipeline(job_id, _deps(settings, repos, events=events, owner=[True, False]), registry)

    assert status == "lost"
    assert calls == []
    assert [event["type"] for event in events] == ["status"]
    assert await repos.jobs.list_stages(job_id) == []
    assert files.read_json(files.stage(Stage.INGEST)) is None
    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.status == "running"


async def test_lost_ownership_after_a_stage_writes_nothing_for_the_next(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    calls: list[str] = []
    events: list[dict[str, Any]] = []
    files = JobFiles(tmp_path, job_id)
    registry = {
        Stage.INGEST: _stage_fn(Stage.INGEST, calls, output={"ok": True}),
        Stage.NORMALIZE: _stage_fn(Stage.NORMALIZE, calls),
    }

    status = await run_pipeline(
        job_id,
        # initial check, ingest pre-check, ingest post-check, normalize pre-check
        _deps(settings, repos, events=events, owner=[True, True, True, False]),
        registry,
    )

    assert status == "lost"
    assert calls == ["ingest"]
    stages = {row.name: row for row in await repos.jobs.list_stages(job_id)}
    assert set(stages) == {"ingest"}
    assert stages["ingest"].status == "done"
    assert files.read_json(files.stage(Stage.INGEST)) == {
        "status": "done",
        "reason": None,
        "output": {"ok": True},
    }
    assert files.read_json(files.stage(Stage.NORMALIZE)) is None
    assert [event["type"] for event in events] == ["status", "stage", "stage"]
    assert events[-1]["status"] == "done"
    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.status == "running"


async def test_stage_result_is_discarded_when_ownership_is_lost_while_it_runs(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    files = JobFiles(tmp_path, job_id)
    events: list[dict[str, Any]] = []
    lost = False

    async def stage(ctx: StageContext) -> StageResult:
        nonlocal lost
        lost = True
        return StageResult(status="done", output={"must": "not be written"})

    async def still_owner() -> bool:
        return not lost

    async def publish(job_id: uuid.UUID, event: dict[str, Any]) -> None:
        events.append(dict(event))

    async def never_cancelled(job_id: uuid.UUID) -> bool:
        return False

    deps = RunnerDeps(
        settings=settings,
        repos=repos,
        queue_events=publish,
        cancel_check=never_cancelled,
        still_owner=still_owner,
    )

    status = await run_pipeline(job_id, deps, {Stage.INGEST: stage})

    assert status == "lost"
    assert files.read_json(files.stage(Stage.INGEST)) is None
    stages = {row.name: row for row in await repos.jobs.list_stages(job_id)}
    assert stages["ingest"].status == "running"
    assert [event["type"] for event in events] == ["status", "stage"]
    assert events[-1]["status"] == "running"
    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.status == "running"


async def test_job_timeout_fails_the_job(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path, CARACAL_JOB_TIMEOUT_S=1)

    async def slow(ctx: StageContext) -> StageResult:
        await asyncio.sleep(10)
        return StageResult(status="done")

    status = await run_pipeline(job_id, _deps(settings, repos), {Stage.INGEST: slow})

    assert status == "failed"
    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.status == "failed"
    assert job.error is not None
    assert job.error["code"] == "internal"
    assert job.error["message"] == "job timeout"
    assert job.error["stage"] == "ingest"


async def test_terminal_job_is_a_noop(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    await repos.jobs.set_status(job_id, "done")
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    calls: list[str] = []
    events: list[dict[str, Any]] = []

    status = await run_pipeline(
        job_id,
        _deps(settings, repos, events=events),
        {Stage.INGEST: _stage_fn(Stage.INGEST, calls)},
    )

    assert status == "done"
    assert calls == []
    assert events == []


async def test_missing_job_raises_not_found(make_settings: MakeSettings, tmp_path: Path) -> None:
    repos = make_fake_repos()
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    with pytest.raises(CaracalError) as excinfo:
        await run_pipeline(uuid.uuid4(), _deps(settings, repos))
    assert excinfo.value.code is ErrorCode.NOT_FOUND


async def test_progress_callback_updates_job_and_publishes_event(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    events: list[dict[str, Any]] = []

    async def stage(ctx: StageContext) -> StageResult:
        await ctx.progress(55.5, "halfway")
        return StageResult(status="done")

    await run_pipeline(job_id, _deps(settings, repos, events=events), {Stage.INGEST: stage})

    job = await repos.jobs.get(job_id)
    assert job is not None
    assert job.progress == {"pct": 55.5, "stage": "ingest", "message": "halfway"}
    progress_events = [event for event in events if event["type"] == "progress"]
    assert progress_events == [
        {
            "type": "progress",
            "stage": "ingest",
            "message": "halfway",
            "ts": progress_events[0]["ts"],
            "pct": 0.0,
        }
    ]


async def test_default_registry_skips_all_stages_and_finishes_done(
    make_settings: MakeSettings, tmp_path: Path
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    files = JobFiles(tmp_path, job_id)

    status = await run_pipeline(job_id, _deps(settings, repos), default_registry())

    assert status == "done"
    stages = {row.name: row for row in await repos.jobs.list_stages(job_id)}
    assert set(stages) == {stage.value for stage in Stage}
    assert all(row.status == "skipped" for row in stages.values())
    assert all(row.reason == "not implemented in M0" for row in stages.values())
    checkpoint = files.read_json(files.stage(Stage.ASSEMBLE))
    assert checkpoint == {"status": "skipped", "reason": "not implemented in M0", "output": {}}


def test_job_options_match_the_contract_defaults() -> None:
    options = JobOptions()
    assert options.language == "auto"
    assert options.target_langs is None
    assert options.depth == "standard"
    assert options.diarize == "auto"
    assert options.verify == "auto"
    assert options.speaker_hints == []
    assert options.context_hint is None


def test_job_options_forbid_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        JobOptions.model_validate({"depth": "standard", "unknown": True})


def test_job_options_limit_speaker_hints() -> None:
    with pytest.raises(ValidationError):
        JobOptions(speaker_hints=[f"hint-{index}" for index in range(21)])
    with pytest.raises(ValidationError):
        JobOptions(speaker_hints=["x" * 201])


def test_job_options_limit_context_hint() -> None:
    with pytest.raises(ValidationError):
        JobOptions(context_hint="x" * 2001)


async def test_runner_logs_every_stage_outcome_without_transcript_text(
    make_settings: MakeSettings,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repos = make_fake_repos()
    job_id = await _new_job(repos)
    settings = make_settings(CARACAL_DATA_DIR=tmp_path)

    async def done(ctx: StageContext) -> StageResult:
        return StageResult(status="done", output={})

    async def skipped(ctx: StageContext) -> StageResult:
        return StageResult(status="skipped", reason="not needed")

    async def broken(ctx: StageContext) -> StageResult:
        raise CaracalError(ErrorCode.DEPENDENCY_DOWN, "whisper down", retryable=True)

    with caplog.at_level("INFO", logger="caracal.pipeline.runner"):
        status = await run_pipeline(
            job_id,
            _deps(settings, repos, sleeps=[0.0, 0.0]),
            {Stage.INGEST: done, Stage.NORMALIZE: skipped, Stage.DIARIZE: broken},
        )

    assert status == "done"
    records = {record.getMessage(): record for record in caplog.records}
    assert "stage.done" in records
    assert records["stage.done"].levelname == "INFO"
    assert records["stage.done"].stage == "ingest"
    assert records["stage.done"].duration_s >= 0.0
    assert "stage.skipped" in records
    assert records["stage.skipped"].levelname == "INFO"
    assert records["stage.skipped"].reason == "not needed"
    failed = next(record for record in caplog.records if record.msg == "stage.failed %s: %s")
    assert failed.getMessage() == "stage.failed dependency_down: whisper down"
    assert failed.levelname == "WARNING"
    assert failed.stage == "diarize"
    assert failed.code == "dependency_down"
    assert failed.error == "whisper down"
    assert failed.retryable is True
    assert failed.attempts == 3
    assert "transcript" not in caplog.text.lower()
