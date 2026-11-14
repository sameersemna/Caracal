"""``/v1/jobs`` contract: creation, dedup, rate limit, listing, status, delete (docs/03 §2)."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from caracal.errors import CaracalError, ErrorCode
from caracal.pipeline.stages import JobOptions, Stage
from caracal.storage import JobFiles
from tests.contract.conftest import AUTH, Harness

URL = "https://example.com/talk"


def _body(url: str = URL, **options: object) -> dict[str, object]:
    body: dict[str, object] = {"source": {"url": url}}
    if options:
        body["options"] = options
    return body


def _links(base: str, job_id: uuid.UUID) -> dict[str, object]:
    return {
        "report": f"{base}/v1/jobs/{job_id}/report",
        "transcript": f"{base}/v1/jobs/{job_id}/transcript",
        "events": f"{base}/v1/jobs/{job_id}/events",
        "subtitles": {},
    }


async def test_url_job_is_queued_and_enqueued(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post("/v1/jobs", headers=AUTH, json=_body())
    assert response.status_code == 202
    payload = response.json()
    job_id = uuid.UUID(payload["job_id"])
    assert payload == {
        "job_id": str(job_id),
        "status": "queued",
        "deduplicated": False,
        "links": _links("http://127.0.0.1:11404", job_id),
    }

    job = await h.repos.jobs.get(job_id)
    assert job is not None
    assert job.status == "queued"
    assert job.source_kind == "platform"
    assert job.source_url == URL
    assert job.upload_id is None
    assert job.progress == {}

    entries = await h.redis.xrange(h.keys.jobs_stream())
    assert len(entries) == 1
    _entry_id, fields = entries[0]
    assert fields[b"job_id"].decode() == str(job_id)
    assert fields[b"kind"].decode() == "pipeline"
    assert h.keys.jobs_stream().startswith(h.settings.REDIS_PREFIX)


async def test_direct_media_url_is_classified_direct(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/jobs", headers=AUTH, json=_body("https://example.com/audio/song.MP3?download=1")
        )
    assert response.status_code == 202
    job = await h.repos.jobs.get(uuid.UUID(response.json()["job_id"]))
    assert job is not None
    assert job.source_kind == "direct"


async def test_upload_job_consumes_the_upload(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    upload = await h.repos.uploads.create(
        filename="talk.wav", size_bytes=3, sha256="sha-abc", path="/uploads/talk.wav"
    )
    async with h.client() as client:
        response = await client.post("/v1/jobs", headers=AUTH, json={"source": {"upload_id": str(upload.id)}})
        assert response.status_code == 202
        job_id = uuid.UUID(response.json()["job_id"])

        consumed = await h.repos.uploads.get(upload.id)
        assert consumed is not None
        assert consumed.consumed_by == job_id

        job = await h.repos.jobs.get(job_id)
        assert job is not None
        assert job.source_kind == "upload"
        assert job.sha256 == "sha-abc"
        assert job.upload_id == upload.id

        second = await client.post("/v1/jobs", headers=AUTH, json={"source": {"upload_id": str(upload.id)}})
    assert second.status_code == 422
    assert second.json()["error"]["code"] == "invalid_input"
    jobs, _ = await h.repos.jobs.list()
    assert len(jobs) == 1


async def test_unknown_upload_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/jobs", headers=AUTH, json={"source": {"upload_id": str(uuid.uuid4())}}
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_dedup_returns_the_existing_done_job(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    options = JobOptions().model_dump()
    existing = await h.repos.jobs.create(source_kind="platform", source_url=URL, options=options)
    await h.repos.jobs.set_status(existing.id, "done")

    async with h.client() as client:
        response = await client.post("/v1/jobs", headers=AUTH, json=_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == str(existing.id)
    assert payload["status"] == "done"
    assert payload["deduplicated"] is True
    assert payload["links"] == _links("http://127.0.0.1:11404", existing.id)
    jobs, _ = await h.repos.jobs.list()
    assert len(jobs) == 1
    assert await h.redis.xlen(h.keys.jobs_stream()) == 0


async def test_depth_deep_bypasses_dedup(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    existing = await h.repos.jobs.create(
        source_kind="platform", source_url=URL, options=JobOptions().model_dump()
    )
    await h.repos.jobs.set_status(existing.id, "done")

    async with h.client() as client:
        response = await client.post("/v1/jobs", headers=AUTH, json=_body(depth="deep"))

    assert response.status_code == 202
    payload = response.json()
    assert payload["deduplicated"] is False
    assert payload["job_id"] != str(existing.id)
    jobs, _ = await h.repos.jobs.list()
    assert len(jobs) == 2


async def test_rate_limit_returns_429_with_retry_after(make_harness: Callable[..., Harness]) -> None:
    h = make_harness(settings_overrides={"CARACAL_RATE_LIMIT_PER_MIN": 1})
    async with h.client() as client:
        first = await client.post("/v1/jobs", headers=AUTH, json=_body("https://example.com/one"))
        second = await client.post("/v1/jobs", headers=AUTH, json=_body("https://example.com/two"))

    assert first.status_code == 202
    assert second.status_code == 429
    payload = second.json()
    assert payload["error"]["code"] == "rate_limited"
    assert int(second.headers["retry-after"]) >= 1
    jobs, _ = await h.repos.jobs.list()
    assert len(jobs) == 1


async def test_enqueue_failure_marks_job_failed_and_returns_503(
    make_harness: Callable[..., Harness],
) -> None:
    class FailingQueue:
        async def enqueue(self, job_id: uuid.UUID, kind: str) -> str:
            raise CaracalError(ErrorCode.DEPENDENCY_DOWN, "xadd failed", retryable=True)

    h = make_harness(queue=FailingQueue())
    async with h.client() as client:
        response = await client.post("/v1/jobs", headers=AUTH, json=_body())

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["code"] == "dependency_down"
    jobs, _ = await h.repos.jobs.list()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.status == "failed"
    assert job.error is not None
    assert job.error["code"] == "dependency_down"


async def test_body_over_64_kib_is_413(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/jobs",
            headers={**AUTH, "Content-Type": "application/json"},
            content=b"{" + b" " * 70_000 + b"}",
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "too_large"


@pytest.mark.parametrize(
    "url",
    [
        "ftp://secret-value.example/audio.mp3",
        "http:///no-host.mp3",
        "https://example.com/" + "a" * 2048,
    ],
)
async def test_invalid_url_is_422_without_echoing_the_value(
    make_harness: Callable[..., Harness], url: str
) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post("/v1/jobs", headers=AUTH, json=_body(url))
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "invalid_input"
    assert payload["error"]["details"]["errors"]
    assert "secret-value.example" not in response.text
    assert "a" * 100 not in response.text


async def test_source_requires_exactly_one_of_url_and_upload(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    async with h.client() as client:
        both = await client.post(
            "/v1/jobs",
            headers=AUTH,
            json={"source": {"url": URL, "upload_id": str(uuid.uuid4())}},
        )
        neither = await client.post("/v1/jobs", headers=AUTH, json={"source": {}})
    for response in (both, neither):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"


async def test_validation_error_never_echoes_submitted_values(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    distinctive = "distinctive-secret-value-1234567890"
    async with h.client() as client:
        response = await client.post(
            "/v1/jobs",
            headers=AUTH,
            json={"source": {"url": URL}, "unexpected_field": distinctive},
        )
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "invalid_input"
    assert payload["error"]["details"]["errors"]
    assert distinctive not in response.text


async def test_unknown_route_uses_the_error_envelope(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/nope", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_unexpected_exception_is_500_without_details(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()

    @h.app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom secret-value")

    async with h.client() as client:
        response = await client.get("/boom")
    assert response.status_code == 500
    assert response.json() == {"error": {"code": "internal", "message": "internal error", "details": {}}}
    assert "kaboom" not in response.text
    assert "Traceback" not in response.text


async def test_list_jobs_filters_and_paginates(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    first = await h.repos.jobs.create(source_kind="platform", source_url="https://e/1", options={})
    second = await h.repos.jobs.create(source_kind="direct", source_url="https://e/2", options={})
    third = await h.repos.jobs.create(source_kind="platform", source_url="https://e/3", options={})
    await h.repos.jobs.set_status(second.id, "failed")

    async with h.client() as client:
        failed = await client.get("/v1/jobs?status=failed", headers=AUTH)
        page1 = await client.get("/v1/jobs?limit=1", headers=AUTH)
        page2 = await client.get("/v1/jobs?limit=1&cursor=" + page1.json()["next_cursor"], headers=AUTH)

    assert failed.status_code == 200
    assert [item["job_id"] for item in failed.json()["items"]] == [str(second.id)]
    assert set(failed.json()["items"][0]) == {
        "job_id",
        "status",
        "created_at",
        "title",
        "source_kind",
    }

    assert len(page1.json()["items"]) == 1
    assert page1.json()["next_cursor"] is not None
    assert len(page2.json()["items"]) == 1
    seen = {page1.json()["items"][0]["job_id"], page2.json()["items"][0]["job_id"]}
    assert len(seen) == 2
    assert seen <= {str(first.id), str(second.id), str(third.id)}


async def test_list_jobs_rejects_an_unknown_status(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get("/v1/jobs?status=exploded", headers=AUTH)
    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "invalid_input"
    assert payload["error"]["details"]["allowed"] == ["queued", "running", "done", "failed", "cancelled"]


async def test_get_job_status_shape(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})
    await h.repos.jobs.upsert_stage(job.id, Stage.INGEST.value, status="done")
    await h.repos.jobs.upsert_stage(job.id, Stage.NORMALIZE.value, status="running")

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job.id}", headers=AUTH)

    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == str(job.id)
    assert payload["status"] == "queued"
    assert payload["source"] == {"kind": "platform", "url": URL}
    assert payload["progress"] == {"pct": 0, "stage": None, "message": None}
    assert payload["links"] == _links("http://127.0.0.1:11404", job.id)
    assert "error" not in payload
    stages = payload["stages"]
    assert [stage["name"] for stage in stages] == [stage.value for stage in Stage]
    assert stages[0]["status"] == "done"
    assert stages[1]["status"] == "running"
    assert "started_at" in stages[1]
    assert all(stage["status"] == "pending" for stage in stages[2:])


async def test_get_job_progress_and_error(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})
    await h.repos.jobs.set_progress(job.id, {"pct": 42, "stage": "transcribe", "message": "working"})
    await h.repos.jobs.set_status(job.id, "failed", error={"code": "dependency_down", "message": "ollama"})

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job.id}", headers=AUTH)

    payload = response.json()
    assert payload["status"] == "failed"
    assert payload["progress"] == {"pct": 42, "stage": "transcribe", "message": "working"}
    assert payload["error"] == {"code": "dependency_down", "message": "ollama"}


async def test_unknown_and_invalid_job_ids(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        unknown = await client.get(f"/v1/jobs/{uuid.uuid4()}", headers=AUTH)
        invalid = await client.get("/v1/jobs/not-a-uuid", headers=AUTH)

    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_input"
    assert "not-a-uuid" not in invalid.text


async def test_delete_running_job_sets_cancel_flag(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})

    async with h.client() as client:
        response = await client.delete(f"/v1/jobs/{job.id}", headers=AUTH)

    assert response.status_code == 202
    assert response.json() == {"job_id": str(job.id), "status": "cancelling"}
    assert await h.redis.exists(h.keys.cancel(job.id)) == 1
    assert await h.repos.jobs.get(job.id) is not None


async def test_delete_terminal_job_removes_rows_and_directory(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})
    files = JobFiles(h.settings.CARACAL_DATA_DIR, job.id)
    files.ensure()
    (files.root / "report.json").write_text("{}", encoding="utf-8")
    await h.repos.jobs.set_status(job.id, "done")
    assert files.root.is_dir()

    async with h.client() as client:
        response = await client.delete(f"/v1/jobs/{job.id}", headers=AUTH)

    assert response.status_code == 204
    assert response.content == b""
    assert await h.repos.jobs.get(job.id) is None
    assert not files.root.exists()


async def test_delete_never_follows_a_symlink_outside_the_data_dir(
    make_harness: Callable[..., Harness], tmp_path: Path
) -> None:
    h = make_harness()
    data_dir = h.settings.CARACAL_DATA_DIR.resolve()
    jobs_dir = data_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")

    job = await h.repos.jobs.create(source_kind="platform", source_url=URL, options={})
    (jobs_dir / str(job.id)).symlink_to(outside, target_is_directory=True)
    await h.repos.jobs.set_status(job.id, "done")

    async with h.client() as client:
        response = await client.delete(f"/v1/jobs/{job.id}", headers=AUTH)

    assert response.status_code == 204
    assert await h.repos.jobs.get(job.id) is None
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
