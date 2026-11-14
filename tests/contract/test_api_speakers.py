"""``/v1/speakers`` contract: list, enroll, delete and their error envelopes (docs/03 §2).

Enrollment stores biometric data, so it is only reachable through this explicit route and
only for a finished job (docs/06 §2). Everything runs against the fake repositories.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from typing import Any

from caracal.pipeline.stages import JobOptions, Stage
from caracal.storage import JobFiles
from tests.contract.conftest import AUTH, Harness

URL = "https://example.com/talk"


def _unit(similarity: float) -> list[float]:
    rest = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return [similarity, rest, *([0.0] * 254)]


async def _job(h: Harness, *, status: str = "done") -> uuid.UUID:
    record = await h.repos.jobs.create(
        source_kind="platform",
        source_url=URL,
        title="A talk",
        options=JobOptions().model_dump(),
    )
    if status != "queued":
        await h.repos.jobs.set_status(record.id, status)
    return record.id


def _write_diarize(h: Harness, job_id: uuid.UUID, embeddings: dict[str, list[float]]) -> None:
    files = JobFiles(h.settings.CARACAL_DATA_DIR, job_id)
    files.ensure()
    files.write_json(
        files.stage(Stage.DIARIZE),
        {
            "status": "done",
            "reason": None,
            "output": {
                "num_speakers": len(embeddings),
                "turns": [],
                "exclusive_turns": [],
                "embeddings": embeddings,
            },
        },
    )


def _enroll_body(job_id: uuid.UUID, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "job_id": str(job_id),
        "speaker_label": "S1",
        "name": "Daniel  Levitin",
        "notes": "Neuroscientist",
    }
    body.update(overrides)
    return body


async def test_list_is_empty_until_something_is_enrolled(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    async with h.client() as client:
        empty = await client.get("/v1/speakers", headers=AUTH)
        assert empty.status_code == 200
        assert empty.json() == {"items": []}

        job_id = await _job(h)
        _write_diarize(h, job_id, {"S1": _unit(0.97), "S2": _unit(0.2)})
        enrolled = await client.post("/v1/speakers/enroll", headers=AUTH, json=_enroll_body(job_id))

        assert enrolled.status_code == 201
        created = enrolled.json()
        assert created["name"] == "Daniel  Levitin"
        assert created["notes"] == "Neuroscientist"
        assert created["samples"] == 1
        assert uuid.UUID(created["id"])
        assert created["created_at"] and created["updated_at"]

        listed = await client.get("/v1/speakers", headers=AUTH)
        assert listed.status_code == 200
        assert listed.json() == {"items": [created]}

        filtered = await client.get("/v1/speakers", headers=AUTH, params={"q": "levitin"})
        assert filtered.json() == {"items": [created]}
        missing = await client.get("/v1/speakers", headers=AUTH, params={"q": "nobody"})
        assert missing.json() == {"items": []}


async def test_notes_are_omitted_from_the_wire_model_when_absent(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h)
    _write_diarize(h, job_id, {"S1": _unit(0.9)})

    async with h.client() as client:
        response = await client.post(
            "/v1/speakers/enroll",
            headers=AUTH,
            json=_enroll_body(job_id, notes=None),
        )

    assert response.status_code == 201
    assert "notes" not in response.json()


async def test_enrolling_the_same_normalised_name_appends_a_sample(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    first_job = await _job(h)
    _write_diarize(h, first_job, {"S1": _unit(0.97)})
    second_job = await _job(h)
    _write_diarize(h, second_job, {"S1": _unit(0.95)})

    async with h.client() as client:
        first = await client.post(
            "/v1/speakers/enroll",
            headers=AUTH,
            json=_enroll_body(first_job, name="Daniel  Levitin"),
        )
        second = await client.post(
            "/v1/speakers/enroll",
            headers=AUTH,
            json=_enroll_body(second_job, name="daniel levitin.", notes=None),
        )
        listed = await client.get("/v1/speakers", headers=AUTH)

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["id"] == first.json()["id"]
    assert second.json()["samples"] == 2
    assert second.json()["notes"] == "Neuroscientist"
    assert listed.json() == {"items": [second.json()]}


async def test_enroll_requires_a_finished_job(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()

    async with h.client() as client:
        for status in ("queued", "running", "failed"):
            job_id = await _job(h, status=status)
            _write_diarize(h, job_id, {"S1": _unit(0.9)})
            response = await client.post(
                "/v1/speakers/enroll",
                headers=AUTH,
                json=_enroll_body(job_id),
            )
            assert response.status_code == 409, status
            assert response.json()["error"]["code"] == "job_not_ready"

        assert (await client.get("/v1/speakers", headers=AUTH)).json() == {"items": []}


async def test_enroll_unknown_job_or_label_is_404(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h)
    _write_diarize(h, job_id, {"S1": _unit(0.9)})

    async with h.client() as client:
        unknown_job = await client.post(
            "/v1/speakers/enroll",
            headers=AUTH,
            json=_enroll_body(uuid.uuid4()),
        )
        unknown_label = await client.post(
            "/v1/speakers/enroll",
            headers=AUTH,
            json=_enroll_body(job_id, speaker_label="S9"),
        )

    assert unknown_job.status_code == 404
    assert unknown_job.json()["error"]["code"] == "not_found"
    assert unknown_label.status_code == 404
    assert unknown_label.json()["error"]["code"] == "not_found"
    assert unknown_label.json()["error"]["details"]["speaker_label"] == "S9"


async def test_enroll_body_validation_is_422(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h)
    _write_diarize(h, job_id, {"S1": _unit(0.9)})

    async with h.client() as client:
        blank_name = await client.post(
            "/v1/speakers/enroll",
            headers=AUTH,
            json=_enroll_body(job_id, name=""),
        )
        bad_job = await client.post(
            "/v1/speakers/enroll",
            headers=AUTH,
            json={**_enroll_body(job_id), "job_id": "not-a-uuid"},
        )
        extra = await client.post(
            "/v1/speakers/enroll",
            headers=AUTH,
            json={**_enroll_body(job_id), "unknown": True},
        )

    for response in (blank_name, bad_job, extra):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"


async def test_delete_removes_the_speaker_and_its_voiceprints(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h)
    _write_diarize(h, job_id, {"S1": _unit(0.97)})

    async with h.client() as client:
        created = await client.post("/v1/speakers/enroll", headers=AUTH, json=_enroll_body(job_id))
        speaker_id = created.json()["id"]

        deleted = await client.delete(f"/v1/speakers/{speaker_id}", headers=AUTH)
        assert deleted.status_code == 204
        assert deleted.content == b""

        listed = await client.get("/v1/speakers", headers=AUTH)
        assert listed.json() == {"items": []}

        again = await client.delete(f"/v1/speakers/{speaker_id}", headers=AUTH)
        assert again.status_code == 404
        assert again.json()["error"]["code"] == "not_found"

        invalid = await client.delete("/v1/speakers/not-a-uuid", headers=AUTH)
        assert invalid.status_code == 422


async def test_speaker_routes_require_the_bearer_token(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()

    async with h.client() as client:
        for method, path in (
            ("GET", "/v1/speakers"),
            ("POST", "/v1/speakers/enroll"),
            ("DELETE", f"/v1/speakers/{uuid.uuid4()}"),
        ):
            response = await client.request(method, path, json={})
            assert response.status_code == 401, (method, path)
            assert response.json()["error"]["code"] == "unauthorized"
