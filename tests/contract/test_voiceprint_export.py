"""``GET /v1/jobs/{id}/voiceprints``: calibration export of raw diarization embeddings.

The export is the only place raw speaker embeddings leave the process, so it is
bearer-protected and REST-only (card T4.3, docs/06 §2). These tests drive the route against
the fake repositories and the diarize checkpoint on disk.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

from caracal.pipeline.stages import Stage
from caracal.storage import JobFiles
from tests.contract.conftest import AUTH, Harness
from tests.contract.test_api_speakers import _job, _unit, _write_diarize


def _skipped_checkpoint(h: Harness, job_id: uuid.UUID) -> None:
    files = JobFiles(h.settings.CARACAL_DATA_DIR, job_id)
    files.ensure()
    files.write_json(
        files.stage(Stage.DIARIZE),
        {"status": "skipped", "reason": "unavailable", "output": {}},
    )


async def test_export_returns_every_diarized_label_and_its_256_floats(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h)
    first = _unit(0.97)
    second = _unit(0.2)
    _write_diarize(h, job_id, {"S2": second, "S1": first})

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/voiceprints", headers=AUTH)

    assert response.status_code == 200
    items = response.json()["items"]
    assert [item["label"] for item in items] == ["S1", "S2"]
    assert items[0]["embedding"] == first
    assert items[1]["embedding"] == second
    assert all(len(item["embedding"]) == 256 for item in items)


async def test_a_skipped_diarize_checkpoint_exports_nothing(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h)
    _skipped_checkpoint(h, job_id)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/voiceprints", headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"items": []}


async def test_unknown_job_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{uuid.uuid4()}/voiceprints", headers=AUTH)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_missing_checkpoint_is_409_while_running_and_404_when_failed(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    running = await _job(h, status="running")
    failed = await _job(h, status="failed")

    async with h.client() as client:
        running_response = await client.get(f"/v1/jobs/{running}/voiceprints", headers=AUTH)
        failed_response = await client.get(f"/v1/jobs/{failed}/voiceprints", headers=AUTH)

    assert running_response.status_code == 409
    assert running_response.json()["error"]["code"] == "job_not_ready"
    assert failed_response.status_code == 404
    assert failed_response.json()["error"]["code"] == "not_found"


async def test_a_malformed_embedding_is_an_internal_error(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h)
    _write_diarize(h, job_id, {"S1": _unit(0.9)[:255]})

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/voiceprints", headers=AUTH)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal"


async def test_export_requires_the_bearer_token(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h)
    _write_diarize(h, job_id, {"S1": _unit(0.9)})

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/voiceprints")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
