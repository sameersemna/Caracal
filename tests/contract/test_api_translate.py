"""``POST /v1/jobs/{id}/translate`` contract (docs/03 §2, docs/04 §10)."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

from caracal.pipeline.stages import JobOptions
from tests.contract.conftest import AUTH, Harness

URL = "https://example.com/talk"


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


def _body(*langs: str) -> dict[str, Any]:
    return {"target_langs": list(langs)}


async def test_translate_queues_a_done_job_and_keeps_it_done(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h, status="done")

    async with h.client() as client:
        response = await client.post(
            f"/v1/jobs/{job_id}/translate",
            headers=AUTH,
            json=_body("de", "ar"),
        )
        status = await client.get(f"/v1/jobs/{job_id}", headers=AUTH)

    assert response.status_code == 202
    payload = response.json()
    assert payload["job_id"] == str(job_id)
    assert payload["status"] == "queued"
    assert payload["links"]["report"].endswith(f"/v1/jobs/{job_id}/report")

    entries = await h.redis.xrange(h.keys.jobs_stream())
    assert len(entries) == 1
    fields = entries[0][1]
    assert fields[b"job_id"].decode() == str(job_id)
    assert fields[b"kind"].decode() == "translate"
    assert json.loads(fields[b"payload"]) == {"target_langs": ["de", "ar"]}
    # The report and its links stay available while the extra languages are produced.
    assert status.status_code == 200
    assert status.json()["status"] == "done"


async def test_two_translate_requests_each_carry_their_own_languages(
    make_harness: Callable[..., Harness],
) -> None:
    """Verdict 35: back-to-back REST requests must not overwrite a shared request file."""
    h = make_harness()
    job_id = await _job(h, status="done")

    async with h.client() as client:
        first = await client.post(f"/v1/jobs/{job_id}/translate", headers=AUTH, json=_body("de"))
        second = await client.post(f"/v1/jobs/{job_id}/translate", headers=AUTH, json=_body("ar"))

    assert (first.status_code, second.status_code) == (202, 202)
    entries = await h.redis.xrange(h.keys.jobs_stream())
    payloads = [json.loads(fields[b"payload"]) for _entry_id, fields in entries]
    assert payloads == [{"target_langs": ["de"]}, {"target_langs": ["ar"]}]


async def test_translate_rejects_a_job_that_is_not_done(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    queued_id = await _job(h, status="queued")
    running_id = await _job(h, status="running")

    async with h.client() as client:
        queued = await client.post(
            f"/v1/jobs/{queued_id}/translate",
            headers=AUTH,
            json=_body("de"),
        )
        running = await client.post(
            f"/v1/jobs/{running_id}/translate",
            headers=AUTH,
            json=_body("de"),
        )

    assert queued.status_code == 409
    assert queued.json()["error"]["code"] == "job_not_ready"
    assert running.status_code == 409
    assert running.json()["error"]["code"] == "job_not_ready"
    assert await h.redis.xrange(h.keys.jobs_stream()) == []


async def test_translate_validates_language_codes(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h)

    for langs in (["english"], ["DE"], ["de-DE"], ["xx"], []):
        async with h.client() as client:
            response = await client.post(
                f"/v1/jobs/{job_id}/translate",
                headers=AUTH,
                json={"target_langs": langs},
            )
        assert response.status_code == 422, langs
        assert response.json()["error"]["code"] == "invalid_input"

    assert await h.redis.xrange(h.keys.jobs_stream()) == []


async def test_translate_unknown_job_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            f"/v1/jobs/{uuid.uuid4()}/translate",
            headers=AUTH,
            json=_body("de"),
        )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
