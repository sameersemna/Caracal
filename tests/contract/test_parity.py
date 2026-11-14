"""REST <-> MCP parity, encoded from the docs/03 §0 table (card 15).

Every capability must exist on both sides. Rows whose milestone has not landed are listed
with ``pending``: the test fails if one side is implemented without the row being promoted
to available, so a capability can never be added to only one front door.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from mcp.client import Client

from caracal.db.repo import SegmentRow
from caracal.pipeline.stages import JobOptions, Stage
from caracal.storage import JobFiles
from tests.contract.conftest import AUTH, Harness
from tests.contract.test_api_qa_search import _ScriptedLlm
from tests.contract.test_api_report import _report

_URL = "https://example.com/talk"


@dataclass(frozen=True)
class Row:
    """One docs/03 §0 capability: its REST routes and its MCP tool(s)."""

    capability: str
    rest: tuple[tuple[str, str], ...]
    mcp: tuple[str, ...]
    pending: str | None = None
    rest_only: bool = False
    reason: str | None = None


AVAILABLE: tuple[Row, ...] = (
    Row(
        "submit audio",
        (("POST", "/v1/uploads"), ("POST", "/v1/jobs")),
        ("analyze_audio",),
    ),
    Row("job status / progress", (("GET", "/v1/jobs/{job_id}"),), ("get_job",)),
    Row("list jobs", (("GET", "/v1/jobs"),), ("list_jobs",)),
    Row(
        "transcript / subtitles",
        (
            ("GET", "/v1/jobs/{job_id}/transcript"),
            ("GET", "/v1/jobs/{job_id}/subtitles/{filename:path}"),
        ),
        ("get_transcript",),
    ),
    Row("cancel / delete", (("DELETE", "/v1/jobs/{job_id}"),), ("cancel_job",)),
    Row("report", (("GET", "/v1/jobs/{job_id}/report"),), ("get_report",)),
    Row(
        "speakers",
        (
            ("GET", "/v1/speakers"),
            ("POST", "/v1/speakers/enroll"),
            ("DELETE", "/v1/speakers/{speaker_id}"),
        ),
        ("list_speakers", "enroll_speaker", "delete_speaker"),
    ),
    Row("translate", (("POST", "/v1/jobs/{job_id}/translate"),), ("translate_job",)),
    Row("ask", (("POST", "/v1/jobs/{job_id}/ask"),), ("ask_about_audio",)),
    Row("full-text search", (("GET", "/v1/search"),), ("search_transcripts",)),
    # Biometric exception (card T4.3): raw embeddings are calibration data and stay REST-only.
    Row(
        "voiceprint export (calibration)",
        (("GET", "/v1/jobs/{job_id}/voiceprints"),),
        (),
        rest_only=True,
        reason="speaker embeddings are biometric data: explicit bearer-protected REST only",
    ),
)

PENDING: tuple[Row, ...] = ()


def _iter_routes(routes: Any) -> Any:
    """Yield leaf routes; FastAPI keeps included routers nested (``_IncludedRouter``)."""
    for route in routes:
        original = getattr(route, "original_router", None)
        if original is not None:
            yield from _iter_routes(original.routes)
        else:
            yield route


def _rest_route_exists(app: FastAPI, method: str, template: str) -> bool:
    for route in _iter_routes(app.routes):
        if getattr(route, "path", None) != template:
            continue
        methods = getattr(route, "methods", None) or set()
        if method in methods:
            return True
    return False


async def _tool_names(h: Harness) -> set[str]:
    server = h.app.state.mcp_server
    async with Client(server) as client:
        listed = await client.list_tools()
    return {tool.name for tool in listed.tools}


def _write_artifacts(h: Harness, job_id: uuid.UUID) -> None:
    files = JobFiles(h.settings.CARACAL_DATA_DIR, job_id)
    files.ensure()
    files.write_json(
        files.transcript_json,
        {
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 12.5,
                    "text": "my fellow americans",
                    "words": [],
                    "avg_logprob": -0.1,
                    "no_speech_prob": 0.0,
                    "lang": "en",
                }
            ],
            "language_code": "en",
            "dropped": {},
        },
    )
    files.write_text(files.transcript_txt, "[00:00] my fellow americans\n")
    files.write_json(
        files.stage(Stage.LANGID),
        {
            "status": "done",
            "reason": None,
            "output": {
                "languages": [{"code": "en", "name": "English", "share": 1.0, "confidence": 0.93}],
                "caption_agreement": None,
                "caption_lang": None,
            },
        },
    )
    files.write_text(files.subtitle("source", "srt"), "1\n00:00:00,000 --> 00:00:12,500\nhello\n")


async def _done_job(h: Harness) -> uuid.UUID:
    job = await h.repos.jobs.create(
        source_kind="platform",
        source_url=_URL,
        title="A talk",
        options=JobOptions().model_dump(),
    )
    await h.repos.jobs.upsert_stage(job.id, Stage.INGEST.value, status="done")
    await h.repos.jobs.set_status(job.id, "done")
    _write_artifacts(h, job.id)
    return job.id


async def _stored_job(h: Harness) -> uuid.UUID:
    job_id = await _done_job(h)
    await h.repos.reports.put(job_id, _report(job_id).model_dump(mode="json"), "1.0")
    return job_id


async def _mcp_call(h: Harness, name: str, arguments: dict[str, Any]) -> Any:
    server = h.app.state.mcp_server
    async with Client(server) as client:
        return await client.call_tool(name, arguments)


async def test_every_available_row_exists_on_both_sides(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    tools = await _tool_names(h)

    for row in AVAILABLE:
        for method, template in row.rest:
            assert _rest_route_exists(h.app, method, template), (row.capability, method, template)
        if row.rest_only:
            assert row.reason, row.capability
            assert row.mcp == (), row.capability
            assert not any("voiceprint" in name for name in tools), row.capability
            continue
        for name in row.mcp:
            assert name in tools, (row.capability, name)


async def test_every_pending_row_is_absent_on_both_sides(
    make_harness: Callable[..., Harness],
) -> None:
    """A route or tool implemented without promoting the row to available must fail here."""
    h = make_harness()
    tools = await _tool_names(h)

    for row in PENDING:
        assert row.pending is not None
        for method, template in row.rest:
            assert not _rest_route_exists(h.app, method, template), (
                f"{row.capability} REST route {method} {template} exists but the row is pending {row.pending}"
            )
        for name in row.mcp:
            assert name not in tools, (
                f"{row.capability} MCP tool {name} exists but the row is pending {row.pending}"
            )


async def test_get_job_returns_equal_data(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _done_job(h)

    mcp = await _mcp_call(h, "get_job", {"job_id": str(job_id)})
    async with h.client() as client:
        rest = await client.get(f"/v1/jobs/{job_id}", headers=AUTH)

    assert rest.status_code == 200
    assert mcp.structured_content == rest.json()


async def test_list_jobs_returns_equal_data(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    await _done_job(h)

    mcp = await _mcp_call(h, "list_jobs", {})
    async with h.client() as client:
        rest = await client.get("/v1/jobs", headers=AUTH)

    assert rest.status_code == 200
    assert mcp.structured_content == rest.json()


async def test_transcript_json_returns_equal_data(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _done_job(h)

    mcp = await _mcp_call(h, "get_transcript", {"job_id": str(job_id), "format": "json"})
    async with h.client() as client:
        rest = await client.get(f"/v1/jobs/{job_id}/transcript", headers=AUTH)

    assert rest.status_code == 200
    assert mcp.structured_content == rest.json()


async def test_report_json_returns_equal_data(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _stored_job(h)

    mcp = await _mcp_call(h, "get_report", {"job_id": str(job_id)})
    async with h.client() as client:
        rest = await client.get(f"/v1/jobs/{job_id}/report", headers=AUTH)

    assert rest.status_code == 200
    assert mcp.structured_content == rest.json()


async def test_translate_queues_equal_data(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _done_job(h)

    mcp = await _mcp_call(h, "translate_job", {"job_id": str(job_id), "target_langs": ["de"]})
    async with h.client() as client:
        rest = await client.post(
            f"/v1/jobs/{job_id}/translate",
            headers=AUTH,
            json={"target_langs": ["de"]},
        )

    assert rest.status_code == 202
    assert mcp.structured_content == rest.json()
    assert rest.json()["status"] == "queued"


def _diarize_checkpoint(h: Harness, job_id: uuid.UUID) -> None:
    files = JobFiles(h.settings.CARACAL_DATA_DIR, job_id)
    files.ensure()
    files.write_json(
        files.stage(Stage.DIARIZE),
        {
            "status": "done",
            "reason": None,
            "output": {
                "num_speakers": 1,
                "turns": [],
                "exclusive_turns": [],
                "embeddings": {"S1": [1.0, *([0.0] * 255)]},
            },
        },
    )


async def test_speakers_return_equal_data(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _done_job(h)
    _diarize_checkpoint(h, job_id)

    enrolled = await _mcp_call(
        h,
        "enroll_speaker",
        {
            "job_id": str(job_id),
            "speaker_label": "S1",
            "name": "Daniel  Levitin",
            "notes": "Neuroscientist",
        },
    )
    assert enrolled.is_error is False

    mcp = await _mcp_call(h, "list_speakers", {})
    async with h.client() as client:
        rest = await client.get("/v1/speakers", headers=AUTH)

    assert rest.status_code == 200
    assert rest.json() == {"items": [enrolled.structured_content]}
    assert mcp.structured_content == rest.json()

    deleted = await _mcp_call(h, "delete_speaker", {"speaker_id": enrolled.structured_content["id"]})
    assert deleted.structured_content == {"deleted": True, "voiceprints_removed": 1}
    async with h.client() as client:
        assert (await client.get("/v1/speakers", headers=AUTH)).json() == {"items": []}


async def _segmented_job(
    h: Harness,
    *,
    status: str,
    texts: list[str],
    title: str,
) -> uuid.UUID:
    record = await h.repos.jobs.create(
        source_kind="platform",
        source_url=_URL,
        title=title,
        options=JobOptions().model_dump(),
    )
    if status != "queued":
        await h.repos.jobs.set_status(record.id, status)
    await h.repos.segments.replace_for_job(
        record.id,
        [
            SegmentRow(record.id, index, float(index), float(index) + 1.0, "S1", "en", text)
            for index, text in enumerate(texts)
        ],
    )
    return record.id


async def test_ask_returns_equal_data(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _stored_job(h)
    await h.repos.segments.replace_for_job(
        job_id,
        [SegmentRow(job_id, 1, 0.0, 1.0, "S1", "en", "stress one")],
    )
    h.app.state.service.llm = _ScriptedLlm(
        answer={
            "answer": "The talk is about stress.",
            "confidence": 0.8,
            "citations": [{"seg_id": "seg:1"}],
        },
    )

    mcp = await _mcp_call(h, "ask_about_audio", {"job_id": str(job_id), "question": "stress"})
    async with h.client() as client:
        rest = await client.post(
            f"/v1/jobs/{job_id}/ask",
            headers=AUTH,
            json={"question": "stress"},
        )

    assert rest.status_code == 200
    assert mcp.structured_content == rest.json()


async def test_search_returns_equal_data(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    done_id = await _segmented_job(h, status="done", texts=["stress alpha", "stress beta"], title="Done")
    await _segmented_job(h, status="running", texts=["stress ignored"], title="Running")

    mcp = await _mcp_call(h, "search_transcripts", {"query": "stress", "limit": 20})
    async with h.client() as client:
        rest = await client.get("/v1/search", headers=AUTH, params={"q": "stress", "limit": 20})

    assert rest.status_code == 200
    assert mcp.structured_content == rest.json()
    assert {item["job_id"] for item in rest.json()["items"]} == {str(done_id)}
