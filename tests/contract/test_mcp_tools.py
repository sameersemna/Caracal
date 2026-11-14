"""In-memory MCP tool contract: every tool, async analysis and error envelopes (card 15).

The tools run against the same fake repositories, fake Redis and temporary data directory as
the REST contract tests: no network, Postgres or real Redis is involved.
"""

from __future__ import annotations

import asyncio
import base64
import math
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client

from caracal.analysis.translate import TRANSLATE_REQUEST_FILENAME
from caracal.mcp.server import build_mcp
from caracal.pipeline.stages import JobOptions, Stage
from caracal.queue import publish_event
from caracal.services import JobService
from caracal.storage import JobFiles
from tests.contract.conftest import AUTH, Harness
from tests.contract.test_api_report import _report

URL = "https://example.com/talk"


def _service(h: Harness) -> JobService:
    return JobService(
        h.settings,
        h.repos,
        h.redis,
        h.keys,
        h.queue,
        h.settings.CARACAL_DATA_DIR,
    )


def _write_artifacts(h: Harness, job_id: uuid.UUID) -> JobFiles:
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
    files.write_json(
        files.stage(Stage.NORMALIZE),
        {"status": "done", "reason": None, "output": {"stats": {"duration_s": 12.5}}},
    )
    files.write_text(files.subtitle("source", "srt"), "1\n00:00:00,000 --> 00:00:12,500\nhello\n")
    files.write_text(files.subtitle("source", "vtt"), "WEBVTT\n\n00:00:00.000 --> 00:00:12.500\nhello\n")
    return files


async def _job(h: Harness, *, status: str = "queued", title: str | None = "A talk") -> uuid.UUID:
    record = await h.repos.jobs.create(
        source_kind="platform",
        source_url=URL,
        title=title,
        options=JobOptions().model_dump(),
    )
    if status != "queued":
        await h.repos.jobs.set_status(record.id, status)
    return record.id


def _text(result: Any) -> str:
    return "".join(block.text for block in result.content if block.type == "text")


async def test_get_job_returns_the_shared_status_model(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h, status="done")
    _write_artifacts(h, job_id)

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool("get_job", {"job_id": str(job_id)})

    assert result.is_error is False
    payload = result.structured_content
    assert payload is not None
    assert payload["job_id"] == str(job_id)
    assert payload["status"] == "done"
    assert payload["source"] == {"kind": "platform", "url": URL, "title": "A talk"}
    assert "error" not in payload
    assert payload["links"]["report"].endswith(f"/v1/jobs/{job_id}/report")
    assert next(stage["name"] for stage in payload["stages"]) == "ingest"
    assert set(payload["links"]["subtitles"]) == {"source.srt", "source.vtt"}


async def test_list_jobs_returns_items_and_filters(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    failed_id = await _job(h, status="failed")
    done_id = await _job(h, status="done")

    async with Client(build_mcp(_service(h), h.settings)) as client:
        everything = await client.call_tool("list_jobs", {})
        failed = await client.call_tool("list_jobs", {"status": "failed"})

    assert everything.structured_content is not None
    assert {item["job_id"] for item in everything.structured_content["items"]} == {
        str(failed_id),
        str(done_id),
    }
    assert failed.structured_content is not None
    assert [item["job_id"] for item in failed.structured_content["items"]] == [str(failed_id)]


async def test_every_registered_tool_declares_an_object_output_schema(
    make_harness: Callable[..., Harness],
) -> None:
    """MCP rejects a non-object ``structuredContent`` over real transports (verdict 20).

    Tool descriptions and inputs are not enough: every tool must publish an object-rooted
    ``outputSchema`` so a real client can validate the result. The expected-name set keeps
    the loop from going vacuous if a tool is ever unregistered.
    """
    h = make_harness()

    async with Client(build_mcp(_service(h), h.settings)) as client:
        listed = await client.list_tools()

    names = {tool.name for tool in listed.tools}
    assert names == {
        "analyze_audio",
        "get_job",
        "list_jobs",
        "get_transcript",
        "get_report",
        "translate_job",
        "list_speakers",
        "enroll_speaker",
        "delete_speaker",
        "cancel_job",
    }
    for tool in listed.tools:
        schema = tool.output_schema
        assert isinstance(schema, dict), tool.name
        assert schema.get("type") == "object", (tool.name, schema)


async def _store_report(h: Harness, job_id: uuid.UUID) -> None:
    await h.repos.reports.put(job_id, _report(job_id).model_dump(mode="json"), "1.0")


async def test_get_report_returns_the_full_report(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h, status="done")
    await _store_report(h, job_id)

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool("get_report", {"job_id": str(job_id)})

    assert result.is_error is False
    assert result.structured_content == _report(job_id).model_dump(mode="json")


async def test_get_report_filters_top_level_sections(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h, status="done")
    await _store_report(h, job_id)

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool(
            "get_report",
            {"job_id": str(job_id), "sections": ["summary", "context"]},
        )

    assert result.is_error is False
    assert set(result.structured_content or {}) == {"summary", "context"}
    assert result.structured_content["summary"]["one_line"] == "A short talk."


async def test_get_report_markdown_returns_format_and_content(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h, status="done")
    await _store_report(h, job_id)

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool(
            "get_report",
            {"job_id": str(job_id), "format": "markdown"},
        )

    assert result.is_error is False
    assert result.structured_content is not None
    assert result.structured_content["format"] == "markdown"
    assert result.structured_content["content"].startswith("# A talk")


async def test_get_report_rejects_an_unknown_section(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h, status="done")
    await _store_report(h, job_id)

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool(
            "get_report",
            {"job_id": str(job_id), "sections": ["not_a_section"]},
        )

    assert result.is_error is True
    assert "invalid_input" in _text(result)
    assert "not_a_section" in _text(result)


async def test_get_report_before_assembly_is_job_not_ready(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h, status="running")

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool("get_report", {"job_id": str(job_id)})

    assert result.is_error is True
    assert "job_not_ready" in _text(result)


async def test_get_transcript_all_formats(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h, status="done")
    _write_artifacts(h, job_id)

    async with Client(build_mcp(_service(h), h.settings)) as client:
        text = await client.call_tool("get_transcript", {"job_id": str(job_id)})
        structured = await client.call_tool("get_transcript", {"job_id": str(job_id), "format": "json"})
        srt = await client.call_tool("get_transcript", {"job_id": str(job_id), "format": "srt"})
        vtt = await client.call_tool("get_transcript", {"job_id": str(job_id), "format": "vtt"})

    assert text.structured_content == {
        "format": "text",
        "lang": "source",
        "content": "[00:00] my fellow americans\n",
    }
    assert structured.structured_content is not None
    assert structured.structured_content["job_id"] == str(job_id)
    assert structured.structured_content["segments"][0]["text"] == "my fellow americans"
    assert [language["code"] for language in structured.structured_content["languages"]] == ["en"]
    assert srt.structured_content is not None
    assert srt.structured_content["content"].startswith("1\n00:00:00,000 --> ")
    assert vtt.structured_content is not None
    assert vtt.structured_content["content"].startswith("WEBVTT")


async def test_analyze_audio_url_returns_a_pending_job(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool(
            "analyze_audio",
            {"url": URL, "wait_seconds": 0},
        )

    assert result.is_error is False
    payload = result.structured_content
    assert payload is not None
    assert payload["status"] == "queued"
    assert payload["hint"] == "call get_job / get_transcript later"
    assert payload["progress"] == {"pct": 0, "stage": None, "message": None}
    job = await h.repos.jobs.get(uuid.UUID(payload["job_id"]))
    assert job is not None
    assert job.source_url == URL


async def test_analyze_audio_upload_id_consumes_the_upload(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    upload = await h.repos.uploads.create(
        filename="talk.wav", size_bytes=3, sha256="sha-abc", path="/uploads/talk.wav"
    )

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool(
            "analyze_audio",
            {"upload_id": str(upload.id), "wait_seconds": 0},
        )

    assert result.is_error is False
    payload = result.structured_content
    assert payload is not None
    consumed = await h.repos.uploads.get(upload.id)
    assert consumed is not None
    assert consumed.consumed_by == uuid.UUID(payload["job_id"])


async def test_analyze_audio_base64_stores_a_tiny_wav(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    wav = b"RIFF$\x00\x00\x00WAVEfmt "
    encoded = base64.b64encode(wav).decode("ascii")

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool(
            "analyze_audio",
            {"audio_base64": encoded, "filename": "talk.wav", "wait_seconds": 0},
        )

    assert result.is_error is False
    payload = result.structured_content
    assert payload is not None
    jobs, _ = await h.repos.jobs.list()
    assert len(jobs) == 1
    assert jobs[0].source_kind == "upload"
    record = await h.repos.uploads.get(jobs[0].upload_id or uuid.uuid4())
    assert record is not None
    assert record.filename == "talk.wav"
    assert Path(record.path).read_bytes() == wav  # noqa: ASYNC240 - test-local file


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"url": URL, "upload_id": str(uuid.uuid4())},
        {"url": URL, "audio_base64": "AAAA"},
    ],
)
async def test_analyze_audio_requires_exactly_one_source(
    make_harness: Callable[..., Harness], arguments: dict[str, object]
) -> None:
    h = make_harness()

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool("analyze_audio", arguments)

    assert result.is_error is True
    assert "invalid_input" in _text(result)


async def test_analyze_audio_waits_reports_progress_and_returns_done(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    seen: list[tuple[float, float | None, str | None]] = []

    async def worker() -> None:
        # Let analyze_audio create the job and subscribe to its event channel first.
        await asyncio.sleep(0.05)
        while True:
            jobs, _ = await h.repos.jobs.list()
            if jobs:
                job = jobs[0]
                await h.repos.jobs.set_progress(
                    job.id,
                    {"pct": 50, "stage": "transcribe", "message": "half way"},
                )
                await publish_event(
                    h.redis,
                    h.keys,
                    job.id,
                    {
                        "type": "stage",
                        "stage": "transcribe",
                        "status": "running",
                        "pct": 50,
                        "message": "half way",
                    },
                )
                _write_artifacts(h, job.id)
                await h.repos.jobs.set_title(job.id, "A talk")
                await h.repos.reports.put(job.id, _report(job.id).model_dump(mode="json"), "1.0")
                await h.repos.jobs.set_status(job.id, "done")
                await publish_event(h.redis, h.keys, job.id, {"type": "end"})
                return
            await asyncio.sleep(0.005)

    async def progress_callback(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))

    worker_task = asyncio.create_task(worker())
    try:
        async with Client(build_mcp(_service(h), h.settings)) as client:
            result = await client.call_tool(
                "analyze_audio",
                {"url": URL, "wait_seconds": 10},
                progress_callback=progress_callback,
            )
    finally:
        await worker_task

    assert result.is_error is False
    payload = result.structured_content
    assert payload is not None
    assert payload["status"] == "done"
    assert payload["summary"]["title"] == "A talk"
    assert payload["summary"]["duration_s"] == 12.5
    assert payload["summary"]["transcript_preview"].startswith("[00:00] my fellow americans")
    assert payload["report_summary"]["title"] == "A talk"
    assert payload["report_summary"]["languages"] == ["en"]
    assert payload["report_summary"]["summary"] == "A short talk."
    assert payload["links"]["report"].endswith("/report")
    assert seen == [(50.0, 100.0, "half way")]


async def test_analyze_audio_oversize_base64_is_rejected_without_decoding(
    make_harness: Callable[..., Harness],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = make_harness()
    calls: list[str] = []

    def spy(data: str, *args: Any, **kwargs: Any) -> bytes:
        calls.append(data)
        raise AssertionError("b64decode must not be reached for an oversize payload")

    monkeypatch.setattr("caracal.services.base64.b64decode", spy)
    server = build_mcp(_service(h), h.settings, max_inline_bytes=1024)

    async with Client(server) as client:
        result = await client.call_tool(
            "analyze_audio",
            {"audio_base64": "A" * 2000, "wait_seconds": 0},
        )

    assert result.is_error is True
    assert "too_large" in _text(result)
    assert calls == []
    jobs, _ = await h.repos.jobs.list()
    assert jobs == []


async def test_tool_errors_carry_the_code_and_never_a_traceback(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    running_id = await _job(h, status="running")

    async with Client(build_mcp(_service(h), h.settings)) as client:
        unknown = await client.call_tool("get_job", {"job_id": str(uuid.uuid4())})
        malformed = await client.call_tool("get_job", {"job_id": "not-a-uuid"})
        not_ready = await client.call_tool("get_transcript", {"job_id": str(running_id)})

    unknown_text = _text(unknown)
    assert unknown.is_error is True
    assert "not_found" in unknown_text
    assert "Traceback" not in unknown_text
    assert "Error executing tool get_job" in unknown_text

    assert malformed.is_error is True
    assert "invalid_input" in _text(malformed)

    assert not_ready.is_error is True
    assert "job_not_ready" in _text(not_ready)


async def test_cancel_job_cancels_and_deletes(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    active_id = await _job(h, status="queued")
    done_id = await _job(h, status="done")
    files = _write_artifacts(h, done_id)

    async with Client(build_mcp(_service(h), h.settings)) as client:
        cancel = await client.call_tool("cancel_job", {"job_id": str(active_id)})
        delete = await client.call_tool(
            "cancel_job",
            {"job_id": str(done_id), "delete_artifacts": True},
        )

    assert cancel.structured_content == {"job_id": str(active_id), "status": "cancelling"}
    assert await h.redis.exists(h.keys.cancel(active_id)) == 1
    assert delete.structured_content == {"job_id": str(done_id), "status": "deleted"}
    assert await h.repos.jobs.get(done_id) is None
    assert not files.root.exists()


def _unit(similarity: float) -> list[float]:
    rest = math.sqrt(max(0.0, 1.0 - similarity * similarity))
    return [similarity, rest, *([0.0] * 254)]


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


async def test_speaker_tools_round_trip_and_match_rest(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h, status="done")
    _write_diarize(h, job_id, {"S1": _unit(0.97)})

    async with Client(build_mcp(_service(h), h.settings)) as client:
        enrolled = await client.call_tool(
            "enroll_speaker",
            {
                "job_id": str(job_id),
                "speaker_label": "S1",
                "name": "Daniel  Levitin",
                "notes": "Neuroscientist",
            },
        )
        assert enrolled.is_error is False
        created = enrolled.structured_content
        assert created is not None
        assert created["name"] == "Daniel  Levitin"
        assert created["notes"] == "Neuroscientist"
        assert created["samples"] == 1
        assert uuid.UUID(created["id"])

        async with h.client() as rest_client:
            rest = await rest_client.get("/v1/speakers", headers=AUTH)
        listed = await client.call_tool("list_speakers", {})
        assert rest.status_code == 200
        assert listed.structured_content == rest.json() == {"items": [created]}

        deleted = await client.call_tool("delete_speaker", {"speaker_id": created["id"]})
        assert deleted.is_error is False
        assert deleted.structured_content == {"deleted": True, "voiceprints_removed": 1}

        empty = await client.call_tool("list_speakers", {})
        assert empty.structured_content == {"items": []}

    async with h.client() as rest_client:
        assert (await rest_client.get("/v1/speakers", headers=AUTH)).json() == {"items": []}


async def test_speaker_tool_errors_carry_the_error_code(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    queued_id = await _job(h, status="queued")
    _write_diarize(h, queued_id, {"S1": _unit(0.9)})
    done_id = await _job(h, status="done")
    _write_diarize(h, done_id, {"S1": _unit(0.9)})

    async with Client(build_mcp(_service(h), h.settings)) as client:
        not_ready = await client.call_tool(
            "enroll_speaker",
            {"job_id": str(queued_id), "speaker_label": "S1", "name": "Too Early"},
        )
        unknown_label = await client.call_tool(
            "enroll_speaker",
            {"job_id": str(done_id), "speaker_label": "S9", "name": "Wrong Label"},
        )
        bad_job = await client.call_tool(
            "enroll_speaker",
            {"job_id": "not-a-uuid", "speaker_label": "S1", "name": "Bad Job"},
        )
        bad_speaker = await client.call_tool("delete_speaker", {"speaker_id": "not-a-uuid"})

    assert not_ready.is_error is True
    assert "job_not_ready" in _text(not_ready)
    assert unknown_label.is_error is True
    assert "not_found" in _text(unknown_label)
    assert bad_job.is_error is True
    assert "invalid_input: job_id must be a UUID" in _text(bad_job)
    assert bad_speaker.is_error is True
    assert "invalid_input: speaker_id must be a UUID" in _text(bad_speaker)


async def test_translate_job_queues_the_requested_languages(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    done_id = await _job(h, status="done")
    running_id = await _job(h, status="running")

    async with Client(build_mcp(_service(h), h.settings)) as client:
        result = await client.call_tool(
            "translate_job",
            {"job_id": str(done_id), "target_langs": ["de", "ar"]},
        )
        not_ready = await client.call_tool(
            "translate_job",
            {"job_id": str(running_id), "target_langs": ["de"]},
        )
        bad_code = await client.call_tool(
            "translate_job",
            {"job_id": str(done_id), "target_langs": ["english"]},
        )

    assert result.is_error is False
    payload = result.structured_content
    assert payload is not None
    assert payload["job_id"] == str(done_id)
    assert payload["status"] == "queued"
    entries = await h.redis.xrange(h.keys.jobs_stream())
    assert len(entries) == 1
    assert entries[0][1][b"kind"] == b"translate"

    files = JobFiles(h.settings.CARACAL_DATA_DIR, done_id)
    assert files.read_json(files.root / TRANSLATE_REQUEST_FILENAME) == {"target_langs": ["de", "ar"]}

    assert not_ready.is_error is True
    assert "job_not_ready" in _text(not_ready)
    assert bad_code.is_error is True
    assert "invalid_input" in _text(bad_code)


async def test_speaker_tool_descriptions_state_the_biometric_consent_rule(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()

    async with Client(build_mcp(_service(h), h.settings)) as client:
        listed = await client.list_tools()

    descriptions = {tool.name: tool.description or "" for tool in listed.tools}
    for name in ("list_speakers", "enroll_speaker", "delete_speaker"):
        assert "biometric" in descriptions[name].lower(), name
        assert "consent" in descriptions[name].lower(), name
