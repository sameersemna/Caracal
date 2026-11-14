"""REST and MCP Q&A plus cross-job transcript search (card T3.5, docs/03 §2-§3).

The scripted LLM is injected into the shared ``JobService``, so both transports exercise the
same retrieval, citation checks and response models — no LLM request leaves the process.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from mcp.client import Client

from caracal.db.repo import SegmentRow
from caracal.llm.client import CallMeta, ToolResult
from caracal.pipeline.stages import JobOptions
from tests.contract.conftest import AUTH, Harness
from tests.contract.test_api_report import _report

URL = "https://example.com/talk"


class _ScriptedLlm:
    """``LLMClient`` stand-in: keyword expansion and one repeated answer payload."""

    def __init__(self, *, keywords: list[str] | None = None, answer: dict[str, Any] | None = None) -> None:
        self.keywords = list(keywords or [])
        self.answer = answer or {"answer": "", "confidence": 0.0, "citations": []}
        self.calls: list[str] = []

    async def call_tool(self, **kwargs: Any) -> ToolResult[Any]:
        model_cls = kwargs["model_cls"]
        tool_name = kwargs["tool_name"]
        self.calls.append(tool_name)
        payload = {"keywords": self.keywords} if tool_name == "qa_keywords" else self.answer
        return ToolResult(
            value=model_cls.model_validate(payload),
            meta=CallMeta(
                model="scripted",
                endpoint="local",
                attempts=1,
                coerced=False,
                fallback_used=False,
                duration_s=0.0,
                mode_used="json",
            ),
        )


async def _job(h: Harness, *, status: str = "done", title: str = "A talk") -> uuid.UUID:
    record = await h.repos.jobs.create(
        source_kind="platform",
        source_url=URL,
        title=title,
        options=JobOptions().model_dump(),
    )
    if status != "queued":
        await h.repos.jobs.set_status(record.id, status)
    return record.id


def _rows(
    job_id: uuid.UUID,
    texts: list[str],
    speakers: list[str | None] | None = None,
) -> list[SegmentRow]:
    speakers = speakers or ["S1"] * len(texts)
    return [
        SegmentRow(
            job_id=job_id,
            seg_id=index,
            start_s=float(index) * 2.0,
            end_s=float(index) * 2.0 + 1.5,
            speaker=speakers[index],
            lang="en",
            text=text,
        )
        for index, text in enumerate(texts)
    ]


async def _seeded_job(
    h: Harness,
    *,
    status: str = "done",
    texts: list[str] | None = None,
    speakers: list[str | None] | None = None,
    title: str = "A talk",
) -> uuid.UUID:
    job_id = await _job(h, status=status, title=title)
    await h.repos.segments.replace_for_job(
        job_id,
        _rows(job_id, texts or ["alpha one", "alpha two", "alpha three"], speakers),
    )
    if status == "done":
        await h.repos.reports.put(job_id, _report(job_id).model_dump(mode="json"), "1.0")
    return job_id


def _use_llm(h: Harness, *, keywords: list[str] | None = None, answer: dict[str, Any]) -> None:
    h.app.state.service.llm = _ScriptedLlm(keywords=keywords, answer=answer)


async def _mcp(h: Harness, name: str, arguments: dict[str, Any]) -> Any:
    async with Client(h.app.state.mcp_server) as client:
        return await client.call_tool(name, arguments)


def _cited() -> dict[str, Any]:
    return {
        "answer": "The talk is about stress.",
        "confidence": 0.8,
        "citations": [{"seg_id": "seg:1", "start": 999.0, "text": "fabricated text"}],
    }


async def test_ask_returns_answer_with_real_citation(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _seeded_job(h, texts=["stress one", "stress two", "stress three"])
    _use_llm(h, answer=_cited())

    async with h.client() as client:
        response = await client.post(
            f"/v1/jobs/{job_id}/ask",
            headers=AUTH,
            json={"question": "stress"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["answer"] == "The talk is about stress."
    assert payload["confidence"] == pytest.approx(0.8)
    assert payload["evidence"] == ["seg:1"]
    assert payload["citations"] == [
        {"seg_id": "seg:1", "start": 2.0, "text": "stress two"},
    ]


@pytest.mark.parametrize("status", ["queued", "running", "failed"])
async def test_ask_requires_a_done_job(
    make_harness: Callable[..., Harness],
    status: str,
) -> None:
    h = make_harness()
    job_id = await _seeded_job(h, status=status)

    async with h.client() as client:
        response = await client.post(
            f"/v1/jobs/{job_id}/ask",
            headers=AUTH,
            json={"question": "stress"},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_ready"


async def test_ask_unknown_job_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()

    async with h.client() as client:
        response = await client.post(
            f"/v1/jobs/{uuid.uuid4()}/ask",
            headers=AUTH,
            json={"question": "stress"},
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_ask_rejects_bad_questions(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _seeded_job(h)

    async with h.client() as client:
        empty = await client.post(f"/v1/jobs/{job_id}/ask", headers=AUTH, json={"question": ""})
        blank = await client.post(f"/v1/jobs/{job_id}/ask", headers=AUTH, json={"question": "   "})
        too_long = await client.post(
            f"/v1/jobs/{job_id}/ask",
            headers=AUTH,
            json={"question": "x" * 1001},
        )

    for response in (empty, blank, too_long):
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_input"


async def test_ask_without_a_valid_citation_is_not_found(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _seeded_job(h)
    _use_llm(
        h,
        answer={
            "answer": "France won the 1998 World Cup.",
            "confidence": 0.99,
            "citations": [{"seg_id": "seg:99"}],
            "evidence": ["seg:99", "meta:summary"],
        },
    )

    async with h.client() as client:
        response = await client.post(
            f"/v1/jobs/{job_id}/ask",
            headers=AUTH,
            json={"question": "Who won the 1998 World Cup?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "answer": "Not found in the audio.",
        "confidence": 0.0,
        "citations": [],
        "evidence": [],
    }


async def test_search_returns_only_done_jobs_with_speaker(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    done_id = await _seeded_job(
        h,
        texts=["alpha one", "alpha two"],
        speakers=["S1", None],
        title="Finished",
    )
    running_id = await _seeded_job(
        h,
        status="running",
        texts=["alpha three"],
        title="Still running",
    )

    async with h.client() as client:
        response = await client.get("/v1/search", headers=AUTH, params={"q": "alpha"})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {"items"}
    assert [item["job_id"] for item in payload["items"]] == [str(done_id), str(done_id)]
    assert all(item["job_id"] != str(running_id) for item in payload["items"])
    first = payload["items"][0]
    assert first["title"] == "Finished"
    assert first["seg_id"] == 0
    assert first["start"] == 0.0
    assert first["speaker"] == "S1"
    assert first["text"] == "alpha one"
    assert first["rank"] > 0
    assert payload["items"][1]["speaker"] is None

    async with h.client() as client:
        limited = await client.get("/v1/search", headers=AUTH, params={"q": "alpha", "limit": 1})
    assert len(limited.json()["items"]) == 1


async def test_search_rejects_blank_query_and_bad_limit(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    await _seeded_job(h)

    async with h.client() as client:
        for params in ({"q": ""}, {"q": "   "}, {}, {"q": "alpha", "limit": 51}, {"q": "alpha", "limit": 0}):
            response = await client.get("/v1/search", headers=AUTH, params=params)
            assert response.status_code == 422, (params, response.text)
            assert response.json()["error"]["code"] == "invalid_input"


async def test_mcp_ask_matches_rest(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _seeded_job(h)
    _use_llm(h, answer=_cited())

    async with h.client() as client:
        rest = await client.post(
            f"/v1/jobs/{job_id}/ask",
            headers=AUTH,
            json={"question": "stress"},
        )
    mcp = await _mcp(h, "ask_about_audio", {"job_id": str(job_id), "question": "stress"})

    assert rest.status_code == 200
    assert mcp.is_error is False
    assert mcp.structured_content == rest.json()


async def test_mcp_ask_and_search_errors(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    queued_id = await _seeded_job(h, status="queued")

    not_done = await _mcp(h, "ask_about_audio", {"job_id": str(queued_id), "question": "stress"})
    assert not_done.is_error is True
    assert "job_not_ready" in "".join(block.text for block in not_done.content if block.type == "text")

    blank = await _mcp(h, "search_transcripts", {"query": "   "})
    assert blank.is_error is True
    assert "invalid_input" in "".join(block.text for block in blank.content if block.type == "text")

    bad_limit = await _mcp(h, "search_transcripts", {"query": "alpha", "limit": 51})
    assert bad_limit.is_error is True
    assert "invalid_input" in "".join(block.text for block in bad_limit.content if block.type == "text")


async def test_mcp_search_returns_items(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    done_id = await _seeded_job(h)
    await _seeded_job(h, status="running", texts=["alpha ignored"], title="Running")

    result = await _mcp(h, "search_transcripts", {"query": "alpha", "limit": 20})

    assert result.is_error is False
    payload = result.structured_content
    assert set(payload) == {"items"}
    assert {item["job_id"] for item in payload["items"]} == {str(done_id)}
    assert payload["items"][0]["speaker"] == "S1"
