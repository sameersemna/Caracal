"""Live Q&A and transcript search against a running ``caracal serve`` + worker (card T3.5).

The judge runs this with ``CARACAL_LIVE_BASE_URL`` pointing at the real server and a real
worker (GPU, whisper.cpp, gemma4). It submits the public TED talk, then asks the two questions
the card names, checks that ``search_transcripts`` finds the TED transcript, and — when a
German job is available — asks an English question about it to exercise keyword expansion.

Set ``CARACAL_LIVE_GERMAN_URL`` to submit a German news clip; otherwise the test looks for an
existing finished German job.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

import httpx
import pytest

from caracal.config import Settings, get_settings

pytestmark = pytest.mark.live

_TED_URL = "https://www.youtube.com/watch?v=8jPQjjsBbIc"
_WAIT_TIMEOUT_S = 1200.0
_POLL_S = 3.0
_ENGLISH_MARKERS = re.compile(
    r"\b(?:the|and|was|were|of|about|during|after|took|place|meeting|protest)\b",
    re.IGNORECASE,
)


def _base_url() -> str:
    base = os.environ.get("CARACAL_LIVE_BASE_URL")
    if not base:
        pytest.skip("CARACAL_LIVE_BASE_URL is not set")
    return base.rstrip("/")


def _headers() -> dict[str, str]:
    get_settings.cache_clear()
    settings: Settings = get_settings()
    return {"Authorization": f"Bearer {settings.CARACAL_API_TOKEN.get_secret_value()}"}


async def _submit(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    url: str,
    *,
    language: str | None,
) -> str:
    options: dict[str, Any] = {"diarize": "off", "verify": "off", "depth": "standard"}
    if language is not None:
        options["language"] = language
    response = await client.post(
        f"{base}/v1/jobs",
        headers=headers,
        json={"source": {"url": url}, "options": options},
    )
    assert response.status_code in (200, 202), response.text
    job_id = response.json()["job_id"]
    assert isinstance(job_id, str)
    return job_id


async def _wait_done(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    job_id: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while True:
        response = await client.get(f"{base}/v1/jobs/{job_id}", headers=headers)
        assert response.status_code == 200, response.text
        status = response.json()
        if status["status"] in {"done", "failed", "cancelled"}:
            return status
        assert time.monotonic() < deadline, status
        await asyncio.sleep(_POLL_S)


async def _ask(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
    job_id: str,
    question: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{base}/v1/jobs/{job_id}/ask",
        headers=headers,
        json={"question": question},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


async def _find_german_job(
    client: httpx.AsyncClient,
    base: str,
    headers: dict[str, str],
) -> str | None:
    listed = await client.get(f"{base}/v1/jobs", headers=headers, params={"status": "done", "limit": 50})
    if listed.status_code != 200:
        return None
    for item in listed.json().get("items", []):
        report = await client.get(f"{base}/v1/jobs/{item['job_id']}/report", headers=headers)
        if report.status_code != 200:
            continue
        languages = report.json().get("languages", [])
        if any(
            str(language.get("code", "")).startswith("de") and float(language.get("share", 0.0)) > 0.5
            for language in languages
        ):
            return str(item["job_id"])
    return None


async def test_ted_question_is_cited_and_search_finds_cortisol() -> None:
    base = _base_url()
    headers = _headers()
    async with httpx.AsyncClient(timeout=180.0) as client:
        job_id = await _submit(client, base, headers, _TED_URL, language="en")
        status = await _wait_done(client, base, headers, job_id)
        assert status["status"] == "done", status

        temperature = await _ask(
            client,
            base,
            headers,
            job_id,
            "What temperature was it outside when he was locked out?",
        )
        answer_text = temperature["answer"].casefold()
        assert "40" in answer_text or "forty" in answer_text
        assert temperature["confidence"] > 0
        assert temperature["citations"]
        assert any("40" in citation["text"] for citation in temperature["citations"])

        missing = await _ask(client, base, headers, job_id, "Who won the 1998 World Cup?")
        assert missing["answer"] == "Not found in the audio."
        assert missing["confidence"] == 0
        assert missing["citations"] == []

        searched = await client.get(f"{base}/v1/search", headers=headers, params={"q": "cortisol"})
        assert searched.status_code == 200, searched.text
        items = searched.json()["items"]
        assert any(item["job_id"] == job_id and "cortisol" in item["text"].lower() for item in items)


async def test_german_transcript_question_is_answered_in_english() -> None:
    base = _base_url()
    headers = _headers()
    german_url = os.environ.get("CARACAL_LIVE_GERMAN_URL")
    async with httpx.AsyncClient(timeout=180.0) as client:
        if german_url:
            job_id = await _submit(client, base, headers, german_url, language=None)
            status = await _wait_done(client, base, headers, job_id)
            assert status["status"] == "done", status
        else:
            found = await _find_german_job(client, base, headers)
            if found is None:
                pytest.skip("no German job available; set CARACAL_LIVE_GERMAN_URL")
            job_id = found

        answer = await _ask(client, base, headers, job_id, "What happened in Krakow?")
        assert answer["citations"]
        assert answer["answer"] != "Not found in the audio."
        assert _ENGLISH_MARKERS.search(answer["answer"]), answer["answer"]
