"""Live report assembly against a running ``caracal serve`` + worker (card T2.7).

The judge runs this with ``CARACAL_LIVE_BASE_URL`` pointing at the real server and a real
worker consuming the queue (GPU, whisper.cpp, pyannote, gemma4). It submits the public TED
talk, waits for the pipeline to finish, then reads ``GET /v1/jobs/{id}/report`` in both
formats and validates the JSON against the report model.
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx
import pytest

from caracal.config import Settings, get_settings
from caracal.report.model import Report
from caracal.report.render_md import md_escape

pytestmark = pytest.mark.live

_TED_URL = "https://www.youtube.com/watch?v=8jPQjjsBbIc"
_WAIT_TIMEOUT_S = 900.0
_POLL_S = 3.0


def _base_url() -> str:
    base = os.environ.get("CARACAL_LIVE_BASE_URL")
    if not base:
        pytest.skip("CARACAL_LIVE_BASE_URL is not set")
    return base.rstrip("/")


async def test_ted_report_validates_and_is_served_as_json_and_markdown() -> None:
    get_settings.cache_clear()
    settings: Settings = get_settings()
    token = settings.CARACAL_API_TOKEN.get_secret_value()
    headers = {"Authorization": f"Bearer {token}"}
    base = _base_url()

    async with httpx.AsyncClient(timeout=120.0) as client:
        created = await client.post(
            f"{base}/v1/jobs",
            headers=headers,
            json={
                "source": {"url": _TED_URL},
                "options": {"language": "en", "diarize": "on", "verify": "on", "depth": "standard"},
            },
        )
        assert created.status_code == 202, created.text
        job_id = created.json()["job_id"]

        deadline = time.monotonic() + _WAIT_TIMEOUT_S
        while True:
            status_response = await client.get(f"{base}/v1/jobs/{job_id}", headers=headers)
            assert status_response.status_code == 200, status_response.text
            status = status_response.json()
            if status["status"] in {"done", "failed", "cancelled"}:
                break
            assert time.monotonic() < deadline, status
            await asyncio.sleep(_POLL_S)

        assert status["status"] == "done", status
        report_response = await client.get(f"{base}/v1/jobs/{job_id}/report", headers=headers)
        assert report_response.status_code == 200, report_response.text
        report = Report.model_validate(report_response.json())

        markdown = await client.get(
            f"{base}/v1/jobs/{job_id}/report?format=md",
            headers=headers,
        )

    assert report.quality.stages["assemble"] == "done"
    assert report.speakers
    assert report.speakers[0].name == "Daniel Levitin"
    assert len(report.entities) >= 3
    assert any(claim.checkworthy for claim in report.claims)
    assert report.evidence_index
    assert markdown.status_code == 200, markdown.text
    assert markdown.headers["content-type"].startswith("text/markdown")
    first_line = markdown.text.splitlines()[0]
    assert first_line == f"# {md_escape(report.source.title or '')}"
