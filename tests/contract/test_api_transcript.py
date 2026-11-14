"""Transcript and subtitle REST endpoints (docs/03 §2, docs/06 S10)."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from caracal.asr import Segment, Transcript, Word
from caracal.pipeline.stages import Stage
from caracal.storage import JobFiles
from tests.contract.conftest import AUTH, Harness
from tests.repo_contract import Repos


def _transcript() -> Transcript:
    words = (
        Word(text="hello", start=0.0, end=0.4, prob=0.9, space_before=False),
        Word(text="world", start=0.4, end=0.8, prob=0.9, space_before=True),
    )
    segment = Segment(
        id=0,
        start=0.0,
        end=0.8,
        text="hello world",
        words=words,
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        lang="en",
    )
    return Transcript(segments=(segment,), language_code="en", dropped={})


async def _job(repos: Repos, status: str = "done") -> uuid.UUID:
    record = await repos.jobs.create(
        source_kind="direct",
        source_url="https://example.com/a.wav",
        options={"depth": "standard"},
    )
    if status != "queued":
        await repos.jobs.set_status(record.id, status)
    return record.id


def _files(h: Harness, job_id: uuid.UUID) -> JobFiles:
    files = JobFiles(h.settings.CARACAL_DATA_DIR, job_id)
    files.ensure()
    return files


def _write_transcript(h: Harness, job_id: uuid.UUID) -> JobFiles:
    files = _files(h, job_id)
    files.write_json(files.transcript_json, _transcript().to_dict())
    files.write_text(files.transcript_txt, "[00:00] hello world\n")
    files.write_json(
        files.stage(Stage.LANGID),
        {
            "status": "done",
            "reason": None,
            "output": {
                "languages": [
                    {
                        "code": "en",
                        "name": "English",
                        "share": 1.0,
                        "spans": [{"start": 0.0, "end": 0.8}],
                        "confidence": 0.93,
                    }
                ],
                "caption_agreement": None,
                "caption_lang": None,
            },
        },
    )
    return files


def _write_subtitles(h: Harness, job_id: uuid.UUID) -> JobFiles:
    files = _files(h, job_id)
    files.write_text(files.subtitle("source", "srt"), "1\n00:00:00,000 --> 00:00:00,800\nhello world\n")
    files.write_text(
        files.subtitle("source", "vtt"), "WEBVTT\n\n00:00:00.000 --> 00:00:00.800\nhello world\n"
    )
    return files


async def test_transcript_json_has_the_job_id_and_languages(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h.repos)
    _write_transcript(h, job_id)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/transcript", headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert body["job_id"] == str(job_id)
    assert body["language_code"] == "en"
    assert len(body["segments"]) == 1
    assert body["segments"][0]["text"] == "hello world"
    assert [language["code"] for language in body["languages"]] == ["en"]


async def test_transcript_txt_returns_plain_text(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h.repos)
    _write_transcript(h, job_id)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/transcript?format=txt", headers=AUTH)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "[00:00] hello world\n"


async def test_transcript_missing_while_running_is_409(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h.repos, status="running")
    _files(h, job_id)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/transcript", headers=AUTH)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_ready"


async def test_transcript_for_a_failed_job_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h.repos, status="failed")

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/transcript", headers=AUTH)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_subtitles_serve_srt_and_vtt_with_filename_and_content_type(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h.repos)
    _write_subtitles(h, job_id)

    async with h.client() as client:
        srt = await client.get(f"/v1/jobs/{job_id}/subtitles/source.srt", headers=AUTH)
        vtt = await client.get(f"/v1/jobs/{job_id}/subtitles/source.vtt", headers=AUTH)

    assert srt.status_code == 200
    assert srt.headers["content-type"] == "application/x-subrip"
    assert srt.headers["content-disposition"] == f'attachment; filename="{job_id}.source.srt"'
    assert srt.text.startswith("1\n00:00:00,000 --> ")

    assert vtt.status_code == 200
    assert vtt.headers["content-type"] == "text/vtt; charset=utf-8"
    assert vtt.headers["content-disposition"] == f'attachment; filename="{job_id}.source.vtt"'
    assert vtt.text.startswith("WEBVTT\n")


async def test_subtitles_missing_while_running_is_409(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h.repos, status="running")
    _files(h, job_id)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/subtitles/source.srt", headers=AUTH)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "job_not_ready"


async def test_unknown_subtitle_language_on_a_done_job_is_404(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h.repos)
    _write_subtitles(h, job_id)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/subtitles/de.srt", headers=AUTH)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_subtitle_for_a_failed_job_is_404(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h.repos, status="failed")

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/subtitles/source.srt", headers=AUTH)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_traversal_language_tag_is_422(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    job_id = await _job(h.repos)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}/subtitles/..%2Fx.srt", headers=AUTH)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert response.json()["error"]["details"]["filename"] == "../x.srt"


async def test_job_status_links_only_the_subtitles_that_exist(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    job_id = await _job(h.repos)
    _write_subtitles(h, job_id)

    async with h.client() as client:
        response = await client.get(f"/v1/jobs/{job_id}", headers=AUTH)

    assert response.status_code == 200
    links = response.json()["links"]["subtitles"]
    assert set(links) == {"source.srt", "source.vtt"}
    assert links["source.srt"].endswith(f"/v1/jobs/{job_id}/subtitles/source.srt")
    assert links["source.vtt"].endswith(f"/v1/jobs/{job_id}/subtitles/source.vtt")
