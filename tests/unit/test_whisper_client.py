"""WhisperClient against the recorded fixtures through an ``httpx`` MockTransport.

The fake is a real HTTP boundary: the client under test builds the multipart request,
handles statuses/transport errors and parses the real ``verbose_json`` bodies.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from tests.fakes.whisper import FakeWhisper, html_response, json_response

from caracal.asr.whisper import WhisperClient
from caracal.errors import CaracalError, ErrorCode

AUDIO = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "audio" / "jfk.wav"
AUDIO_BYTES = AUDIO.read_bytes()


async def test_request_is_explicit_multipart_and_parses_real_fixture() -> None:
    fake = FakeWhisper.from_fixtures("jfk")
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        result = await client.transcribe(AUDIO)

    request = fake.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://whisper.test/inference"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")

    form = fake.form()
    assert form["response_format"] == "verbose_json"
    assert form["language"] == "auto"  # the server default is en: always send it explicitly
    assert form["translate"] == "false"
    assert form["temperature"] == "0.0"
    upload = fake.parts()["file"]
    assert upload.filename == "jfk.wav"
    assert upload.content_type == "audio/wav"
    assert upload.content == AUDIO_BYTES

    assert result.language_code == "en"
    assert result.language_prob == pytest.approx(0.9352319240570068)
    assert result.duration == 11.0
    assert len(result.segments) == 3
    assert result.segments[0].text == "And so my fellow Americans,"
    assert result.segments[0].words[0].text == "And"
    assert all(word.text == word.text.strip() for segment in result.segments for word in segment.words)
    assert result.segments[0].lang == "en"


async def test_forced_language_uses_real_detection_not_the_echo() -> None:
    fake = FakeWhisper.from_fixtures("talk_c0_forced_de")
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        result = await client.transcribe(AUDIO, language="de")

    assert fake.form()["language"] == "de"
    assert result.language_code == "en"
    assert result.language_prob == pytest.approx(0.9970652461051941)


async def test_silence_yields_no_segments_and_low_confidence() -> None:
    fake = FakeWhisper.from_fixtures("silence20")
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        result = await client.transcribe(AUDIO)

    assert result.segments == ()
    assert result.language_prob == pytest.approx(0.3209926187992096)
    assert result.language_code == "en"
    assert result.duration == 20.0


@pytest.mark.parametrize("failure", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
async def test_transport_failures_are_retryable_dependency_down(failure: Exception) -> None:
    fake = FakeWhisper([failure])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        with pytest.raises(CaracalError) as excinfo:
            await client.transcribe(AUDIO)
    assert excinfo.value.code is ErrorCode.DEPENDENCY_DOWN
    assert excinfo.value.retryable is True


async def test_5xx_is_retryable_dependency_down() -> None:
    fake = FakeWhisper([httpx.Response(503, text="overloaded")])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        with pytest.raises(CaracalError) as excinfo:
            await client.transcribe(AUDIO)
    assert excinfo.value.code is ErrorCode.DEPENDENCY_DOWN
    assert excinfo.value.retryable is True


async def test_4xx_is_internal_and_not_retryable() -> None:
    fake = FakeWhisper([httpx.Response(400, text="bad request")])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        with pytest.raises(CaracalError) as excinfo:
            await client.transcribe(AUDIO)
    assert excinfo.value.code is ErrorCode.INTERNAL
    assert excinfo.value.retryable is False


@pytest.mark.parametrize(
    "response",
    [
        html_response("<html>proxy error</html>"),
        json_response({"language": "english"}),
        json_response({"language": "english", "segments": "nope"}),
    ],
)
async def test_malformed_bodies_are_internal(response: httpx.Response) -> None:
    fake = FakeWhisper([response])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        with pytest.raises(CaracalError) as excinfo:
            await client.transcribe(AUDIO)
    assert excinfo.value.code is ErrorCode.INTERNAL
    assert excinfo.value.retryable is False


@pytest.mark.parametrize("language", ["english", "EN", "e", "de-DE", ""])
async def test_invalid_language_never_reaches_the_server(language: str) -> None:
    fake = FakeWhisper([])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        with pytest.raises(CaracalError) as excinfo:
            await client.transcribe(AUDIO, language=language)
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert fake.requests == []


async def test_request_happens_inside_the_lock() -> None:
    events: list[str] = []

    class RecordingLock:
        async def __aenter__(self) -> None:
            events.append("enter")

        async def __aexit__(self, *_exc: object) -> None:
            events.append("exit")

    @asynccontextmanager
    async def lock() -> AsyncIterator[None]:
        async with RecordingLock() as entered:
            yield entered

    fake = FakeWhisper.from_fixtures("jfk", events=events)
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0, lock=lock)
        await client.transcribe(AUDIO)
    assert events == ["enter", "request", "exit"]


async def test_language_falls_back_to_the_detected_name() -> None:
    payload = {
        "segments": [
            {
                "id": 0,
                "text": "hello",
                "start": 0.0,
                "end": 1.0,
                "words": [],
                "avg_logprob": 0.0,
                "no_speech_prob": 0.0,
            }
        ],
        "duration": 1.0,
        "detected_language": "english",
        "detected_language_probability": 0.7,
    }
    fake = FakeWhisper([json_response(payload)])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        result = await client.transcribe(AUDIO)
    assert result.language_probs == {}
    assert result.language_code == "en"
    assert result.language_prob == pytest.approx(0.7)


async def test_health_reports_the_root_status() -> None:
    fake = FakeWhisper([httpx.Response(200)])
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        assert await client.health() is True
    assert str(fake.requests[0].url) == "http://whisper.test/"

    fake_down = FakeWhisper([httpx.Response(500)])
    async with httpx.AsyncClient(transport=fake_down.transport()) as http:
        assert await WhisperClient("http://whisper.test", http, timeout_s=5.0).health() is False

    fake_error = FakeWhisper([httpx.ConnectError("offline")])
    async with httpx.AsyncClient(transport=fake_error.transport()) as http:
        assert await WhisperClient("http://whisper.test", http, timeout_s=5.0).health() is False


async def test_word_space_before_records_whisper_whitespace() -> None:
    """The parser keeps whisper's leading-space signal; the first word of a segment has none."""
    fake = FakeWhisper.from_fixtures("jfk")
    async with httpx.AsyncClient(transport=fake.transport()) as http:
        client = WhisperClient("http://whisper.test", http, timeout_s=5.0)
        result = await client.transcribe(AUDIO)

    first = result.segments[0]
    assert [(word.text, word.space_before) for word in first.words] == [
        ("And", False),
        ("so", True),
        ("my", True),
        ("fellow", True),
        ("Americans", True),
        (",", False),  # attached punctuation: no space before it
    ]
    assert all(segment.words[0].space_before is False for segment in result.segments)
    assert result.segments[1].words[1].space_before is True
    assert first.to_dict()["words"][1]["space_before"] is True
