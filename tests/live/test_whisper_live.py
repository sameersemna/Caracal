"""Live whisper.cpp check (``@pytest.mark.live``, run by the judge, never by the gate).

Uses the real server from ``WHISPER_URL`` with the recorded public-domain ``jfk.wav``.
stdout must stay empty because it is the MCP stdio channel.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from caracal.asr.whisper import WhisperClient
from caracal.config import Settings, get_settings

pytestmark = pytest.mark.live

AUDIO = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "jfk.wav"


@pytest.fixture(autouse=True)
def isolated_environment() -> Iterator[None]:
    """Live tests need the real environment/``.env`` rather than the unit-suite isolation."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


async def test_live_jfk_transcription(capsys: pytest.CaptureFixture[str]) -> None:
    settings: Settings = get_settings()
    async with httpx.AsyncClient() as http:
        client = WhisperClient(settings.WHISPER_URL, http, timeout_s=settings.WHISPER_TIMEOUT_S)
        assert await client.health() is True
        result = await client.transcribe(AUDIO, language="auto")

    text = " ".join(segment.text for segment in result.segments).lower()
    assert result.language_code == "en"
    assert result.language_prob > 0.8
    assert len(result.segments) >= 3
    assert "country" in text
    assert all(segment.words for segment in result.segments)
    assert capsys.readouterr().out == ""
