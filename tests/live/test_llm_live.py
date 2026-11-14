"""Live Ollama checks with the recorded README transcript (judge-run, never by the gate).

The cloud run proves the tool-call + coercion recipe against the real ``gemma4:31b``. The
forced-fallback run points the primary at an unreachable port so the client must switch to
the local daemon and its local model.
"""

from __future__ import annotations

from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, Field, SecretStr

from caracal.config import Settings, get_settings
from caracal.llm.client import LLMClient, ToolResult
from caracal.llm.prompts import load_prompt, wrap_data
from caracal.llm.schema import DEFAULT_SYNONYMS
from tests.fakes.ollama import load_fixture

pytestmark = pytest.mark.live


class Entity(BaseModel):
    text: str
    type: Literal["person", "org", "place", "date", "other"]
    evidence: list[str]


class Out(BaseModel):
    language: str
    speaker_name: str | None = None
    entities: list[Entity]
    confidence: float = Field(ge=0.0, le=1.0)


def _prompt() -> str:
    transcript = load_fixture("README")["transcript"]
    return load_prompt("probe", transcript=wrap_data("transcript", transcript))


async def _analyse(settings: Settings) -> ToolResult[Out]:
    async with httpx.AsyncClient() as http:
        client = LLMClient(settings, http)
        return await client.call_tool(
            prompt=_prompt(),
            system="You are a careful audio analyst.",
            model_cls=Out,
            tool_name="record_analysis",
            tool_description="Record the language, speaker and entities found in the transcript.",
            synonyms=DEFAULT_SYNONYMS,
        )


def _assert_merkel_and_berlin(result: ToolResult[Out]) -> None:
    assert result.value.speaker_name is not None
    assert "Merkel" in result.value.speaker_name
    assert any(entity.type == "place" and "Berlin" in entity.text for entity in result.value.entities), (
        result.value.entities
    )
    assert all(entity.evidence for entity in result.value.entities)


async def test_live_cloud_tool_call_is_valid_after_coercion() -> None:
    settings = get_settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")

    result = await _analyse(settings)

    _assert_merkel_and_berlin(result)
    assert result.meta.endpoint == "cloud"
    assert result.meta.model == settings.LLM_MODEL
    assert result.meta.fallback_used is False
    assert result.meta.attempts >= 1


async def test_live_forced_fallback_uses_local_ollama() -> None:
    settings = get_settings()
    forced = settings.model_copy(
        update={
            "OLLAMA_CLOUD_URL": "http://127.0.0.1:9",
            "OLLAMA_CLOUD_API_KEY": SecretStr("invalid-cloud-key"),
        }
    )

    result = await _analyse(forced)

    _assert_merkel_and_berlin(result)
    assert result.meta.endpoint == "local"
    assert result.meta.model == settings.LLM_FALLBACK_MODEL
    assert result.meta.fallback_used is True
