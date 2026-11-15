"""Live Ollama checks with the recorded README transcript (judge-run, never by the gate).

The cloud run proves the tool-call + coercion recipe against the real ``gemma4:31b``. The
forced-fallback run points the primary at an unreachable port so the client must switch to
the local daemon and its local model. The content-analysis run reproduces the measured
reserved-name finding: an aliased ``entity_type`` plus normalised evidence ids must make the
realistic task valid on at least 5 of 6 parallel calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
from pydantic import BaseModel, Field, SecretStr

from caracal.asr import Segment
from caracal.config import Route, Settings, get_settings
from caracal.llm.client import LLMClient, ToolResult
from caracal.llm.prompts import load_prompt, render_transcript_lines, wrap_data
from caracal.llm.schema import DEFAULT_SYNONYMS
from tests.fakes.ollama import load_fixture

pytestmark = pytest.mark.live

_WHISPER_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "whisper" / "talk_c0.json"


class Entity(BaseModel):
    """A report field named ``type`` is exposed to the LLM as ``entity_type``."""

    text: str
    type: Literal["person", "org", "place", "date", "other"] = Field(validation_alias="entity_type")
    evidence: list[str]


class Out(BaseModel):
    language: str
    speaker_name: str | None = None
    entities: list[Entity]
    confidence: float = Field(ge=0.0, le=1.0)


class ContentEntity(BaseModel):
    name: str
    type: Literal["person", "org", "place", "date", "work", "other"] = Field(validation_alias="entity_type")
    evidence: list[str]


class Claim(BaseModel):
    text: str
    checkworthy: bool
    evidence: list[str]


class ContentAnalysis(BaseModel):
    summary: str
    entities: list[ContentEntity]
    claims: list[Claim]


def _prompt() -> str:
    transcript = load_fixture("README")["transcript"]
    return load_prompt("probe", transcript=wrap_data("transcript", transcript))


def _talk_segments() -> list[Segment]:
    payload = json.loads(_WHISPER_FIXTURE.read_text(encoding="utf-8"))
    return [
        Segment(
            id=int(item["id"]),
            start=float(item["start"]),
            end=float(item["end"]),
            text=str(item["text"]).strip(),
            words=(),
            avg_logprob=float(item.get("avg_logprob", 0.0)),
            no_speech_prob=float(item.get("no_speech_prob", 0.0)),
            lang="en",
            speaker="S1",
        )
        for item in payload["segments"]
    ]


def _content_prompt() -> tuple[str, set[str]]:
    segments = _talk_segments()
    instruction = (
        "Analyse the transcript in the DATA block below. Record a two-sentence summary, every "
        "named entity with the evidence id that supports it, and every checkworthy factual claim "
        "with its evidence ids. Only state what the evidence supports. Cite evidence ids (like "
        "seg:12, meta:title) for every item. If you are not sure, say so with a low confidence "
        "instead of guessing. Never invent names, dates or sources.\n\n"
    )
    return instruction + wrap_data("transcript", render_transcript_lines(segments)), {
        f"seg:{segment.id}" for segment in segments
    }


async def _analyse(settings: Settings, *, route: Route = Route.DEFAULT) -> ToolResult[Out]:
    async with httpx.AsyncClient() as http:
        client = LLMClient(settings, http)
        return await client.call_tool(
            prompt=_prompt(),
            system="You are a careful audio analyst.",
            model_cls=Out,
            tool_name="record_analysis",
            tool_description="Record the language, speaker and entities found in the transcript.",
            synonyms=DEFAULT_SYNONYMS,
            route=route,
        )


def _default_chain(settings: Settings) -> list[str]:
    """The pre-routing chain: cloud with the key, otherwise the local ``-cloud`` proxy, then local."""
    if settings.OLLAMA_CLOUD_API_KEY is not None:
        return [f"cloud:{settings.LLM_MODEL}", f"local:{settings.LLM_FALLBACK_MODEL}"]
    return [f"local:{settings.LLM_LOCAL_PROXY_MODEL}", f"local:{settings.LLM_FALLBACK_MODEL}"]


def _settings(chain: list[str] | None = None, **overrides: Any) -> Settings:
    """A ``Settings`` whose LLM routing is explicit, so the owner's ``.env`` cannot leak into a test.

    Every ``LLM_ROUTE_*`` variable and the allow-list are overwritten instead of inherited; the
    remaining values still come from the owner's real configuration (endpoints, keys, models).
    """
    base = get_settings()
    refs = list(chain) if chain is not None else _default_chain(base)
    routing: dict[str, Any] = {"LLM_ALLOWED_MODELS": refs}
    for route in Route:
        routing[f"LLM_ROUTE_{route.name}"] = list(refs)
    routing.update(overrides)
    return Settings(_env_file=None, **{**base.model_dump(), **routing})


def _assert_merkel_and_berlin(result: ToolResult[Out]) -> None:
    assert result.value.speaker_name is not None
    assert "Merkel" in result.value.speaker_name
    assert any(entity.type == "place" and "Berlin" in entity.text for entity in result.value.entities), (
        result.value.entities
    )
    assert all(entity.evidence for entity in result.value.entities)


async def test_live_cloud_tool_call_is_valid_after_coercion() -> None:
    settings = _settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")

    result = await _analyse(settings)

    _assert_merkel_and_berlin(result)
    assert result.meta.endpoint == "cloud"
    assert result.meta.model == settings.LLM_MODEL
    assert result.meta.fallback_used is False
    assert result.meta.attempts >= 1


async def test_live_forced_fallback_uses_local_ollama() -> None:
    base = get_settings()
    settings = _settings(
        chain=[f"cloud:{base.LLM_MODEL}", f"local:{base.LLM_FALLBACK_MODEL}"],
        OLLAMA_CLOUD_URL="http://127.0.0.1:9",
        OLLAMA_CLOUD_API_KEY=SecretStr("invalid-cloud-key"),
    )

    result = await _analyse(settings)

    _assert_merkel_and_berlin(result)
    assert result.meta.endpoint == "local"
    assert result.meta.model == base.LLM_FALLBACK_MODEL
    assert result.meta.fallback_used is True


async def test_live_content_analysis_is_valid_and_cites_only_transcript_ids() -> None:
    settings = _settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")

    prompt, valid_ids = _content_prompt()

    async with httpx.AsyncClient() as http:
        client = LLMClient(settings, http)

        async def analyse_once() -> ToolResult[ContentAnalysis]:
            return await client.call_tool(
                prompt=prompt,
                system="You are a careful audio analyst.",
                model_cls=ContentAnalysis,
                tool_name="record_content",
                tool_description="Record the summary, entities and claims found in the transcript.",
                synonyms=DEFAULT_SYNONYMS,
            )

        results = await asyncio.gather(*(analyse_once() for _ in range(6)), return_exceptions=True)

    valid = [result for result in results if isinstance(result, ToolResult)]
    assert len(valid) >= 5, results

    cited: set[str] = set()
    for result in valid:
        for entity in result.value.entities:
            cited.update(entity.evidence)
        for claim in result.value.claims:
            cited.update(claim.evidence)

    assert any(result.value.entities for result in valid)
    assert cited
    assert cited <= valid_ids, sorted(cited - valid_ids)


async def test_live_every_route_answers_with_its_preferred_model() -> None:
    """Each task route is answered by the first ref of its configured chain (docs/02)."""
    settings = _settings()

    for route in Route:
        result = await _analyse(settings, route=route)

        _assert_merkel_and_berlin(result)
        assert result.meta.route is route
        assert result.meta.model_ref == settings.llm_route_chain(route)[0].normalised
        assert result.meta.refs_tried == 1
        assert result.meta.fallback_used is False


async def test_live_qa_route_falls_back_to_the_local_model() -> None:
    """A QA chain whose first ref cannot serve falls over to local ``gemma4:31b``."""
    routed = _settings(chain=["local:nonexistent-model:1b", "local:gemma4:31b"])

    result = await _analyse(routed, route=Route.QA)

    _assert_merkel_and_berlin(result)
    assert result.meta.model_ref == "local:gemma4:31b"
    assert result.meta.endpoint == "local"
    assert result.meta.refs_tried == 2
    assert result.meta.fallback_used is True
