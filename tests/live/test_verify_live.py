"""Live cross-verification checks (judge-run, never by the gate).

Real Brave/Exa through the real router, real Redis with a throwaway prefix and the real
Ollama LLM. The checkpoints are the two M3 acceptance cases: the All Ears English host
("Lindsay" -> "Lindsay McMahon") and the TED claim about -40 being where Celsius and
Fahrenheit meet.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import httpx
import pytest
from redis.asyncio import Redis

from caracal.analysis.verify import VerifyResult, verify
from caracal.asr import Segment, Transcript
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.config import Settings, get_settings
from caracal.llm.client import LLMClient
from caracal.queue import Keys, make_redis
from caracal.report.model import Claim, SpeakerCandidate, SpeakerIdentity
from caracal.search import BraveProvider, ExaProvider, SearchRouter
from tests.fakes.whisper import load_fixture

pytestmark = pytest.mark.live


def _keys(settings: Settings) -> Keys:
    return Keys(f"{settings.REDIS_PREFIX}live-verify-{uuid.uuid4().hex[:8]}:")


async def _cleanup(redis: Redis, keys: Keys) -> None:
    found = [key async for key in redis.scan_iter(match=f"{keys.prefix}*")]
    if found:
        await redis.delete(*found)


def _router(
    settings: Settings,
    redis: Redis,
    keys: Keys,
    http: httpx.AsyncClient,
) -> SearchRouter:
    providers = []
    if settings.BRAVE_API_KEY is not None:
        providers.append(BraveProvider(settings.BRAVE_API_KEY.get_secret_value(), http))
    if settings.EXA_API_KEY is not None:
        providers.append(ExaProvider(settings.EXA_API_KEY.get_secret_value(), http))
    return SearchRouter(settings, redis, keys, providers)


def _transcript(rows: list[tuple[str, float, float, str]]) -> Transcript:
    return Transcript(
        segments=tuple(
            Segment(
                id=index,
                start=start,
                end=end,
                text=text,
                words=(),
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                lang="en",
                speaker=speaker,
            )
            for index, (speaker, start, end, text) in enumerate(rows)
        ),
        language_code="en",
        dropped={},
    )


def _ted_transcript() -> Transcript:
    chunks = [
        Chunk(index=0, start_s=0.0, end_s=61.0),
        Chunk(index=1, start_s=60.0, end_s=120.0),
    ]
    merged = merge_chunks(
        [
            parse_whisper_response(load_fixture("talk_c0")),
            parse_whisper_response(load_fixture("talk_c1")),
        ],
        chunks,
    )
    segments = tuple(
        replace(
            segment,
            speaker="S1",
            words=tuple(replace(word, speaker="S1") for word in segment.words),
        )
        for segment in merged.segments
    )
    return Transcript(segments=segments, language_code=merged.language_code, dropped=dict(merged.dropped))


async def _run(
    settings: Settings,
    *,
    transcript: Transcript,
    claims: list[Claim],
    speakers: list[SpeakerIdentity],
) -> tuple[VerifyResult, Redis, Keys]:
    redis: Redis = make_redis(settings)
    keys = _keys(settings)
    job_id = uuid.uuid4()

    async def budget_exhausted() -> bool:
        raw = await redis.get(keys.search_budget(job_id))
        return False if raw is None else int(raw) > settings.SEARCH_MAX_QUERIES_PER_JOB

    async with httpx.AsyncClient(timeout=120.0) as http:
        llm = LLMClient(settings, http)
        result = await verify(
            transcript=transcript,
            claims=claims,
            speakers=speakers,
            flags=[],
            key_quotes=[],
            metadata={
                "title": "All Ears English Podcast",
                "channel": "All Ears English",
                "uploader": "All Ears English",
            },
            depth="standard",
            settings=settings,
            llm=llm,
            search=_router(settings, redis, keys, http),
            job_id=job_id,
            budget_exhausted=budget_exhausted,
        )
    return result, redis, keys


@pytest.mark.live
async def test_live_all_ears_english_resolves_lindsay_mcmahon() -> None:
    settings = get_settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")
    if settings.BRAVE_API_KEY is None and settings.EXA_API_KEY is None:
        pytest.skip("a search provider is required")

    speaker = SpeakerIdentity(
        label="S1",
        talk_time_s=600.0,
        share=1.0,
        first_seen_s=0.0,
        name=None,
        confidence=0.0,
        candidates=[
            SpeakerCandidate(
                name="Lindsay",
                confidence=0.55,
                sources=["self_intro"],
                evidence=["seg:0"],
            )
        ],
        evidence=["seg:0"],
    )
    transcript = _transcript([("S1", 0.0, 4.0, "I'm Lindsay, and this is the All Ears English podcast.")])

    result, redis, keys = await _run(
        settings,
        transcript=transcript,
        claims=[],
        speakers=[speaker],
    )
    await _cleanup(redis, keys)
    await redis.aclose()

    assert result.speakers[0].name == "Lindsay McMahon"
    assert result.speakers[0].confidence >= 0.6
    candidate = result.speakers[0].candidates[0]
    assert "web" in candidate.sources
    assert any(evidence.startswith("web:") for evidence in candidate.evidence)
    assert result.verifications[0].verdict == "supported"


@pytest.mark.live
async def test_live_ted_claim_about_minus_forty_is_supported() -> None:
    settings = get_settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")
    if settings.BRAVE_API_KEY is None and settings.EXA_API_KEY is None:
        pytest.skip("a search provider is required")

    claim = Claim(
        id="c1",
        text="Minus 40 is where the Celsius and Fahrenheit scales meet.",
        speaker="S1",
        seg_ids=["seg:0"],
        kind="fact",
        checkworthy=True,
        evidence=["seg:0"],
    )

    result, redis, keys = await _run(
        settings,
        transcript=_ted_transcript(),
        claims=[claim],
        speakers=[],
    )
    await _cleanup(redis, keys)
    await redis.aclose()

    assert len(result.verifications) == 1
    verification = result.verifications[0]
    assert verification.target == "claim:c1"
    assert verification.verdict == "supported"
    assert verification.sources
    assert all(source.evidence_id.startswith("web:") for source in verification.sources)
    assert all(source.url.startswith("http") for source in verification.sources)
