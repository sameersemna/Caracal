"""Live identity resolution on the merged real TED transcript (judge-run, never by the gate).

The transcript is the recorded two-chunk TED talk (``talk_c0`` + ``talk_c1``) merged exactly
as the pipeline merges whisper results, then labelled as one speaker. The metadata is the
real talk title, which names the speaker after the ``|`` separator. Content analysis runs
live first, because the sole-speaker metadata rule needs a ``person`` entity that the talk's
metadata actually names (docs/04 §7).
"""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from caracal.analysis.content import analyse_content
from caracal.analysis.identity import resolve_identities
from caracal.asr import Transcript
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.config import get_settings
from caracal.llm.client import LLMClient
from caracal.pipeline.stages import JobOptions
from caracal.speakers.align import speaker_stats
from tests.fakes.whisper import load_fixture

pytestmark = pytest.mark.live

_METADATA = {
    "title": "How to stay calm when you know you'll be stressed | Daniel Levitin | TED",
    "channel": "TED",
    "uploader": "TED",
    "description": "Neuroscientist Daniel Levitin offers strategies for staying calm under stress.",
}


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


async def test_live_identity_names_the_single_ted_speaker() -> None:
    settings = get_settings()
    if settings.OLLAMA_CLOUD_API_KEY is None:
        pytest.skip("OLLAMA_CLOUD_API_KEY is not configured")

    transcript = _ted_transcript()
    stats = speaker_stats(transcript)
    assert set(stats) == {"S1"}
    assert float(stats["S1"]["share"]) == pytest.approx(1.0)

    async with httpx.AsyncClient() as http:
        client = LLMClient(settings, http)
        analysis = await analyse_content(
            transcript,
            _METADATA,
            [],
            JobOptions(depth="standard"),
            settings,
            llm=client,
        )
        result = await resolve_identities(
            transcript,
            _METADATA,
            [],
            analysis.entities,
            {},
            stats,
            settings,
            llm=client,
        )

    person_names = {entity.text for entity in analysis.entities if entity.type == "person"}
    assert "Daniel Levitin" in person_names, person_names
    assert len(result.speakers) == 1
    identity = result.speakers[0]
    assert identity.label == "S1"
    assert identity.name == "Daniel Levitin"
    assert identity.confidence == pytest.approx(0.75)
    assert identity.candidates[0].sources == ["metadata"]
    assert identity.voiceprint_similarity is None
    assert identity.known_speaker_id is None
    lowered = (identity.description or "").casefold()
    for word in ("male", "female", "woman", "man"):
        assert word not in lowered.split()
