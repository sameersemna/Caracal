"""Live translation of the TED fixture into German and Arabic (judge-run, never by the gate).

Real Ollama with the real ``translate.md`` prompt over the recorded 0-120 s TED transcript.
German proves sentence context survives unit grouping ("A few years ago" -> "Vor ... Jahren");
Arabic proves RTL text is kept without injected bidi marks. Every translated cue must keep
its source cue's timings and contain text.
"""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from caracal.analysis.translate import redistribute, translate_units, units_from_cues
from caracal.asr import Transcript
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.config import get_settings
from caracal.llm.client import LLMClient
from caracal.subtitles import Cue, cues_from_transcript
from tests.fakes.whisper import load_fixture

pytestmark = pytest.mark.live


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


def _rendered(units: object, translated: list[str], lang: str) -> list[Cue]:
    cues: list[Cue] = []
    for unit, text in zip(units, translated, strict=True):  # type: ignore[arg-type]
        unit_cues, _overlong = redistribute(unit, text, lang)
        cues.extend(unit_cues)
    return cues


async def test_ted_translation_to_german_and_arabic() -> None:
    settings = get_settings()
    transcript = _ted_transcript()
    source_cues = cues_from_transcript(transcript)
    units = units_from_cues(source_cues)
    assert units

    async with httpx.AsyncClient(timeout=240.0) as http:
        llm = LLMClient(settings, http)
        german = await translate_units(
            units,
            "de",
            glossary=["Daniel Levitin", "TED"],
            summary="A talk about how to stay calm under stress.",
            llm=llm,
        )
        arabic = await translate_units(
            units,
            "ar",
            glossary=["Daniel Levitin", "TED"],
            summary="A talk about how to stay calm under stress.",
            llm=llm,
        )

    assert len(german) == len(units)
    assert len(arabic) == len(units)
    for lang, translated in (("de", german), ("ar", arabic)):
        rendered = _rendered(units, list(translated), lang)
        assert len(rendered) == len(source_cues)
        assert all(any(line.strip() for line in cue.lines) for cue in rendered)
        assert [(cue.start, cue.end) for cue in rendered] == [(cue.start, cue.end) for cue in source_cues]

    first_unit_cues = [cue for cue in sorted(units[0].cue_indices)]
    german_cues = _rendered(units, list(german), "de")
    germany_text = " ".join(" ".join(german_cues[index].lines) for index in first_unit_cues)
    assert "Vor" in germany_text
    assert "Jahren" in germany_text

    arabic_text = " ".join(" ".join(cue.lines) for cue in _rendered(units, list(arabic), "ar"))
    assert not any("\u202a" <= character <= "\u202e" for character in arabic_text)
    assert "\u200e" not in arabic_text and "\u200f" not in arabic_text
