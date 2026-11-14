"""Content analysis: single call, map-reduce, evidence and time post-checks, depth variants."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from tests.fakes.ollama import FakeOllama, json_response

from caracal.analysis import Depth
from caracal.analysis.content import ContentAnalysis, analyse_content
from caracal.asr import Segment, Transcript
from caracal.config import Settings
from caracal.llm.client import LLMClient
from caracal.pipeline.stages import JobOptions

MakeSettings = Callable[..., Settings]
_META = {"title": "A talk"}


async def _no_sleep(_seconds: float) -> None:
    return None


def _segment(index: int, text: str, *, start: float | None = None, length: float = 5.0) -> Segment:
    begin = float(index) * length if start is None else start
    return Segment(
        id=index,
        start=begin,
        end=begin + length,
        text=text,
        words=(),
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        lang="en",
        speaker="S1",
    )


def _transcript(texts: Sequence[str], *, length: float = 5.0) -> Transcript:
    return Transcript(
        segments=tuple(_segment(index, text, length=length) for index, text in enumerate(texts)),
        language_code="en",
        dropped={},
    )


def _full_args(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "summary": {
            "one_line": "A short talk about letters.",
            "short": "The speaker explains a few Greek letters.",
            "detailed": "The speaker walks through Greek letters and their order.",
            "evidence": ["seg:0"],
        },
        "context": {
            "genre": "lecture",
            "setting": "studio",
            "audience": "students",
            "purpose": "explain",
            "tone": ["calm"],
            "recorded_when": None,
            "recorded_where": None,
            "evidence": ["meta:title"],
            "confidence": 0.8,
        },
        "chapters": [
            {
                "title": "Introduction",
                "summary": "The first letters.",
                "first_seg": "seg:0",
                "last_seg": "seg:1",
                "evidence": ["seg:0", "seg:1"],
            }
        ],
        "topics": [{"label": "letters", "weight": 0.9, "evidence": ["seg:1 S1"]}],
        "entities": [
            {
                "text": "Alpha",
                "entity_type": "Location",
                "mentions": ["seg:0 S1"],
                "evidence": ["seg:0"],
            }
        ],
        "key_quotes": [
            {
                "text": "Alpha beta gamma delta",
                "seg_ids": ["seg:0"],
                "why_notable": "opening",
            }
        ],
        "claims": [
            {
                "id": "x",
                "text": "Alpha is first.",
                "speaker": "S1",
                "seg_ids": ["seg:0"],
                "kind": "fact",
                "checkworthy": True,
                "evidence": ["seg:0"],
            }
        ],
    }
    arguments.update(overrides)
    return arguments


def _response(arguments: object) -> httpx.Response:
    """A JSON-mode reply; content analysis never uses tool mode (card 25)."""
    return json_response(arguments)


def _client(settings: Settings, fake: FakeOllama) -> tuple[httpx.AsyncClient, LLMClient]:
    http = httpx.AsyncClient(transport=fake.transport())
    return http, LLMClient(settings, http, sleep=_no_sleep)


async def _analyse(
    settings: Settings,
    fake: FakeOllama,
    transcript: Transcript,
    *,
    depth: Depth = "standard",
) -> ContentAnalysis:
    http, client = _client(settings, fake)
    try:
        return await analyse_content(
            transcript,
            _META,
            ["a caller hint"],
            JobOptions(depth=depth),
            settings,
            llm=client,
        )
    finally:
        await http.aclose()


async def test_single_call_produces_validated_analysis_with_normalised_ids(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_response(_full_args())])
    transcript = _transcript(["alpha beta gamma delta", "epsilon zeta eta theta", "iota kappa lambda mu"])

    analysis = await _analyse(settings, fake, transcript)

    assert analysis.summary.one_line == "A short talk about letters."
    assert analysis.context.genre == "lecture"
    assert analysis.context.confidence == 0.8
    assert len(analysis.chapters) == 1
    assert (analysis.chapters[0].start, analysis.chapters[0].end) == (0.0, 10.0)
    assert analysis.entities[0].type == "place"
    assert analysis.entities[0].evidence == ["seg:0"]
    assert analysis.entities[0].mentions == ["seg:0"]
    assert analysis.topics[0].evidence == ["seg:1"]
    assert analysis.key_quotes[0].seg_id == "seg:0"
    assert analysis.key_quotes[0].speaker == "S1"
    assert analysis.claims[0].evidence == ["seg:0"]
    assert analysis.claims[0].checkworthy is True
    assert analysis.llm_meta.calls == 1
    assert analysis.llm_meta.attempts == 1
    assert analysis.llm_meta.coerced is True

    assert len(fake.requests) == 1
    body = fake.body()
    assert body["format"] == "json"
    assert "tools" not in body
    schema = json.loads(body["messages"][0]["content"].split("(field names and allowed values):\n", 1)[1])
    assert set(schema["properties"]) >= {"summary", "context", "chapters", "entities", "claims"}
    prompt = body["messages"][1]["content"]
    assert "[seg:0] S1 00:00.0 | alpha beta gamma delta" in prompt
    assert "[meta:title] A talk" in prompt
    assert "[hint:0] a caller hint" in prompt
    assert "<<<DATA:transcript>>>" in prompt


async def test_map_reduce_windows_overlap_by_one_segment_and_reduce_merges_deterministically(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(LLM_CONTEXT_CHARS=110)
    transcript = _transcript(["alpha beta gamma delta"] * 6)
    window_arguments = [
        {
            "entities": [
                {
                    "text": f"Entity{index}",
                    "entity_type": "person",
                    "mentions": [f"seg:{index}"],
                    "evidence": [f"seg:{index}"],
                }
            ],
            "topics": [{"label": f"topic{index}", "weight": 0.5, "evidence": [f"seg:{index}"]}],
        }
        for index in range(5)
    ]
    fake = FakeOllama(
        [
            *[_response(arguments) for arguments in window_arguments],
            _response(_full_args()),
        ]
    )

    analysis = await _analyse(settings, fake, transcript)

    assert len(fake.requests) == 6
    prompts = [request.content for request in fake.requests]
    first = fake.body(0)["messages"][1]["content"]
    second = fake.body(1)["messages"][1]["content"]
    assert "[seg:0]" in first and "[seg:1]" in first
    assert "[seg:2]" not in first
    assert "[seg:1]" in second and "[seg:2]" in second
    assert "[seg:0]" not in second
    assert len(prompts) == 6

    reduce_prompt = fake.body(-1)["messages"][1]["content"]
    for index in range(5):
        assert f"Entity{index}" in reduce_prompt
    assert "<<<DATA:partials>>>" in reduce_prompt
    assert fake.body(-1)["format"] == "json"
    assert "tools" not in fake.body(-1)
    # The reduce call has a small schema: summary and context only (verdict 25 D1).
    schema = json.loads(
        fake.body(-1)["messages"][0]["content"].split("(field names and allowed values):\n", 1)[1]
    )
    assert set(schema["properties"]) == {"summary", "context"}

    assert analysis.llm_meta.calls == 6
    assert analysis.llm_meta.attempts == 6
    assert [entity.text for entity in analysis.entities] == [
        "Entity0",
        "Entity1",
        "Entity2",
        "Entity3",
        "Entity4",
    ]
    assert analysis.entities[0].evidence == ["seg:0"]
    assert [topic.label for topic in analysis.topics] == [
        "topic0",
        "topic1",
        "topic2",
        "topic3",
        "topic4",
    ]


async def test_deterministic_merge_dedupes_entities_topics_chapters_quotes_and_claims(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(LLM_CONTEXT_CHARS=110)
    transcript = _transcript(["alpha beta gamma delta"] * 6)
    window_arguments = [
        {
            "chapters": [
                {
                    "title": "Introduction",
                    "summary": "Opening.",
                    "first_seg": "seg:0",
                    "last_seg": "seg:0",
                    "evidence": ["seg:0"],
                }
            ],
            "topics": [{"label": "Letters", "weight": 0.3, "evidence": ["seg:0"]}],
            "entities": [
                {
                    "text": "Alpha",
                    "entity_type": "person",
                    "mentions": ["seg:0"],
                    "evidence": ["seg:0"],
                }
            ],
            "key_quotes": [
                {"text": "alpha beta gamma delta", "seg_ids": ["seg:0"], "why_notable": "opening"}
            ],
            "claims": [
                {
                    "text": "Alpha is first.",
                    "speaker": "S1",
                    "seg_ids": ["seg:0"],
                    "kind": "fact",
                    "checkworthy": True,
                    "evidence": ["seg:0"],
                }
            ],
        },
        {
            "chapters": [
                {
                    "title": "Introduction.",
                    "summary": "Still opening.",
                    "first_seg": "seg:1",
                    "last_seg": "seg:1",
                    "evidence": ["seg:1"],
                }
            ],
            "topics": [{"label": "letters", "weight": 0.9, "evidence": ["seg:1"]}],
            "entities": [
                {
                    "text": "alpha",
                    "entity_type": "place",
                    "mentions": ["seg:1"],
                    "evidence": ["seg:1"],
                }
            ],
            "key_quotes": [{"text": "Alpha beta gamma delta", "seg_ids": ["seg:1"], "why_notable": "repeat"}],
            "claims": [
                {
                    "text": "alpha is first!",
                    "speaker": "",
                    "seg_ids": ["seg:1"],
                    "kind": "fact",
                    "checkworthy": False,
                    "evidence": ["seg:1"],
                }
            ],
        },
        {
            "chapters": [
                {
                    "title": "Middle",
                    "summary": "The middle.",
                    "first_seg": "seg:2",
                    "last_seg": "seg:2",
                    "evidence": ["seg:2"],
                }
            ],
            "topics": [{"label": "Greek", "weight": 0.5, "evidence": ["seg:2"]}],
            "entities": [
                {
                    "text": "Alpha",
                    "entity_type": "person",
                    "mentions": ["seg:2"],
                    "evidence": ["seg:2"],
                }
            ],
        },
    ]
    fake = FakeOllama(
        [
            *[_response(arguments) for arguments in window_arguments],
            *[_response({}) for _ in range(2)],
            _response(_full_args()),
        ]
    )

    analysis = await _analyse(settings, fake, transcript)

    # Entity: one entry, majority type (person 2 vs place 1), evidence unioned.
    assert len(analysis.entities) == 1
    assert analysis.entities[0].text == "Alpha"
    assert analysis.entities[0].type == "person"
    assert analysis.entities[0].evidence == ["seg:0", "seg:1", "seg:2"]
    assert analysis.entities[0].mentions == ["seg:0", "seg:1", "seg:2"]

    # Topic: one per normalised label, weight = max, evidence unioned.
    assert [(topic.label, topic.weight) for topic in analysis.topics] == [
        ("Letters", 0.9),
        ("Greek", 0.5),
    ]
    assert analysis.topics[0].evidence == ["seg:0", "seg:1"]

    # Chapters: time order, adjacent near-identical titles merged.
    assert [(chapter.title, chapter.start, chapter.end) for chapter in analysis.chapters] == [
        ("Introduction", 0.0, 10.0),
        ("Middle", 10.0, 15.0),
    ]
    assert analysis.chapters[0].evidence == ["seg:0", "seg:1"]

    # Quote: deduped by normalised text, matched against the best cited segment.
    assert len(analysis.key_quotes) == 1
    assert analysis.key_quotes[0].seg_id == "seg:0"

    # Claim: deduped by normalised text, evidence unioned, checkworthy preserved.
    assert len(analysis.claims) == 1
    assert analysis.claims[0].text == "Alpha is first."
    assert analysis.claims[0].evidence == ["seg:0", "seg:1"]
    assert analysis.claims[0].checkworthy is True


async def test_music_mode_analyses_metadata_only_and_flags_non_speech_audio(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_response(_full_args())])
    transcript = _transcript(["🎵🎵🎵"] * 3)

    http, client = _client(settings, fake)
    try:
        analysis = await analyse_content(
            transcript,
            _META,
            ["a caller hint"],
            JobOptions(),
            settings,
            llm=client,
            music=True,
        )
    finally:
        await http.aclose()

    assert len(fake.requests) == 1
    prompt = fake.body()["messages"][1]["content"]
    assert "[meta:title] A talk" in prompt
    assert "[seg:0]" not in prompt
    assert "non-speech audio" in prompt
    assert analysis.key_quotes == []
    assert analysis.claims == []
    assert analysis.chapters == []
    assert [flag.message for flag in analysis.review_flags] == ["non-speech audio"]
    assert analysis.review_flags[0].severity == "warn"


async def test_chapters_are_derived_from_first_and_last_segment_ids(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    arguments = _full_args(
        chapters=[
            {
                "start": 999.0,
                "end": 1000.0,
                "title": "Far away",
                "summary": "x",
                "first_seg": "seg:0",
                "last_seg": "seg:0",
                "evidence": ["seg:0"],
            },
            {
                "title": "Overlap",
                "summary": "y",
                "first_seg": "seg:0",
                "last_seg": "seg:0",
                "evidence": ["seg:0"],
            },
            {
                "title": "Late",
                "summary": "z",
                "first_seg": "seg:2",
                "last_seg": "seg:2",
                "evidence": ["seg:2"],
            },
            {
                "title": "No segment",
                "summary": "w",
                "first_seg": "seg:9",
                "last_seg": "seg:9",
                "evidence": ["meta:title"],
            },
        ]
    )
    fake = FakeOllama([_response(arguments)])
    transcript = _transcript(["one", "two", "three"])

    analysis = await _analyse(settings, fake, transcript)

    # The model's numeric times are ignored: only first_seg/last_seg decide the bounds.
    assert [(chapter.title, chapter.start, chapter.end) for chapter in analysis.chapters] == [
        ("Far away", 0.0, 5.0),
        ("Late", 10.0, 15.0),
    ]
    assert analysis.chapters[0].evidence == ["seg:0"]


async def test_chapter_with_reversed_segment_ids_uses_the_cited_span(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    arguments = _full_args(
        chapters=[
            {
                "title": "Reversed",
                "summary": "s",
                "first_seg": "seg:1",
                "last_seg": "seg:0",
                "evidence": ["seg:0", "seg:1"],
            }
        ]
    )
    fake = FakeOllama([_response(arguments)])
    transcript = _transcript(["one", "two", "three"])

    analysis = await _analyse(settings, fake, transcript)

    assert [(chapter.start, chapter.end) for chapter in analysis.chapters] == [(0.0, 10.0)]
    assert analysis.chapters[0].evidence == ["seg:0", "seg:1"]


async def test_quotes_are_kept_only_when_they_fuzzy_match_the_cited_segments(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    arguments = _full_args(
        key_quotes=[
            {
                "text": "Epsilon zeta eta theta",
                "seg_ids": ["seg:1 S1"],
                "why_notable": "exact",
            },
            {
                "text": "totally unrelated words",
                "seg_ids": ["seg:2"],
                "why_notable": "invented",
            },
            {"text": "Alpha", "seg_ids": ["garbage"], "why_notable": "bad id"},
        ]
    )
    fake = FakeOllama([_response(arguments)])
    transcript = _transcript(["alpha beta", "epsilon zeta eta theta", "iota kappa"])

    analysis = await _analyse(settings, fake, transcript)

    assert len(analysis.key_quotes) == 1
    quote = analysis.key_quotes[0]
    assert quote.seg_id == "seg:1"
    assert quote.speaker == "S1"
    assert quote.start == 5.0
    assert quote.text == "Epsilon zeta eta theta"
    assert quote.why_notable == "exact"


async def test_quote_spanning_two_segments_matches_the_joined_text(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    arguments = _full_args(
        key_quotes=[
            {
                "text": "epsilon zeta eta theta",
                "seg_ids": ["seg:1", "seg:2"],
                "why_notable": "spans two segments",
            }
        ]
    )
    fake = FakeOllama([_response(arguments)])
    transcript = _transcript(["alpha beta", "epsilon zeta", "eta theta"])

    analysis = await _analyse(settings, fake, transcript)

    assert len(analysis.key_quotes) == 1
    assert analysis.key_quotes[0].seg_id == "seg:1"
    assert analysis.key_quotes[0].start == 5.0


async def test_items_without_a_resolving_evidence_id_are_dropped(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    arguments = _full_args(
        entities=[
            {"text": "No evidence", "entity_type": "person", "evidence": ["garbage"]},
            {"text": "Valid", "entity_type": "person", "evidence": ["seg:2 S1"]},
            {"text": "Missing", "entity_type": "person", "evidence": ["seg:99"]},
        ],
        topics=[
            {"label": "kept", "weight": 0.4, "evidence": ["seg:1."]},
            {"label": "dropped", "weight": 0.4, "evidence": ["seg:99"]},
        ],
        claims=[
            {
                "text": "Missing evidence",
                "speaker": "S1",
                "seg_ids": ["seg:99"],
                "checkworthy": True,
                "evidence": ["seg:99"],
            },
            {
                "text": "Kept evidence",
                "speaker": "S1",
                "seg_ids": ["seg:0."],
                "checkworthy": True,
                "evidence": ["seg:0."],
            },
        ],
    )
    fake = FakeOllama([_response(arguments)])
    transcript = _transcript(["one", "two", "three"])

    analysis = await _analyse(settings, fake, transcript)

    assert [entity.text for entity in analysis.entities] == ["Valid"]
    assert analysis.entities[0].evidence == ["seg:2"]
    assert [topic.label for topic in analysis.topics] == ["kept"]
    assert analysis.topics[0].evidence == ["seg:1"]
    assert [claim.text for claim in analysis.claims] == ["Kept evidence"]
    assert analysis.claims[0].evidence == ["seg:0"]
    assert analysis.claims[0].seg_ids == ["seg:0"]


async def test_quick_depth_requests_summary_and_context_only(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_response(_full_args())])
    transcript = _transcript(["alpha beta", "epsilon zeta"])

    analysis = await _analyse(settings, fake, transcript, depth="quick")

    assert analysis.summary.one_line == "A short talk about letters."
    assert analysis.context.genre == "lecture"
    assert analysis.chapters == []
    assert analysis.topics == []
    assert analysis.entities == []
    assert analysis.key_quotes == []
    assert analysis.claims == []
    prompt = fake.body()["messages"][1]["content"]
    assert "Do not produce chapters." in prompt
    assert "at most 0" in prompt


async def test_deep_depth_allows_fifteen_claims_and_asks_for_fine_chapters(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    claims = [
        {
            "id": f"c{index}",
            "text": f"Claim number {index}.",
            "speaker": "S1",
            "seg_ids": ["seg:0"],
            "kind": "fact",
            "checkworthy": True,
            "evidence": ["seg:0"],
        }
        for index in range(16)
    ]
    transcript = _transcript(["alpha beta", "epsilon zeta"])

    deep_fake = FakeOllama([_response(_full_args(claims=claims))])
    deep = await _analyse(settings, deep_fake, transcript, depth="deep")
    assert len(deep.claims) == 15
    assert "2-4 minutes" in deep_fake.body()["messages"][1]["content"]

    standard_fake = FakeOllama([_response(_full_args(claims=claims))])
    standard = await _analyse(settings, standard_fake, transcript, depth="standard")
    assert len(standard.claims) == 8
    assert "5-10 minutes" in standard_fake.body()["messages"][1]["content"]


async def test_completeness_reasks_only_the_missing_sections_and_merges_them(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    first = _full_args(chapters=[], entities=[])
    missing = {
        "chapters": [
            {
                "title": "Opening",
                "summary": "The first letters.",
                "first_seg": "seg:0",
                "last_seg": "seg:2",
                "evidence": ["seg:0", "seg:2"],
            }
        ],
        "entities": [
            {
                "text": "Twenty",
                "entity_type": "other",
                "mentions": ["seg:3"],
                "evidence": ["seg:3"],
            }
        ],
    }
    fake = FakeOllama([_response(first), _response(missing)])
    transcript = _transcript([f"line number {index}" for index in range(20)])

    analysis = await _analyse(settings, fake, transcript)

    assert len(fake.requests) == 2
    reask = fake.body(1)
    assert reask["format"] == "json"
    assert "tools" not in reask
    schema = json.loads(reask["messages"][0]["content"].split("(field names and allowed values):\n", 1)[1])
    assert set(schema["properties"]) == {"chapters", "entities"}
    assert "Fill EVERY field" in reask["messages"][1]["content"]
    assert "These sections are still missing: chapters, entities" in reask["messages"][1]["content"]

    assert [chapter.title for chapter in analysis.chapters] == ["Opening"]
    assert (analysis.chapters[0].start, analysis.chapters[0].end) == (0.0, 15.0)
    assert [entity.text for entity in analysis.entities] == ["Twenty"]
    assert analysis.llm_meta.reasked == ["chapters", "entities"]
    assert analysis.llm_meta.completeness == {
        "chapters": 1,
        "entities": 1,
        "claims": 1,
        "topics": 1,
    }
    assert analysis.llm_meta.calls == 2


async def test_completeness_is_skipped_for_short_transcripts(make_settings: MakeSettings) -> None:
    settings = make_settings()
    fake = FakeOllama([_response(_full_args(chapters=[], entities=[]))])
    transcript = _transcript(["alpha beta", "epsilon zeta", "iota kappa"])

    analysis = await _analyse(settings, fake, transcript)

    assert len(fake.requests) == 1
    assert analysis.llm_meta.reasked == []
    assert analysis.llm_meta.completeness["chapters"] == 0
    assert analysis.llm_meta.completeness["entities"] == 0


async def test_completeness_skips_the_reask_when_every_section_is_present(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_response(_full_args())])
    transcript = _transcript([f"line number {index}" for index in range(20)])

    analysis = await _analyse(settings, fake, transcript)

    assert len(fake.requests) == 1
    assert analysis.llm_meta.reasked == []
    assert analysis.llm_meta.completeness == {
        "chapters": 1,
        "entities": 1,
        "claims": 1,
        "topics": 1,
    }
