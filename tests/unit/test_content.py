"""Content analysis: single call, map-reduce, evidence and time post-checks, depth variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import httpx
from tests.fakes.ollama import FakeOllama, message_response, tool_call_message

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
                "start": 0.0,
                "end": 10.0,
                "title": "Introduction",
                "summary": "The first letters.",
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


def _response(arguments: object, *, name: str = "record_content") -> httpx.Response:
    return message_response(tool_call_message(arguments, name=name))


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
    assert body["tools"][0]["function"]["name"] == "record_content"
    prompt = body["messages"][1]["content"]
    assert "[seg:0] S1 00:00.0 | alpha beta gamma delta" in prompt
    assert "[meta:title] A talk" in prompt
    assert "[hint:0] a caller hint" in prompt
    assert "<<<DATA:transcript>>>" in prompt


async def test_map_reduce_windows_overlap_by_one_segment_and_reduce_sees_every_partial(
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
    reduce_arguments = _full_args(
        entities=[
            {"text": "Alpha", "entity_type": "person", "evidence": ["seg:0"]},
            {"text": "alpha", "entity_type": "person", "evidence": ["seg:1"]},
        ]
    )
    fake = FakeOllama(
        [
            *[_response(arguments, name="record_content_window") for arguments in window_arguments],
            _response(reduce_arguments),
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
    assert fake.body(-1)["tools"][0]["function"]["name"] == "record_content"
    assert analysis.llm_meta.calls == 6
    assert analysis.llm_meta.attempts == 6
    assert len(analysis.entities) == 1
    assert analysis.entities[0].text == "Alpha"
    assert analysis.entities[0].evidence == ["seg:0", "seg:1"]


async def test_chapters_are_clamped_rederived_and_overlaps_removed(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    arguments = _full_args(
        chapters=[
            {"start": 999.0, "end": 1000.0, "title": "Far away", "summary": "x", "evidence": ["seg:0"]},
            {"start": 5.0, "end": 9.0, "title": "Overlap", "summary": "y", "evidence": ["seg:0"]},
            {"start": 100.0, "end": 200.0, "title": "Late", "summary": "z", "evidence": ["seg:2"]},
            {"start": 0.0, "end": 1.0, "title": "No segment", "summary": "w", "evidence": ["meta:title"]},
        ]
    )
    fake = FakeOllama([_response(arguments)])
    transcript = _transcript(["one", "two", "three"])

    analysis = await _analyse(settings, fake, transcript)

    assert [(chapter.title, chapter.start, chapter.end) for chapter in analysis.chapters] == [
        ("Far away", 0.0, 5.0),
        ("Late", 10.0, 15.0),
    ]
    assert analysis.chapters[0].evidence == ["seg:0"]


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
