"""Identity resolution: mention verification, source scores, fusion and promotion (docs/04 §7).

Every LLM answer is scripted through ``FakeOllama`` in JSON mode; nothing here touches the
network. The tests assert concrete confidences and evidence so removing any part of the
resolution (verification, scoring, fusion, promotion or sanitisation) makes them fail.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from tests.fakes.ollama import FakeOllama, json_response

from caracal.analysis.identity import (
    IdentityResult,
    VoiceprintMatch,
    resolve_identities,
)
from caracal.asr import Segment, Transcript
from caracal.config import Settings
from caracal.llm.client import LLMClient
from caracal.llm.prompts import load_prompt
from caracal.report.model import Entity, SpeakerIdentity

MakeSettings = Callable[..., Settings]


async def _no_sleep(_seconds: float) -> None:
    return None


def _segment(index: int, text: str, speaker: str | None, *, length: float = 4.0) -> Segment:
    start = float(index) * length
    return Segment(
        id=index,
        start=start,
        end=start + length,
        text=text,
        words=(),
        avg_logprob=-0.1,
        no_speech_prob=0.0,
        lang="en",
        speaker=speaker,
    )


def _transcript(rows: Sequence[tuple[str | None, str]]) -> Transcript:
    return Transcript(
        segments=tuple(_segment(index, text, speaker) for index, (speaker, text) in enumerate(rows)),
        language_code="en",
        dropped={},
    )


def _stats(**shares: float) -> dict[str, dict[str, float | int]]:
    return {
        label: {
            "talk_time_s": share * 100.0,
            "share": share,
            "first_seen_s": 0.0,
            "words": int(share * 100.0),
        }
        for label, share in shares.items()
    }


def _args(
    mentions: Sequence[dict[str, Any]] = (),
    speakers: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    return {"mentions": list(mentions), "speakers": list(speakers)}


async def _resolve(
    settings: Settings,
    transcript: Transcript,
    arguments: dict[str, Any],
    *,
    stats: dict[str, dict[str, float | int]] | None = None,
    metadata: dict[str, str] | None = None,
    hints: Sequence[str] = (),
    entities: Sequence[Entity] = (),
    voiceprints: dict[str, list[VoiceprintMatch]] | None = None,
) -> tuple[IdentityResult, FakeOllama]:
    fake = FakeOllama([json_response(arguments)])
    http = httpx.AsyncClient(transport=fake.transport())
    client = LLMClient(settings, http, sleep=_no_sleep)
    try:
        result = await resolve_identities(
            transcript,
            metadata or {},
            hints,
            entities,
            voiceprints or {},
            stats if stats is not None else _stats(S1=1.0),
            settings,
            llm=client,
        )
    finally:
        await http.aclose()
    return result, fake


def _identity(result: IdentityResult, label: str) -> SpeakerIdentity:
    return next(identity for identity in result.speakers if identity.label == label)


async def test_self_intro_in_german_promotes_the_name(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Hallo, ich bin Angela Merkel und heute spreche ich über Europa.")])
    arguments = _args(
        mentions=[
            {
                "speaker_label": "S1",
                "name": "Angela Merkel",
                "relation": "self_intro",
                "seg_ids": ["seg:0"],
            }
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments, stats=_stats(S1=1.0))

    identity = _identity(result, "S1")
    assert identity.name == "Angela Merkel"
    assert identity.confidence == pytest.approx(0.85)
    assert len(identity.candidates) == 1
    assert identity.candidates[0].sources == ["self_intro"]
    assert identity.candidates[0].confidence == pytest.approx(0.85)
    assert identity.candidates[0].evidence == ["seg:0"]


async def test_addressed_name_before_the_turn_is_a_weak_candidate(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript(
        [
            ("S1", "Thanks for having me."),
            ("S1", "thanks, Anna"),
            ("S2", "It is great to be here."),
        ]
    )
    arguments = _args(
        mentions=[
            {
                "speaker_label": "S2",
                "name": "Anna",
                "relation": "addressed",
                "seg_ids": ["seg:1"],
            }
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments, stats=_stats(S1=0.5, S2=0.5))

    identity = _identity(result, "S2")
    assert identity.name == "Anna"
    assert identity.confidence == pytest.approx(0.6)
    assert identity.candidates[0].sources == ["addressed_by_name"]
    assert identity.candidates[0].confidence == pytest.approx(0.6)
    assert identity.candidates[0].evidence == ["seg:1"]


async def test_hallucinated_mention_whose_name_is_not_in_the_segment_is_dropped(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Hallo zusammen, willkommen zur Sendung.")])
    arguments = _args(
        mentions=[
            {
                "speaker_label": "S1",
                "name": "Angela Merkel",
                "relation": "self_intro",
                "seg_ids": ["seg:0"],
            }
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments)

    identity = _identity(result, "S1")
    assert identity.candidates == []
    assert identity.name is None
    assert identity.confidence == 0.0


async def test_referred_name_is_never_a_source(make_settings: MakeSettings) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Ich habe Angela Merkel gestern getroffen.")])
    arguments = _args(
        mentions=[
            {
                "speaker_label": "S1",
                "name": "Angela Merkel",
                "relation": "referred",
                "seg_ids": ["seg:0"],
            }
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments)

    assert _identity(result, "S1").candidates == []


async def test_self_intro_and_addressed_fuse_to_094(make_settings: MakeSettings) -> None:
    settings = make_settings()
    transcript = _transcript(
        [
            ("S2", "Welcome to the show, Anna."),
            ("S1", "Ich bin Anna und freue mich hier zu sein."),
        ]
    )
    arguments = _args(
        mentions=[
            {"speaker_label": "S1", "name": "Anna", "relation": "self_intro", "seg_ids": ["seg:1"]},
            {"speaker_label": "S1", "name": "Anna", "relation": "addressed", "seg_ids": ["seg:0"]},
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments, stats=_stats(S1=0.6, S2=0.4))

    identity = _identity(result, "S1")
    assert identity.name == "Anna"
    assert len(identity.candidates) == 1
    assert identity.candidates[0].sources == ["self_intro", "addressed_by_name"]
    assert identity.candidates[0].confidence == pytest.approx(0.94)
    assert identity.confidence == pytest.approx(0.94)


async def test_fusion_is_capped_at_098(make_settings: MakeSettings) -> None:
    settings = make_settings()
    transcript = _transcript(
        [
            ("S2", "Welcome to the show, Anna."),
            ("S1", "Ich bin Anna und freue mich hier zu sein."),
        ]
    )
    arguments = _args(
        mentions=[
            {"speaker_label": "S1", "name": "Anna", "relation": "self_intro", "seg_ids": ["seg:1"]},
            {"speaker_label": "S1", "name": "Anna", "relation": "addressed", "seg_ids": ["seg:0"]},
        ],
        speakers=[
            {"speaker_label": "S1", "role": "host", "description": "hosts the show"},
            {"speaker_label": "S2", "role": "guest", "description": "appears as a guest"},
        ],
    )

    result, _fake = await _resolve(
        settings,
        transcript,
        arguments,
        stats=_stats(S1=0.6, S2=0.4),
        hints=["host: Anna"],
    )

    identity = _identity(result, "S1")
    assert identity.candidates[0].sources == ["self_intro", "addressed_by_name", "hint"]
    assert identity.candidates[0].confidence == pytest.approx(0.98)


async def test_conflicting_voiceprint_and_self_intro_are_not_promoted(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    known_id = uuid.uuid4()
    transcript = _transcript([("S1", "Ich bin Angela Merkel.")])
    arguments = _args(
        mentions=[
            {
                "speaker_label": "S1",
                "name": "Angela Merkel",
                "relation": "self_intro",
                "seg_ids": ["seg:0"],
            }
        ]
    )
    matches = {
        "S1": [VoiceprintMatch(known_id, "Jane Doe", 0.96, "match")],
    }

    result, _fake = await _resolve(
        settings,
        transcript,
        arguments,
        voiceprints=matches,
    )

    identity = _identity(result, "S1")
    assert identity.name is None
    assert identity.confidence == 0.0
    assert [candidate.name for candidate in identity.candidates] == ["Jane Doe", "Angela Merkel"]
    assert identity.candidates[0].confidence == pytest.approx(0.96)
    assert [flag.message for flag in result.flags] == ["ambiguous identity"]
    assert result.flags[0].field == "speakers.S1.name"
    assert identity.known_speaker_id == known_id
    assert identity.voiceprint_similarity == pytest.approx(0.96)


async def test_two_equal_candidates_fail_the_lead_margin(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Ich bin Anna Schmidt. Man nennt mich auch Anna Weber.")])
    arguments = _args(
        mentions=[
            {
                "speaker_label": "S1",
                "name": "Anna Schmidt",
                "relation": "self_intro",
                "seg_ids": ["seg:0"],
            },
            {
                "speaker_label": "S1",
                "name": "Anna Weber",
                "relation": "self_intro",
                "seg_ids": ["seg:0"],
            },
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments)

    identity = _identity(result, "S1")
    assert identity.name is None
    assert {candidate.name for candidate in identity.candidates} == {"Anna Schmidt", "Anna Weber"}
    assert [flag.message for flag in result.flags] == ["ambiguous identity"]


async def test_metadata_only_name_is_a_candidate_but_is_not_promoted(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Welcome to the interview."), ("S2", "Thank you.")])
    arguments = _args(
        speakers=[
            {"speaker_label": "S1", "role": "host", "description": "hosts the interview"},
            {"speaker_label": "S2", "role": "guest", "description": "appears as a guest"},
        ]
    )
    entities = [Entity(text="Jane Doe", type="person", evidence=["meta:title"])]

    result, _fake = await _resolve(
        settings,
        transcript,
        arguments,
        stats=_stats(S1=0.7, S2=0.3),
        metadata={"title": "Interview with Jane Doe"},
        entities=entities,
    )

    identity = _identity(result, "S1")
    assert identity.name is None
    assert len(identity.candidates) == 1
    assert identity.candidates[0].name == "Jane Doe"
    assert identity.candidates[0].sources == ["metadata"]
    assert identity.candidates[0].confidence == pytest.approx(0.5)
    assert identity.candidates[0].evidence == ["meta:title"]
    assert result.flags == []
    assert _identity(result, "S2").candidates == []


async def test_sole_speaker_ted_case_promotes_the_metadata_name_at_075(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript(
        [
            ("S1", "A few years ago, I broke into my own house."),
            ("S1", "It was around midnight in the dead of Montreal winter."),
        ]
    )
    arguments = _args(
        speakers=[
            {"speaker_label": "S1", "role": "narrator", "description": "narrates the talk"},
        ]
    )
    entities = [Entity(text="Daniel Levitin", type="person", evidence=["meta:title"])]

    result, _fake = await _resolve(
        settings,
        transcript,
        arguments,
        stats=_stats(S1=1.0),
        metadata={
            "title": "How to stay calm when you know you will be stressed | Daniel Levitin | TED",
            "channel": "TED",
        },
        entities=entities,
    )

    identity = _identity(result, "S1")
    assert identity.name == "Daniel Levitin"
    assert identity.confidence == pytest.approx(0.75)
    assert identity.candidates[0].sources == ["metadata"]
    assert identity.candidates[0].confidence == pytest.approx(0.75)
    assert identity.candidates[0].evidence == ["meta:title"]
    assert result.flags == []


async def test_voiceprint_match_sets_the_known_speaker_and_similarity(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    known_id = uuid.uuid4()
    transcript = _transcript([("S1", "Good evening and welcome.")])
    matches = {"S1": [VoiceprintMatch(known_id, "Jane Doe", 0.962, "match")]}

    result, _fake = await _resolve(
        settings,
        transcript,
        _args(),
        voiceprints=matches,
    )

    identity = _identity(result, "S1")
    assert identity.name == "Jane Doe"
    assert identity.confidence == pytest.approx(0.962)
    assert identity.candidates[0].sources == ["voiceprint"]
    assert identity.candidates[0].evidence == ["vp:S1"]
    assert identity.known_speaker_id == known_id
    assert identity.voiceprint_similarity == pytest.approx(0.962)


async def test_weak_voiceprint_candidate_is_scored_similarity_times_06(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    known_id = uuid.uuid4()
    transcript = _transcript([("S1", "Good evening and welcome.")])
    matches = {"S1": [VoiceprintMatch(known_id, "Jane Doe", 0.55, "candidate")]}

    result, _fake = await _resolve(settings, transcript, _args(), voiceprints=matches)

    identity = _identity(result, "S1")
    assert identity.name == "Jane Doe"
    assert identity.confidence == pytest.approx(0.33)
    assert identity.candidates[0].sources == ["voiceprint"]
    assert identity.known_speaker_id is None
    assert identity.voiceprint_similarity is None


async def test_hint_maps_to_the_label_with_the_matching_role(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Welcome to the show."), ("S2", "Thanks for having me.")])
    arguments = _args(
        speakers=[
            {"speaker_label": "S1", "role": "host", "description": "hosts the show"},
            {"speaker_label": "S2", "role": "guest", "description": "appears as a guest"},
        ]
    )

    result, _fake = await _resolve(
        settings,
        transcript,
        arguments,
        stats=_stats(S1=0.65, S2=0.35),
        hints=["host: Lex Fridman"],
    )

    host = _identity(result, "S1")
    assert host.name == "Lex Fridman"
    assert host.candidates[0].sources == ["hint"]
    assert host.candidates[0].confidence == pytest.approx(0.7)
    assert host.candidates[0].evidence == ["hint:0"]
    assert _identity(result, "S2").candidates == []


async def test_bare_hint_applies_to_every_label_when_several_speak(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Welcome."), ("S2", "Thanks.")])

    result, _fake = await _resolve(
        settings,
        transcript,
        _args(),
        stats=_stats(S1=0.6, S2=0.4),
        hints=["Lex Fridman"],
    )

    for label in ("S1", "S2"):
        identity = _identity(result, label)
        assert identity.candidates[0].sources == ["hint"]
        assert identity.candidates[0].confidence == pytest.approx(0.4)


async def test_protected_attribute_in_a_description_is_replaced_by_the_role_fallback(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Welcome to the show.")])
    arguments = _args(
        speakers=[
            {
                "speaker_label": "S1",
                "role": "host",
                "description": "adult male, calm, appears to be the channel owner",
            }
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments)

    identity = _identity(result, "S1")
    assert identity.description == "hosts the recording"
    for word in ("male", "female", "woman", "man"):
        assert re.search(rf"\b{word}\b", identity.description or "", re.IGNORECASE) is None


def test_prompt_forbids_protected_attributes() -> None:
    prompt = load_prompt(
        "identity_mentions",
        metadata="<<<DATA:metadata>>>\n<<<END:metadata>>>",
        hints="<<<DATA:hints>>>\n<<<END:hints>>>",
        transcript="<<<DATA:transcript>>>\n<<<END:transcript>>>",
        roles="host, guest",
        report_language="English",
    )

    assert "never infer or mention protected attributes" in prompt
    assert "gender" in prompt
    assert "male" in prompt and "female" in prompt
    assert "role and the context" in prompt


def _captured_transcript_block(fake: FakeOllama) -> str:
    body = fake.body()
    messages = body["messages"]
    assert isinstance(messages, list)
    prompt = messages[1]["content"]
    assert isinstance(prompt, str)
    match = re.search(r"<<<DATA:transcript>>>\n(.*?)\n<<<END:transcript>>>", prompt, re.DOTALL)
    assert match is not None
    return match.group(1)


async def test_long_transcript_sends_first_40_segments_plus_the_run_up_to_turns(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings(LLM_CONTEXT_CHARS=3000)
    rows: list[tuple[str | None, str]] = [
        ("S1", f"Segment {index} of the long recording.") for index in range(100)
    ]
    rows.append(("S2", "One question from the host."))
    rows.append(("S1", "And the answer after the question."))
    transcript = _transcript(rows)

    _result, fake = await _resolve(settings, transcript, _args())

    block = _captured_transcript_block(fake)
    assert len(block) <= settings.LLM_CONTEXT_CHARS
    assert "[seg:0] S1" in block
    assert "[seg:39] S1" in block
    assert "[seg:40] S1" not in block
    # The three segments before S1's second turn (positions 98, 99, 100) are included.
    assert "[seg:98] S1" in block
    assert "[seg:99] S1" in block
    assert "[seg:100] S2" in block


async def test_short_transcript_is_sent_whole(make_settings: MakeSettings) -> None:
    settings = make_settings(LLM_CONTEXT_CHARS=100_000)
    transcript = _transcript([("S1", "One line."), ("S2", "Two.")])

    _result, fake = await _resolve(settings, transcript, _args())

    block = _captured_transcript_block(fake)
    assert "[seg:0] S1" in block
    assert "[seg:1] S2" in block


async def test_addressed_mention_too_far_before_the_turn_is_dropped(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript(
        [
            ("S1", "thanks, Anna"),
            ("S1", "we have a lot to cover"),
            ("S1", "first the news"),
            ("S1", "then the interview"),
            ("S1", "and later the weather"),
            ("S2", "It is great to be here."),
        ]
    )
    arguments = _args(
        mentions=[{"speaker_label": "S2", "name": "Anna", "relation": "addressed", "seg_ids": ["seg:0"]}]
    )

    result, _fake = await _resolve(settings, transcript, arguments, stats=_stats(S1=0.6, S2=0.4))

    assert _identity(result, "S2").candidates == []


async def test_self_intro_cited_from_another_speaker_is_dropped(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Welcome to the show."), ("S2", "I am Angela Merkel.")])
    arguments = _args(
        mentions=[
            {
                "speaker_label": "S1",
                "name": "Angela Merkel",
                "relation": "self_intro",
                "seg_ids": ["seg:1"],
            }
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments)

    assert _identity(result, "S1").candidates == []


async def test_name_matching_ignores_case_and_diacritics_and_allows_a_first_name(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S1", "Hallo, ich bin JOSÉ Alvarez und heute geht es um Musik.")])
    arguments = _args(
        mentions=[
            {
                "speaker_label": "S1",
                "name": "Jose Alvarez",
                "relation": "self_intro",
                "seg_ids": ["seg:0"],
            },
            {
                "speaker_label": "S1",
                "name": "Jose",
                "relation": "referred",
                "seg_ids": ["seg:0"],
            },
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments)

    identity = _identity(result, "S1")
    assert identity.name == "Jose Alvarez"
    assert identity.candidates[0].sources == ["self_intro"]


async def test_a_first_name_merges_into_the_full_name_it_starts(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    transcript = _transcript([("S2", "Welcome to the show, Angela."), ("S1", "Ich bin Angela Merkel.")])
    arguments = _args(
        mentions=[
            {"speaker_label": "S1", "name": "Angela Merkel", "relation": "self_intro", "seg_ids": ["seg:1"]},
            {"speaker_label": "S1", "name": "Angela", "relation": "addressed", "seg_ids": ["seg:0"]},
        ]
    )

    result, _fake = await _resolve(settings, transcript, arguments, stats=_stats(S1=0.6, S2=0.4))

    identity = _identity(result, "S1")
    assert len(identity.candidates) == 1
    assert identity.name == "Angela Merkel"
    assert identity.candidates[0].sources == ["self_intro", "addressed_by_name"]
    assert identity.candidates[0].confidence == pytest.approx(0.94)
