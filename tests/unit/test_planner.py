"""The rule-first stage planner: music, speech ratio, options, verify, translate, depth.

The LLM is only ever allowed to fill ``content_profile``; every test that checks a rule also
checks that no LLM call was made when the rule should have short-circuited.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import httpx
from tests.fakes.ollama import FakeOllama, message_response, tool_call_message

from caracal.analysis.planner import (
    AMBIENT_REASON,
    DIARIZE_OFF_REASON,
    DIARIZER_UNAVAILABLE_REASON,
    MUSIC_REASON,
    NO_TRANSLATION_REASON,
    QUICK_REASON,
    PlanInputs,
    plan,
)
from caracal.asr import Segment, Transcript
from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.llm.client import LLMClient
from caracal.pipeline.stages import JobOptions

MakeSettings = Callable[..., Settings]
_PROFILE = {
    "likely_genre": "interview",
    "expected_speakers": 2,
    "needs_identity": True,
    "checkworthy_domain": True,
}


async def _no_sleep(_seconds: float) -> None:
    return None


def _segment(index: int, text: str, *, start: float | None = None, length: float = 10.0) -> Segment:
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
        lang_prob=0.9,
        speaker="S1",
    )


def _transcript(texts: list[str], *, length: float = 10.0) -> Transcript:
    segments = tuple(_segment(index, text, length=length) for index, text in enumerate(texts))
    return Transcript(segments=segments, language_code="en", dropped={})


def _profile_response(**overrides: object) -> httpx.Response:
    arguments = {**_PROFILE, **overrides}
    return message_response(
        tool_call_message(arguments, name="record_content_profile"),
    )


def _languages(code: str = "en", share: float = 1.0, end: float = 60.0) -> list[dict[str, object]]:
    return [
        {
            "code": code,
            "name": code,
            "share": share,
            "spans": [{"start": 0.0, "end": end}],
            "confidence": 0.9,
        }
    ]


def _inputs(
    *,
    transcript: Transcript | None = None,
    options: JobOptions | None = None,
    **overrides: object,
) -> PlanInputs:
    values: dict[str, Any] = {
        "duration": 120.0,
        "speech_ratio": 0.9,
        "options": options if options is not None else JobOptions(),
        "transcript": transcript,
        "languages": _languages(),
        "metadata": {"title": "A talk"},
        "diarizer_available": True,
        "search_configured": True,
        "default_target_langs": ["de"],
        "search_max_queries": 24,
        "report_language": "en",
    }
    values.update(overrides)
    return PlanInputs(**values)


def _client(settings: Settings, fake: FakeOllama) -> tuple[httpx.AsyncClient, LLMClient]:
    http = httpx.AsyncClient(transport=fake.transport())
    return http, LLMClient(settings, http, sleep=_no_sleep)


async def test_music_rule_skips_speakers_and_rewrites_languages_without_an_llm_call(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([])
    http, client = _client(settings, fake)
    language_items = _languages("nn", end=60.0)
    try:
        result = await plan(
            _inputs(
                transcript=_transcript(["🎵🎵🎵"] * 6),
                languages=language_items,
            ),
            llm=client,
        )
    finally:
        await http.aclose()

    assert fake.requests == []
    assert result.music_detected is True
    # Diarization runs before the plan, so the planner never skips it (verdict 25 D3).
    assert result.run["diarize"] is True
    assert "diarize" not in result.reasons
    assert result.run["align"] is False
    assert result.run["voiceprint"] is False
    assert result.run["identity"] is False
    assert result.run["verify"] is False
    assert result.run["translate"] is False
    assert result.run["content"] is True
    assert result.reasons["align"] == MUSIC_REASON
    assert result.reasons["content"] == "music: describe the track, not speech"
    assert result.languages == [
        {
            "code": "und",
            "name": "Unknown",
            "share": 1.0,
            "spans": [{"start": 0.0, "end": 60.0}],
            "confidence": 1.0,
        }
    ]


def test_music_text_fallback_is_seventy_percent_of_speech_time() -> None:
    music = _transcript(["🎵🎵🎵"] * 7 + ["hello there"] * 3)
    lyrics = _transcript(["la la la"] * 7 + ["🎵🎵🎵"] * 3)

    from caracal.analysis.planner import _music_detected

    assert _music_detected(music, None) is True
    assert _music_detected(lyrics, None) is False


async def test_diarization_speech_share_decides_music_without_an_llm_call(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    music_fake = FakeOllama([])
    music_http, music_client = _client(settings, music_fake)
    try:
        music = await plan(
            _inputs(
                transcript=_transcript(["Utak lyder", "Norske musik"] * 3),
                speech_share=0.02,
            ),
            llm=music_client,
        )
    finally:
        await music_http.aclose()

    speech_fake = FakeOllama([_profile_response()])
    speech_http, speech_client = _client(settings, speech_fake)
    try:
        speech = await plan(
            _inputs(
                transcript=_transcript(["🎵🎵🎵"] * 6),
                speech_share=0.87,
            ),
            llm=speech_client,
        )
    finally:
        await speech_http.aclose()

    assert music_fake.requests == []
    assert music.music_detected is True
    assert music.run["content"] is True
    assert music.languages is not None and music.languages[0]["code"] == "und"
    assert len(speech_fake.requests) == 1
    assert speech.music_detected is False
    assert speech.run["diarize"] is True
    assert speech.run["content"] is True
    assert speech.languages is None


async def test_lyrics_with_words_are_not_music_and_the_profile_is_used(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_profile_response()])
    http, client = _client(settings, fake)
    try:
        result = await plan(_inputs(transcript=_transcript(["la la la"] * 6)), llm=client)
    finally:
        await http.aclose()

    assert result.music_detected is False
    assert result.run["diarize"] is True
    assert result.run["content"] is True
    assert result.content_profile.likely_genre == "interview"
    assert result.content_profile.expected_speakers == 2
    assert result.reasons["profile"] == "content profile from the LLM"
    assert len(fake.requests) == 1
    body = fake.body()
    assert body["tools"][0]["function"]["name"] == "record_content_profile"
    prompt = body["messages"][1]["content"]
    assert "<<<DATA:transcript>>>" in prompt
    assert "[seg:0] S1 00:00.0 | la la la" in prompt


async def test_quick_depth_makes_no_llm_call_and_skips_identity_work(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([])
    http, client = _client(settings, fake)
    try:
        result = await plan(
            _inputs(
                transcript=_transcript(["hello there"] * 3),
                options=JobOptions(depth="quick"),
            ),
            llm=client,
        )
    finally:
        await http.aclose()

    assert fake.requests == []
    assert result.reasons["profile"] == "quick depth: default profile"
    assert result.content_profile.likely_genre == "unknown"
    assert result.run["identity"] is False
    assert result.run["verify"] is False
    assert result.run["translate"] is False
    assert result.reasons["identity"] == QUICK_REASON
    assert result.budgets.search_queries == 0
    assert result.budgets.llm_calls == 1


class _FailingLLM:
    async def call_tool(self, **_kwargs: object) -> object:
        raise CaracalError(ErrorCode.DEPENDENCY_DOWN, "ollama is down")


async def test_llm_failure_falls_back_to_the_default_profile(
    make_settings: MakeSettings,
) -> None:
    failing = cast("LLMClient", _FailingLLM())

    result = await plan(_inputs(transcript=_transcript(["hello there"] * 3)), llm=failing)

    assert result.reasons["profile"] == "llm unavailable"
    assert result.content_profile.likely_genre == "unknown"
    assert result.content_profile.needs_identity is True
    assert result.run["diarize"] is True


async def test_llm_output_cannot_flip_run_flags(make_settings: MakeSettings) -> None:
    settings = make_settings()
    fake = FakeOllama(
        [
            _profile_response(
                likely_genre="news",
                run={"diarize": False, "verify": False, "translate": True},
            )
        ]
    )
    http, client = _client(settings, fake)
    try:
        result = await plan(_inputs(transcript=_transcript(["hello there"] * 3)), llm=client)
    finally:
        await http.aclose()

    assert len(fake.requests) == 1
    assert result.content_profile.likely_genre == "news"
    assert result.run["diarize"] is True
    assert result.run["verify"] is True
    assert result.run["translate"] is True


async def test_diarize_off_rule_skips_diarize_align_and_voiceprint(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_profile_response()])
    http, client = _client(settings, fake)
    try:
        result = await plan(
            _inputs(
                transcript=_transcript(["hello there"] * 3),
                options=JobOptions(diarize="off"),
            ),
            llm=client,
        )
    finally:
        await http.aclose()

    assert result.run["diarize"] is False
    assert result.run["align"] is False
    assert result.run["voiceprint"] is False
    assert result.reasons["diarize"] == DIARIZE_OFF_REASON
    assert result.reasons["align"] == DIARIZE_OFF_REASON
    assert result.reasons["voiceprint"] == DIARIZE_OFF_REASON


async def test_missing_diarizer_rule_skips_the_speaker_stages(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_profile_response()])
    http, client = _client(settings, fake)
    try:
        result = await plan(
            _inputs(
                transcript=_transcript(["hello there"] * 3),
                diarizer_available=False,
            ),
            llm=client,
        )
    finally:
        await http.aclose()

    assert result.run["diarize"] is False
    assert result.run["align"] is False
    assert result.run["voiceprint"] is False
    assert result.reasons["diarize"] == DIARIZER_UNAVAILABLE_REASON


async def test_low_speech_ratio_skips_identity_and_verify(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_profile_response()])
    http, client = _client(settings, fake)
    try:
        result = await plan(
            _inputs(
                transcript=_transcript(["hello there"] * 3),
                speech_ratio=0.01,
            ),
            llm=client,
        )
    finally:
        await http.aclose()

    assert result.run["identity"] is False
    assert result.run["verify"] is False
    assert result.run["translate"] is False
    assert result.reasons["identity"] == AMBIENT_REASON
    assert result.reasons["content"] == "music/ambient: short description only"


async def test_short_clip_keeps_automatic_diarization(make_settings: MakeSettings) -> None:
    settings = make_settings()
    fake = FakeOllama([_profile_response()])
    http, client = _client(settings, fake)
    try:
        result = await plan(
            _inputs(
                transcript=_transcript(["hello there"] * 2),
                duration=12.0,
                options=JobOptions(diarize="auto"),
            ),
            llm=client,
        )
    finally:
        await http.aclose()

    assert result.run["diarize"] is True
    assert result.reasons["diarize"] == "short clip: diarization is cheap"
    assert result.reasons["identity"] == "short clip: single-speaker identity logic"


async def test_verify_modes(make_settings: MakeSettings) -> None:
    settings = make_settings()
    cases = [
        (JobOptions(verify="off"), True, False, "verification disabled by options"),
        (JobOptions(verify="on"), True, True, "verification requested"),
        (
            JobOptions(verify="auto"),
            True,
            True,
            "auto verification: checkworthy domain and search configured",
        ),
        (JobOptions(verify="auto"), False, False, "auto verification: no checkworthy domain or search"),
    ]
    for options, checkworthy, expected, reason in cases:
        fake = FakeOllama([_profile_response(checkworthy_domain=checkworthy)])
        http, client = _client(settings, fake)
        try:
            result = await plan(
                _inputs(transcript=_transcript(["hello there"] * 3), options=options),
                llm=client,
            )
        finally:
            await http.aclose()
        assert result.run["verify"] is expected
        assert result.reasons["verify"] == reason


async def test_translate_only_runs_for_a_target_that_differs_from_the_source(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    for targets, expected, reason in (
        (None, True, "translate to de"),
        (["en"], False, NO_TRANSLATION_REASON),
    ):
        fake = FakeOllama([_profile_response()])
        http, client = _client(settings, fake)
        try:
            result = await plan(
                _inputs(
                    transcript=_transcript(["hello there"] * 3),
                    languages=_languages("en"),
                    options=JobOptions(target_langs=targets),
                    default_target_langs=["de"],
                ),
                llm=client,
            )
        finally:
            await http.aclose()
        assert result.run["translate"] is expected
        assert result.reasons["translate"] == reason


async def test_deep_depth_uses_the_search_budget_and_fifteen_claims(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    fake = FakeOllama([_profile_response()])
    http, client = _client(settings, fake)
    try:
        result = await plan(
            _inputs(
                transcript=_transcript(["hello there"] * 3),
                options=JobOptions(depth="deep"),
                search_max_queries=30,
            ),
            llm=client,
        )
    finally:
        await http.aclose()

    assert result.budgets.search_queries == 30
    assert result.budgets.llm_calls == 30
    assert result.budgets.claims == 15
