"""Sentence-unit translation and cue redistribution (docs/04 §10, card T3.3)."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from tests.fakes.ollama import json_response

from caracal.analysis.translate import (
    TranslationUnit,
    is_valid_target_lang,
    redistribute,
    resolve_target_langs,
    translate_units,
    units_from_cues,
)
from caracal.config import Settings
from caracal.llm.client import LLMClient
from caracal.pipeline.stages import JobOptions
from caracal.subtitles.segmenter import Cue, wrap_lines

MakeSettings = Callable[..., Settings]


async def _no_sleep(_seconds: float) -> None:
    return None


def _cue(index: int, text: str, *, start: float | None = None, speaker: str = "S1") -> Cue:
    begin = float(index) if start is None else start
    return Cue(
        index=index,
        start=begin,
        end=begin + 2.0,
        lines=[text],
        speaker=speaker,
        word_span=(0, 0),
    )


def _unit(cues: Sequence[Cue]) -> TranslationUnit:
    return TranslationUnit(
        index=0,
        cues=tuple(cues),
        cue_indices=tuple(range(len(cues))),
        text=" ".join(cue.lines[0] for cue in cues),
        speaker=cues[0].speaker,
    )


def _payload(request: httpx.Request) -> dict[str, Any]:
    request.read()
    body = json.loads(request.content)
    user = next(message for message in body["messages"] if message["role"] == "user")
    return json.loads(user["content"])


async def _translate(
    settings: Settings,
    units: Sequence[TranslationUnit],
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    lang: str = "de",
) -> tuple[list[str], int, list[dict[str, Any]]]:
    bodies: list[dict[str, Any]] = []

    def recording(request: httpx.Request) -> httpx.Response:
        request.read()
        bodies.append(json.loads(request.content))
        return handler(request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(recording))
    llm = LLMClient(settings, http, sleep=_no_sleep)
    try:
        result = await translate_units(
            units,
            lang,
            glossary=["Daniel Levitin"],
            summary="A talk about staying calm.",
            llm=llm,
        )
        return list(result), result.failed_units, bodies
    finally:
        await http.aclose()


# --- units ----------------------------------------------------------------------------


def test_units_group_same_speaker_until_sentence_punctuation() -> None:
    cues = [
        _cue(1, "A few"),
        _cue(2, "years ago, I"),
        _cue(3, "started running."),
        _cue(4, "It was hard."),
    ]
    units = units_from_cues(cues)

    assert [unit.cue_indices for unit in units] == [(0, 1, 2), (3,)]
    assert units[0].text == "A few years ago, I started running."
    assert units[0].speaker == "S1"


def test_units_break_on_a_speaker_change() -> None:
    cues = [_cue(1, "Hello there", speaker="S1"), _cue(2, "Hi", speaker="S2")]
    units = units_from_cues(cues)

    assert [unit.speaker for unit in units] == ["S1", "S2"]
    assert [unit.cue_indices for unit in units] == [(0,), (1,)]


def test_units_cap_at_four_cues() -> None:
    cues = [_cue(index, f"word{index}") for index in range(1, 6)]
    units = units_from_cues(cues)

    assert [len(unit.cues) for unit in units] == [4, 1]
    assert [unit.cue_indices for unit in units] == [(0, 1, 2, 3), (4,)]


def test_units_cap_at_two_hundred_characters() -> None:
    first = "x" * 120
    second = "y" * 120
    units = units_from_cues([_cue(1, first), _cue(2, second)])

    assert len(units) == 2
    assert [unit.text for unit in units] == [first, second]


def test_sentence_end_ignores_a_trailing_quote() -> None:
    units = units_from_cues([_cue(1, 'He said "go."'), _cue(2, "Then he left.")])
    assert [len(unit.cues) for unit in units] == [1, 1]


# --- translate_units ------------------------------------------------------------------


async def test_translate_units_batches_twenty_five_and_keeps_order(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    units = [_unit([_cue(index + 1, f"unit {index}")]) for index in range(30)]

    def handler(request: httpx.Request) -> httpx.Response:
        cues = _payload(request)["cues"]
        return json_response({"cues": [{"i": item["i"], "text": f"de:{item['text']}"} for item in cues]})

    translated, failed, bodies = await _translate(settings, units, handler)

    assert len(bodies) == 2
    assert sorted(len(_payload_body(body)["cues"]) for body in bodies) == [5, 25]
    assert translated == [f"de:unit {index}" for index in range(30)]
    assert failed == 0


async def test_translate_units_sends_json_mode_with_target_and_glossary(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    units = [_unit([_cue(1, "A few years ago")])]

    def handler(request: httpx.Request) -> httpx.Response:
        _payload(request)
        return json_response({"cues": [{"i": 0, "text": "Vor ein paar Jahren"}]})

    translated, failed, bodies = await _translate(settings, units, handler)

    assert translated == ["Vor ein paar Jahren"]
    assert failed == 0
    body = bodies[0]
    assert body["format"] == "json"
    assert "tools" not in body
    system = body["messages"][0]["content"]
    assert "German (Deutsch)" in system
    assert "Daniel Levitin" in system
    assert "staying calm" in system
    assert _payload_body(body) == {"cues": [{"i": 0, "text": "A few years ago"}]}


def _payload_body(body: dict[str, Any]) -> dict[str, Any]:
    return json.loads(body["messages"][1]["content"])


async def test_translate_units_retries_an_index_mismatch_once(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    units = [_unit([_cue(1, "hello")]), _unit([_cue(2, "world")])]
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return json_response({"cues": [{"i": 0, "text": "only one"}]})
        return json_response({"cues": [{"i": 0, "text": "hallo"}, {"i": 1, "text": "welt"}]})

    translated, failed, bodies = await _translate(settings, units, handler)

    assert len(bodies) == 2
    assert translated == ["hallo", "welt"]
    assert failed == 0


async def test_translate_units_falls_back_per_unit_and_keeps_source_on_failure(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    units = [_unit([_cue(1, "one")]), _unit([_cue(2, "source two")])]

    def handler(request: httpx.Request) -> httpx.Response:
        cues = _payload(request)["cues"]
        if len(cues) > 1:
            return json_response({"cues": [{"i": 0, "text": "wrong index set"}]})
        if cues[0]["text"] == "source two":
            return json_response({"cues": []})
        return json_response({"cues": [{"i": 0, "text": "eins"}]})

    translated, failed, bodies = await _translate(settings, units, handler)

    assert len(bodies) == 4
    assert translated == ["eins", "source two"]
    assert failed == 1


# --- redistribution -------------------------------------------------------------------


def test_redistribute_is_proportional_keeps_timings_and_fills_every_cue() -> None:
    cues = [_cue(1, "a" * 10), _cue(2, "b" * 30, start=10.0), _cue(3, "c" * 20, start=20.0)]
    unit = _unit(cues)
    translated = "one two three four five six seven eight nine ten"

    result, overlong = redistribute(unit, translated, "de")

    assert [(cue.start, cue.end) for cue in result] == [(cue.start, cue.end) for cue in cues]
    joined = [" ".join(cue.lines) for cue in result]
    assert [len(item.split()) for item in joined] == [2, 5, 3]
    assert all(item for item in joined)
    assert all(len(line) <= 42 for cue in result for line in cue.lines)
    assert overlong == 0


def test_redistribute_splits_cjk_by_characters() -> None:
    cues = [_cue(1, "a"), _cue(2, "b", start=5.0)]
    unit = _unit(cues)

    result, overlong = redistribute(unit, "你好 世界 再见", "zh")

    assert ["".join(cue.lines) for cue in result] == ["你好世", "界再见"]
    assert overlong == 0


def test_redistribute_marks_a_cue_that_cannot_fit_two_lines() -> None:
    unit = _unit([_cue(1, "hello")])
    translated = " ".join(["word"] * 30)

    result, overlong = redistribute(unit, translated, "de")

    assert overlong == 1
    assert len(result[0].lines) > 2
    assert " ".join(result[0].lines) == translated


def test_redistribute_keeps_rtl_text_without_injected_marks() -> None:
    unit = _unit([_cue(1, "hello")])
    translated = "مرحبا بالعالم"

    result, overlong = redistribute(unit, translated, "ar")

    assert overlong == 0
    text = " ".join(result[0].lines)
    assert text == translated
    assert not any("\u202a" <= character <= "\u202e" for character in text)
    assert "\u200e" not in text and "\u200f" not in text


def test_wrap_lines_returns_at_most_two_balanced_lines() -> None:
    lines, overlong = wrap_lines("a short translated subtitle line", max_chars=42)
    assert len(lines) <= 2
    assert overlong is False
    assert " ".join(lines) == "a short translated subtitle line"


def test_wrap_lines_breaks_a_long_line_into_two_within_the_limit() -> None:
    text = "the quick brown fox jumps over the lazy dog and keeps running"
    lines, overlong = wrap_lines(text, max_chars=42)

    assert len(lines) == 2
    assert all(len(line) <= 42 for line in lines)
    assert " ".join(lines) == text
    assert overlong is False


# --- target languages -----------------------------------------------------------------


def test_resolve_target_langs_handles_null_empty_and_source() -> None:
    languages = [{"code": "en", "share": 0.9}, {"code": "de", "share": 0.1}]

    assert resolve_target_langs(None, ["en", "de"], languages, "en") == ["de"]
    assert resolve_target_langs([], ["en"], languages, "en") == []
    assert resolve_target_langs(["de", "de"], [], languages, "en") == ["de"]
    assert resolve_target_langs(None, ["en"], [], "en") == []


def test_target_language_codes_are_validated_at_job_creation() -> None:
    assert JobOptions(target_langs=["de", "ar", "de"]).target_langs == ["de", "ar"]
    with pytest.raises(ValidationError):
        JobOptions(target_langs=["english"])
    with pytest.raises(ValidationError):
        JobOptions(target_langs=["DE"])
    with pytest.raises(ValidationError):
        JobOptions(target_langs=["xx"])
    assert is_valid_target_lang("de")
    assert not is_valid_target_lang("de-DE")
