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
    segment_token_starts,
    translate_units,
    units_from_cues,
)
from caracal.asr import Segment, Transcript, Word, join_words
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


def _word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end, prob=0.9, space_before=True)


def _transcript(*segments: tuple[Word, ...]) -> Transcript:
    return Transcript(
        segments=tuple(
            Segment(
                id=index,
                start=words[0].start,
                end=words[-1].end,
                text=join_words(words),
                words=words,
                avg_logprob=0.0,
                no_speech_prob=0.0,
                lang="en",
            )
            for index, words in enumerate(segments)
        ),
        language_code="en",
        dropped={"no_speech": 0, "repetition": 0, "empty": 0},
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


@pytest.mark.parametrize(
    ("lang", "translated"),
    [
        ("zh", "这是一个非常长的翻译句子用来检查每一行不超过十六个字符"),
        ("ja", "これは非常に長い翻訳文で各行が十六文字を超えないことを確認します"),
        ("th", "นี่คือคำแปลที่ยาวมากเพื่อทดสอบว่าบรรทัดไม่เกินสิบหกตัวอักษร"),
    ],
)
def test_translated_cjk_and_thai_lines_stay_within_sixteen_characters(lang: str, translated: str) -> None:
    """A translated cue obeys the same script-aware line limit as a source cue."""
    unit = _unit([_cue(1, "hello")])

    result, _overlong = redistribute(unit, translated, lang)

    lines = result[0].lines
    assert len(lines) >= 2
    assert all(len(line) <= 16 for line in lines)
    assert "".join(lines) == translated


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


async def test_translate_prompt_keeps_names_verbatim_for_a_latin_target(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    units = [_unit([_cue(1, "A few years ago")])]

    def handler(request: httpx.Request) -> httpx.Response:
        _payload(request)
        return json_response({"cues": [{"i": 0, "text": "Vor ein paar Jahren"}]})

    _translated, _failed, bodies = await _translate(settings, units, handler, lang="de")

    system = bodies[0]["messages"][0]["content"]
    assert "German (Deutsch)" in system
    assert "keep every name and place name exactly as written in the source, in the Latin script" in system
    assert "conventional form used in the target script" not in system


async def test_translate_prompt_uses_target_script_names_for_arabic(
    make_settings: MakeSettings,
) -> None:
    settings = make_settings()
    units = [_unit([_cue(1, "A few years ago")])]

    def handler(request: httpx.Request) -> httpx.Response:
        _payload(request)
        return json_response({"cues": [{"i": 0, "text": "قبل بضع سنوات"}]})

    _translated, _failed, bodies = await _translate(settings, units, handler, lang="ar")

    system = bodies[0]["messages"][0]["content"]
    assert "Arabic (العربية)" in system
    assert "write names and place names in the conventional form used in the target script" in system
    assert "مونتريال" in system
    assert "keep every name and place name exactly as written" not in system


# --- unit boundaries: pauses and whisper segments (card 46 item 7) ----------------------


def _zh_cue(index: int, text: str, *, start: float, end: float, first_token: int) -> Cue:
    return Cue(
        index=index,
        start=start,
        end=end,
        lines=[text],
        speaker="S1",
        word_span=(first_token, first_token + 3),
    )


_JUDGE_FIRST = "有4人被殺 很多人受傷"
_JUDGE_SECOND = "在坦桑尼亞 警方說"


def test_a_long_pause_ends_a_unit_so_separate_news_items_are_not_merged() -> None:
    """Measured: two cues 4.8 s apart were translated as one false sentence (card 46)."""
    first = _zh_cue(1, _JUDGE_FIRST, start=0.0, end=4.8, first_token=0)
    second = _zh_cue(2, _JUDGE_SECOND, start=9.6, end=13.0, first_token=4)

    units = units_from_cues([first, second])

    assert [unit.text for unit in units] == [_JUDGE_FIRST, _JUDGE_SECOND]
    assert [unit.cue_indices for unit in units] == [(0,), (1,)]


async def test_separate_units_are_translated_as_separate_batch_items(
    make_settings: MakeSettings,
) -> None:
    """The two news items stay two translation units, so neither translation can merge them."""
    settings = make_settings()
    first = _zh_cue(1, _JUDGE_FIRST, start=0.0, end=4.8, first_token=0)
    second = _zh_cue(2, _JUDGE_SECOND, start=9.6, end=13.0, first_token=4)
    units = units_from_cues([first, second])
    payloads: list[list[str]] = []
    english = {
        _JUDGE_FIRST: "Four people were killed and many were injured.",
        _JUDGE_SECOND: "In Tanzania, police said.",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        cues = _payload(request)["cues"]
        payloads.append([item["text"] for item in cues])
        return json_response({"cues": [{"i": item["i"], "text": english[item["text"]]} for item in cues]})

    translated, failed, _bodies = await _translate(settings, units, handler, lang="en")

    # Both units travel in one batch, but as two independent items: one unit each.
    assert payloads == [[_JUDGE_FIRST, _JUDGE_SECOND]]
    assert translated == [english[_JUDGE_FIRST], english[_JUDGE_SECOND]]
    assert "Tanzania" not in translated[0]
    assert failed == 0


def test_a_short_pause_does_not_end_a_unit() -> None:
    cues = [
        _zh_cue(1, "第一部分", start=0.0, end=2.0, first_token=0),
        _zh_cue(2, "第二部分", start=2.9, end=5.0, first_token=3),
    ]

    units = units_from_cues(cues)

    assert [unit.cue_indices for unit in units] == [(0, 1)]


def test_a_whisper_segment_boundary_ends_a_cjk_unit_without_punctuation() -> None:
    first = _zh_cue(1, _JUDGE_FIRST, start=0.0, end=3.0, first_token=0)
    second = _zh_cue(2, _JUDGE_SECOND, start=3.0, end=6.0, first_token=4)

    units = units_from_cues([first, second], segment_starts={4})

    assert [unit.text for unit in units] == [_JUDGE_FIRST, _JUDGE_SECOND]


def test_a_segment_boundary_does_not_end_a_latin_unit() -> None:
    first = _cue(1, "the police said")
    second = _cue(2, "that the search continued", start=3.5)

    units = units_from_cues([first, second], segment_starts={0, 3})

    assert [unit.cue_indices for unit in units] == [(0, 1)]


def test_a_segment_boundary_does_not_end_a_cjk_unit_that_ends_a_sentence() -> None:
    first = _zh_cue(1, "有4人被殺。", start=0.0, end=3.0, first_token=0)
    second = _zh_cue(2, _JUDGE_SECOND, start=3.0, end=6.0, first_token=4)

    units = units_from_cues([first, second], segment_starts={4})

    assert [len(unit.cues) for unit in units] == [1, 1]


def test_segment_token_starts_counts_words_not_cues() -> None:
    words_a = (_word("one", 0.0, 0.4), _word("two", 0.4, 0.8))
    words_b = (_word("three", 0.8, 1.2),)
    transcript = _transcript(words_a, words_b)

    starts = segment_token_starts(transcript)

    assert starts == {0, 2}


async def test_translate_calls_use_the_translate_route(make_settings: MakeSettings) -> None:
    settings = make_settings(
        LLM_ALLOWED_MODELS="local:tr-model",
        LLM_ROUTE_TRANSLATE="local:tr-model",
    )
    units = [_unit([_cue(1, "A few years ago")])]
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        bodies.append(json.loads(request.content))
        return json_response({"cues": [{"i": 0, "text": "Vor ein paar Jahren"}]})

    translated, failed, _bodies = await _translate(settings, units, handler)

    assert translated == ["Vor ein paar Jahren"]
    assert failed == 0
    assert [body["model"] for body in bodies] == ["tr-model"]
