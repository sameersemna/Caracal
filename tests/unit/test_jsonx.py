"""Tolerant JSON extraction from LLM replies (docs/04 §1)."""

from __future__ import annotations

import pytest

from caracal.errors import CaracalError, ErrorCode
from caracal.llm.jsonx import extract_json


def test_fenced_json_is_unwrapped() -> None:
    text = 'Here you go:\n```json\n{"language": "de", "confidence": 0.9}\n```\nDone.'
    assert extract_json(text) == {"language": "de", "confidence": 0.9}


def test_fence_without_language_tag() -> None:
    assert extract_json('```\n{"ok": true}\n```') == {"ok": True}


def test_prose_around_json() -> None:
    text = 'Sure. {"status": "ok", "count": 2} That is all.'
    assert extract_json(text) == {"status": "ok", "count": 2}


def test_braces_inside_strings_are_not_counted() -> None:
    text = 'prefix {"text": "a } b { c", "nested": {"x": 1}} suffix'
    assert extract_json(text) == {"text": "a } b { c", "nested": {"x": 1}}


def test_escaped_quotes_and_backslashes() -> None:
    text = r'{"quote": "she said \"hi\"", "path": "c:\\tmp"}'
    assert extract_json(text) == {"quote": 'she said "hi"', "path": "c:\\tmp"}


def test_nested_object_returns_outermost() -> None:
    assert extract_json('{"a": {"b": {"c": 3}}, "d": 4}') == {"a": {"b": {"c": 3}}, "d": 4}


def test_prose_with_stray_brace_before_json() -> None:
    assert extract_json('the set {a, b} and then {"ok": 1}') == {"ok": 1}


def test_no_json_raises_internal_error() -> None:
    with pytest.raises(CaracalError) as caught:
        extract_json("I cannot answer that.")
    assert caught.value.code is ErrorCode.INTERNAL
    assert caught.value.retryable is False


def test_json_array_is_not_an_object() -> None:
    with pytest.raises(CaracalError) as caught:
        extract_json("[1, 2, 3]")
    assert caught.value.code is ErrorCode.INTERNAL
