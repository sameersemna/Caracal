"""Schema preparation and coercion from the measured gemma4 behaviour (docs/04 §1)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError
from tests.fakes.ollama import load_fixture

from caracal.llm.schema import DEFAULT_SYNONYMS, coerce, prepare_schema

ENTITY_TYPES = ["person", "org", "place", "date", "other"]
ENTITY_SCHEMA: dict[str, Any] = {"type": "string", "enum": ENTITY_TYPES}


class Inner(BaseModel):
    label: Literal["alpha", "beta"]
    score: float = Field(ge=0.0, le=1.0)


class Middle(BaseModel):
    inner: Inner
    tags: list[str]


class Outer(BaseModel):
    middle: Middle
    items: list[Inner]
    kind: Literal["a", "b", "c"] = "a"


class Entity(BaseModel):
    text: str
    type: Literal["person", "org", "place", "date", "other"]
    evidence: list[str]


class Out(BaseModel):
    language: str
    speaker_name: str | None = None
    entities: list[Entity]
    confidence: float = Field(ge=0.0, le=1.0)


def _nodes(value: object) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nodes(item)


def test_prepare_schema_inlines_refs_in_three_level_model() -> None:
    prepared = prepare_schema(Outer)

    nodes = list(_nodes(prepared))
    assert all("$ref" not in node for node in nodes)
    assert "$defs" not in prepared
    assert "$defs" not in {key for node in nodes for key in node}
    assert prepared["properties"]["middle"]["properties"]["inner"]["properties"]["label"]["type"] == "string"
    assert prepared["properties"]["items"]["items"]["properties"]["label"]["type"] == "string"


def test_prepare_schema_drops_titles_without_dropping_a_field_named_title() -> None:
    class WithTitle(BaseModel):
        title: str
        value: int

    prepared = prepare_schema(WithTitle)
    assert prepared["properties"]["title"] == {"type": "string"}
    assert "title" not in prepared


def test_prepare_schema_adds_enum_hints() -> None:
    prepared = prepare_schema(Outer)

    enum_nodes = [node for node in _nodes(prepared) if "enum" in node]
    assert len(enum_nodes) == 3  # label, kind and none for the float/list
    hints = {tuple(node["enum"]): node["description"] for node in enum_nodes}
    assert hints[("alpha", "beta")].endswith("Must be exactly one of: alpha, beta.")
    assert hints[("a", "b", "c")].endswith("Must be exactly one of: a, b, c.")


def test_prepare_schema_is_cached() -> None:
    assert prepare_schema(Outer) is prepare_schema(Outer)


class Node(BaseModel):
    name: str
    child: Node | None = None


Node.model_rebuild()


def test_prepare_schema_rejects_recursive_models() -> None:
    with pytest.raises(ValueError, match="recursive"):
        prepare_schema(Node)


@pytest.mark.parametrize(
    ("value", "schema", "expected"),
    [
        ("PERSON", ENTITY_SCHEMA, "person"),
        ("location", ENTITY_SCHEMA, "place"),
        ("ORGANIZATION", ENTITY_SCHEMA, "org"),
        ("organisation", ENTITY_SCHEMA, "org"),
        ("time", ENTITY_SCHEMA, "date"),
        ("0.9", {"type": "number"}, 0.9),
        (1.4, {"type": "number", "maximum": 1.0}, 1.0),
        (2, {"type": "integer", "minimum": 3}, 3),
        ("5", {"type": "integer"}, 5),
        ("null", {"anyOf": [{"type": "string"}, {"type": "null"}]}, None),
        ("", {"anyOf": [{"type": "string"}, {"type": "null"}]}, None),
        ("x", {"type": "array", "items": {"type": "string"}}, ["x"]),
    ],
)
def test_coerce_table(value: object, schema: dict[str, Any], expected: object) -> None:
    assert coerce(value, schema, synonyms=DEFAULT_SYNONYMS) == expected


def test_coerce_merges_per_call_synonyms() -> None:
    schema: dict[str, Any] = {"type": "string", "enum": ENTITY_TYPES}
    assert coerce("town", schema, synonyms={"town": "place"}) == "place"


def test_coerce_leaves_unknown_object_keys() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    assert coerce({"count": "7", "extra": {"keep": True}}, schema) == {
        "count": 7,
        "extra": {"keep": True},
    }


def test_inlined_tool_call_fixture_coerces_to_valid_model() -> None:
    message = load_fixture("tool_call_inlined_schema")["message"]
    arguments = message["tool_calls"][0]["function"]["arguments"]

    prepared = prepare_schema(Out)
    coerced = coerce(arguments, prepared, synonyms=DEFAULT_SYNONYMS)
    result = Out.model_validate(coerced)

    assert result.language == "German"
    assert result.speaker_name == "Angela Merkel"
    assert [entity.text for entity in result.entities] == ["Angela Merkel", "Berlin", "Siemens"]
    assert [entity.type for entity in result.entities] == ["person", "place", "org"]
    assert [entity.evidence for entity in result.entities] == [["seg:0"], ["seg:1"], ["seg:1"]]
    assert result.confidence == 1.0


def test_defs_tool_call_fixture_does_not_validate_after_coercion() -> None:
    """The $defs schema made gemma4 invent start/end instead of evidence: inlining is mandatory."""
    message = load_fixture("tool_call_defs_schema")["message"]
    arguments = message["tool_calls"][0]["function"]["arguments"]

    prepared = prepare_schema(Out)
    coerced = coerce(arguments, prepared, synonyms=DEFAULT_SYNONYMS)
    with pytest.raises(ValidationError) as caught:
        Out.model_validate(coerced)

    assert any(error["loc"][-1] == "evidence" for error in caught.value.errors())
