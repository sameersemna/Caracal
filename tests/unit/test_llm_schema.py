"""Schema preparation and coercion from the measured gemma4 behaviour (docs/04 §1)."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model
from tests.fakes.ollama import load_fixture

from caracal.llm.schema import (
    DEFAULT_SYNONYMS,
    RESERVED_FIELD_NAMES,
    coerce,
    prepare_schema,
)

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
    entries: list[Inner]
    kind: Literal["a", "b", "c"] = "a"


class Entity(BaseModel):
    """The report keeps a ``type`` field; the LLM sees the reserved-name-free alias."""

    text: str
    type: Literal["person", "org", "place", "date", "other"] = Field(validation_alias="entity_type")
    evidence: list[str]


class Out(BaseModel):
    language: str
    speaker_name: str | None = None
    entities: list[Entity]
    confidence: float = Field(ge=0.0, le=1.0)


class Claim(BaseModel):
    text: str
    evidence: list[str]


class Quote(BaseModel):
    text: str
    quote_evidence: list[str]


class MarkedCitation(BaseModel):
    text: str
    citations: list[Annotated[str, Field(json_schema_extra={"x-evidence": True})]]


class Tagged(BaseModel):
    tags: list[str]


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
    assert prepared["properties"]["entries"]["items"]["properties"]["label"]["type"] == "string"


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


@pytest.mark.parametrize("name", sorted(RESERVED_FIELD_NAMES))
def test_prepare_schema_rejects_reserved_wire_names(name: str) -> None:
    model = create_model(f"Reserved{name.replace('$', 'D')}", **{name: (str, ...)})
    with pytest.raises(ValueError, match=f"reserved field name: {re.escape(name)}"):
        prepare_schema(model)


def test_prepare_schema_rejects_reserved_name_in_nested_model() -> None:
    nested = create_model("NestedReserved", enum=(str, ...))
    wrapper = create_model("WrapperReserved", payload=(nested, ...))
    with pytest.raises(ValueError, match="reserved field name: enum"):
        prepare_schema(wrapper)


def test_prepare_schema_checks_the_alias_not_the_python_field_name() -> None:
    model = create_model("ReservedAlias", entity_type=(str, Field(validation_alias="type")))
    with pytest.raises(ValueError, match="reserved field name: type"):
        prepare_schema(model)


def test_validation_alias_is_the_wire_name_and_keeps_the_field_name() -> None:
    prepared = prepare_schema(Entity)
    entity_props = prepared["properties"]

    assert set(entity_props) == {"text", "entity_type", "evidence"}
    assert entity_props["entity_type"]["enum"] == ["person", "org", "place", "date", "other"]
    assert prepared["required"] == ["text", "entity_type", "evidence"]

    value = Entity.model_validate(
        {"text": "Berlin", "entity_type": "place", "evidence": ["seg:1"]},
        by_alias=True,
        by_name=True,
    )
    assert value.type == "place"
    assert value.model_dump() == {"text": "Berlin", "type": "place", "evidence": ["seg:1"]}


def test_aliased_nested_model_validates_wire_name_into_field_name() -> None:
    value = Out.model_validate(
        {
            "language": "de",
            "entities": [{"text": "Berlin", "entity_type": "place", "evidence": ["seg:1"]}],
            "confidence": 0.9,
        },
        by_alias=True,
        by_name=True,
    )

    assert value.entities[0].type == "place"
    assert value.model_dump()["entities"] == [{"text": "Berlin", "type": "place", "evidence": ["seg:1"]}]


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (["seg:2 S1"], ["seg:2"]),
        (["seg:2, seg:5"], ["seg:2", "seg:5"]),
        (["see segment 2"], []),
        (["meta:title"], ["meta:title"]),
        (["seg:2", "seg:2", "web:3"], ["seg:2", "web:3"]),
        (["web:3", "seg:1"], ["web:3", "seg:1"]),
        (["vp:S1", "caption:4", "hint:0"], ["vp:S1", "caption:4", "hint:0"]),
        # Typed ids only (round 17 carry-forward): pyannote labels are renamed S1..Sn,
        # and sentence punctuation is never part of an id.
        (["vp:SPEAKER_00"], []),
        (["seg:2."], ["seg:2"]),
        (["caption:4."], ["caption:4"]),
        (["meta:title."], ["meta:title"]),
        ("seg:7 S3 00:05.0 | hi", ["seg:7"]),
        ([42, "seg:1"], ["seg:1"]),
    ],
)
def test_coerce_normalises_evidence_ids(value: object, expected: list[str]) -> None:
    prepared = prepare_schema(Claim)
    coerced = coerce({"text": "x", "evidence": value}, prepared)

    assert coerced["evidence"] == expected


def test_coerce_normalises_evidence_suffix_and_x_evidence_marker() -> None:
    quote_schema = prepare_schema(Quote)
    assert coerce({"text": "x", "quote_evidence": ["seg:9 S1"]}, quote_schema)["quote_evidence"] == ["seg:9"]

    marker_schema = prepare_schema(MarkedCitation)
    assert marker_schema["properties"]["citations"]["items"]["x-evidence"] is True
    coerced = coerce({"text": "x", "citations": ["caption:2, web:1"]}, marker_schema)
    assert coerced["citations"] == ["caption:2", "web:1"]


def test_coerce_leaves_string_arrays_that_are_not_evidence_untouched() -> None:
    prepared = prepare_schema(Tagged)
    assert coerce({"tags": ["seg:2, seg:5", "plain"]}, prepared) == {"tags": ["seg:2, seg:5", "plain"]}


def _wire_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Recorded before the reserved-name rule: entity ``type`` is now the ``entity_type`` alias."""
    converted = copy.deepcopy(arguments)
    for entity in converted["entities"]:
        entity["entity_type"] = entity.pop("type")
    return converted


def test_inlined_tool_call_fixture_coerces_to_valid_model() -> None:
    message = load_fixture("tool_call_inlined_schema")["message"]
    arguments = _wire_arguments(message["tool_calls"][0]["function"]["arguments"])

    prepared = prepare_schema(Out)
    coerced = coerce(arguments, prepared, synonyms=DEFAULT_SYNONYMS)
    result = Out.model_validate(coerced, by_alias=True, by_name=True)

    assert result.language == "German"
    assert result.speaker_name == "Angela Merkel"
    assert [entity.text for entity in result.entities] == ["Angela Merkel", "Berlin", "Siemens"]
    assert [entity.type for entity in result.entities] == ["person", "place", "org"]
    assert [entity.evidence for entity in result.entities] == [["seg:0"], ["seg:1"], ["seg:1"]]
    assert result.confidence == 1.0


def test_defs_tool_call_fixture_does_not_validate_after_coercion() -> None:
    """The $defs schema made gemma4 invent start/end instead of evidence: inlining is mandatory."""
    message = load_fixture("tool_call_defs_schema")["message"]
    arguments = _wire_arguments(message["tool_calls"][0]["function"]["arguments"])

    prepared = prepare_schema(Out)
    coerced = coerce(arguments, prepared, synonyms=DEFAULT_SYNONYMS)
    with pytest.raises(ValidationError) as caught:
        Out.model_validate(coerced, by_alias=True, by_name=True)

    assert any(error["loc"][-1] == "evidence" for error in caught.value.errors())
