"""JSON-schema preparation and output coercion for the LLM client (docs/04 §1).

Measured 2026-11-14: gemma4 invents field names when the tool schema uses pydantic's
default ``$defs``/``$ref`` layout, and even with an inlined schema it emits enum synonyms
and upper-cased enum values. It also omits any property literally named ``type`` because
that collides with the JSON-Schema keyword (0/6 valid; renaming gave 6/6), so
:func:`prepare_schema` rejects such wire names and pydantic validation aliases are the
escape hatch. :func:`coerce` repairs the model's arguments before pydantic validation,
including evidence lists, which gemma4 copies as whole transcript line prefixes
(``"seg:2 S1"``) or with sentence punctuation (``"seg:2."``). The ids themselves are typed
and shared with the report through :data:`caracal.report.evidence.EVIDENCE_ID_RE`.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from functools import cache
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from caracal.report.evidence import EVIDENCE_ID_RE

ModelT = TypeVar("ModelT", bound=BaseModel)

#: JSON-Schema keyword names that gemma4 will not emit as model properties (docs/04 §1).
RESERVED_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "type",
        "items",
        "properties",
        "enum",
        "format",
        "required",
        "default",
        "const",
        "anyOf",
        "oneOf",
        "allOf",
        "$ref",
    }
)

DEFAULT_SYNONYMS: dict[str, str] = {
    # entity type: place
    "location": "place",
    "loc": "place",
    "gpe": "place",
    "city": "place",
    "country": "place",
    # entity type: organisation
    "organization": "org",
    "organisation": "org",
    "company": "org",
    "institution": "org",
    # entity type: person
    "per": "person",
    "people": "person",
    "human": "person",
    # entity type: date
    "time": "date",
    "year": "date",
}

_ENUM_HINT = "Must be exactly one of: {}."
_REF_PREFIX = "#/$defs/"


def _inline(value: Any, defs: Mapping[str, Any], stack: tuple[str, ...]) -> Any:
    """Recursively replace ``$ref`` nodes with a copy of their ``$defs`` definition."""
    if isinstance(value, dict):
        node = cast(dict[str, Any], value)
        if "$ref" in node:
            return _resolve_ref(node, defs, stack)
        return {key: _inline(item, defs, stack) for key, item in node.items()}
    if isinstance(value, list):
        return [_inline(item, defs, stack) for item in value]
    return value


def _resolve_ref(node: dict[str, Any], defs: Mapping[str, Any], stack: tuple[str, ...]) -> Any:
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith(_REF_PREFIX):
        raise ValueError(f"unsupported JSON schema $ref: {ref!r}")
    name = ref[len(_REF_PREFIX) :]
    if name in stack:
        raise ValueError(f"recursive JSON schema {name!r} cannot be inlined")
    definition = defs.get(name)
    if not isinstance(definition, dict):
        raise ValueError(f"unresolved JSON schema $ref: {ref!r}")
    resolved = _inline(copy.deepcopy(definition), defs, (*stack, name))
    siblings = {key: _inline(item, defs, stack) for key, item in node.items() if key != "$ref"}
    if not siblings:
        return resolved
    merged = dict(resolved)
    merged.update(siblings)
    return merged


def _check_reserved_names(node: object) -> None:
    """Raise for every declared property whose **wire** name is a JSON-Schema keyword.

    Runs on the raw ``model_json_schema()`` output so nested models (still under ``$defs``)
    are checked too; pydantic already renders ``validation_alias`` as the property name, which
    is exactly the name the model sees and therefore the name that must not be reserved.
    """
    if isinstance(node, dict):
        node_dict = cast(dict[str, Any], node)
        properties = node_dict.get("properties")
        if isinstance(properties, dict):
            for name in cast(dict[str, Any], properties):
                if name in RESERVED_FIELD_NAMES:
                    raise ValueError(f"reserved field name: {name}")
        for value in node_dict.values():
            _check_reserved_names(value)
    elif isinstance(node, list):
        for item in node:
            _check_reserved_names(item)


def _finish(node: dict[str, Any]) -> None:
    """Drop ``title`` and append the enum hint, walking schema nodes structurally."""
    node.pop("title", None)
    options = node.get("enum")
    if isinstance(options, list) and options:
        hint = _ENUM_HINT.format(", ".join(str(option) for option in options))
        description = node.get("description")
        if isinstance(description, str) and description.strip():
            node["description"] = f"{description.rstrip()} {hint}"
        else:
            node["description"] = hint
    properties = node.get("properties")
    if isinstance(properties, dict):
        for subschema in cast(dict[str, Any], properties).values():
            if isinstance(subschema, dict):
                _finish(cast(dict[str, Any], subschema))
    items = node.get("items")
    if isinstance(items, dict):
        _finish(cast(dict[str, Any], items))
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        branches = node.get(key)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict):
                    _finish(cast(dict[str, Any], branch))
    additional = node.get("additionalProperties")
    if isinstance(additional, dict):
        _finish(cast(dict[str, Any], additional))


@cache
def prepare_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return the cached tool schema for ``model_cls`` with inlined refs and enum hints.

    Recursive models raise :class:`ValueError`: an inlined schema cannot represent a cycle.
    A field whose wire name (``validation_alias`` if set, else the field name) is a
    JSON-Schema keyword raises :class:`ValueError`; alias the field instead.
    """
    schema = model_cls.model_json_schema(by_alias=True)
    _check_reserved_names(schema)
    defs = schema.pop("$defs", {})
    if not isinstance(defs, dict):
        raise ValueError("JSON schema $defs must be an object")
    inlined = _inline(schema, cast(dict[str, Any], defs), ())
    if not isinstance(inlined, dict):
        raise ValueError("JSON schema root must be an object")
    prepared = cast(dict[str, Any], inlined)
    _finish(prepared)
    return prepared


def _merged_synonyms(synonyms: Mapping[str, str]) -> dict[str, str]:
    merged = dict(DEFAULT_SYNONYMS)
    for key, value in synonyms.items():
        merged[str(key).strip().lower()] = value
    return merged


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return not text or text.lower() == "null"
    return False


def _type_matches(value: Any, type_name: Any) -> bool:
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "array":
        return isinstance(value, (list, tuple))
    if type_name == "object":
        return isinstance(value, Mapping)
    return False


def _coerce_enum(value: Any, options: list[Any], synonyms: Mapping[str, str]) -> Any:
    if value is None:
        return None
    for option in options:
        if value == option:
            return option
        if isinstance(value, str) and str(option).casefold() == value.strip().casefold():
            return option
    if not isinstance(value, str):
        return value
    folded = str(value).strip().casefold()
    for option in options:
        if str(option).casefold() == folded:
            return option
    target = synonyms.get(folded)
    if target is not None:
        for option in options:
            if str(option).casefold() == str(target).casefold():
                return option
    return value


def _coerce_number(value: Any, schema: Mapping[str, Any], *, integer: bool) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            number = float(text)
        except ValueError:
            return value
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        return value

    coerced: int | float
    if integer:
        coerced = int(number) if number.is_integer() else number
    else:
        coerced = number

    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        if float(coerced) < float(minimum):
            coerced = int(minimum) if integer else float(minimum)
    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        if float(coerced) > float(maximum):
            coerced = int(maximum) if integer else float(maximum)
    return coerced


def _coerce_union(value: Any, branches: list[Any], synonyms: Mapping[str, str]) -> Any:
    typed = [branch for branch in branches if isinstance(branch, Mapping)]
    candidates = [cast(Mapping[str, Any], branch) for branch in typed]
    nullable = any(branch.get("type") == "null" for branch in candidates)
    if nullable and _is_nullish(value):
        return None
    non_null = [branch for branch in candidates if branch.get("type") != "null"]
    if not non_null:
        return value
    if isinstance(value, str):
        for branch in non_null:
            options = branch.get("enum")
            if isinstance(options, list):
                coerced = _coerce_enum(value, options, synonyms)
                if coerced in options:
                    return coerced
    for branch in non_null:
        if _type_matches(value, branch.get("type")):
            return coerce(value, branch, synonyms=synonyms)
    return coerce(value, non_null[0], synonyms=synonyms)


def _array_schema(schema: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the array branch of a property schema, looking through ``anyOf``/``oneOf``."""
    if schema.get("type") == "array":
        return schema
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if isinstance(branch, Mapping) and cast(Mapping[str, Any], branch).get("type") == "array":
                return cast(Mapping[str, Any], branch)
    return None


def _is_evidence_property(name: str, schema: Mapping[str, Any]) -> bool:
    """True when a property holds evidence ids (by name, ``_evidence`` suffix or marker)."""
    array = _array_schema(schema)
    if array is None:
        return False
    if name == "evidence" or name.endswith("_evidence"):
        return True
    items = array.get("items")
    return isinstance(items, Mapping) and cast(Mapping[str, Any], items).get("x-evidence") is True


def _normalise_evidence(value: Any) -> list[str]:
    """Flatten a value into de-duplicated evidence ids, dropping strings without one.

    gemma4 copies the whole rendered line prefix (``"seg:2 S1"``), a comma-separated list
    (``"seg:2, seg:5"``) or sentence punctuation (``"seg:2."``); only the typed id itself is
    evidence (docs/04 §1, §9).
    """
    if isinstance(value, str):
        candidates: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        candidates = [value]
    ids: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        for found in EVIDENCE_ID_RE.findall(candidate):
            if found in seen:
                continue
            seen.add(found)
            ids.append(found)
    return ids


def coerce(
    value: Any, schema: Mapping[str, Any], *, synonyms: Mapping[str, str] = DEFAULT_SYNONYMS
) -> object:
    """Repair a model-supplied value against its JSON schema (docs/04 §1 step 2).

    Handles enum case/synonyms, numeric strings, ``"null"``/``""`` for nullable fields,
    scalar-to-list promotion, numeric clamping, recursive object/array coercion and evidence
    lists (ids extracted from rendered transcript prefixes, de-duplicated). Unknown extra keys
    are left untouched for pydantic's own model config to handle.
    """
    if not isinstance(schema, Mapping):
        return value
    mapping = schema
    merged = _merged_synonyms(synonyms)

    union = mapping.get("anyOf") or mapping.get("oneOf")
    if isinstance(union, list):
        return _coerce_union(value, union, merged)

    type_name = mapping.get("type")
    if isinstance(type_name, list):
        nullable = "null" in type_name
        if nullable and _is_nullish(value):
            return None
        for candidate in type_name:
            if candidate != "null":
                return coerce(value, {**mapping, "type": candidate}, synonyms=merged)
        return value
    if mapping.get("nullable") is True and _is_nullish(value):
        return None

    options = mapping.get("enum")
    if isinstance(options, list):
        return _coerce_enum(value, options, merged)

    if type_name == "array":
        items = mapping.get("items")
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if isinstance(items, Mapping):
            item_schema = cast(Mapping[str, Any], items)
            return [coerce(item, item_schema, synonyms=merged) for item in values]
        return values

    if type_name == "object":
        if not isinstance(value, Mapping):
            return value
        source = cast(Mapping[str, Any], value)
        result = dict(source)
        properties = mapping.get("properties")
        if isinstance(properties, Mapping):
            for key, subschema in cast(Mapping[str, Any], properties).items():
                if key in result and isinstance(subschema, Mapping):
                    node = cast(Mapping[str, Any], subschema)
                    if _is_evidence_property(key, node):
                        result[key] = _normalise_evidence(result[key])
                    else:
                        result[key] = coerce(result[key], node, synonyms=merged)
        additional = mapping.get("additionalProperties")
        if isinstance(additional, Mapping):
            known = set(properties) if isinstance(properties, Mapping) else set()
            for key in result:
                if key not in known:
                    result[key] = coerce(result[key], cast(Mapping[str, Any], additional), synonyms=merged)
        return result

    if type_name in ("number", "integer"):
        return _coerce_number(value, mapping, integer=type_name == "integer")

    if type_name == "boolean" and isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false"):
            return text == "true"

    return value


__all__ = ["DEFAULT_SYNONYMS", "RESERVED_FIELD_NAMES", "coerce", "prepare_schema"]
