"""Tolerant JSON extraction from LLM text replies (docs/04 §1 step 3).

gemma4 sometimes wraps JSON in a ```json fence or adds prose around it, even when asked for
a tool call. :func:`extract_json` strips fences, then finds the outermost balanced object
while respecting string literals (so braces inside strings are safe).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, cast

from caracal.errors import CaracalError, ErrorCode

_FENCE_RE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\r?\n?(.*?)```", re.DOTALL)


def _balanced_objects(text: str) -> Iterator[str]:
    """Yield every balanced ``{...}`` substring, outermost starts first."""
    for start, character in enumerate(text):
        if character != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            current = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif current == "\\":
                    escaped = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : position + 1]
                    break
        else:
            continue


def _parse_candidate(candidate: str) -> dict[str, Any] | None:
    for blob in _balanced_objects(candidate):
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    return None


def extract_json(text: str) -> dict[str, Any]:
    """Return the first JSON object found in ``text``, fences and prose tolerated."""
    if not isinstance(text, str):
        raise CaracalError(ErrorCode.INTERNAL, "LLM reply is not text")
    candidates = [match.group(1) for match in _FENCE_RE.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        parsed = _parse_candidate(candidate)
        if parsed is not None:
            return parsed
    raise CaracalError(ErrorCode.INTERNAL, "LLM reply contains no JSON object")


__all__ = ["extract_json"]
