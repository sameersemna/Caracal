"""Prompt loading with safe substitution and DATA-block wrapping (docs/04 §1, docs/06 S9).

Templates use ``{placeholder}`` tokens. Substitution is a two-phase ``str.replace`` so braces
inside substituted DATA are never interpreted as placeholders (``str.format`` would do that).
"""

from __future__ import annotations

import re
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent
_GUARD = "_guard"
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _read(name: str) -> str:
    if _NAME_RE.fullmatch(name) is None:
        raise ValueError(f"invalid prompt name: {name!r}")
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def load_prompt(name: str, **variables: object) -> str:
    """Load ``<name>.md``, substitute ``{placeholders}`` and prepend the guard block.

    Raises :class:`KeyError` when a placeholder in the template has no variable. Values are
    inserted verbatim; braces inside them are data, not placeholders.
    """
    template = _read(name)
    for match in _PLACEHOLDER_RE.finditer(template):
        if match.group(1) not in variables:
            raise KeyError(match.group(1))

    rendered = template
    sentinels: dict[str, str] = {}
    for index, (key, value) in enumerate(variables.items()):
        sentinel = f"\x00{index}\x00"
        rendered = rendered.replace("{" + key + "}", sentinel)
        sentinels[sentinel] = str(value)
    for sentinel, value in sentinels.items():
        rendered = rendered.replace(sentinel, value)

    guard = _read(_GUARD).rstrip()
    return f"{guard}\n\n{rendered.lstrip()}"


def wrap_data(label: str, text: str) -> str:
    """Wrap untrusted text in a DATA block that it cannot close (docs/06 S9)."""
    safe = text.replace("<<<", "").replace(">>>", "")
    return f"<<<DATA:{label}>>>\n{safe}\n<<<END:{label}>>>"


__all__ = ["load_prompt", "wrap_data"]
