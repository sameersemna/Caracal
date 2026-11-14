"""Sentence-unit translation and cue redistribution (docs/04 §10).

Per-cue translation produces grammatical fragments ("A few" becomes "Jahren" in German), so
translation works on *units*: consecutive cues of one speaker up to a sentence end, at most
four cues or 200 characters. Units are translated in JSON-mode batches of about 25 (the
measured gemma4 recipe: exact index sets on 4/4 batches), retried once on an index mismatch
and then one call per unit. A unit that still fails keeps its source text and is counted.

The translated unit is redistributed over its source cues at word boundaries (characters for
CJK), proportionally to the source cues' character counts, keeping the original timings, and
re-wrapped to at most 42 characters per line. Everything in this module is deterministic
except the LLM call; no filesystem or network access happens here.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from caracal.asr import LANGUAGE_NAMES
from caracal.errors import CaracalError
from caracal.llm.client import LLMClient
from caracal.llm.prompts import load_prompt
from caracal.log import get_logger
from caracal.subtitles.segmenter import Cue, wrap_lines

logger = get_logger("caracal.analysis.translate")

#: The file a queued ``translate`` delivery reads its languages from (written by the service).
TRANSLATE_REQUEST_FILENAME: Final = "translate-request.json"

_TARGET_LANG_RE: Final = re.compile(r"^[a-z]{2,3}$")
_SENTENCE_ENDINGS: Final = frozenset(".?!…。！？؟")  # noqa: RUF001 - CJK/Arabic punctuation
_CLOSING_CHARS: Final = "\"'”’»)]}"  # noqa: RUF001 - typographic quotes
_MAX_UNIT_CUES: Final = 4
_MAX_UNIT_CHARS: Final = 200
_BATCH_SIZE: Final = 25
_MAX_GLOSSARY: Final = 60
_MAX_LINE_CHARS: Final = 42
_MAX_LINES: Final = 2
_CJK_LANGS: Final = frozenset({"zh", "ja", "ko", "yue"})

#: Native names make the target unambiguous for the model ("German (Deutsch)").
_NATIVE_NAMES: Final[dict[str, str]] = {
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
    "it": "Italiano",
    "pt": "Português",
    "nl": "Nederlands",
    "sv": "Svenska",
    "da": "Dansk",
    "no": "Norsk",
    "fi": "Suomi",
    "pl": "Polski",
    "cs": "Čeština",
    "sk": "Slovenčina",
    "hu": "Magyar",
    "ro": "Română",
    "bg": "Български",
    "hr": "Hrvatski",
    "sr": "Српски",
    "sl": "Slovenščina",
    "uk": "Українська",
    "ru": "Русский",
    "el": "Ελληνικά",
    "tr": "Türkçe",
    "ar": "العربية",
    "he": "עברית",
    "fa": "فارسی",
    "ur": "اردو",
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "ml": "മലയാളം",
    "kn": "ಕನ್ನಡ",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "pa": "ਪੰਜਾਬੀ",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
    "vi": "Tiếng Việt",
    "th": "ไทย",
    "id": "Bahasa Indonesia",
    "ms": "Bahasa Melayu",
    "sw": "Kiswahili",
    "af": "Afrikaans",
    "ca": "Català",
    "gl": "Galego",
    "eu": "Euskara",
    "is": "Íslenska",
    "lt": "Lietuvių",
    "lv": "Latviešu",
    "et": "Eesti",
    "sq": "Shqip",
    "mk": "Македонски",
    "bs": "Bosanski",
    "ka": "ქართული",
    "hy": "Հայերեն",
    "az": "Azərbaycan",
    "kk": "Қазақ",
    "uz": "O'zbek",
    "mn": "Монгол",
    "ne": "नेपाली",
    "si": "සිංහල",
    "km": "ខ្មែរ",
    "lo": "ລາວ",
    "my": "မြန်မာ",
    "tl": "Tagalog",
    "cy": "Cymraeg",
    "ga": "Gaeilge",
    "mt": "Malti",
    "yi": "ייִדיש",
}


class _TranslatedCue(BaseModel):
    """One translated unit: the input index and its translated text."""

    model_config = ConfigDict(extra="ignore")

    i: int
    text: str = Field(min_length=1)


class _TranslationBatch(BaseModel):
    """The JSON-mode output object. ``cues`` is deliberate: ``items`` is reserved."""

    model_config = ConfigDict(extra="ignore")

    cues: list[_TranslatedCue] = Field(default_factory=list)


class TranslatedUnits(list[str]):
    """Translated texts in unit order (a ``list[str]``) plus the fallback indices."""

    def __init__(self, values: Iterable[str], *, failed_indices: Iterable[int] = ()) -> None:
        super().__init__(values)
        self.failed_indices: frozenset[int] = frozenset(failed_indices)

    @property
    def failed_units(self) -> int:
        """How many units kept their source text."""
        return len(self.failed_indices)


@dataclass(frozen=True, slots=True)
class TranslationUnit:
    """Consecutive cues of one speaker up to a sentence boundary (docs/04 §10)."""

    index: int
    cues: tuple[Cue, ...]
    cue_indices: tuple[int, ...]
    text: str
    speaker: str | None


def is_valid_target_lang(code: str) -> bool:
    """True for a known ISO 639 code matching ``^[a-z]{2,3}$`` (docs/03 §2)."""
    return bool(_TARGET_LANG_RE.fullmatch(code)) and code in LANGUAGE_NAMES


def normalise_target_langs(codes: Sequence[str]) -> list[str]:
    """De-duplicate language codes, preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


def resolve_target_langs(
    requested: Sequence[str] | None,
    defaults: Sequence[str],
    languages: Sequence[Mapping[str, object]],
    transcript_language: str | None,
) -> list[str]:
    """Target languages minus the dominant/source language (docs/03 §2, docs/04 §2)."""
    candidates = list(defaults) if requested is None else list(requested)
    dominant: str | None = None
    best_share = -1.0
    for language in languages:
        code = language.get("code")
        share = language.get("share")
        if not isinstance(code, str) or code == "und":
            continue
        value = float(share) if isinstance(share, (int, float)) and not isinstance(share, bool) else 0.0
        if value > best_share:
            dominant = code
            best_share = value
    if dominant is None:
        dominant = transcript_language
    return [
        code for code in normalise_target_langs([code for code in candidates if code]) if code != dominant
    ]


def cue_text(cue: Cue) -> str:
    """The cue's visible text with single spaces between its rendered lines."""
    return " ".join(part.strip() for part in cue.lines if part.strip()).strip()


def _ends_sentence(text: str) -> bool:
    trimmed = text.rstrip().rstrip(_CLOSING_CHARS)
    return bool(trimmed) and trimmed[-1] in _SENTENCE_ENDINGS


def _unit(entries: Sequence[tuple[int, Cue]]) -> TranslationUnit:
    text = " ".join(cue_text(cue) for _position, cue in entries if cue_text(cue))
    return TranslationUnit(
        index=0,
        cues=tuple(cue for _position, cue in entries),
        cue_indices=tuple(position for position, _cue in entries),
        text=text,
        speaker=entries[0][1].speaker,
    )


def units_from_cues(cues: Sequence[Cue]) -> list[TranslationUnit]:
    """Group cues into sentence units: same speaker, sentence end, 4 cues / 200 chars."""
    units: list[TranslationUnit] = []
    current: list[tuple[int, Cue]] = []
    for position, cue in enumerate(cues):
        if not current:
            current = [(position, cue)]
            continue
        previous = current[-1][1]
        combined = " ".join([*(cue_text(item) for _pos, item in current), cue_text(cue)])
        too_long = len(combined) > _MAX_UNIT_CHARS
        too_many = len(current) >= _MAX_UNIT_CUES
        if cue.speaker != previous.speaker or _ends_sentence(cue_text(previous)) or too_long or too_many:
            units.append(_unit(current))
            current = [(position, cue)]
        else:
            current.append((position, cue))
    if current:
        units.append(_unit(current))
    return [
        TranslationUnit(
            index=index,
            cues=unit.cues,
            cue_indices=unit.cue_indices,
            text=unit.text,
            speaker=unit.speaker,
        )
        for index, unit in enumerate(units)
    ]


def _tokens(text: str, *, break_chars: bool) -> list[str]:
    if break_chars:
        return [character for character in text if not character.isspace()]
    return text.split()


def _counts(total: int, weights: Sequence[int]) -> list[int]:
    """Proportional token counts with at least one token per cue (``total >= len``)."""
    if not weights:
        return []
    remaining = total
    counts: list[int] = []
    for index, weight in enumerate(weights[:-1]):
        remaining_weight = sum(weights[index:])
        ideal = remaining * weight / remaining_weight if remaining_weight else 1.0
        reserve = len(weights) - index - 1
        count = max(1, round(ideal))
        count = min(count, max(1, remaining - reserve))
        counts.append(count)
        remaining -= count
    counts.append(max(1, remaining))
    return counts


def _char_chunks(text: str, weights: Sequence[int]) -> list[str]:
    """Character fallback when a translation has fewer words than source cues."""
    visible = [character for character in text if not character.isspace()]
    counts = _counts(len(visible), weights)
    chunks: list[str] = []
    offset = 0
    for count in counts:
        chunks.append("".join(visible[offset : offset + count]))
        offset += count
    return chunks


def redistribute(
    unit: TranslationUnit,
    translated: str,
    target_lang: str,
) -> tuple[list[Cue], int]:
    """Spread one translated unit over its source cues, keeping their timings.

    Returns the translated cues and the number of cues that needed more than two lines (or
    a line wider than 42 characters). RTL languages get no injected bidi marks: the model's
    text is kept exactly as returned.
    """
    if not unit.cues:
        return [], 0
    cjk = target_lang in _CJK_LANGS
    tokens = _tokens(translated, break_chars=cjk)
    weights = [max(1, len(cue_text(cue))) for cue in unit.cues]
    if len(tokens) >= len(unit.cues):
        counts = _counts(len(tokens), weights)
        chunks: list[str] = []
        offset = 0
        for count in counts:
            chunk_tokens = tokens[offset : offset + count]
            chunks.append("".join(chunk_tokens) if cjk else " ".join(chunk_tokens))
            offset += count
    else:
        chunks = _char_chunks(translated, weights)

    cues: list[Cue] = []
    overlong = 0
    for cue, chunk in zip(unit.cues, chunks, strict=True):
        lines, is_overlong = wrap_lines(
            chunk,
            max_chars=_MAX_LINE_CHARS,
            max_lines=_MAX_LINES,
            break_chars=cjk,
        )
        if is_overlong:
            overlong += 1
        cues.append(
            Cue(
                index=cue.index,
                start=cue.start,
                end=cue.end,
                lines=lines or [chunk],
                speaker=cue.speaker,
                word_span=cue.word_span,
            )
        )
    return cues, overlong


def _payload(texts: Sequence[str]) -> str:
    return json.dumps(
        {"cues": [{"i": index, "text": text} for index, text in enumerate(texts)]},
        ensure_ascii=False,
    )


def _validate_batch(cues: Sequence[_TranslatedCue], size: int) -> dict[int, str] | None:
    mapping: dict[int, str] = {}
    for cue in cues:
        if cue.i in mapping or not 0 <= cue.i < size:
            return None
        text = cue.text.strip()
        if not text:
            return None
        mapping[cue.i] = text
    if set(mapping) != set(range(size)):
        return None
    return mapping


async def _call(
    llm: LLMClient,
    *,
    system: str,
    texts: Sequence[str],
) -> dict[int, str] | None:
    result = await llm.call_tool(
        prompt=_payload(texts),
        system=system,
        model_cls=_TranslationBatch,
        tool_name="translate_cues",
        tool_description="Return the translated subtitle units with their input indices i.",
        temperature=0.2,
        max_tokens=4096,
        mode="json",
    )
    return _validate_batch(result.value.cues, len(texts))


async def _translate_one(
    llm: LLMClient,
    *,
    system: str,
    text: str,
) -> str | None:
    try:
        mapping = await _call(llm, system=system, texts=[text])
    except CaracalError as exc:
        logger.warning("translate.unit_failed", extra={"code": exc.code.value})
        return None
    return mapping.get(0) if mapping is not None else None


async def _translate_batch(
    units: Sequence[TranslationUnit],
    *,
    system: str,
    llm: LLMClient,
    offset: int,
) -> tuple[list[str], set[int]]:
    texts = [unit.text for unit in units]
    for _attempt in range(2):
        try:
            mapping = await _call(llm, system=system, texts=texts)
        except CaracalError as exc:
            logger.warning("translate.batch_failed", extra={"code": exc.code.value, "units": len(units)})
            mapping = None
            break
        if mapping is not None:
            return [mapping[index] for index in range(len(units))], set()

    fallbacks = await asyncio.gather(*(_translate_one(llm, system=system, text=unit.text) for unit in units))
    translated: list[str] = []
    failed: set[int] = set()
    for local, (unit, value) in enumerate(zip(units, fallbacks, strict=True)):
        if value is None:
            translated.append(unit.text)
            failed.add(offset + local)
        else:
            translated.append(value)
    return translated, failed


async def translate_units(
    units: Sequence[TranslationUnit],
    target_lang: str,
    *,
    glossary: Sequence[str],
    summary: str,
    llm: LLMClient,
) -> list[str]:
    """Translate sentence units into ``target_lang`` (docs/04 §10).

    Units are sent in batches of 25; a batch whose index set does not match is retried once
    and then re-sent one unit at a time. Unrecoverable units keep their source text and are
    reported in :attr:`TranslatedUnits.failed_indices`.
    """
    if not units:
        return TranslatedUnits([], failed_indices=set())

    english = LANGUAGE_NAMES.get(target_lang, target_lang)
    native = _NATIVE_NAMES.get(target_lang, english)
    names = normalise_target_langs([name.strip() for name in glossary if name.strip()])
    system = load_prompt(
        "translate",
        target_language=english,
        target_native=native,
        glossary=", ".join(names[:_MAX_GLOSSARY]) or "none",
        summary=summary.strip() or "none",
    )

    batches = [units[start : start + _BATCH_SIZE] for start in range(0, len(units), _BATCH_SIZE)]
    results = await asyncio.gather(
        *(
            _translate_batch(batch, system=system, llm=llm, offset=start)
            for start, batch in zip(range(0, len(units), _BATCH_SIZE), batches, strict=True)
        )
    )
    translated: list[str] = []
    failed: set[int] = set()
    for values, failures in results:
        translated.extend(values)
        failed |= failures
    return TranslatedUnits(translated, failed_indices=failed)


__all__ = [
    "TRANSLATE_REQUEST_FILENAME",
    "TranslatedUnits",
    "TranslationUnit",
    "cue_text",
    "is_valid_target_lang",
    "normalise_target_langs",
    "redistribute",
    "resolve_target_langs",
    "translate_units",
    "units_from_cues",
]
