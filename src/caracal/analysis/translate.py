"""Sentence-unit translation and cue redistribution (docs/04 §10).

Per-cue translation produces grammatical fragments ("A few" becomes "Jahren" in German), so
translation works on *units*: consecutive cues of one speaker up to a sentence end, at most
four cues or 200 characters. A unit also ends at a pause longer than one second between cues
in every script, and at a whisper segment boundary when the cue before it is Han/Kana/Hangul/
Thai text without sentence punctuation (whisper writes those scripts without punctuation, so
the segment boundary is the only sentence signal). Units are translated in JSON-mode batches
of about 25 (the measured gemma4 recipe: exact index sets on 4/4 batches), retried once on an
index mismatch and then one call per unit. A unit that still fails keeps its source text and
is counted.

The translated unit is redistributed over its source cues at word boundaries (characters for
CJK), proportionally to the source cues' character counts, keeping the original timings, and
re-wrapped to the same script-aware line limit as source cues (42 columns, 16 characters for
Han/Kana/Hangul/Thai). Everything in this module is deterministic
except the LLM call; no filesystem or network access happens here.
"""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from caracal.asr import LANGUAGE_NAMES, Transcript
from caracal.config import Route
from caracal.errors import CaracalError
from caracal.llm.client import LLMClient
from caracal.llm.prompts import load_prompt
from caracal.log import get_logger
from caracal.subtitles.segmenter import Cue, line_chars, wrap_lines

logger = get_logger("caracal.analysis.translate")

_TARGET_LANG_RE: Final = re.compile(r"^[a-z]{2,3}$")
_SENTENCE_ENDINGS: Final = frozenset(".?!…。！？؟")  # noqa: RUF001 - CJK/Arabic punctuation
_CLOSING_CHARS: Final = "\"'”’»)]}"  # noqa: RUF001 - typographic quotes
_MAX_UNIT_CUES: Final = 4
_MAX_UNIT_CHARS: Final = 200
#: A pause this long between consecutive cues ends the unit in every script (measured: separate
#: news items 4.8 s apart were merged into one sentence and translated falsely).
_UNIT_GAP_S: Final = 1.0
_BATCH_SIZE: Final = 25
_MAX_GLOSSARY: Final = 60
_MAX_LINE_CHARS: Final = 42
_MAX_LINES: Final = 2
#: Targets written without word spaces: tokens are characters, never whitespace pieces.
#: Thai is included because its words run together just like Han/Kana/Hangul.
_CHAR_BREAK_LANGS: Final = frozenset({"zh", "ja", "ko", "yue", "th"})

#: Scripts the name rule and the leak check distinguish (docs/04 §10). Everything alphabetic
#: outside the ranges below is Latin or, when it is a script we do not know, ``"other"``.
Script = Literal[
    "latin",
    "han",
    "kana",
    "hangul",
    "thai",
    "arabic",
    "hebrew",
    "cyrillic",
    "devanagari",
    "other",
]

#: Inclusive code-point ranges per non-Latin script. Digits, punctuation and symbols are never
#: letters, so only ``str.isalpha()`` characters are ever counted (Arabic-Indic digits such as
#: ``٣`` live inside the Arabic block but are not letters).
_SCRIPT_RANGES: Final[tuple[tuple[Script, tuple[tuple[int, int], ...]], ...]] = (
    ("han", ((0x3005, 0x3007), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
    ("kana", ((0x3040, 0x309F), (0x30A0, 0x30FF), (0xFF66, 0xFF9D))),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("hebrew", ((0x0590, 0x05FF), (0xFB1D, 0xFB4F))),
    ("cyrillic", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("devanagari", ((0x0900, 0x097F),)),
)

_NON_LATIN_SCRIPTS: Final = frozenset(
    {"han", "kana", "hangul", "thai", "arabic", "hebrew", "cyrillic", "devanagari"}
)

#: The script a language is conventionally written in. Languages whose script is non-Latin but
#: outside the eight above (Greek, Georgian, Khmer, the Indic scripts ...) are ``"other"``: not
#: Latin, so a name still cannot stay in the source script.
_TARGET_SCRIPTS: Final[dict[str, Script]] = {
    "zh": "han",
    "yue": "han",
    "ja": "kana",
    "ko": "hangul",
    "th": "thai",
    "ar": "arabic",
    "fa": "arabic",
    "ur": "arabic",
    "he": "hebrew",
    "yi": "hebrew",
    "ru": "cyrillic",
    "bg": "cyrillic",
    "uk": "cyrillic",
    "sr": "cyrillic",
    "mk": "cyrillic",
    "be": "cyrillic",
    "kk": "cyrillic",
    "mn": "cyrillic",
    "hi": "devanagari",
    "mr": "devanagari",
    "ne": "devanagari",
}
_OTHER_SCRIPT_LANGS: Final = frozenset(
    {"am", "bn", "el", "gu", "hy", "ka", "km", "kn", "lo", "ml", "my", "pa", "si", "ta", "te"}
)

_LATIN_SAME_SCRIPT_RULE: Final = (
    "keep every name and place name exactly as written in the source, in the Latin script."
)
_SAME_SCRIPT_RULE: Final = "keep every name and place name exactly as written in the source."
_LOCALISED_LATIN_RULE: Final = (
    "write every name and place name in the conventional form used in {language} when one "
    "exists (an exonym such as Moscow for Москва, Tokyo for 東京, or Cairo for القاهرة); "
    "otherwise use a standard romanisation such as pinyin or Hepburn. Never leave the name "
    "in the non-Latin source script."
)
_LOCALISED_SCRIPT_RULE: Final = (
    "write names and place names in the conventional form used in the target script when one "
    "exists (for example Montreal becomes مونتريال in Arabic); otherwise use a standard "
    "transliteration."
)
#: Extra system line for a per-unit retry after a translation leaked source-script letters
#: into a Latin-script target (docs/04 §10, verdict 48).
_LATIN_NAMES_LINE: Final = "Write every name in the Latin script."


def _script_of_char(char: str) -> Script:
    """The script of one alphabetic character (``"other"`` when it is none of the known ones)."""
    code = ord(char)
    for script, ranges in _SCRIPT_RANGES:
        if any(low <= code <= high for low, high in ranges):
            return script
    if "LATIN" in unicodedata.name(char, ""):
        return "latin"
    return "other"


def script_of_text(text: str) -> Script:
    """The script most of ``text``'s letters are written in (``"latin"`` when it has none).

    Detected from the units being translated, not from a language code: a zh job's text is
    Han whatever the chunk-level language guess was.
    """
    counts: dict[Script, int] = {}
    for char in text:
        if not char.isalpha():
            continue
        script = _script_of_char(char)
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "latin"
    return max(counts, key=counts.__getitem__)


def target_script(target_lang: str) -> Script:
    """The script ``target_lang`` is conventionally written in (Latin unless mapped)."""
    return _TARGET_SCRIPTS.get(target_lang, "other" if target_lang in _OTHER_SCRIPT_LANGS else "latin")


def has_script_leak(text: str) -> bool:
    """True when ``text`` still holds letters of a non-Latin script (docs/04 §10).

    Used for Latin-script targets: digits, punctuation, ``°``, ``€`` and every symbol are not
    letters and never count, so ``12 °C`` and ``€5`` are clean.
    """
    return any(char.isalpha() and _script_of_char(char) in _NON_LATIN_SCRIPTS for char in text)


def name_rule(target_lang: str, source_script: Script = "latin") -> str:
    """The prompt rule for proper names, chosen from the source and target script.

    Same script: the name is already in the reader's script, so it stays exactly as written.
    Different script: the conventional target-language form (exonym), else a standard
    romanisation or transliteration, so a zh→en line never mixes Han characters into English
    (verdict 48: a Han name stayed inside an English sentence because the old rule demanded
    the Latin script while also demanding the source spelling).
    """
    target = target_script(target_lang)
    if target == source_script:
        return _LATIN_SAME_SCRIPT_RULE if source_script == "latin" else _SAME_SCRIPT_RULE
    if target == "latin":
        return _LOCALISED_LATIN_RULE.format(language=LANGUAGE_NAMES.get(target_lang, target_lang))
    return _LOCALISED_SCRIPT_RULE


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
    """Translated texts in unit order (a ``list[str]``), fallbacks and script leaks."""

    def __init__(
        self,
        values: Iterable[str],
        *,
        failed_indices: Iterable[int] = (),
        script_leaks: int = 0,
    ) -> None:
        super().__init__(values)
        self.failed_indices: frozenset[int] = frozenset(failed_indices)
        #: Units kept as returned although they still hold non-Latin letters for a Latin target.
        self.script_leaks = script_leaks

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


def _is_no_space_script(text: str) -> bool:
    """Han/Kana/Hangul/Thai text: the segmenter wraps those scripts at 16 characters."""
    return line_chars(text, _MAX_LINE_CHARS) < _MAX_LINE_CHARS


def segment_token_starts(transcript: Transcript) -> set[int]:
    """Global token index at which every whisper segment starts (docs/04 §10).

    Cues carry global token spans (``Cue.word_span``), so membership in this set says whether
    a cue boundary coincides with a whisper segment boundary.
    """
    starts: set[int] = set()
    offset = 0
    for segment in transcript.segments:
        if not segment.words:
            continue
        starts.add(offset)
        offset += len(segment.words)
    return starts


def _ends_unit(
    next_cue: Cue,
    previous: Cue,
    current_len: int,
    cue_count: int,
    segment_starts: Collection[int],
) -> bool:
    """True when ``next_cue`` starts a new translation unit (docs/04 §10)."""
    previous_text = cue_text(previous)
    if next_cue.speaker != previous.speaker:
        return True
    if _ends_sentence(previous_text):
        return True
    if next_cue.start - previous.end > _UNIT_GAP_S:
        return True
    if (
        next_cue.word_span[0] in segment_starts
        and not _ends_sentence(previous_text)
        and _is_no_space_script(previous_text)
    ):
        # Whisper writes CJK/Thai without punctuation; the segment boundary is the sentence.
        return True
    return current_len > _MAX_UNIT_CHARS or cue_count >= _MAX_UNIT_CUES


def units_from_cues(
    cues: Sequence[Cue],
    *,
    segment_starts: Collection[int] = (),
) -> list[TranslationUnit]:
    """Group cues into sentence units: same speaker, sentence end, gap, segment, 4 cues/200 chars."""
    units: list[TranslationUnit] = []
    current: list[tuple[int, Cue]] = []
    for position, cue in enumerate(cues):
        if not current:
            current = [(position, cue)]
            continue
        previous = current[-1][1]
        combined = " ".join([*(cue_text(item) for _pos, item in current), cue_text(cue)])
        if _ends_unit(cue, previous, len(combined), len(current), segment_starts):
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
    a line wider than the script-aware limit). RTL languages get no injected bidi marks: the
    model's text is kept exactly as returned.
    """
    if not unit.cues:
        return [], 0
    char_break = target_lang in _CHAR_BREAK_LANGS
    tokens = _tokens(translated, break_chars=char_break)
    weights = [max(1, len(cue_text(cue))) for cue in unit.cues]
    if len(tokens) >= len(unit.cues):
        counts = _counts(len(tokens), weights)
        chunks: list[str] = []
        offset = 0
        for count in counts:
            chunk_tokens = tokens[offset : offset + count]
            chunks.append("".join(chunk_tokens) if char_break else " ".join(chunk_tokens))
            offset += count
    else:
        chunks = _char_chunks(translated, weights)

    cues: list[Cue] = []
    overlong = 0
    for cue, chunk in zip(unit.cues, chunks, strict=True):
        lines, is_overlong = wrap_lines(
            chunk,
            max_chars=line_chars(chunk, _MAX_LINE_CHARS),
            max_lines=_MAX_LINES,
            break_chars=char_break,
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
    """Index-exact mapping of one batch, or ``None`` when the index set is wrong."""
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


def _batch_leaks(mapping: Mapping[int, str], latin_target: bool) -> bool:
    """True when a Latin-script target's batch still holds non-Latin letters (docs/04 §10)."""
    return latin_target and any(has_script_leak(text) for text in mapping.values())


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
        route=Route.TRANSLATE,
    )
    return _validate_batch(result.value.cues, len(texts))


async def _translate_one(
    llm: LLMClient,
    *,
    system: str,
    text: str,
    latin_target: bool,
) -> tuple[str | None, bool]:
    """One unit on its own; returns ``(text or None, still_leaking)``.

    A leaked non-Latin name gets one retry with an explicit Latin-script system line. If it
    leaks again the translation is kept (never the source text) and reported as a leak.
    """
    try:
        mapping = await _call(llm, system=system, texts=[text])
    except CaracalError as exc:
        logger.warning("translate.unit_failed", extra={"code": exc.code.value})
        return None, False
    if mapping is None:
        return None, False
    value = mapping[0]
    if not latin_target or not has_script_leak(value):
        return value, False
    try:
        retry = await _call(
            llm,
            system=f"{system}\n{_LATIN_NAMES_LINE}",
            texts=[text],
        )
    except CaracalError as exc:
        logger.warning("translate.unit_retry_failed", extra={"code": exc.code.value})
        return value, True
    if retry is not None:
        value = retry[0]
        if not has_script_leak(value):
            return value, False
    return value, True


async def _translate_batch(
    units: Sequence[TranslationUnit],
    *,
    system: str,
    llm: LLMClient,
    offset: int,
    latin_target: bool,
) -> tuple[list[str], set[int], int]:
    """Translate one batch; returns ``(texts, failed indices, script leaks)``."""
    texts = [unit.text for unit in units]
    for _attempt in range(2):
        try:
            mapping = await _call(llm, system=system, texts=texts)
        except CaracalError as exc:
            logger.warning("translate.batch_failed", extra={"code": exc.code.value, "units": len(units)})
            mapping = None
            break
        if mapping is not None and not _batch_leaks(mapping, latin_target):
            return [mapping[index] for index in range(len(units))], set(), 0

    fallbacks = await asyncio.gather(
        *(_translate_one(llm, system=system, text=unit.text, latin_target=latin_target) for unit in units)
    )
    translated: list[str] = []
    failed: set[int] = set()
    leaks = 0
    for local, (unit, (value, leaking)) in enumerate(zip(units, fallbacks, strict=True)):
        if value is None:
            translated.append(unit.text)
            failed.add(offset + local)
        else:
            translated.append(value)
            if leaking:
                leaks += 1
    return translated, failed, leaks


async def translate_units(
    units: Sequence[TranslationUnit],
    target_lang: str,
    *,
    glossary: Sequence[str],
    summary: str,
    llm: LLMClient,
) -> list[str]:
    """Translate sentence units into ``target_lang`` (docs/04 §10).

    Units are sent in batches of 25; a batch whose index set does not match, or (for a
    Latin-script target) whose text still holds non-Latin letters, is retried once and then
    re-sent one unit at a time. Unrecoverable units keep their source text and are reported in
    :attr:`TranslatedUnits.failed_indices`; a leaked unit that survives its retry keeps its
    translation and is counted in :attr:`TranslatedUnits.script_leaks`.
    """
    if not units:
        return TranslatedUnits([], failed_indices=set())

    english = LANGUAGE_NAMES.get(target_lang, target_lang)
    native = _NATIVE_NAMES.get(target_lang, english)
    names = normalise_target_langs([name.strip() for name in glossary if name.strip()])
    source_script = script_of_text(" ".join(unit.text for unit in units))
    latin_target = target_script(target_lang) == "latin"
    system = load_prompt(
        "translate",
        target_language=english,
        target_native=native,
        glossary=", ".join(names[:_MAX_GLOSSARY]) or "none",
        name_rule=name_rule(target_lang, source_script),
        summary=summary.strip() or "none",
    )

    batches = [units[start : start + _BATCH_SIZE] for start in range(0, len(units), _BATCH_SIZE)]
    results = await asyncio.gather(
        *(
            _translate_batch(
                batch,
                system=system,
                llm=llm,
                offset=start,
                latin_target=latin_target,
            )
            for start, batch in zip(range(0, len(units), _BATCH_SIZE), batches, strict=True)
        )
    )
    translated: list[str] = []
    failed: set[int] = set()
    leaks = 0
    for values, failures, batch_leaks in results:
        translated.extend(values)
        failed |= failures
        leaks += batch_leaks
    return TranslatedUnits(translated, failed_indices=failed, script_leaks=leaks)


__all__ = [
    "TranslatedUnits",
    "TranslationUnit",
    "cue_text",
    "has_script_leak",
    "is_valid_target_lang",
    "name_rule",
    "normalise_target_langs",
    "redistribute",
    "resolve_target_langs",
    "script_of_text",
    "segment_token_starts",
    "target_script",
    "translate_units",
    "units_from_cues",
]
