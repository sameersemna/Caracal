"""ASR data model and the whisper language table (docs/01 §3, docs/04 §4).

Every persisted model is a frozen dataclass with :meth:`to_dict` so a stage checkpoint can
be written as JSON without a custom encoder. Whisper text and word text are stored stripped;
:attr:`Word.space_before` keeps whisper's leading-whitespace signal so a rebuilt segment can
reproduce the original punctuation and contraction spacing.

Whisper writes some scripts without spaces between words (Han, Kana, Thai, Lao, Khmer,
Myanmar, Tibetan). There ``space_before`` is ``False`` for every token, so it cannot delimit
words; every token that begins with a character of such a script is treated as one word
instead (:func:`word_groups`), and the cue/line breaker can split between characters.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

# Whisper's own language table: ISO 639-1/3 code -> English display name. The inverse is
# used when a response has no ``language_probabilities`` but does carry ``detected_language``.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
    "yue": "Cantonese",
}

_NAME_TO_CODE: dict[str, str] = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


def language_code_for_name(name: str) -> str | None:
    """Map whisper's full language name (``"english"``) to its ISO code, or ``None``."""
    return _NAME_TO_CODE.get(name.strip().lower())


@dataclass(frozen=True, slots=True)
class Word:
    """One transcribed word with its timing and acoustic confidence.

    ``space_before`` is whisper's own spacing signal: ``True`` when the raw token started
    with whitespace, ``False`` for tokens such as ``"'t"`` or ``","`` that are attached to
    the preceding word. It is recorded for every word, including the first word of a
    segment, so words joined across a segment seam keep their separating space.
    """

    text: str
    start: float
    end: float
    prob: float
    speaker: str | None = None
    space_before: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", self.text.strip())

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "prob": self.prob,
            "speaker": self.speaker,
            "space_before": self.space_before,
        }
        return payload


# A token made only of closing punctuation, quotes or brackets (e.g. ``."``, ``...``,
# ``"``, ``»``) closes the previous word, so it never gets a leading space.
_CLOSING_TOKEN_RE = re.compile(r"[.,?!;:…\"'”’»)\]]+")  # noqa: RUF001 - unicode quotes are intentional


def _is_closing_token(text: str) -> bool:
    return bool(text) and _CLOSING_TOKEN_RE.fullmatch(text) is not None


# Scripts whisper writes without spaces between words, as inclusive code-point ranges.
# Every token that starts with one of these characters is its own breakable word.
_NO_SPACE_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x2E80, 0x2FFF),  # CJK radicals supplement, Kangxi radicals
    (0x3005, 0x3007),  # ideographic iteration/zero marks
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK unified ideographs extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF66, 0xFF9D),  # halfwidth Katakana
)

#: whisper's replacement character: a multi-byte character that was split across tokens.
REPLACEMENT_CHAR = "\ufffd"


def is_no_space_char(char: str) -> bool:
    """True for a character of a script that whisper writes without word spaces."""
    code = ord(char)
    return any(low <= code <= high for low, high in _NO_SPACE_RANGES)


def _is_punctuation(text: str) -> bool:
    """True when every character of ``text`` is punctuation (``。``, ``、``, ``:``, ``「``)."""
    return bool(text) and all(unicodedata.category(char).startswith("P") for char in text)


def _starts_no_space_word(word: Word) -> bool:
    """True when a token opens a new word in a script written without spaces."""
    if not word.text or word.text.startswith(REPLACEMENT_CHAR):
        return False  # empty/fragment tokens never open a word
    if _is_punctuation(word.text):
        return False  # punctuation attaches to the word it closes
    first = word.text[0]
    if unicodedata.category(first).startswith("M"):
        return False  # a combining mark (Thai tone, Tibetan vowel) belongs to the base before it
    return is_no_space_char(first)


def join_words(words: Sequence[Word]) -> str:
    """Rebuild text from words using whisper's spacing signal.

    One space is inserted before a word only when :attr:`Word.space_before` is ``True``,
    so contractions (``"'t"``) and attached punctuation (``","``) survive the rebuild.
    A token made only of closing punctuation, quotes or brackets (whisper's detached
    ``"..."``, ``"."`` or ``."``) is attached to the preceding word instead of being
    rendered as ``"and ..."``. The result is stripped; the first kept word never receives
    a leading space.
    """
    parts: list[str] = []
    for word in words:
        if parts and word.space_before and not _is_closing_token(word.text):
            parts.append(" ")
        parts.append(word.text)
    return "".join(parts).strip()


@dataclass(frozen=True, slots=True)
class WordGroup:
    """Consecutive whisper tokens that spell one word (docs/04 §10).

    Whisper emits sub-word pieces: ``Tag`` + ``ess`` + ``chau`` and ``Sek`` + ``unden`` +
    ``.``. A token whose raw text had no leading space (``space_before is False``) continues
    the previous word, so grouping never breaks a word in the middle — except in scripts
    written without spaces, where every token that begins with such a character is its own
    word. ``tokens`` is the inclusive global index span of the group's tokens.
    """

    text: str
    start: float
    end: float
    tokens: tuple[int, int]
    speaker: str | None
    space_before: bool


def _majority_speaker(tokens: Sequence[Word]) -> str | None:
    """The speaker most of the group's tokens carry; a tie goes to the first token's."""
    counts: dict[str, int] = {}
    for token in tokens:
        if token.speaker is not None:
            counts[token.speaker] = counts.get(token.speaker, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    leaders = [label for label, count in counts.items() if count == best]
    first = tokens[0].speaker
    if first is not None and first in leaders:
        return first
    return leaders[0]


def _group(tokens: Sequence[Word], start_index: int) -> WordGroup:
    return WordGroup(
        text=join_words(tokens),
        start=tokens[0].start,
        end=tokens[-1].end,
        tokens=(start_index, start_index + len(tokens) - 1),
        speaker=_majority_speaker(tokens),
        space_before=tokens[0].space_before,
    )


def word_groups(words: Sequence[Word]) -> list[WordGroup]:
    """Group tokens into words using whisper's leading-whitespace signal (docs/04 §10).

    A token with ``space_before is False`` continues the current group, which also attaches
    detached punctuation (``,``, ``.``, ``."``) to the word it closes. In a script whispered
    without spaces (Han, Kana, Thai …) a token that begins with such a character starts a new
    group even though ``space_before`` is ``False``, so a cue can break between characters.
    The first token of a sequence always starts a new group.
    """
    groups: list[WordGroup] = []
    current: list[Word] = []
    start_index = 0
    for index, word in enumerate(words):
        if current and (word.space_before or _starts_no_space_word(word)):
            groups.append(_group(current, start_index))
            current = []
            start_index = index
        current.append(word)
    if current:
        groups.append(_group(current, start_index))
    return groups


@dataclass(frozen=True, slots=True)
class Segment:
    """One whisper segment. ``lang``/``lang_prob`` describe the chunk that produced it."""

    id: int
    start: float
    end: float
    text: str
    words: tuple[Word, ...]
    avg_logprob: float
    no_speech_prob: float
    lang: str | None
    speaker: str | None = None
    lang_prob: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", self.text.strip())

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
            "lang": self.lang,
            "speaker": self.speaker,
            "lang_prob": self.lang_prob,
        }
        return payload


@dataclass(frozen=True, slots=True)
class WhisperResult:
    """The parsed ``verbose_json`` of one whisper call (timings are chunk-local)."""

    language_code: str | None
    language_prob: float
    language_probs: dict[str, float]
    duration: float
    segments: tuple[Segment, ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "language_code": self.language_code,
            "language_prob": self.language_prob,
            "language_probs": dict(self.language_probs),
            "duration": self.duration,
            "segments": [segment.to_dict() for segment in self.segments],
        }
        return payload


@dataclass(frozen=True, slots=True)
class Transcript:
    """Merged, time-absolute transcript with per-language attribution."""

    segments: tuple[Segment, ...]
    language_code: str | None
    dropped: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "segments": [segment.to_dict() for segment in self.segments],
            "language_code": self.language_code,
            "dropped": dict(self.dropped),
        }
        return payload


@dataclass(frozen=True, slots=True)
class LangSpan:
    """A contiguous run of segments that share one detected language."""

    code: str
    start: float
    end: float
    confidence: float

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "code": self.code,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }
        return payload
