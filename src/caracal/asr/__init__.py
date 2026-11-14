"""ASR data model and the whisper language table (docs/01 §3, docs/04 §4).

Every model is a frozen dataclass with :meth:`to_dict` so a stage checkpoint can be written
as JSON without a custom encoder. Whisper text and word text are stored stripped;
:attr:`Word.space_before` keeps whisper's leading-whitespace signal so a rebuilt segment can
reproduce the original punctuation and contraction spacing.
"""

from __future__ import annotations

import re
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
