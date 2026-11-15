"""whisper.cpp HTTP client (docs/04 §4).

One call per chunk. The server is single-slot, so the caller may pass a lock factory (the
transcribe stage passes ``queue.whisper_lock``); the request is always made inside it. The
server always receives an explicit ``language`` because its default is ``en``.

whisper.cpp emits CJK and other multi-byte characters as UTF-8 byte fragments: a character
whose bytes straddle a token boundary arrives as several tokens that each decode to U+FFFD,
while ``segment.text`` is correct (measured: 74 of 140 tokens in a Chinese news clip). The
parser re-derives those token texts from the segment text (fix 2 of verdict 42), anchoring
the intact tokens inside it, so no word ever carries U+FFFD into a subtitle.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx

from caracal.asr import (
    REPLACEMENT_CHAR,
    Segment,
    WhisperResult,
    Word,
    language_code_for_name,
)
from caracal.errors import CaracalError, ErrorCode
from caracal.log import get_logger

logger = get_logger("caracal.asr.whisper")

_LANGUAGE_RE = re.compile(r"(?:auto|[a-z]{2,3})")
_INFERENCE_PATH = "/inference"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _argmax_language(probs: dict[str, float]) -> str | None:
    if not probs:
        return None
    return max(probs, key=probs.__getitem__)


def _mean_prob(words: Sequence[Word]) -> float:
    if not words:
        return 0.0
    return sum(word.prob for word in words) / len(words)


def _spread_words(text: str, start: float, end: float, prob: float) -> tuple[Word, ...]:
    """Fallback repair: one token per character, timed evenly across the segment span."""
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return ()
    step = max(0.0, end - start) / len(chars)
    return tuple(
        Word(text=char, start=start + index * step, end=start + (index + 1) * step, prob=prob)
        for index, char in enumerate(chars)
    )


def _assign_run(gap: str, texts: list[str], first: int, stop: int) -> bool:
    """Hand the characters of ``gap`` to the fragment tokens ``first .. stop - 1``.

    Character ``j`` goes to fragment ``first + j * count // len(gap)``, so the first character
    of a run lands on its first fragment token and a later character of the same run lands on
    the token that starts its bytes. Returns ``False`` when the run cannot carry the gap.
    """
    count = stop - first
    if not gap:
        return True
    if count <= 0:
        return False
    for offset, char in enumerate(gap):
        texts[first + offset * count // len(gap)] += char
    return True


def _repair_words(text: str, words: Sequence[Word], *, start: float, end: float) -> tuple[Word, ...]:
    """Re-derive token texts from the segment text when whisper split a character.

    whisper.cpp tokenises UTF-8 bytes, so a multi-byte character whose bytes straddle a token
    boundary arrives as several tokens that each decode to U+FFFD while ``segment.text`` is
    correct. The intact tokens (anchors) are located in the whitespace-free segment text and
    the characters between two anchors are handed to the fragment tokens between them. The
    continuation fragments end up empty and are merged into the preceding token's time span,
    so no word carries U+FFFD. When the anchors do not line up, every character of the segment
    text becomes one token spread evenly over the segment span.

    A segment without U+FFFD (every Latin-script fixture) is returned unchanged.
    """
    if not any(REPLACEMENT_CHAR in word.text for word in words):
        return tuple(words)

    flat = "".join(text.split())
    if not flat:
        return tuple(words)  # nothing to align against; never discard a segment's words
    anchors: list[tuple[int, int, int]] = []
    cursor = 0
    for index, word in enumerate(words):
        if not word.text or REPLACEMENT_CHAR in word.text:
            continue
        position = flat.find(word.text, cursor)
        if position < 0:
            return _spread_words(text, start, end, _mean_prob(words))
        anchors.append((index, position, position + len(word.text)))
        cursor = position + len(word.text)
    if not anchors:
        return _spread_words(text, start, end, _mean_prob(words))

    texts = [""] * len(words)
    for index, _, _ in anchors:
        texts[index] = words[index].text

    previous_char = 0
    previous_index = -1
    for index, position, after in [*anchors, (len(words), len(flat), len(flat))]:
        if not _assign_run(flat[previous_char:position], texts, previous_index + 1, index):
            return _spread_words(text, start, end, _mean_prob(words))
        previous_char = after
        previous_index = index

    if any(REPLACEMENT_CHAR in value for value in texts):
        return _spread_words(text, start, end, _mean_prob(words))

    repaired: list[Word] = []
    for word, value in zip(words, texts, strict=True):
        if not value:
            if not repaired:
                continue
            repaired[-1] = replace(repaired[-1], end=max(repaired[-1].end, word.end))
            continue
        repaired.append(replace(word, text=value))
    return tuple(repaired)


def parse_whisper_response(payload: object) -> WhisperResult:
    """Parse a ``verbose_json`` body, raising ``internal`` for a structurally bad one."""
    if not isinstance(payload, dict):
        raise CaracalError(ErrorCode.INTERNAL, "whisper response is not a JSON object")
    body = cast(dict[str, Any], payload)
    raw_segments = body.get("segments")
    if not isinstance(raw_segments, list):
        raise CaracalError(ErrorCode.INTERNAL, "whisper response has no segments")

    raw_probs = body.get("language_probabilities")
    language_probs: dict[str, float] = {}
    if isinstance(raw_probs, dict):
        for code, value in raw_probs.items():
            if isinstance(code, str):
                language_probs[code] = _as_float(value)

    language_code = _argmax_language(language_probs)
    if language_code is None:
        detected = body.get("detected_language")
        if isinstance(detected, str):
            language_code = language_code_for_name(detected)
    language_prob = _as_float(body.get("detected_language_probability"))
    duration = _as_float(body.get("duration"))

    segments: list[Segment] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            continue
        segment = cast(dict[str, Any], raw_segment)
        words: list[Word] = []
        raw_words = segment.get("words")
        if isinstance(raw_words, list):
            pending_space = False
            for raw_word in raw_words:
                if not isinstance(raw_word, dict):
                    continue
                item = cast(dict[str, Any], raw_word)
                raw_text = str(item.get("word", ""))
                text = raw_text.strip()
                if not text:
                    # A whitespace-only token carries the word separator (measured on the
                    # Chinese fixture: " " is its own token); it belongs to the next token.
                    pending_space = pending_space or raw_text[:1].isspace()
                    continue
                words.append(
                    Word(
                        text=text,
                        start=_as_float(item.get("start")),
                        end=_as_float(item.get("end")),
                        prob=_as_float(item.get("probability")),
                        space_before=raw_text[:1].isspace() or pending_space,
                    )
                )
                pending_space = False
        segment_start = _as_float(segment.get("start"))
        segment_end = _as_float(segment.get("end"))
        segment_text = str(segment.get("text", ""))
        segments.append(
            Segment(
                id=int(_as_float(segment.get("id"), float(index))),
                start=segment_start,
                end=segment_end,
                text=segment_text,
                words=_repair_words(segment_text, words, start=segment_start, end=segment_end),
                avg_logprob=_as_float(segment.get("avg_logprob")),
                no_speech_prob=_as_float(segment.get("no_speech_prob")),
                lang=language_code,
                lang_prob=language_prob,
            )
        )

    return WhisperResult(
        language_code=language_code,
        language_prob=language_prob,
        language_probs=language_probs,
        duration=duration,
        segments=tuple(segments),
    )


class WhisperClient:
    """Async client for the whisper.cpp server (``POST /inference``, ``GET /``)."""

    def __init__(
        self,
        base_url: str,
        http: httpx.AsyncClient,
        *,
        timeout_s: float,
        lock: Callable[[], AbstractAsyncContextManager[None]] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._timeout_s = timeout_s
        self._lock = lock

    async def transcribe(
        self,
        wav: Path,
        *,
        language: str = "auto",
        translate: bool = False,
        temperature: float = 0.0,
    ) -> WhisperResult:
        """Transcribe one WAV file, always sending an explicit language."""
        if _LANGUAGE_RE.fullmatch(language) is None:
            raise CaracalError(
                ErrorCode.INVALID_INPUT,
                f"invalid language {language!r}: expected 'auto' or a 2-3 letter ISO code",
                {"language": language},
            )
        if self._lock is None:
            return await self._request(wav, language, translate, temperature)
        async with self._lock():
            return await self._request(wav, language, translate, temperature)

    async def health(self) -> bool:
        """True when ``GET /`` answers 200."""
        try:
            response = await self._http.get(f"{self._base_url}/", timeout=self._timeout_s)
        except httpx.TransportError:
            return False
        return response.status_code == 200

    async def _request(
        self,
        wav: Path,
        language: str,
        translate: bool,
        temperature: float,
    ) -> WhisperResult:
        data = {
            "response_format": "verbose_json",
            "language": language,
            "temperature": str(temperature),
            "translate": "true" if translate else "false",
        }
        content = await asyncio.to_thread(wav.read_bytes)
        files = {"file": (wav.name, content, "audio/wav")}
        try:
            response = await self._http.post(
                f"{self._base_url}{_INFERENCE_PATH}",
                files=files,
                data=data,
                timeout=self._timeout_s,
            )
        except httpx.TransportError as exc:
            raise CaracalError(
                ErrorCode.DEPENDENCY_DOWN,
                "whisper server is unreachable",
                {"error": type(exc).__name__},
                retryable=True,
            ) from exc

        if response.status_code >= 500:
            raise CaracalError(
                ErrorCode.DEPENDENCY_DOWN,
                f"whisper server failed with HTTP {response.status_code}",
                {"status": response.status_code},
                retryable=True,
            )
        if response.status_code >= 400:
            raise CaracalError(
                ErrorCode.INTERNAL,
                f"whisper rejected the request with HTTP {response.status_code}",
                {"status": response.status_code},
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise CaracalError(
                ErrorCode.INTERNAL,
                "whisper returned a body that is not JSON",
                {"status": response.status_code},
            ) from exc

        result = parse_whisper_response(payload)
        logger.info(
            "whisper transcription",
            extra={
                "language_code": result.language_code,
                "language_prob": result.language_prob,
                "duration_s": result.duration,
                "segments": len(result.segments),
            },
        )
        return result
