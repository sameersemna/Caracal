"""whisper.cpp HTTP client (docs/04 §4).

One call per chunk. The server is single-slot, so the caller may pass a lock factory (the
transcribe stage passes ``queue.whisper_lock``); the request is always made inside it. The
server always receives an explicit ``language`` because its default is ``en``.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any, cast

import httpx

from caracal.asr import (
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
            for raw_word in raw_words:
                if not isinstance(raw_word, dict):
                    continue
                item = cast(dict[str, Any], raw_word)
                text = str(item.get("word", "")).strip()
                if not text:
                    continue
                words.append(
                    Word(
                        text=text,
                        start=_as_float(item.get("start")),
                        end=_as_float(item.get("end")),
                        prob=_as_float(item.get("probability")),
                    )
                )
        segments.append(
            Segment(
                id=int(_as_float(segment.get("id"), float(index))),
                start=_as_float(segment.get("start")),
                end=_as_float(segment.get("end")),
                text=str(segment.get("text", "")),
                words=tuple(words),
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
