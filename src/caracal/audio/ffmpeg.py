"""ffmpeg/ffprobe wrappers for ingest, normalization and analysis (docs/04 §3, docs/06 S5).

Every call that reads user-supplied media passes both ``-protocol_whitelist file`` and
``-format_whitelist`` before the input, so a playlist or ffconcat file cannot make ffmpeg
open a URL or an arbitrary local path. Subprocesses are always argv lists, never a shell.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from caracal.errors import CaracalError, ErrorCode

# Demuxers a media upload is allowed to use. Anything outside this list (hls, ffconcat,
# image2, ...) is rejected by the demuxer itself before it can follow a reference.
_ALLOWED_FORMATS = (
    "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,ogg,wav,mp3,flac,aac,asf,avi,amr,mpegts,flv,w64,aiff,caf"
)
_ALLOWED_TAGS = frozenset(
    {
        "title",
        "artist",
        "album",
        "album_artist",
        "date",
        "comment",
        "language",
        "genre",
        "encoder",
        "creation_time",
        "composer",
        "publisher",
    }
)
_STDERR_TAIL = 4096
_PROBE_TIMEOUT_S = 30.0
_ANALYSE_TIMEOUT_S = 600.0
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")
_LOUDNESS_RE = re.compile(r"\bI:\s*(-?\d+(?:\.\d+)?)\s*LUFS")


@dataclass(frozen=True, slots=True)
class ProbeInfo:
    """The parts of an ffprobe result the pipeline needs (docs/04 §3)."""

    format_name: str
    duration_s: float | None
    bit_rate: int | None
    audio_streams: int
    video_streams: int
    codec: str | None
    sample_rate: int | None
    channels: int | None
    tags: dict[str, str]


@dataclass(frozen=True, slots=True)
class AudioStats:
    """Silence, speech ratio and loudness measured in one ffmpeg pass (docs/04 §3)."""

    duration_s: float
    silences: list[tuple[float, float]]
    speech_ratio: float
    loudness_lufs: float | None
    quality: Literal["good", "fair", "poor"]


class _ProcessFailed(Exception):
    """A non-zero ffmpeg/ffprobe exit; callers map it to the right public error code."""

    def __init__(self, returncode: int, stdout: bytes, stderr: bytes) -> None:
        super().__init__(f"process exited with {returncode}")
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _unsupported(message: str) -> CaracalError:
    return CaracalError(ErrorCode.UNSUPPORTED_MEDIA, message)


def _part_path(dst: Path) -> Path:
    return dst.with_name(f"{dst.name}.part")


def _discard(path: Path) -> None:
    path.unlink(missing_ok=True)


async def _run(
    argv: list[str],
    *,
    timeout_s: float,
    stderr_limit: int | None = _STDERR_TAIL,
) -> tuple[bytes, bytes]:
    """Run one ffmpeg/ffprobe argv, returning stdout and stderr.

    The binary is resolved with :func:`shutil.which` at call time. A missing binary is a
    ``dependency_down`` error; a timeout or cancellation kills and reaps the process and
    raises ``internal``. ``stderr_limit`` keeps error tails small; analysis passes ``None``
    because it parses the full stderr for silence boundaries.
    """
    binary = shutil.which(argv[0])
    if binary is None:
        raise CaracalError(
            ErrorCode.DEPENDENCY_DOWN,
            f"{argv[0]} is not installed",
            {"binary": argv[0]},
            retryable=False,
        )
    process = await asyncio.create_subprocess_exec(
        binary,
        *argv[1:],
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout_s):
            stdout, stderr = await process.communicate()
    except TimeoutError as exc:
        _kill(process)
        await process.wait()
        raise CaracalError(
            ErrorCode.INTERNAL,
            "ffmpeg timed out",
            {"timeout_s": timeout_s},
            retryable=False,
        ) from exc
    except asyncio.CancelledError:
        _kill(process)
        await process.wait()
        raise
    returncode = process.returncode
    if returncode is None:
        raise CaracalError(
            ErrorCode.INTERNAL,
            "ffmpeg process state is unavailable",
            retryable=False,
        )
    if returncode != 0:
        tail = stderr[-stderr_limit:] if stderr_limit is not None else stderr
        raise _ProcessFailed(returncode, stdout, tail)
    if stderr_limit is not None:
        stderr = stderr[-stderr_limit:]
    return stdout, stderr


def _kill(process: asyncio.subprocess.Process) -> None:
    """Kill a still-running process, tolerating one that exited in the meantime."""
    if process.returncode is not None:
        return
    try:
        process.kill()
    except ProcessLookupError:
        return


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _format_tags(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    tags: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.lower()
        if key in _ALLOWED_TAGS and isinstance(raw_value, (str, int, float)):
            tags[key] = str(raw_value)[:500]
    return tags


def _parse_probe(stdout: bytes) -> ProbeInfo:
    try:
        payload: object = json.loads(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _unsupported("unsupported media") from exc
    if not isinstance(payload, dict):
        raise _unsupported("unsupported media")
    raw_streams = payload.get("streams")
    streams = raw_streams if isinstance(raw_streams, list) else []
    audio = [stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"]
    video_count = sum(
        1 for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"
    )
    if not audio:
        raise _unsupported("no audio stream")
    first = audio[0]
    raw_format = payload.get("format")
    fmt = raw_format if isinstance(raw_format, dict) else {}
    format_name = _string(fmt.get("format_name")) or ""
    return ProbeInfo(
        format_name=format_name,
        duration_s=_optional_float(fmt.get("duration")),
        bit_rate=_optional_int(fmt.get("bit_rate")),
        audio_streams=len(audio),
        video_streams=video_count,
        codec=_string(first.get("codec_name")),
        sample_rate=_optional_int(first.get("sample_rate")),
        channels=_optional_int(first.get("channels")),
        tags=_format_tags(fmt.get("tags")),
    )


async def probe(path: Path) -> ProbeInfo:
    """Inspect ``path`` with ffprobe, rejecting playlists, concat lists and non-media."""
    argv = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        _ALLOWED_FORMATS,
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        stdout, _ = await _run(argv, timeout_s=_PROBE_TIMEOUT_S)
    except _ProcessFailed as exc:
        raise _unsupported("unsupported media") from exc
    return _parse_probe(stdout)


async def normalize(src: Path, dst: Path, *, max_duration_s: float) -> ProbeInfo:
    """Convert ``src`` to 16 kHz mono PCM WAV at ``dst`` atomically.

    The source is probed first so an over-duration upload is rejected before any
    conversion work. The returned :class:`ProbeInfo` describes the **source**.
    """
    info = await probe(src)
    duration = info.duration_s
    if duration is not None and duration > max_duration_s:
        raise CaracalError(
            ErrorCode.TOO_LARGE,
            "media duration exceeds the limit",
            {"duration_s": duration, "max_duration_s": max_duration_s},
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(dst)
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        _ALLOWED_FORMATS,
        "-i",
        str(src),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(part),
    ]
    timeout_s = max(120.0, 0.5 * (duration or 0.0))
    try:
        await _run(argv, timeout_s=timeout_s)
    except _ProcessFailed as exc:
        _discard(part)
        raise _unsupported("media could not be decoded") from exc
    except BaseException:
        _discard(part)
        raise
    os.replace(part, dst)
    return info


async def cut(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    """Write the ``[start_s, end_s)`` range of ``src`` to ``dst`` as 16 kHz mono PCM WAV."""
    if not (0.0 <= start_s < end_s):
        raise CaracalError(
            ErrorCode.INVALID_INPUT,
            "invalid cut range",
            {"start_s": start_s, "end_s": end_s},
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(dst)
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        _ALLOWED_FORMATS,
        "-ss",
        f"{start_s:.6f}",
        "-to",
        f"{end_s:.6f}",
        "-i",
        str(src),
        "-c:a",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-f",
        "wav",
        str(part),
    ]
    timeout_s = max(120.0, 0.5 * (end_s - start_s))
    try:
        await _run(argv, timeout_s=timeout_s)
    except _ProcessFailed as exc:
        _discard(part)
        raise CaracalError(
            ErrorCode.INTERNAL,
            "audio cut failed",
            {"returncode": exc.returncode},
            retryable=False,
        ) from exc
    except BaseException:
        _discard(part)
        raise
    os.replace(part, dst)


def _parse_duration(stderr: str) -> float | None:
    match = _DURATION_RE.search(stderr)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_silences(stderr: str, duration_s: float) -> list[tuple[float, float]]:
    silences: list[tuple[float, float]] = []
    start: float | None = None
    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match is not None:
            start = float(start_match.group(1))
            continue
        end_match = _SILENCE_END_RE.search(line)
        if end_match is not None and start is not None:
            silences.append((start, float(end_match.group(1))))
            start = None
    if start is not None:
        # The file ended during a silence: silencedetect never emits silence_end.
        silences.append((start, duration_s))
    return [(begin, end) for begin, end in silences if begin < end]


def _quality(speech_ratio: float, loudness_lufs: float | None) -> Literal["good", "fair", "poor"]:
    if speech_ratio < 0.05 or (loudness_lufs is not None and loudness_lufs < -45):
        return "poor"
    if loudness_lufs is not None and loudness_lufs < -30:
        return "fair"
    return "good"


async def analyse(
    wav: Path,
    *,
    noise_db: float = -35,
    min_silence_s: float = 0.4,
) -> AudioStats:
    """Measure silences, speech ratio and integrated loudness of our normalized WAV."""
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "info",
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        _ALLOWED_FORMATS,
        "-i",
        str(wav),
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_s},ebur128=framelog=quiet",
        "-f",
        "null",
        "-",
    ]
    try:
        _, stderr = await _run(
            argv,
            timeout_s=_ANALYSE_TIMEOUT_S,
            stderr_limit=None,
        )
    except _ProcessFailed as exc:
        raise CaracalError(
            ErrorCode.INTERNAL,
            "audio analysis failed",
            {"returncode": exc.returncode},
            retryable=False,
        ) from exc
    text = stderr.decode("utf-8", errors="replace")
    duration_s = _parse_duration(text)
    if duration_s is None or duration_s <= 0:
        raise CaracalError(
            ErrorCode.INTERNAL,
            "audio duration is unavailable",
            retryable=False,
        )
    silences = _parse_silences(text, duration_s)
    silent = sum(max(0.0, min(end, duration_s) - max(begin, 0.0)) for begin, end in silences)
    speech_ratio = min(1.0, max(0.0, 1.0 - silent / duration_s))
    loudness_match = _LOUDNESS_RE.search(text)
    loudness = float(loudness_match.group(1)) if loudness_match is not None else None
    return AudioStats(
        duration_s=duration_s,
        silences=silences,
        speech_ratio=speech_ratio,
        loudness_lufs=loudness,
        quality=_quality(speech_ratio, loudness),
    )
