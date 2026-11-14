"""yt-dlp ingest: platform metadata, audio download and platform captions (docs/04 §3, docs/06 S4).

yt-dlp is used through its Python API only. Every public entry point first runs the same
host-level SSRF check as direct media downloads, then builds one hardened options dict
(:func:`build_options`) and runs the blocking extractor in a worker thread under an
overall timeout. The progress hook is the only cancellation point: when the caller's
:class:`threading.Event` is set, it raises :class:`_Cancelled`, which aborts the download.

The generic extractor is disabled: it is the SSRF hole because it fetches arbitrary URLs
and follows embeds. The output template is fixed to ``source.%(ext)s`` so no
user-controlled value can reach a path, and no post-processor, external downloader or
``exec`` hook is ever configured.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qs, urlsplit

import yt_dlp  # type: ignore[import-untyped]
from yt_dlp import utils as ytdlp_utils

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.ssrf import Resolver, resolve_and_check
from caracal.log import get_logger, redact
from caracal.storage import JobFiles

Purpose = Literal["metadata", "audio", "captions"]
CaptionKind = Literal["manual", "auto"]

_METADATA_TIMEOUT_S = 60.0
_CAPTIONS_TIMEOUT_S = 60.0
_DESCRIPTION_MAX_CHARS = 5_000
_MIB = 1024 * 1024
_LIVE_STATUSES = frozenset({"is_live", "is_upcoming"})
_AUDIO_TEMPLATE = "source.%(ext)s"

_NETWORK_MARKERS = ("timed out", "timeout", "connection", "http error 5", "network is unreachable")
_HTTP_5XX_RE = re.compile(r"http error 5\d{2}")


@dataclass(frozen=True)
class PlatformMetadata:
    """The metadata subset the pipeline keeps (docs/04 §3)."""

    id: str | None
    title: str | None
    uploader: str | None
    channel: str | None
    channel_url: str | None
    upload_date: str | None
    duration: float | None
    description: str | None
    tags: list[str]
    categories: list[str]
    chapters: list[dict[str, Any]]
    language: str | None
    view_count: int | None
    webpage_url: str | None
    extractor: str | None


@dataclass(frozen=True)
class DownloadedAudio:
    """One downloaded source file under ``files.source_dir``."""

    path: Path
    ext: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class CaptionTrack:
    """One platform caption file saved as ``captions/<lang>.vtt``."""

    lang: str
    kind: CaptionKind
    path: Path


class _Cancelled(Exception):
    """Raised from the progress hook so yt-dlp aborts the transfer."""


class _YtdlpLogger:
    """Route yt-dlp output to the JSON stderr logger, never to stdout.

    yt-dlp sends both ``[debug]`` and ``[info]`` chatter to ``debug``; warnings and
    errors are kept at WARNING because ``docs/06 S8`` forbids INFO-level transcript or
    secret leakage. Every message is redacted before it reaches the logger.
    """

    def __init__(self) -> None:
        self._log = get_logger("caracal.ingest.ytdlp")

    def debug(self, message: str) -> None:
        self._log.debug(redact(str(message)))

    def info(self, message: str) -> None:
        self._log.debug(redact(str(message)))

    def warning(self, message: str) -> None:
        self._log.warning(redact(str(message)))

    def error(self, message: str) -> None:
        self._log.warning(redact(str(message)))


class _MatchFilter:
    """Reject live streams and over-long media, remembering why yt-dlp skipped it.

    yt-dlp does not raise when ``match_filter`` returns a reason: it silently skips the
    download. The reason is therefore stored here and inspected after extraction so the
    caller gets a typed :class:`CaracalError` instead of a missing file.
    """

    def __init__(self, max_duration_s: int) -> None:
        self.max_duration_s = max_duration_s
        self.reject_code: ErrorCode | None = None
        self.reject_message: str | None = None

    def __call__(self, info: dict[str, Any], *, incomplete: bool = False) -> str | None:
        verdict = _reject_reason(info, self.max_duration_s)
        if verdict is None:
            return None
        code, message = verdict
        self.reject_code = code
        self.reject_message = message
        return message


def _reject_reason(info: dict[str, Any], max_duration_s: int) -> tuple[ErrorCode, str] | None:
    live_status = info.get("live_status")
    if (
        info.get("is_live")
        or (info.get("was_live") is False and live_status in _LIVE_STATUSES)
        or live_status in _LIVE_STATUSES
    ):
        return ErrorCode.INVALID_INPUT, "live streams are not supported"
    duration = info.get("duration")
    if isinstance(duration, (int, float)) and duration > max_duration_s:
        return ErrorCode.TOO_LARGE, "source exceeds the maximum duration"
    return None


def _progress_hook(cancel: threading.Event) -> Callable[[dict[str, Any]], None]:
    """Return a progress hook that aborts the transfer once ``cancel`` is set."""

    def hook(_status: dict[str, Any]) -> None:
        if cancel.is_set():
            raise _Cancelled("download cancelled")

    return hook


def build_options(
    settings: Settings,
    *,
    dest_dir: Path,
    cancel: threading.Event,
    purpose: Purpose,
    lang: str | None = None,
) -> dict[str, Any]:
    """Build the single hardened yt-dlp options dict used by every purpose.

    ``lang`` is required for ``purpose="captions"``: it is the base language whose
    original track (and ``<lang>-orig`` automatic track) should be fetched. It is
    keyword-only and optional so the metadata/audio paths keep the documented signature.
    """
    options: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _YtdlpLogger(),
        "noplaylist": True,
        "allowed_extractors": ["default", "-generic"],
        "enable_file_urls": False,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "restrictfilenames": True,
        "paths": {"home": str(dest_dir)},
        "outtmpl": {"default": _AUDIO_TEMPLATE},
        "max_filesize": settings.CARACAL_MAX_UPLOAD_MB * _MIB,
        "format": "bestaudio/best",
        "postprocessors": [],
        "match_filter": _MatchFilter(settings.CARACAL_MAX_DURATION_S),
        "progress_hooks": [_progress_hook(cancel)],
    }
    if settings.YTDLP_COOKIES_FILE is not None:
        options["cookiefile"] = str(settings.YTDLP_COOKIES_FILE)

    if purpose == "metadata":
        options["skip_download"] = True
    elif purpose == "captions":
        if lang is None:
            raise CaracalError(ErrorCode.INTERNAL, "caption language is required")
        options.update(
            {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitlesformat": "vtt",
                "subtitleslangs": [lang, f"{lang}-orig"],
            }
        )
    elif purpose != "audio":
        raise CaracalError(ErrorCode.INTERNAL, "unknown yt-dlp purpose", {"purpose": str(purpose)})
    return options


def _url_is_playlist(url: str) -> bool:
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    query = parse_qs(parts.query, keep_blank_values=True)
    return "list" in query and "v" not in query


def _ensure_single_item(url: str, info: dict[str, Any] | None = None) -> None:
    if _url_is_playlist(url):
        raise CaracalError(ErrorCode.INVALID_INPUT, "one item per job")
    if info is not None and info.get("_type") in {"playlist", "multi_video"}:
        raise CaracalError(ErrorCode.INVALID_INPUT, "one item per job")


def _extract(url: str, options: dict[str, Any], download: bool) -> dict[str, Any] | None:
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=download)
    if info is None:
        return None
    return dict(info)


async def _run(
    url: str,
    options: dict[str, Any],
    *,
    download: bool,
    timeout_s: float,
    cancel: threading.Event,
) -> dict[str, Any] | None:
    """Run yt-dlp in a thread under an overall timeout, mapping its failures."""
    if cancel.is_set():
        raise CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "download cancelled")
    try:
        async with asyncio.timeout(timeout_s):
            return await asyncio.to_thread(_extract, url, options, download)
    except TimeoutError as exc:
        cancel.set()
        raise CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "download timed out", retryable=True) from exc
    except (_Cancelled, ytdlp_utils.DownloadCancelled) as exc:
        cancel.set()
        raise CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "download cancelled") from exc
    except (ytdlp_utils.DownloadError, ytdlp_utils.ExtractorError) as exc:
        raise _map_error(exc) from exc


def _map_error(exc: BaseException) -> CaracalError:
    """Translate a yt-dlp failure into a short, URL-free :class:`CaracalError`."""
    text = str(exc).lower()
    if "no suitable extractor" in text or "unsupported url" in text:
        return CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "unsupported site", {"category": "unsupported"})
    if "max-filesize" in text or "larger than max-filesize" in text or "file is larger" in text:
        return CaracalError(ErrorCode.TOO_LARGE, "source exceeds the maximum size")
    if any(marker in text for marker in _NETWORK_MARKERS) or _HTTP_5XX_RE.search(text) is not None:
        return CaracalError(
            ErrorCode.SOURCE_UNAVAILABLE, "network error", {"category": "network"}, retryable=True
        )
    if "private" in text:
        return CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "video is private", {"category": "private"})
    if any(marker in text for marker in ("removed", "deleted", "no longer available", "unavailable")):
        return CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "video is unavailable", {"category": "unavailable"})
    if any(
        marker in text for marker in ("geo", "not available in your country", "country", "region", "location")
    ):
        return CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "video is geo-restricted", {"category": "geo"})
    if any(
        marker in text
        for marker in ("age-restricted", "age restricted", "confirm your age", "age gate", "age verification")
    ):
        return CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "video is age-restricted", {"category": "age"})
    if any(
        marker in text
        for marker in ("sign in", "login", "log in", "authentication", "cookies", "members-only")
    ):
        return CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "authentication required", {"category": "auth"})
    return CaracalError(
        ErrorCode.SOURCE_UNAVAILABLE, "source could not be downloaded", {"category": "unknown"}
    )


def _finalize(
    info: dict[str, Any] | None,
    *,
    url: str,
    options: dict[str, Any],
    max_duration_s: int,
) -> dict[str, Any]:
    """Apply the playlist check, match-filter verdict and the defensive info checks."""
    if info is not None:
        # A playlist/channel URL must win over a per-entry live/size rejection.
        _ensure_single_item(url, info)
    match_filter = options.get("match_filter")
    if isinstance(match_filter, _MatchFilter) and match_filter.reject_code is not None:
        raise CaracalError(match_filter.reject_code, match_filter.reject_message or "source rejected")
    if info is None:
        raise CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "source could not be extracted")
    verdict = _reject_reason(info, max_duration_s)
    if verdict is not None:
        code, message = verdict
        raise CaracalError(code, message)
    return info


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item is not None]


def _chapters(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    chapters: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            chapters.append(dict(item))
    return chapters


def _metadata_from_info(info: dict[str, Any]) -> PlatformMetadata:
    description = _optional_str(info.get("description"))
    if description is not None:
        description = description[:_DESCRIPTION_MAX_CHARS]
    return PlatformMetadata(
        id=_optional_str(info.get("id")),
        title=_optional_str(info.get("title")),
        uploader=_optional_str(info.get("uploader")),
        channel=_optional_str(info.get("channel")),
        channel_url=_optional_str(info.get("channel_url")),
        upload_date=_optional_str(info.get("upload_date")),
        duration=_optional_float(info.get("duration")),
        description=description,
        tags=_string_list(info.get("tags")),
        categories=_string_list(info.get("categories")),
        chapters=_chapters(info.get("chapters")),
        language=_optional_str(info.get("language")),
        view_count=_optional_int(info.get("view_count")),
        webpage_url=_optional_str(info.get("webpage_url")),
        extractor=_optional_str(info.get("extractor")),
    )


async def fetch_metadata(url: str, settings: Settings, *, resolver: Resolver) -> PlatformMetadata:
    """Return the documented metadata subset for one platform item (docs/04 §3)."""
    await resolve_and_check(url, resolver=resolver)
    _ensure_single_item(url)
    cancel = threading.Event()
    with tempfile.TemporaryDirectory(prefix="caracal-ytdlp-meta-") as directory:
        options = build_options(settings, dest_dir=Path(directory), cancel=cancel, purpose="metadata")
        info = await _run(
            url,
            options,
            download=False,
            timeout_s=_METADATA_TIMEOUT_S,
            cancel=cancel,
        )
    finalized = _finalize(info, url=url, options=options, max_duration_s=settings.CARACAL_MAX_DURATION_S)
    return _metadata_from_info(finalized)


def _clean_source(directory: Path) -> None:
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        return
    except OSError:
        return
    for path in entries:
        if not path.name.startswith("source"):
            continue
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
        except OSError:
            continue


def _source_files(directory: Path) -> list[Path]:
    try:
        entries = list(directory.iterdir())
    except OSError:
        return []
    return sorted(path for path in entries if path.is_file() and path.name.startswith("source"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


async def download_audio(
    url: str,
    files: JobFiles,
    settings: Settings,
    *,
    resolver: Resolver,
    cancel: threading.Event,
) -> DownloadedAudio:
    """Download ``url`` as the single ``source.*`` file in ``files.source_dir``."""
    await resolve_and_check(url, resolver=resolver)
    _ensure_single_item(url)
    files.ensure()
    _clean_source(files.source_dir)
    options = build_options(settings, dest_dir=files.source_dir, cancel=cancel, purpose="audio")
    try:
        info = await _run(
            url,
            options,
            download=True,
            timeout_s=float(settings.CARACAL_JOB_TIMEOUT_S),
            cancel=cancel,
        )
        _finalize(info, url=url, options=options, max_duration_s=settings.CARACAL_MAX_DURATION_S)
        sources = _source_files(files.source_dir)
        if len(sources) != 1 or sources[0].name.endswith(".part"):
            raise CaracalError(
                ErrorCode.INTERNAL,
                "download did not produce exactly one source file",
                {"count": len(sources)},
            )
        path = sources[0]
        size_bytes = path.stat().st_size
        sha256 = await asyncio.to_thread(_sha256_file, path)
    except BaseException:
        _clean_source(files.source_dir)
        raise
    return DownloadedAudio(
        path=path,
        ext=path.suffix.lstrip("."),
        size_bytes=size_bytes,
        sha256=sha256,
    )


def _subtitle_map(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _base_lang(key: str) -> str:
    return key[: -len("-orig")] if key.endswith("-orig") else key


def _select_track(
    preferred: str | None,
    manual: dict[str, object],
    automatic: dict[str, object],
) -> tuple[str, str, CaptionKind] | None:
    """Pick the original-language track, preferring manual captions over automatic ones.

    Returns ``(base_lang, requested_key, kind)`` or ``None``. Machine-translated
    automatic tracks (e.g. ``en-de``) are never selected.
    """
    lang = preferred or next(iter(manual), None)
    if lang is None:
        original = next((key for key in automatic if key.endswith("-orig")), None)
        if original is not None:
            lang = _base_lang(original)
        else:
            lang = next((key for key in automatic if "-" not in key), None)
    if lang is None:
        return None

    for key in (lang, f"{lang}-orig"):
        if key in manual:
            return _base_lang(key), key, "manual"
    same_base = next((key for key in manual if _base_lang(key) == lang), None)
    if same_base is not None:
        return _base_lang(same_base), same_base, "manual"
    for key in (f"{lang}-orig", lang):
        if key in automatic:
            return _base_lang(key), key, "auto"
    return None


def _place_caption(source: Path, target: Path) -> bool:
    """Move a downloaded subtitle file into ``captions/<lang>.vtt`` atomically.

    The captions options request both ``<lang>`` and ``<lang>-orig``; only the selected
    track belongs in the artifact directory, so any other yt-dlp ``source.*`` intermediate
    is removed.
    """
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.resolve().relative_to(target.parent.resolve())
    except ValueError:
        return False
    source.replace(target)
    for leftover in target.parent.glob("source.*"):
        try:
            leftover.unlink()
        except OSError:
            continue
    return True


async def fetch_platform_captions(
    url: str,
    meta: PlatformMetadata,
    files: JobFiles,
    settings: Settings,
    *,
    resolver: Resolver,
) -> list[CaptionTrack]:
    """Download the original-language platform captions as ``captions/<lang>.vtt``.

    A metadata probe lists the available tracks; the captions run then fetches only the
    base language and its ``-orig`` automatic variant. No captions is ``[]``, not an error.
    """
    await resolve_and_check(url, resolver=resolver)
    _ensure_single_item(url)
    files.ensure()

    probe_cancel = threading.Event()
    with tempfile.TemporaryDirectory(prefix="caracal-ytdlp-subs-") as directory:
        probe_options = build_options(
            settings, dest_dir=Path(directory), cancel=probe_cancel, purpose="metadata"
        )
        probe = await _run(
            url,
            probe_options,
            download=False,
            timeout_s=_CAPTIONS_TIMEOUT_S,
            cancel=probe_cancel,
        )
    probe_info = _finalize(
        probe,
        url=url,
        options=probe_options,
        max_duration_s=settings.CARACAL_MAX_DURATION_S,
    )

    manual = _subtitle_map(probe_info.get("subtitles"))
    automatic = _subtitle_map(probe_info.get("automatic_captions"))
    selection = _select_track(meta.language, manual, automatic)
    if selection is None:
        return []
    base_lang, requested_key, kind = selection

    download_cancel = threading.Event()
    options = build_options(
        settings,
        dest_dir=files.captions_dir,
        cancel=download_cancel,
        purpose="captions",
        lang=base_lang,
    )
    info = await _run(
        url,
        options,
        download=True,
        timeout_s=_CAPTIONS_TIMEOUT_S,
        cancel=download_cancel,
    )
    finalized = _finalize(info, url=url, options=options, max_duration_s=settings.CARACAL_MAX_DURATION_S)

    requested = finalized.get("requested_subtitles")
    if not isinstance(requested, dict):
        return []
    entry = requested.get(requested_key)
    if not isinstance(entry, dict):
        entry = requested.get(base_lang)
    if not isinstance(entry, dict):
        return []
    filepath = entry.get("filepath")
    if not isinstance(filepath, str):
        return []
    source = Path(filepath)
    target = files.captions(base_lang)
    if not await asyncio.to_thread(_place_caption, source, target):
        return []
    return [CaptionTrack(lang=base_lang, kind=kind, path=target)]


__all__ = [
    "CaptionTrack",
    "DownloadedAudio",
    "PlatformMetadata",
    "build_options",
    "download_audio",
    "fetch_metadata",
    "fetch_platform_captions",
]
