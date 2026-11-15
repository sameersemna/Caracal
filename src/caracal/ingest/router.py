"""Source classification (docs/04 §3).

Two halves: a network-free extension check (:func:`classify_source_static`) and the async
router that also recognises platform hosts and probes otherwise ambiguous URLs with one
pinned ``HEAD`` request. The probe never follows more than five redirects and revalidates
every hop through the SSRF guard.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.direct import HttpHead
from caracal.ingest.ssrf import IpPolicy, Resolver, default_ip_policy

MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {"mp3", "m4a", "aac", "wav", "flac", "ogg", "opus", "webm", "mp4", "mkv", "mov"}
)

PLATFORM_HOSTS: frozenset[str] = frozenset(
    {
        "youtube.com",
        "youtu.be",
        "vimeo.com",
        "soundcloud.com",
        "tiktok.com",
        "instagram.com",
        "facebook.com",
        "fb.watch",
        "x.com",
        "twitter.com",
        "twitch.tv",
        "dailymotion.com",
        "bilibili.com",
        "rumble.com",
    }
)

SourceKind = Literal["direct", "platform"]

_YOUTUBE_HOSTS = frozenset({"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"})
_YOUTUBE_PATH_PREFIXES = ("/shorts/", "/embed/", "/live/", "/v/")
_YOUTUBE_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
_TRACKING_PARAMS = frozenset({"si", "fbclid", "gclid", "feature", "t", "start"})


def _is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("utm_") or lowered in _TRACKING_PARAMS


def _youtube_id(host: str, path: str, query: Sequence[tuple[str, str]]) -> str | None:
    """The 11-character video id of a YouTube URL, or ``None`` when there is no valid one."""
    candidate: str | None = None
    if host == "youtu.be":
        segments = [segment for segment in path.split("/") if segment]
        candidate = segments[0] if segments else None
    elif host in _YOUTUBE_HOSTS:
        if path.rstrip("/") == "/watch":
            candidate = next((value for name, value in query if name == "v"), None)
        else:
            for prefix in _YOUTUBE_PATH_PREFIXES:
                if path.startswith(prefix):
                    candidate = path[len(prefix) :].split("/", 1)[0]
                    break
    if candidate is not None and _YOUTUBE_ID_RE.fullmatch(candidate):
        return candidate
    return None


def canonical_url(url: str) -> str:
    """Canonicalise a source URL for storage and dedup (docs/05 §1), without any network call.

    Scheme and host are lower-cased; the fragment and tracking parameters (``utm_*``,
    ``si``, ``fbclid``, ``gclid``, ``feature``, ``t``, ``start``) are dropped. Every
    YouTube shape — ``watch?v=``, ``youtu.be/<id>``, ``shorts/``, ``embed/``, ``live/``,
    ``v/``, under ``youtube.com``, ``www.``, ``m.`` or ``music.`` — with a valid
    11-character id maps to ``https://www.youtube.com/watch?v=<id>``. Any other URL keeps
    its path and remaining query; a YouTube URL without a valid id is only cleaned.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    query = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not _is_tracking_param(name)
    ]
    host = (parts.hostname or "").lower().rstrip(".")
    video_id = _youtube_id(host, parts.path, query)
    if video_id is not None:
        return f"https://www.youtube.com/watch?v={video_id}"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, urlencode(query, doseq=True), "")
    )


def classify_source_static(url: str) -> SourceKind:
    """Return ``"direct"`` for an http(s) URL ending in a media extension, else ``"platform"``.

    Query strings and fragments are ignored; classification never performs I/O.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "platform"
    if parts.scheme.lower() not in {"http", "https"}:
        return "platform"
    path = parts.path.lower()
    if "." not in path.rsplit("/", 1)[-1]:
        return "platform"
    extension = path.rsplit(".", 1)[-1]
    return "direct" if extension in MEDIA_EXTENSIONS else "platform"


def is_platform_host(host: str) -> bool:
    """True when ``host`` is one of the known platform domains (suffix, label-bound)."""
    candidate = host.lower().rstrip(".")
    return any(candidate == domain or candidate.endswith(f".{domain}") for domain in PLATFORM_HOSTS)


def _media_content_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type.startswith(("audio/", "video/"))


async def classify_source(
    url: str,
    *,
    resolver: Resolver,
    policy: IpPolicy = default_ip_policy,
    http_head: HttpHead,
) -> SourceKind:
    """Classify ``url`` as ``direct`` or ``platform`` (docs/04 §3).

    Known platform hosts win without any network call, as do static media extensions.
    Everything else gets one pinned ``HEAD``: an ``audio/*`` or ``video/*`` response means
    a direct download, any failure means the platform (yt-dlp) path. A blocked destination
    always propagates so no fetch is ever attempted against it.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "platform"
    if parts.scheme.lower() not in {"http", "https"}:
        return "platform"
    host = (parts.hostname or "").lower()
    if host and is_platform_host(host):
        return "platform"
    if classify_source_static(url) == "direct":
        return "direct"
    try:
        content_type = await http_head(url)
    except CaracalError as exc:
        if exc.code is ErrorCode.SOURCE_BLOCKED:
            raise
        return "platform"
    except Exception:
        return "platform"
    return "direct" if _media_content_type(content_type) else "platform"


__all__ = [
    "MEDIA_EXTENSIONS",
    "PLATFORM_HOSTS",
    "SourceKind",
    "canonical_url",
    "classify_source",
    "classify_source_static",
    "is_platform_host",
]
