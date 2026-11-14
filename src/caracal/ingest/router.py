"""Source classification (docs/04 §3).

Two halves: a network-free extension check (:func:`classify_source_static`) and the async
router that also recognises platform hosts and probes otherwise ambiguous URLs with one
pinned ``HEAD`` request. The probe never follows more than five redirects and revalidates
every hop through the SSRF guard.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

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
    "classify_source",
    "classify_source_static",
    "is_platform_host",
]
