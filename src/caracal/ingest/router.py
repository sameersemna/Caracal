"""Static source classification (docs/04 §3).

This is the network-free half of the ingest router: a URL whose path ends in a known
media extension is a direct download, everything else goes through yt-dlp. T1.1 extends
this module with the SSRF checks and the HEAD ``Content-Type`` probe.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit

MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {"mp3", "m4a", "aac", "wav", "flac", "ogg", "opus", "webm", "mp4", "mkv", "mov"}
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
