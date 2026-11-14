"""Search provider contract, hits and text hygiene shared by Brave and Exa (docs/04 §8).

Web text is untrusted data (docs/06 S9): every provider runs the same hygiene pass so a hit
never carries markup, entities or control characters into the report or an LLM prompt.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

SearchErrorKind = Literal["auth", "rate_limited", "unavailable", "bad_response"]

SNIPPET_MAX_CHARS = 500
TEXT_MAX_CHARS = 2_000
TITLE_MAX_CHARS = 300
QUERY_MAX_CHARS = 400

_TAG_RE = re.compile(r"<[^>]*>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE_RE = re.compile(r"\s+")


class SearchError(Exception):
    """A provider failure classified for the router's fallback policy."""

    def __init__(
        self,
        kind: SearchErrorKind,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind: SearchErrorKind = kind
        self.message = message
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One search result with every text field cleaned and clipped."""

    provider: str
    query: str
    rank: int
    url: str
    title: str
    snippet: str
    published: str | None
    text: str | None

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable representation (used for the Redis cache)."""
        return {
            "provider": self.provider,
            "query": self.query,
            "rank": self.rank,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "published": self.published,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, raw: object) -> SearchHit:
        """Rebuild a hit from cache JSON, rejecting anything structurally invalid."""
        if not isinstance(raw, dict):
            raise ValueError("cached hit is not an object")
        provider = raw.get("provider")
        query = raw.get("query")
        rank = raw.get("rank")
        url = raw.get("url")
        title = raw.get("title")
        snippet = raw.get("snippet")
        published = raw.get("published")
        text = raw.get("text")
        if not isinstance(provider, str) or not isinstance(query, str):
            raise ValueError("cached hit has invalid provider/query")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError("cached hit has invalid rank")
        if not isinstance(url, str) or not isinstance(title, str) or not isinstance(snippet, str):
            raise ValueError("cached hit has invalid text fields")
        if published is not None and not isinstance(published, str):
            raise ValueError("cached hit has invalid published")
        if text is not None and not isinstance(text, str):
            raise ValueError("cached hit has invalid text")
        return cls(
            provider=provider,
            query=query,
            rank=rank,
            url=url,
            title=title,
            snippet=snippet,
            published=published,
            text=text,
        )


@runtime_checkable
class SearchProvider(Protocol):
    """One web-search backend. Implementations raise :class:`SearchError` on failure."""

    name: str

    async def search(self, query: str, *, limit: int) -> list[SearchHit]:
        """Return at most ``limit`` hits, or raise :class:`SearchError`."""
        ...


def clean_text(value: object) -> str:
    """Decode HTML entities, strip tags/control characters and collapse whitespace."""
    if not isinstance(value, str):
        return ""
    text = html.unescape(value)
    text = _TAG_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clip(value: object, limit: int) -> str:
    """Truncate to ``limit`` characters after hygiene."""
    return clean_text(value)[:limit]


def clean_url(value: object) -> str | None:
    """Return an absolute http(s) URL, or ``None`` when the hit must be dropped."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    parts = urlsplit(candidate)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None
    return candidate


def normalise_query(value: str) -> str:
    """Trim, collapse whitespace and cap a query at the documented length."""
    collapsed = _WHITESPACE_RE.sub(" ", value).strip()
    return collapsed[:QUERY_MAX_CHARS].strip()


def _first_number(value: str) -> float | None:
    token = value.split(",", 1)[0].strip()
    try:
        return max(0.0, float(token))
    except ValueError:
        return None


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Read ``retry-after`` or ``x-ratelimit-reset``; both are seconds in the measured APIs."""
    for header in ("retry-after", "x-ratelimit-reset"):
        raw = response.headers.get(header)
        if raw is None:
            continue
        seconds = _first_number(raw)
        if seconds is not None:
            return seconds
    return None


def response_error(
    provider: str,
    response: httpx.Response,
    *,
    auth_status: bool = False,
) -> SearchError:
    """Map a non-2xx response to the router's error kinds without exposing the body."""
    status = response.status_code
    if status in {401, 403} or auth_status:
        return SearchError("auth", f"{provider} rejected credentials (HTTP {status})", status=status)
    if status == 429:
        return SearchError(
            "rate_limited",
            f"{provider} is rate limited (HTTP 429)",
            status=status,
            retry_after=retry_after_seconds(response),
        )
    if status >= 500:
        return SearchError("unavailable", f"{provider} is unavailable (HTTP {status})", status=status)
    return SearchError("bad_response", f"{provider} returned HTTP {status}", status=status)
