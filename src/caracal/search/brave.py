"""Brave Web Search provider (docs/04 §8).

Measured contract (2026-11-14): ``GET /res/v1/web/search`` with ``X-Subscription-Token``;
results live at ``web.results[]``. Descriptions arrive with ``<strong>`` markup and HTML
entities, so every field goes through :func:`clean_text` before it becomes a hit.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

from caracal.search.base import (
    SNIPPET_MAX_CHARS,
    TITLE_MAX_CHARS,
    SearchError,
    SearchHit,
    clean_text,
    clean_url,
    clip,
    response_error,
)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
_TIMEOUT_S = 15.0
_MAX_EXTRA_SNIPPETS = 2


def _subscription_invalid(response: httpx.Response) -> bool:
    """Brave reports an invalid key as HTTP 422 with a ``SUBSCRIPTION_TOKEN_INVALID`` code."""
    try:
        payload = response.json()
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    return isinstance(error, dict) and error.get("code") == "SUBSCRIPTION_TOKEN_INVALID"


def _snippet(result: dict[str, Any]) -> str:
    parts: list[str] = []
    description = clean_text(result.get("description"))
    if description:
        parts.append(description)
    extras = result.get("extra_snippets")
    if isinstance(extras, list):
        for extra in extras[:_MAX_EXTRA_SNIPPETS]:
            cleaned = clean_text(extra)
            if cleaned:
                parts.append(cleaned)
    return clip(" … ".join(parts), SNIPPET_MAX_CHARS)


def _published(result: dict[str, Any]) -> str | None:
    for field in ("page_age", "age"):
        value = result.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class BraveProvider:
    """Brave Web Search over an injected ``httpx.AsyncClient``."""

    name = "brave"

    def __init__(self, api_key: str, http: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._http = http
        self.cost_usd = 0.0

    async def search(self, query: str, *, limit: int) -> list[SearchHit]:
        """One Brave search; raises :class:`SearchError` for every failure mode."""
        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }
        params = {"q": query, "count": str(limit), "extra_snippets": "true"}
        try:
            response = await self._http.get(
                BRAVE_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=_TIMEOUT_S,
            )
        except httpx.RequestError:
            raise SearchError("unavailable", "brave request failed") from None

        if response.status_code >= 400:
            auth = response.status_code == 422 and _subscription_invalid(response)
            raise response_error("brave", response, auth_status=auth)

        try:
            payload = response.json()
        except ValueError:
            raise SearchError("bad_response", "brave returned invalid JSON") from None
        if not isinstance(payload, dict):
            raise SearchError("bad_response", "brave returned a non-object body")
        body = cast(dict[str, Any], payload)
        web = body.get("web")
        if web is None:
            return []
        if not isinstance(web, dict):
            raise SearchError("bad_response", "brave returned a malformed web section")
        results = web.get("results")
        if not isinstance(results, list):
            raise SearchError("bad_response", "brave returned no results list")

        hits: list[SearchHit] = []
        for raw in results:
            if not isinstance(raw, dict):
                continue
            url = clean_url(raw.get("url"))
            if url is None:
                continue
            hits.append(
                SearchHit(
                    provider=self.name,
                    query=query,
                    rank=len(hits) + 1,
                    url=url,
                    title=clip(raw.get("title"), TITLE_MAX_CHARS),
                    snippet=_snippet(raw),
                    published=_published(raw),
                    text=None,
                )
            )
        return hits
