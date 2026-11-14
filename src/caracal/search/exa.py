"""Exa search provider (docs/04 §8).

Measured contract (2026-11-14): ``POST /search`` with ``x-api-key`` and a JSON body asking
for ``contents.text.maxCharacters = 2000``. Every response carries ``costDollars.total``
(the measured price is ≈ $0.007 per search), accumulated in :attr:`ExaProvider.cost_usd`.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

from caracal.search.base import (
    SNIPPET_MAX_CHARS,
    TEXT_MAX_CHARS,
    TITLE_MAX_CHARS,
    SearchError,
    SearchHit,
    clean_url,
    clip,
    response_error,
)

EXA_SEARCH_URL = "https://api.exa.ai/search"
_TIMEOUT_S = 15.0
_MAX_TEXT_CHARS = 2_000


def _cost(payload: dict[str, Any]) -> float:
    costs = payload.get("costDollars")
    if not isinstance(costs, dict):
        return 0.0
    total = costs.get("total")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        return 0.0
    return max(0.0, float(total))


class ExaProvider:
    """Exa search over an injected ``httpx.AsyncClient``."""

    name = "exa"

    def __init__(self, api_key: str, http: httpx.AsyncClient) -> None:
        self._api_key = api_key
        self._http = http
        self.cost_usd = 0.0

    async def search(self, query: str, *, limit: int) -> list[SearchHit]:
        """One Exa search; raises :class:`SearchError` for every failure mode."""
        payload = {
            "query": query,
            "numResults": limit,
            "type": "auto",
            "contents": {"text": {"maxCharacters": _MAX_TEXT_CHARS}},
        }
        try:
            response = await self._http.post(
                EXA_SEARCH_URL,
                json=payload,
                headers={"x-api-key": self._api_key},
                timeout=_TIMEOUT_S,
            )
        except httpx.RequestError:
            raise SearchError("unavailable", "exa request failed") from None

        if response.status_code >= 400:
            raise response_error("exa", response)

        try:
            body = response.json()
        except ValueError:
            raise SearchError("bad_response", "exa returned invalid JSON") from None
        if not isinstance(body, dict):
            raise SearchError("bad_response", "exa returned a non-object body")
        document = cast(dict[str, Any], body)
        self.cost_usd += _cost(document)

        results = document.get("results")
        if not isinstance(results, list):
            raise SearchError("bad_response", "exa returned no results list")

        hits: list[SearchHit] = []
        for raw in results:
            if not isinstance(raw, dict):
                continue
            url = clean_url(raw.get("url"))
            if url is None:
                continue
            text = clip(raw.get("text"), TEXT_MAX_CHARS) or None
            raw_published = raw.get("publishedDate")
            published = (
                raw_published.strip() if isinstance(raw_published, str) and raw_published.strip() else None
            )
            hits.append(
                SearchHit(
                    provider=self.name,
                    query=query,
                    rank=len(hits) + 1,
                    url=url,
                    title=clip(raw.get("title"), TITLE_MAX_CHARS),
                    snippet=(text or "")[:SNIPPET_MAX_CHARS],
                    published=published,
                    text=text,
                )
            )
        return hits
