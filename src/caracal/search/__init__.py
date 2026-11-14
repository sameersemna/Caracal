"""Web-search adapters and the router used by cross-verification (docs/04 §8)."""

from caracal.search.base import (
    SearchError,
    SearchErrorKind,
    SearchHit,
    SearchProvider,
    clean_text,
    normalise_query,
)
from caracal.search.brave import BraveProvider
from caracal.search.exa import ExaProvider
from caracal.search.router import SearchResult, SearchRouter

__all__ = [
    "BraveProvider",
    "ExaProvider",
    "SearchError",
    "SearchErrorKind",
    "SearchHit",
    "SearchProvider",
    "SearchResult",
    "SearchRouter",
    "clean_text",
    "normalise_query",
]
