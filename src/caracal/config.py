"""Application configuration (docs/02).

This is the only module allowed to read the process environment: every other module
imports a :class:`Settings` instance from here. Values are validated and normalised on
load so downstream code can trust them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DATABASE_SCHEME = re.compile(
    r"^(?P<scheme>postgresql|postgres)(?:\+[A-Za-z0-9_]+)?://",
    re.IGNORECASE,
)
_DB_SCHEMA = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_OLLAMA_SUFFIXES = ("/api", "/v1")
_LOG_LEVELS = frozenset({"CRITICAL", "DEBUG", "ERROR", "INFO", "WARNING"})
_MODEL_REF_PREFIXES: tuple[Literal["cloud", "local"], ...] = ("cloud", "local")


def _as_plain(value: Any) -> Any:
    """Unwrap a SecretStr so validators can inspect the raw text."""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def normalise_ollama_url(value: str) -> str:
    """Strip a trailing slash and a trailing ``/v1`` or ``/api`` from an Ollama URL."""
    url = value.strip().rstrip("/")
    for suffix in _OLLAMA_SUFFIXES:
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
            break
    if not url:
        raise ValueError("Ollama URL must not be empty")
    return url


def normalise_database_url(value: str) -> str:
    """Rewrite any accepted Postgres scheme to ``postgresql+asyncpg://``.

    The user, password, host, port, database and query string are preserved verbatim.
    """
    candidate = value.strip()
    if not _DATABASE_SCHEME.match(candidate):
        raise ValueError(
            "DATABASE_URL must use postgresql://, postgres://, postgresql+psycopg:// or postgresql+asyncpg://"
        )
    return _DATABASE_SCHEME.sub("postgresql+asyncpg://", candidate, count=1)


def parse_csv(value: Any) -> list[str]:
    """Parse a comma-separated setting into a clean list of lowercase tokens."""
    if value is None:
        return []
    if isinstance(value, str):
        parts: list[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = list(value)
    else:
        raise ValueError("expected a comma-separated string or a list of strings")
    parsed: list[str] = []
    for part in parts:
        token = str(part).strip().lower()
        if token:
            parsed.append(token)
    return parsed


def parse_ref_csv(value: Any) -> list[str]:
    """Parse a comma-separated model-ref list, preserving the case of every ref."""
    if value is None:
        return []
    if isinstance(value, str):
        parts: list[Any] = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        parts = list(value)
    else:
        raise ValueError("expected a comma-separated string or a list of strings")
    return [str(part).strip() for part in parts if str(part).strip()]


class Route(StrEnum):
    """The task a structured LLM call is made for (docs/02 ``LLM_ROUTE_*``)."""

    DEFAULT = "default"
    PLAN = "plan"
    CONTENT = "content"
    IDENTITY = "identity"
    VERIFY = "verify"
    TRANSLATE = "translate"
    REVIEW = "review"
    QA = "qa"


@dataclass(frozen=True, slots=True)
class ModelRef:
    """One model reference: which Ollama endpoint to call and the model name on it.

    ``raw`` is the text the owner wrote (``cloud:gemma4:31b``, ``local:gemma4:31b`` or a bare
    ``gemma4:31b-cloud``); ``normalised`` compares refs regardless of the prefix the owner used.
    """

    endpoint: Literal["cloud", "local"]
    name: str
    raw: str

    @property
    def normalised(self) -> str:
        return f"{self.endpoint}:{self.name}"


def parse_model_ref(raw: str) -> ModelRef:
    """Parse one model reference (docs/02 ``LLM_ALLOWED_MODELS``).

    ``cloud:<name>`` calls ``OLLAMA_CLOUD_URL`` with the bearer key, ``local:<name>`` calls
    ``OLLAMA_LOCAL_URL``, and a bare ``<name>`` is local (the daemon proxies ``-cloud`` tags).
    """
    text = raw.strip()
    if not text:
        raise ValueError("a model reference must not be empty")
    for prefix in _MODEL_REF_PREFIXES:
        marker = f"{prefix}:"
        if text.startswith(marker):
            name = text[len(marker) :].strip()
            if not name:
                raise ValueError(f"model reference {text!r} has no model name")
            return ModelRef(endpoint=prefix, name=name, raw=text)
    return ModelRef(endpoint="local", name=text, raw=text)


#: The environment variable that configures each route.
ROUTE_FIELDS: dict[Route, str] = {
    Route.DEFAULT: "LLM_ROUTE_DEFAULT",
    Route.PLAN: "LLM_ROUTE_PLAN",
    Route.CONTENT: "LLM_ROUTE_CONTENT",
    Route.IDENTITY: "LLM_ROUTE_IDENTITY",
    Route.VERIFY: "LLM_ROUTE_VERIFY",
    Route.TRANSLATE: "LLM_ROUTE_TRANSLATE",
    Route.REVIEW: "LLM_ROUTE_REVIEW",
    Route.QA: "LLM_ROUTE_QA",
}


@dataclass(frozen=True, slots=True)
class LlmRouting:
    """The resolved allow-list and the ordered fallback chain of every route."""

    allowed: tuple[ModelRef, ...]
    chains: dict[Route, tuple[ModelRef, ...]]


def _dedupe(refs: Iterable[ModelRef]) -> tuple[ModelRef, ...]:
    seen: set[str] = set()
    ordered: list[ModelRef] = []
    for ref in refs:
        if ref.normalised not in seen:
            seen.add(ref.normalised)
            ordered.append(ref)
    return tuple(ordered)


def _legacy_chain(settings: Settings) -> tuple[ModelRef, ModelRef]:
    """Today's behaviour: cloud (key) or the local cloud proxy, then the local fallback."""
    if settings.OLLAMA_CLOUD_API_KEY is not None:
        primary = ModelRef(
            endpoint="cloud",
            name=settings.LLM_MODEL,
            raw=f"cloud:{settings.LLM_MODEL}",
        )
    else:
        primary = ModelRef(
            endpoint="local",
            name=settings.LLM_LOCAL_PROXY_MODEL,
            raw=settings.LLM_LOCAL_PROXY_MODEL,
        )
    fallback = ModelRef(
        endpoint="local",
        name=settings.LLM_FALLBACK_MODEL,
        raw=f"local:{settings.LLM_FALLBACK_MODEL}",
    )
    return primary, fallback


def resolve_llm_routing(settings: Settings) -> LlmRouting:
    """Resolve ``LLM_ALLOWED_MODELS`` and every ``LLM_ROUTE_*`` into ordered ref chains.

    With neither the allow-list nor any route configured the result is exactly the
    pre-routing behaviour (cloud or local proxy, then the local fallback model), so an
    existing deployment keeps working unchanged. Every ref of every chain must appear in
    the allow-list (compared as ``endpoint:name``); a violation raises ``ValueError`` that
    names the route's variable and the offending model.
    """
    explicit_allowed = [parse_model_ref(raw) for raw in settings.LLM_ALLOWED_MODELS]
    configured: dict[Route, tuple[ModelRef, ...]] = {}
    for route, field in ROUTE_FIELDS.items():
        refs = tuple(parse_model_ref(raw) for raw in getattr(settings, field))
        if refs:
            configured[route] = refs

    default: tuple[ModelRef, ...]
    if not explicit_allowed and not configured:
        primary, fallback = _legacy_chain(settings)
        allowed = _dedupe([primary, fallback])
        default = (primary, fallback)
    else:
        allowed = _dedupe(
            explicit_allowed if explicit_allowed else [ref for chain in configured.values() for ref in chain]
        )
        default = configured.get(Route.DEFAULT) or allowed

    chains: dict[Route, tuple[ModelRef, ...]] = {Route.DEFAULT: default}
    for route, chain in configured.items():
        if route is not Route.DEFAULT:
            chains[route] = chain
    for route in Route:
        chains.setdefault(route, default)

    allowed_names = {ref.normalised for ref in allowed}
    for route in Route:
        for ref in chains[route]:
            if ref.normalised not in allowed_names:
                listed = ", ".join(sorted(allowed_names)) or "empty"
                raise ValueError(
                    f"{ROUTE_FIELDS[route]} uses model {ref.raw!r} ({ref.normalised}), which is "
                    f"not in LLM_ALLOWED_MODELS ([{listed}])"
                )
    for ref in allowed:
        if ref.endpoint == "cloud" and settings.OLLAMA_CLOUD_API_KEY is None:
            raise ValueError(
                f"model {ref.raw!r} ({ref.normalised}) needs OLLAMA_CLOUD_API_KEY, which is not set"
            )
    return LlmRouting(allowed=allowed, chains=chains)


class Settings(BaseSettings):
    """Validated Caracal configuration. See docs/02 for every variable."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    CARACAL_HOST: str = "127.0.0.1"
    CARACAL_PORT: int = 11404
    CARACAL_API_TOKEN: SecretStr
    CARACAL_PUBLIC_BASE_URL: str = "http://127.0.0.1:11404"
    CARACAL_ALLOWED_HOSTS: Annotated[list[str], NoDecode] = []
    CARACAL_DATA_DIR: Path = Path("./data")
    CARACAL_RETENTION_DAYS: int = 30
    CARACAL_KEEP_MEDIA: bool = False

    DATABASE_URL: SecretStr
    CARACAL_DB_SCHEMA: str = "public"
    REDIS_URL: SecretStr
    REDIS_PREFIX: str = "caracal:"

    WHISPER_URL: str = "http://localhost:8768"
    WHISPER_TIMEOUT_S: int = 900

    OLLAMA_CLOUD_URL: str = "https://ollama.com/api"
    OLLAMA_CLOUD_API_KEY: SecretStr | None = None
    OLLAMA_LOCAL_URL: str = "http://127.0.0.1:11434"
    LLM_MODEL: str = "gemma4:31b"
    LLM_LOCAL_PROXY_MODEL: str = "gemma4:31b-cloud"
    LLM_FALLBACK_MODEL: str = "gemma4:31b"
    LLM_ALLOWED_MODELS: Annotated[list[str], NoDecode] = []
    LLM_ROUTE_DEFAULT: Annotated[list[str], NoDecode] = []
    LLM_ROUTE_PLAN: Annotated[list[str], NoDecode] = []
    LLM_ROUTE_CONTENT: Annotated[list[str], NoDecode] = []
    LLM_ROUTE_IDENTITY: Annotated[list[str], NoDecode] = []
    LLM_ROUTE_VERIFY: Annotated[list[str], NoDecode] = []
    LLM_ROUTE_TRANSLATE: Annotated[list[str], NoDecode] = []
    LLM_ROUTE_REVIEW: Annotated[list[str], NoDecode] = []
    LLM_ROUTE_QA: Annotated[list[str], NoDecode] = []
    LLM_TIMEOUT_S: int = 180
    LLM_KEEP_ALIVE: str = "30m"
    CARACAL_LLM_CONCURRENCY: int = 4
    LLM_CONTEXT_CHARS: int = 60000

    HF_TOKEN: SecretStr | None = None
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-community-1"
    DIARIZATION_DEVICE: str = "cuda"
    VOICEPRINT_MATCH_THRESHOLD: float = 0.72
    VOICEPRINT_CANDIDATE_THRESHOLD: float = 0.55

    SEARCH_PROVIDERS: Annotated[list[str], NoDecode] = ["brave", "exa"]
    BRAVE_API_KEY: SecretStr | None = None
    EXA_API_KEY: SecretStr | None = None
    SEARCH_MAX_QUERIES_PER_JOB: int = 24
    SEARCH_CACHE_TTL_S: int = 86400

    CARACAL_RATE_LIMIT_PER_MIN: int = 30
    CARACAL_WORKER_CONCURRENCY: int = 2
    CARACAL_JOB_TIMEOUT_S: int = 10800
    CARACAL_MAX_UPLOAD_MB: int = 2048
    CARACAL_MAX_DURATION_S: int = 14400
    CARACAL_REPORT_LANGUAGE: str = "en"
    CARACAL_DEFAULT_TARGET_LANGS: Annotated[list[str], NoDecode] = ["en"]

    YTDLP_COOKIES_FILE: Path | None = None
    CARACAL_LOG_LEVEL: str = "INFO"

    @field_validator("CARACAL_API_TOKEN")
    @classmethod
    def _validate_api_token(cls, value: SecretStr) -> SecretStr:
        if len(value.get_secret_value()) < 32:
            raise ValueError("CARACAL_API_TOKEN must be at least 32 characters")
        return value

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _normalise_database_url(cls, value: Any) -> Any:
        if value is None:
            return value
        return normalise_database_url(str(_as_plain(value)))

    @field_validator("CARACAL_DB_SCHEMA")
    @classmethod
    def _validate_db_schema(cls, value: str) -> str:
        schema = value.strip()
        if not _DB_SCHEMA.fullmatch(schema):
            raise ValueError("CARACAL_DB_SCHEMA must match ^[a-z_][a-z0-9_]{0,62}$")
        return schema

    @field_validator("REDIS_PREFIX")
    @classmethod
    def _normalise_redis_prefix(cls, value: str) -> str:
        prefix = value.strip()
        if not prefix:
            raise ValueError("REDIS_PREFIX must not be empty")
        return prefix if prefix.endswith(":") else f"{prefix}:"

    @field_validator("OLLAMA_CLOUD_URL", "OLLAMA_LOCAL_URL")
    @classmethod
    def _normalise_ollama(cls, value: str) -> str:
        return normalise_ollama_url(value)

    @field_validator(
        "SEARCH_PROVIDERS",
        "CARACAL_DEFAULT_TARGET_LANGS",
        "CARACAL_ALLOWED_HOSTS",
        mode="before",
    )
    @classmethod
    def _parse_csv_fields(cls, value: Any) -> list[str]:
        return parse_csv(value)

    @field_validator(
        "LLM_ALLOWED_MODELS",
        "LLM_ROUTE_DEFAULT",
        "LLM_ROUTE_PLAN",
        "LLM_ROUTE_CONTENT",
        "LLM_ROUTE_IDENTITY",
        "LLM_ROUTE_VERIFY",
        "LLM_ROUTE_TRANSLATE",
        "LLM_ROUTE_REVIEW",
        "LLM_ROUTE_QA",
        mode="before",
    )
    @classmethod
    def _parse_ref_csv_fields(cls, value: Any) -> list[str]:
        return parse_ref_csv(value)

    @model_validator(mode="after")
    def _validate_llm_routing(self) -> Settings:
        """Reject a route that uses a model outside ``LLM_ALLOWED_MODELS`` at startup."""
        resolve_llm_routing(self)
        return self

    @field_validator(
        "OLLAMA_CLOUD_API_KEY",
        "HF_TOKEN",
        "BRAVE_API_KEY",
        "EXA_API_KEY",
        mode="before",
    )
    @classmethod
    def _empty_secret_is_none(cls, value: Any) -> Any:
        raw = _as_plain(value)
        if raw is None:
            return None
        if isinstance(raw, str) and not raw.strip():
            return None
        return raw

    @field_validator("YTDLP_COOKIES_FILE", mode="before")
    @classmethod
    def _empty_path_is_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("CARACAL_REPORT_LANGUAGE")
    @classmethod
    def _normalise_report_language(cls, value: str) -> str:
        language = value.strip()
        if not language:
            raise ValueError("CARACAL_REPORT_LANGUAGE must not be empty")
        return language

    @field_validator("CARACAL_LOG_LEVEL")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"CARACAL_LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}")
        return level

    def llm_allowed_models(self) -> tuple[ModelRef, ...]:
        """Every model a route may use, in allow-list order (docs/02)."""
        return resolve_llm_routing(self).allowed

    def llm_route_chain(self, route: Route) -> tuple[ModelRef, ...]:
        """The ordered fallback chain of ``route``; an unconfigured task uses DEFAULT."""
        return resolve_llm_routing(self).chains[route]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the process-wide settings."""
    return Settings()
