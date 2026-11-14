"""Application configuration (docs/02).

This is the only module allowed to read the process environment: every other module
imports a :class:`Settings` instance from here. Values are validated and normalised on
load so downstream code can trust them.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DATABASE_SCHEME = re.compile(
    r"^(?P<scheme>postgresql|postgres)(?:\+[A-Za-z0-9_]+)?://",
    re.IGNORECASE,
)
_DB_SCHEMA = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_OLLAMA_SUFFIXES = ("/api", "/v1")
_LOG_LEVELS = frozenset({"CRITICAL", "DEBUG", "ERROR", "INFO", "WARNING"})


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

    @field_validator("CARACAL_LOG_LEVEL")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"CARACAL_LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}")
        return level


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache the process-wide settings."""
    return Settings()
