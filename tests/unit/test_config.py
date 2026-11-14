"""Configuration validation and normalisation (docs/02)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from caracal.config import Settings, get_settings

MakeSettings = Callable[..., Settings]


def _token(settings: Settings) -> str:
    return settings.CARACAL_API_TOKEN.get_secret_value()


def _database_url(settings: Settings) -> str:
    return settings.DATABASE_URL.get_secret_value()


def test_missing_token_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(
            _env_file=None,
            DATABASE_URL="postgresql://u:p@h:5432/caracal",
            REDIS_URL="redis://:p@h:6379/0",
        )
    assert any("CARACAL_API_TOKEN" in str(error["loc"]) for error in excinfo.value.errors())


def test_short_token_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 32 characters"):
        Settings(
            _env_file=None,
            CARACAL_API_TOKEN="x" * 31,
            DATABASE_URL="postgresql://u:p@h:5432/caracal",
            REDIS_URL="redis://:p@h:6379/0",
        )


def test_empty_token_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            CARACAL_API_TOKEN="",
            DATABASE_URL="postgresql://u:p@h:5432/caracal",
            REDIS_URL="redis://:p@h:6379/0",
        )


def test_token_of_exactly_32_characters_is_accepted(make_settings: MakeSettings) -> None:
    settings = make_settings(CARACAL_API_TOKEN="y" * 32)
    assert _token(settings) == "y" * 32


def test_csv_fields_are_split_stripped_lowercased_and_empties_dropped(make_settings: MakeSettings) -> None:
    settings = make_settings(
        SEARCH_PROVIDERS=" Brave , EXA ,,",
        CARACAL_DEFAULT_TARGET_LANGS="EN, De , ",
    )
    assert settings.SEARCH_PROVIDERS == ["brave", "exa"]
    assert settings.CARACAL_DEFAULT_TARGET_LANGS == ["en", "de"]


def test_csv_defaults(make_settings: MakeSettings) -> None:
    settings = make_settings()
    assert settings.SEARCH_PROVIDERS == ["brave", "exa"]
    assert settings.CARACAL_DEFAULT_TARGET_LANGS == ["en"]


def test_csv_settings_are_parsed_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARACAL_API_TOKEN", "e" * 32)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/caracal")
    monkeypatch.setenv("REDIS_URL", "redis://:p@h:6379/0")
    monkeypatch.setenv("SEARCH_PROVIDERS", " Brave ,EXA, ,")
    monkeypatch.setenv("CARACAL_DEFAULT_TARGET_LANGS", "EN, De")
    settings = Settings(_env_file=None)
    assert settings.SEARCH_PROVIDERS == ["brave", "exa"]
    assert settings.CARACAL_DEFAULT_TARGET_LANGS == ["en", "de"]


def test_empty_optional_secret_from_the_environment_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARACAL_API_TOKEN", "e" * 32)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/caracal")
    monkeypatch.setenv("REDIS_URL", "redis://:p@h:6379/0")
    monkeypatch.setenv("HF_TOKEN", "")
    settings = Settings(_env_file=None)
    assert settings.HF_TOKEN is None


def test_empty_cookies_path_is_none(make_settings: MakeSettings) -> None:
    assert make_settings(YTDLP_COOKIES_FILE="").YTDLP_COOKIES_FILE is None


@pytest.mark.parametrize(
    "field",
    ["OLLAMA_CLOUD_API_KEY", "HF_TOKEN", "BRAVE_API_KEY", "EXA_API_KEY"],
)
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_optional_secrets_become_none(make_settings: MakeSettings, field: str, value: str) -> None:
    settings = make_settings(**{field: value})
    assert getattr(settings, field) is None


def test_optional_secrets_unset_are_none(make_settings: MakeSettings) -> None:
    settings = make_settings()
    assert settings.OLLAMA_CLOUD_API_KEY is None
    assert settings.HF_TOKEN is None
    assert settings.BRAVE_API_KEY is None
    assert settings.EXA_API_KEY is None


def test_optional_secret_value_is_kept(make_settings: MakeSettings) -> None:
    value = "hf_" + "secret_value"
    settings = make_settings(HF_TOKEN=value)
    assert settings.HF_TOKEN is not None
    assert settings.HF_TOKEN.get_secret_value() == value


def test_defaults_match_docs_02(make_settings: MakeSettings) -> None:
    settings = make_settings()
    assert settings.CARACAL_HOST == "127.0.0.1"
    assert settings.CARACAL_PORT == 11404
    assert settings.CARACAL_PUBLIC_BASE_URL == "http://127.0.0.1:11404"
    assert settings.CARACAL_DATA_DIR == Path("data")
    assert settings.CARACAL_RETENTION_DAYS == 30
    assert settings.CARACAL_KEEP_MEDIA is False
    assert settings.REDIS_PREFIX == "caracal:"
    assert settings.WHISPER_URL == "http://localhost:8768"
    assert settings.WHISPER_TIMEOUT_S == 900
    assert settings.OLLAMA_CLOUD_URL == "https://ollama.com"
    assert settings.OLLAMA_LOCAL_URL == "http://127.0.0.1:11434"
    assert settings.LLM_MODEL == "gemma4:31b"
    assert settings.LLM_LOCAL_PROXY_MODEL == "gemma4:31b-cloud"
    assert settings.LLM_FALLBACK_MODEL == "gemma4:31b"
    assert settings.LLM_TIMEOUT_S == 180
    assert settings.LLM_KEEP_ALIVE == "30m"
    assert settings.CARACAL_LLM_CONCURRENCY == 4
    assert settings.LLM_CONTEXT_CHARS == 60000
    assert settings.DIARIZATION_MODEL == "pyannote/speaker-diarization-community-1"
    assert settings.DIARIZATION_DEVICE == "cuda"
    assert settings.VOICEPRINT_MATCH_THRESHOLD == pytest.approx(0.72)
    assert settings.VOICEPRINT_CANDIDATE_THRESHOLD == pytest.approx(0.55)
    assert settings.SEARCH_MAX_QUERIES_PER_JOB == 24
    assert settings.SEARCH_CACHE_TTL_S == 86400
    assert settings.CARACAL_WORKER_CONCURRENCY == 2
    assert settings.CARACAL_JOB_TIMEOUT_S == 10800
    assert settings.CARACAL_MAX_UPLOAD_MB == 2048
    assert settings.CARACAL_MAX_DURATION_S == 14400
    assert settings.YTDLP_COOKIES_FILE is None
    assert settings.CARACAL_LOG_LEVEL == "INFO"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://ollama.com/api", "https://ollama.com"),
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434"),
        ("http://x:11434/", "http://x:11434"),
        ("http://x:11434/api/", "http://x:11434"),
        ("https://ollama.com", "https://ollama.com"),
    ],
)
def test_ollama_cloud_url_normalisation(make_settings: MakeSettings, value: str, expected: str) -> None:
    assert make_settings(OLLAMA_CLOUD_URL=value).OLLAMA_CLOUD_URL == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434"),
        ("http://localhost:11434/api", "http://localhost:11434"),
    ],
)
def test_ollama_local_url_normalisation(make_settings: MakeSettings, value: str, expected: str) -> None:
    assert make_settings(OLLAMA_LOCAL_URL=value).OLLAMA_LOCAL_URL == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("postgresql://u:p@h:5432/caracal", "postgresql+asyncpg://u:p@h:5432/caracal"),
        ("postgres://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
        (
            "postgresql+psycopg://u:p%40ss@h:5432/caracal?sslmode=require",
            "postgresql+asyncpg://u:p%40ss@h:5432/caracal?sslmode=require",
        ),
        (
            "postgresql+asyncpg://u:p@h/db?application_name=caracal",
            "postgresql+asyncpg://u:p@h/db?application_name=caracal",
        ),
    ],
)
def test_database_url_normalisation(make_settings: MakeSettings, value: str, expected: str) -> None:
    assert _database_url(make_settings(DATABASE_URL=value)) == expected


def test_non_postgres_database_url_is_rejected(make_settings: MakeSettings) -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        make_settings(DATABASE_URL="mysql://u:p@h/db")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("caracal", "caracal:"),
        ("caracal:", "caracal:"),
        ("  jobs  ", "jobs:"),
        ("myapp:jobs", "myapp:jobs:"),
    ],
)
def test_redis_prefix_normalisation(make_settings: MakeSettings, value: str, expected: str) -> None:
    assert make_settings(REDIS_PREFIX=value).REDIS_PREFIX == expected


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_redis_prefix_is_rejected(make_settings: MakeSettings, value: str) -> None:
    with pytest.raises(ValidationError, match="REDIS_PREFIX"):
        make_settings(REDIS_PREFIX=value)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARACAL_API_TOKEN", "z" * 32)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5432/caracal")
    monkeypatch.setenv("REDIS_URL", "redis://:p@h:6379/0")
    first = get_settings()
    assert get_settings() is first

    get_settings.cache_clear()
    monkeypatch.setenv("CARACAL_PORT", "12000")
    assert get_settings().CARACAL_PORT == 12000


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("POSTGRESQL://u:p@h:5432/caracal", "postgresql+asyncpg://u:p@h:5432/caracal"),
        ("Postgres://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
        (
            "PostgreSQL+Psycopg://u:p@h/db?sslmode=require",
            "postgresql+asyncpg://u:p@h/db?sslmode=require",
        ),
    ],
)
def test_database_url_scheme_is_case_insensitive(
    make_settings: MakeSettings, value: str, expected: str
) -> None:
    assert _database_url(make_settings(DATABASE_URL=value)) == expected


@pytest.mark.parametrize("scheme", ["MYSQL", "PostgresX", "Postgresql+Weird!"])
def test_unknown_database_scheme_is_rejected(make_settings: MakeSettings, scheme: str) -> None:
    with pytest.raises(ValidationError, match="DATABASE_URL"):
        make_settings(DATABASE_URL=f"{scheme}://u:p@h/db")
