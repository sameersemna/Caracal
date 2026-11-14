# 02 — Configuration

`src/caracal/config.py` defines `class Settings(BaseSettings)` with `env_prefix=""`, `env_file=".env"`.
No other module reads `os.environ` (enforced by `scripts/check_rules.py`). Secrets are `SecretStr`.

| Variable | Type | Default | Notes |
|---|---|---|---|
| `CARACAL_HOST` | str | `127.0.0.1` | `0.0.0.0` for LAN |
| `CARACAL_PORT` | int | `11404` | |
| `CARACAL_API_TOKEN` | SecretStr | — **required** | bearer token for REST and HTTP MCP; min 32 chars |
| `CARACAL_ALLOWED_HOSTS` | csv | `` | extra `host[:port]` values accepted by the MCP transport's Host/Origin check, besides localhost/127.0.0.1 and the `CARACAL_PUBLIC_BASE_URL` host (the SDK answers **421 Invalid Host header** otherwise — measured) |
| `CARACAL_PUBLIC_BASE_URL` | str | `http://127.0.0.1:11404` | used to build download links in MCP results |
| `CARACAL_DATA_DIR` | path | `./data` | artifacts root (docs/05 §3) |
| `CARACAL_RETENTION_DAYS` | int | `30` | media files deleted after N days; reports kept |
| `CARACAL_KEEP_MEDIA` | bool | `false` | keep original download after normalize |
| `DATABASE_URL` | SecretStr | — **required** | any of `postgresql://`, `postgres://`, `postgresql+psycopg://`, `postgresql+asyncpg://`; normalised to `postgresql+asyncpg://` (the owner's `.env` uses `+psycopg`) |
| `CARACAL_DB_SCHEMA` | str | `public` | Postgres schema for all tables (identifier `^[a-z_][a-z0-9_]{0,62}$`). Live tests use a throwaway schema |
| `REDIS_URL` | SecretStr | — **required** | `redis://:***@latitude:6379/0` — DB 0 is **shared** with other apps |
| `REDIS_PREFIX` | str | `caracal:` | prepended to **every** Redis key/channel (docs/05 §2). Must be non-empty and end with `:` (validator appends `:` if missing) |
| `WHISPER_URL` | str | `http://localhost:8768` | whisper.cpp server |
| `WHISPER_TIMEOUT_S` | int | `900` | per chunk |
| `OLLAMA_CLOUD_URL` | str | `https://ollama.com/api` | Ollama cloud, native API |
| `OLLAMA_CLOUD_API_KEY` | SecretStr? | none | sent as `Authorization: Bearer`. Absent → primary goes through the local daemon's cloud proxy (`LLM_LOCAL_PROXY_MODEL`) |
| `OLLAMA_LOCAL_URL` | str | `http://127.0.0.1:11434` | local Ollama daemon |
| `LLM_MODEL` | str | `gemma4:31b` | primary model name **on the cloud** |
| `LLM_LOCAL_PROXY_MODEL` | str | `gemma4:31b-cloud` | primary when no cloud key: the local daemon proxies this tag to the cloud |
| `LLM_FALLBACK_MODEL` | str | `gemma4:31b` | **local** model (installed, runs on the GB10); used when the primary errors/times out twice in a row |
| `LLM_TIMEOUT_S` | int | `180` | |
| `LLM_KEEP_ALIVE` | str | `30m` | passed through to Ollama |
| `CARACAL_LLM_CONCURRENCY` | int | `4` | |
| `LLM_CONTEXT_CHARS` | int | `60000` | transcript chars per LLM call before map-reduce (docs/04 §5) |
| `HF_TOKEN` | SecretStr? | none | pyannote model download; absent → diarize skipped |
| `DIARIZATION_MODEL` | str | `pyannote/speaker-diarization-community-1` | |
| `DIARIZATION_DEVICE` | str | `cuda` | falls back to `cpu` if CUDA is unavailable (logged) |
| `VOICEPRINT_MATCH_THRESHOLD` | float | `0.72` | cosine similarity for a "match" (calibrated in T4.2) |
| `VOICEPRINT_CANDIDATE_THRESHOLD` | float | `0.55` | reported as a weak candidate |
| `SEARCH_PROVIDERS` | csv | `brave,exa` | order = priority |
| `BRAVE_API_KEY` | SecretStr? | none | provider disabled if absent |
| `EXA_API_KEY` | SecretStr? | none | provider disabled if absent |
| `SEARCH_MAX_QUERIES_PER_JOB` | int | `24` | hard budget |
| `SEARCH_CACHE_TTL_S` | int | `86400` | |
| `CARACAL_RATE_LIMIT_PER_MIN` | int | `30` | job creations per token+IP per minute (docs/06 S11) |
| `CARACAL_WORKER_CONCURRENCY` | int | `2` | |
| `CARACAL_JOB_TIMEOUT_S` | int | `10800` | |
| `CARACAL_MAX_UPLOAD_MB` | int | `2048` | |
| `CARACAL_MAX_DURATION_S` | int | `14400` | |
| `CARACAL_DEFAULT_TARGET_LANGS` | csv | `en` | auto-translate target(s) when the source language differs |
| `YTDLP_COOKIES_FILE` | path? | none | optional, for age-gated content the owner is entitled to |
| `CARACAL_LOG_LEVEL` | str | `INFO` | |

**URL normalisation:** both Ollama URL settings are normalised by a validator — strip trailing `/`, then a
trailing `/v1` or `/api` — so the stored value is the server root. The client always calls `<root>/api/chat`
(native API). The owner's `.env` uses `https://ollama.com/api` and `http://127.0.0.1:11434/v1`; both must work.

`.env.example` lists every variable with the default or a placeholder. `caracal doctor` prints each dependency's
reachability (Postgres, pgvector extension, Redis, whisper, Ollama + model, HF/pyannote importable, CUDA, search
keys present) without printing secrets, and exits non-zero if a **required** one fails.
