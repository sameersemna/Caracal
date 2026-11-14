# 05 — Data

## 1. Postgres (database `caracal` on `latitude`, migrations via alembic)

Verified 2026-11-14: PostgreSQL **18.6** on latitude, database `caracal`, user `caracal` with CREATE on the database and
`public` schema; pgvector **0.8.1** available. The first migration runs `CREATE EXTENSION IF NOT EXISTS vector`.

| Table | Columns |
|---|---|
| `jobs` | `id uuid pk`, `status text`, `source_kind text`, `source_url text null`, `upload_id uuid null`, `title text null`, `options jsonb`, `progress jsonb`, `error jsonb null`, `created_at timestamptz`, `updated_at timestamptz`, `finished_at timestamptz null`, `sha256 text null` — index `(status, created_at desc)`, index `sha256` |
| `job_stages` | `job_id uuid fk cascade`, `name text`, `status text`, `reason text null`, `attempts int`, `started_at`, `ended_at`, pk `(job_id, name)` |
| `uploads` | `id uuid pk`, `filename text`, `size_bytes bigint`, `sha256 text`, `path text`, `created_at`, `consumed_by uuid null` |
| `reports` | `job_id uuid pk fk cascade`, `schema_version text`, `report jsonb`, `created_at` |
| `segments` | `job_id uuid fk cascade`, `seg_id int`, `start_s real`, `end_s real`, `speaker text null`, `lang text null`, `text text`, `tsv tsvector generated always as (to_tsvector('simple', text)) stored`, pk `(job_id, seg_id)`, GIN index on `tsv` |
| `known_speakers` | `id uuid pk`, `name text`, `name_norm text unique`, `notes text null`, `created_at`, `updated_at` |
| `voiceprints` | `id uuid pk`, `known_speaker_id uuid fk cascade`, `job_id uuid null`, `speaker_label text`, `embedding vector(EMB_DIM)`, `created_at` — HNSW index `vector_cosine_ops` |

`EMB_DIM = 256` (measured 2026-11-14: community-1 `speaker_embeddings` shape (n, 256)).

Dedup: a new job whose source `sha256` (uploads) or `webpage_url` (platform) matches a `done` job from the last
7 days with identical options returns the existing job id with `"deduplicated": true` unless `options.depth=="deep"`.

Conventions: all ids are `uuid` generated in Python (`uuid.uuid4()`); timestamps `timestamptz` set by the DB
(`server_default=now()`, updated with `now()` in UPDATE statements); `jsonb` for structured columns; names are
snake_case; no column is named with a SQL reserved word (hence `start_s`/`end_s`). Stage semantics: `attempts` increments on each entry into `running`; re-entering `running` clears `ended_at`;
`list_stages` of an unknown job returns `[]`. Everything lives in the schema
`CARACAL_DB_SCHEMA` (default `public`), including alembic's `alembic_version` table.

## 2. Redis — every key starts with `REDIS_PREFIX`

The Redis DB on latitude is **shared with other applications** (DB 0, ~40 k foreign keys measured 2026-11-14).
Rules (enforced by a unit test on `queue.py` with a recording fake and by `check_rules.py`):
- Every key and channel name is built by one helper, `queue.key(*parts) -> f"{REDIS_PREFIX}{':'.join(parts)}"`.
  No string literal key anywhere else.
- Never `FLUSHDB`, `FLUSHALL`, `KEYS`, or unprefixed `SCAN`. Cleanup only via `SCAN MATCH {REDIS_PREFIX}*`.
- **No arq / RQ / Celery** — they hard-code their own key prefixes (`arq:job:` etc., verified in arq 0.28).
  The job queue is our own Redis Streams queue in `queue.py`.

Keys (shown with the default prefix `caracal:`):

| Key | Type | Use |
|---|---|---|
| `caracal:jobs` | stream | job queue: `XADD` `{job_id, kind: "pipeline"|"translate", enqueued_at, payload?}` (`payload` = JSON, e.g. `{"target_langs": ["de"]}` for translate); consumer group `workers` |
| `caracal:worker:<worker_id>` | string, TTL 30 s | worker heartbeat (refreshed every 10 s) |
| `caracal:lock:whisper` | string | whisper single-slot lock (`SET NX PX 960000`, owner token, renewed; release only if owner matches — Lua) |
| `caracal:lock:retention` | string | daily retention job singleton |
| `caracal:events:<job_id>` | pub/sub channel | progress events (SSE + MCP progress) |
| `caracal:search:<provider>:<sha1(query)>` | string, TTL `SEARCH_CACHE_TTL_S` | cached hits |
| `caracal:cancel:<job_id>` | string, TTL 1 d | cancellation flag checked between stages and between chunks |
| `caracal:ratelimit:<sha1(token+ip)>` | string, TTL 60 s | job-creation rate limit counter |

Queue semantics (`queue.py`):
- `enqueue_job(job_id, kind)` → `XADD`. Group created with `XGROUP CREATE ... MKSTREAM` (ignore BUSYGROUP).
- Worker loop: `XREADGROUP GROUP workers <worker_id> COUNT 1 BLOCK 5000`, at most `CARACAL_WORKER_CONCURRENCY`
  in flight. While a job runs, the worker heartbeats its entry every 60 s with an **owner-checked** Lua `XCLAIM … JUSTID`
  (only if `XPENDING` still lists this consumer as owner). If the heartbeat finds another owner, the job is **lost**: the runner
  stops before the next stage (and does not write the result of the stage in progress) and returns `lost` without writing any
  status/stage/checkpoint/event; the worker does not ack. On finish (done/failed/cancelled) → owner-checked Lua `XACK` + `XDEL`.
  Rate-limit counters are updated by one Lua script (INCR + EXPIRE when TTL is −1). Event `pct` = finished stages / total.
- Recovery: every 60 s, `XAUTOCLAIM` entries idle > 300 s (their worker died) and re-run them; the runner resumes
  from checkpoints (docs/01 §4). Entry delivered > 3 times → job `failed` with `internal: "worker crashed repeatedly"`.
- `XTRIM MAXLEN ~ 10000` on enqueue.

## 3. Artifacts (`CARACAL_DATA_DIR/jobs/<job_id>/`)

```
source/<original filename>      (deleted after normalize unless CARACAL_KEEP_MEDIA)
audio.wav                       16 kHz mono PCM
chunks/000.wav ...              deleted after transcribe
captions/<lang>.vtt             platform captions (if any)
stages/<stage>.json             checkpoints (docs/01 §4)
transcript.json  transcript.txt
subtitles/source.srt|vtt  subtitles/<lang>.srt|vtt
report.json  report.md
```
`uploads/<upload_id>/<filename>` for uploads before they are consumed. Writes are atomic (`tmp` + `os.replace`).
Retention job (worker background task, daily, singleton via `caracal:lock:retention`): delete `audio.wav`, `source/`, `chunks/` older than `CARACAL_RETENTION_DAYS`;
unconsumed uploads older than 24 h.
