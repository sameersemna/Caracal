# 01 — Architecture

## 1. Components

```
          REST /v1/*  ─┐                       ┌─> whisper.cpp  (localhost:8768)
 clients               ├─ FastAPI app (11404) ─┤
          MCP  /mcp   ─┘   │ enqueue           ├─> Ollama cloud gemma4:31b (fallback: local gemma4:31b)
                           v                    ├─> Brave / Exa search APIs
                    Redis (latitude)  <── worker process (Streams) ── pyannote (GPU, in-process)
                           │                    │
                    Postgres+pgvector (latitude)    artifacts on disk  (CARACAL_DATA_DIR)
```

- **API process** (`caracal serve`): FastAPI + MCP (streamable HTTP mounted at `/mcp`). Never does heavy work;
  it validates, stores uploads, enqueues, and reads results.
- **Worker process** (`caracal worker`): consumes our Redis Streams queue (docs/05 §2) and runs the pipeline,
  up to `CARACAL_WORKER_CONCURRENCY` (default 2) jobs at once. Whisper calls are serialised by a Redis lock
  (`<REDIS_PREFIX>lock:whisper`) because the server is single-slot.
- **MCP stdio** (`caracal mcp-stdio`): same tools, stdio transport, talks to the same DB/Redis.

One package, three entry points (`[project.scripts] caracal = "caracal.cli:main"`).

## 2. Stack (pinned in pyproject; verify APIs with context7 before use)

Python 3.12, uv. Runtime: `fastapi`, `uvicorn[standard]`, `mcp` 2.3.0 (official SDK; server class `MCPServer` — FastMCP was renamed in 2.x), `pydantic` v2,
`pydantic-settings`, `httpx`, `sqlalchemy[asyncio]` 2.x core (no ORM models needed) + `asyncpg`, `alembic`,
`pgvector`, `redis` (asyncio), `yt-dlp`, `python-multipart`. **No arq/RQ/Celery** (they ignore `REDIS_PREFIX`).
Speaker extra (`[project.optional-dependencies] speakers`): `pyannote.audio` 4.x, `torch`, `torchaudio`.
Dev: `pytest`, `pytest-asyncio`, `respx`, `ruff`, `mypy`.

The speakers extra is optional at import time: if `pyannote` is not importable, the diarize stage reports
`skipped: reason=unavailable` and the job still succeeds (docs/04 §2).

## 3. Repository layout (binding — do not invent other top-level modules)

```
pyproject.toml  uv.lock  alembic.ini  .env.example  README.md
src/caracal/
  __init__.py            __version__
  cli.py                 main(): serve | worker | mcp-stdio | migrate | doctor
  config.py              Settings (the ONLY module that reads os.environ)
  log.py                 get_logger(name); JSON lines to stderr; redact()
  errors.py              CaracalError(code, message, details) + ErrorCode enum (docs/03 §4)
  app.py                 create_app(settings) -> FastAPI
  services.py            JobService: the one implementation of every capability; REST routes and MCP tools are thin adapters over it (docs/03 §0)
  api/auth.py            bearer-token dependency
  api/routes.py          /v1 router
  api/schemas.py         request/response pydantic models (docs/03 §2)
  mcp/server.py          build_mcp(service, settings) -> mcp.server.mcpserver.MCPServer (mcp 2.x — FastMCP no longer exists); tool functions (docs/03 §3)
  db/engine.py           make_engine(settings), session helpers
  db/tables.py           SQLAlchemy Core Table objects (docs/05 §1)
  db/repo.py             Protocols + Postgres impls: JobRepo, UploadRepo, ReportRepo, SegmentRepo (T0.2), SpeakerRepo (T2.3)
  migrations/            alembic env + versions
  queue.py               key(*parts), enqueue_job(), JobQueue (Redis Streams consumer), whisper_lock(), publish_event()
  worker.py              Worker.run(): consume loop, heartbeats, XAUTOCLAIM recovery, retention task; run_job(job_id)
  storage.py             JobFiles(job_id): paths for every artifact (docs/05 §3)
  ingest/ssrf.py         resolve_and_check(url) -> pinned IP or raise
  ingest/upload.py       save_upload(stream, filename) -> UploadInfo
  ingest/direct.py       download_direct(url, dest) (http(s) media URLs)
  ingest/ytdlp.py        fetch_metadata(url), download_audio(url, dest), fetch_platform_captions(url)
  ingest/router.py       classify_source(url) -> "direct" | "platform"
  audio/ffmpeg.py        probe(path), normalize(src, dst), detect_silences(path), cut(src, dst, start, end)
  audio/chunking.py      plan_chunks(duration, silences, target_s, max_s, overlap_s) -> list[Chunk]
  asr/whisper.py         WhisperClient.transcribe(wav, language, translate=False) -> WhisperResult
  asr/merge.py           merge_chunks(results, chunks) -> Transcript (dedupe overlap)
  asr/langid.py          language_spans(transcript) -> list[LangSpan]
  asr/captions.py        parse_vtt(text) -> list[CaptionWord]; caption_agreement(transcript, caption_words) -> float
  speakers/diarize.py    Diarizer.run(wav) -> list[Turn], embeddings per label
  speakers/align.py      assign_speakers(transcript, turns) -> Transcript (word.speaker set)
  speakers/voiceprint.py match(embedding) -> list[VoiceMatch]; enroll(label, job_id, name)
  llm/client.py          LLMClient.call_tool(...), .chat(...)  (docs/04 §1)
  llm/jsonx.py           extract_json(text) -> dict  (fence/prose tolerant)
  llm/schema.py          prepare_schema(model) (inline $refs, enum hints), coerce(value, schema) (measured necessity, docs/04 §1)
  llm/prompts/*.md       one prompt file per stage (loaded by name)
  search/base.py         SearchProvider protocol, SearchHit
  search/brave.py  search/exa.py  search/router.py (fallback + Redis cache)
  analysis/planner.py    plan(job_context) -> StagePlan
  analysis/content.py    analyse_content(...) -> ContentAnalysis
  analysis/identity.py   resolve_identities(...) -> list[SpeakerIdentity]
  analysis/verify.py     verify(...) -> list[Verification]
  analysis/translate.py  translate_segments(...) -> TranslatedTranscript
  analysis/review.py     self_review(report) -> list[ReviewFlag]
  analysis/qa.py         answer(job_id, question) -> Answer
  report/model.py        Report and all sub-models (docs/03 §5) — pydantic
  report/evidence.py     EvidenceIndex, validate_citations(report) -> report (drops uncited claims)
  report/render_md.py    render_markdown(report) -> str
  report/assemble.py     assemble_report(checkpoints, job, settings) -> Report (pure; T2.7)
  subtitles/segmenter.py cues_from_transcript(transcript, max_chars=42, max_lines=2, max_dur=7.0, min_dur=1.0)
  subtitles/formats.py   to_srt(cues), to_vtt(cues)
  pipeline/stages.py     Stage enum + per-stage functions (signature §4)
  pipeline/runner.py     run_pipeline(job_id) — checkpoint/resume, progress events
tests/
  unit/ contract/ live/ fixtures/ conftest.py fakes/
scripts/
  gate.sh  check_rules.py  run-card.sh
```

## 4. Pipeline (DAG)

```
ingest ─> normalize ─> chunk ─> transcribe ─> langid ─> diarize ─> plan ─┬─> align ─> voiceprint ─┐
            │                                                  │   (align regenerates subtitles)  │
            └─ (platform metadata + captions from ingest) ──────┴──────────────────────────────────┤
                                                                                                  v
                           content ─> identity ─> verify ─> translate ─> subtitles ─> review ─> assemble
(The planner runs right after diarize: pyannote's speech share is the measured music discriminator — music 2 % vs speech 87 %. Its rules
then skip align/voiceprint and the analysis stages that do not apply — verdict 25.)
```

Every stage has the signature:

```python
async def stage_<name>(ctx: StageContext) -> StageResult
```

- `StageContext`: `job_id`, `options: JobOptions`, `files: JobFiles`, `settings`, clients (`whisper`, `llm`,
  `search`, `repo`, `diarizer | None`), `load(stage) -> dict` (reads a previous checkpoint), `progress(pct, msg)`.
- `StageResult`: `status: "done" | "skipped"`, `reason: str | None`, `output: dict` (JSON-serialisable).
- The runner writes `output` to `files.stage(<name>)` (`<job>/stages/<name>.json`) atomically and records stage
  status in Postgres **before** starting the next stage.
- **Resume:** on (re)start the runner skips stages whose checkpoint exists and whose status is `done|skipped`.
- A stage raising `CaracalError` with `retryable=True` is retried up to 2× with backoff (2 s, 8 s). Any other
  exception fails the stage. **Failure policy:** `ingest, normalize, transcribe` failures fail the job; every other
  stage failure is recorded as `failed` and the pipeline **continues** (the report notes the missing section).
- `plan` (docs/04 §2) may mark later stages `skipped` with a reason before they run.

## 5. Concurrency and time

- Whisper: global Redis lock per chunk call (not per job), so two jobs interleave chunks fairly.
- LLM: semaphore `CARACAL_LLM_CONCURRENCY` (default 4) per worker process.
- Search: per-provider rate limit (token bucket in-process) and 24 h Redis cache keyed by provider+query hash.
- Job hard timeout `CARACAL_JOB_TIMEOUT_S` (default 3 h). Max media duration `CARACAL_MAX_DURATION_S` (default 4 h).
