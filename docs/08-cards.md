# 08 — Implementation Plan (task cards)

One card = one Kilo session = one branch `task/<ID>-<slug>` = one verdict. Cards run strictly in order unless
marked *parallel*. The boss refines each card into `agents/inbox/<NN>-<ID>.md` before issuing it; that inbox file
overrides this summary where they differ.

Card fields: **Read** (only these doc sections), **Files** (the only files the worker may create/edit),
**Deps** (new dependencies allowed), **Accept** (tests/conditions), **Boss live check** (done by the judge).

---------------------------------------------------------------------------------------------------
## M0 — Foundations  (demo: `caracal serve` + `caracal worker` run; a dummy job goes queued→done)

### T0.1 Scaffold, config, logging, errors, gate
- Read: 00, 01 §2–3, 02, 07 §1–2, 06 S5/S8/S12
- Files: `pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`, `README.md` (stub), `src/caracal/{__init__,cli,config,log,errors}.py`, `scripts/gate.sh`, `scripts/check_rules.py`, `tests/conftest.py`, `tests/unit/test_config.py`, `tests/unit/test_log.py`, `tests/unit/test_check_rules.py`
- Deps: all of docs/01 §2 runtime + dev (not the speakers extra yet)
- Accept: gate green; Settings validates required vars and token length; `redact()` masks the listed secrets; `check_rules.py` catches each forbidden pattern (tests use temp files); `caracal --help` lists the 5 subcommands (subcommands other than `doctor` may print "not yet available" to stderr and exit 2)

### T0.2 Database: tables, migrations, repositories
- Read: 05 §1, 01 §3 (db/*)
- Files: `alembic.ini`, `src/caracal/migrations/**`, `src/caracal/db/{__init__,engine,tables,repo}.py`, `tests/fakes/repo.py`, `tests/unit/test_repo_fake.py`, `tests/live/test_db.py`
- Accept: `alembic upgrade head` creates every table except `voiceprints` (T2.3); repo protocol + in-memory fake share one interface; live test (boss) migrates a scratch schema on latitude and round-trips a job

### T0.3 Queue, worker skeleton, runner with checkpoints, jobs REST
- Read: 01 §1, §4–5, 03 §1–2 (jobs, healthz), 03 §4, 05 §2–3, 06 S1/S2/S10/S11
- Files: `src/caracal/{app,queue,worker,storage}.py`, `src/caracal/api/**`, `src/caracal/pipeline/{__init__,stages,runner}.py`, tests under `tests/unit/` and `tests/contract/`
- Accept: Streams queue per docs/05 §2 (enqueue, consumer group, heartbeat XCLAIM, XAUTOCLAIM recovery, max-deliveries → failed, ack+del), every key via `queue.key()` (test with a recording fake asserts every touched key starts with the configured prefix, using a non-default prefix like `t3st:`); runner executes a list of stages, writes checkpoints atomically, resumes skipping done stages, applies the failure policy, honours cancel flag; REST `POST/GET /v1/jobs`, `/events` (SSE), `/healthz`, auth 401s, Origin check; stages are stubs that return `skipped: reason="not implemented in M0"` (allowed string, not a TODO)
- Boss live check: real Redis+Postgres, a job goes queued→running→done with SSE events; `kill -9` the worker mid-job → another worker reclaims and finishes it; afterwards `SCAN` shows no Caracal key outside `REDIS_PREFIX`

---------------------------------------------------------------------------------------------------
## M1 — Hear  (demo: YouTube link → transcript + source SRT/VTT via REST and MCP)

### T1.1 Uploads + direct URLs + SSRF
- Read: 04 §3 (classify, direct), 06 S3/S6/S7, 03 §2 uploads
- Files: `src/caracal/ingest/{__init__,ssrf,upload,direct,router}.py`, `tests/fakes/resolver.py`, tests
### T1.2 yt-dlp ingest (metadata, audio, platform captions)
- Read: 04 §3, 06 S4
- Files: `src/caracal/ingest/ytdlp.py`, tests (yt-dlp mocked at the `YoutubeDL` class boundary), `tests/live/test_ytdlp.py`
- Boss live check: a public YouTube video + one non-YouTube (e.g. SoundCloud/Vimeo) download and metadata
### T1.3 ffmpeg normalize, loudness, silences, chunk planning
- Read: 04 §3 (normalize, chunk), 06 S5
- Files: `src/caracal/audio/{__init__,ffmpeg,chunking}.py`, tests (real ffmpeg on generated tones is allowed in unit tests — it is local)
### T1.4 Whisper client, merge, language spans, caption agreement
- Read: 00 (whisper facts), 04 §4
- Files: `src/caracal/asr/{__init__,whisper,merge,langid}.py`, `tests/fakes/whisper.py`, tests
### T1.5 Subtitles
- Read: 04 §10 (segmenter, formats)
- Files: `src/caracal/subtitles/{__init__,segmenter,formats}.py`, tests
### T1.6 Wire M1 stages + transcript/subtitle endpoints + MCP server (`analyze_audio`, `get_job`, `get_transcript`)
- Read: 03 §0, §2–3, 01 §4
- Files: `src/caracal/pipeline/stages.py`, `src/caracal/mcp/**`, `src/caracal/api/routes.py`, `src/caracal/cli.py`, contract tests incl. `tests/contract/test_parity.py`
- Accept: parity test (docs/03 §0) covers every capability that exists after M1; rows for later milestones are listed in the test as `pending=<card>` and the test fails if a pending row's card is already merged
- Boss live check: end-to-end on jfk.wav upload and on a YouTube link; MCP via a real MCP client

---------------------------------------------------------------------------------------------------
## M2 — Understand  (demo: two-speaker interview → named speakers, summary, chapters, entities, claims)

### T2.1 LLM client (tool-call protocol + repair + fallback), prompt loader
- Read: 00 (LLM facts), 04 §1, 06 S9
- Files: `src/caracal/llm/**`, `tests/fakes/ollama.py`, tests covering every FakeOllama failure mode
### T2.2 Diarization + alignment (speakers extra)
- Read: 04 §5, 02 (HF/diarization vars)
- Files: `src/caracal/speakers/{__init__,diarize,align}.py`, `pyproject.toml` (extra), `tests/fakes/diarizer.py`, tests, `tests/live/test_diarize.py`
- Boss pre-step: owner provides `HF_TOKEN` + accepts the community-1 license; boss verifies CUDA torch on GB10
### T2.3 Voiceprints + enroll + known speakers API
- Read: 04 §5, 05 §1 voiceprints, 03 speakers endpoints, 06 (biometric note)
- Files: `src/caracal/speakers/voiceprint.py`, new migration, repo additions, routes/MCP tools, tests
### T2.4 Report model + evidence index/validator + markdown renderer
- Read: 03 §5, 04 §9
- Files: `src/caracal/report/**`, tests
### T2.5 Planner + content analysis (map-reduce)
- Read: 04 §2, §6
- Files: `src/caracal/analysis/{__init__,planner,content}.py`, prompts `plan.md content.md content_window.md content_reduce.md`, tests
### T2.6 Identity resolution
- Read: 04 §7
- Files: `src/caracal/analysis/identity.py`, prompt `identity_mentions.md`, tests (fusion math table-driven)
### T2.7 Wire M2 + `get_report` + assemble
- Boss live check: two-speaker fixture; injected-instruction transcript (S9)

---------------------------------------------------------------------------------------------------
## M3 — Verify & translate  (demo: German news clip → verified claims, en+ar subtitles)

### T3.1 Search providers: Brave, Exa, router, cache, rate limit  *(parallel with T3.3)*
- Boss pre-check: fetch current Brave & Exa API docs with context7/web and paste the exact request/response shapes into the inbox card
### T3.2 Cross-verification stage
### T3.3 Translation + multi-language subtitles + `translate_job`  *(parallel with T3.1)*
### T3.4 Self-review + review flags
### T3.5 Q&A + full-text search + `ask_about_audio`, `search_transcripts`

---------------------------------------------------------------------------------------------------
## M4 — Operate  (demo: running as systemd services; eval report)

### T4.1 `caracal doctor`, retention cron, dedup, metrics endpoint, systemd units (`caracal-api.service`, `caracal-worker.service`), README
### T4.2 Eval harness: WER vs reference transcripts, diarization error rate on the two-speaker fixture, identity accuracy on a small labelled set, voiceprint threshold calibration → `docs/findings/eval-<date>.md`
### T4.3 Hardening sweep: fuzz inputs (URLs, filenames, base64, malformed LLM output), concurrency (2 jobs + cancel), resume after `kill -9` of the worker mid-transcribe
