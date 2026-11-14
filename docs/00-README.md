# Caracal — Audio Intelligence API + MCP Server

Caracal takes an audio/video file **or a link** (YouTube or any site yt-dlp supports) and returns a
complete, evidence-backed **audio intelligence report**, not just a transcript:

- transcript with word timestamps, language(s) incl. code-switching
- subtitles / captions (SRT, VTT) in the source language and any number of translations
- **who is speaking** — speaker separation (diarization), voiceprint matching against known speakers, names from
  self-introductions / metadata / web, each with a confidence and the evidence behind it
- context: genre (interview, lecture, sermon, news, podcast …), setting, audience, purpose, tone
- details: summary, chapters, topics, named entities, key quotes, claims
- **cross-verification**: claims and identity hypotheses checked against the web (Brave + Exa) and against the
  platform's own captions/metadata, each marked supported / contradicted / unverified with citations
- a self-review pass that flags contradictions and low-confidence fields

It runs autonomously: one request in, one finished report out. The pipeline decides by itself which stages
are worth running (docs/04 §2).

## Doc map (source of truth — the worker never edits `docs/`)

| Doc | Contents |
|---|---|
| 01-architecture.md | components, repository layout, pipeline DAG, module signatures |
| 02-configuration.md | every environment variable, defaults |
| 03-contract.md | REST API, MCP tools, the Report JSON schema |
| 04-intelligence.md | every pipeline stage in detail, LLM protocol, evidence rules, prompts |
| 05-data.md | Postgres schema, Redis keys, artifact directory layout |
| 06-security.md | auth, SSRF, file limits, subprocess rules, secrets |
| 07-testing.md | the gate, test layers, fakes, fixtures, live tests |
| 08-cards.md | the implementation plan: milestones and task cards |
| 09-harness.md | how Claude (boss/judge) and Kilo (worker) cooperate |
| findings/ | errata the worker records; Claude resolves them into the docs |

## Verified environment facts (measured 2026-11-14 on `promaxgb10-6116`)

These are measurements, not assumptions. Code must match them.

| Fact | Value |
|---|---|
| Host | `promaxgb10-6116`, aarch64, NVIDIA GB10 (unified memory, ~121 GB RAM), 20 cores, Ubuntu, Python 3.12.3, uv, ffmpeg, Node 24 |
| Whisper | **whisper.cpp server** at `http://localhost:8768`, model `ggml-large-v3`, started **without** `--convert`, `--vad`, `--diarize`. Default language flag is `en` → **always send `language` explicitly** (`auto` or ISO code). Endpoint `POST /inference` multipart (`file`, `response_format`, `language`, `temperature`, `translate`). Processes one request at a time. ~0.9 s for 11 s of audio |
| Whisper `verbose_json` | top-level `task, language ("english" = full name!), duration, text, detected_language, detected_language_probability, language_probabilities ({"en":0.93,...} keyed by ISO code), segments[]`; segment `id,text,start,end,tokens,words[],temperature,avg_logprob,no_speech_prob`; word `word,start,end,t_dtw,probability`. Input must be WAV 16 kHz mono PCM (we convert with ffmpeg). **When `language` is forced, `language` echoes the request but `detected_language`, `detected_language_probability` and `language_probabilities` still hold the real detection** → always use those for language ID. Digital silence → 0 segments, detection prob ≈ 0.3 (treat < 0.5 as unknown) |
| LLM | Primary: **Ollama cloud direct**, `https://ollama.com/api/chat`, bearer `OLLAMA_CLOUD_API_KEY`, model **`gemma4:31b`** (verified). Without a key the same model is reachable through the local daemon as `gemma4:31b-cloud` (~0.2–0.6 s for short prompts). Local fallback `gemma4:31b` is installed on this host |
| LLM JSON | **`format: <json schema>` is IGNORED by gemma4 on the cloud** — verified both direct (ollama.com/api) and via the local proxy (native and OpenAI-compatible endpoints all return markdown). `format: "json"` returns JSON wrapped in a ```json fence. **Tool calling works** and returns structured `arguments` (a dict). → docs/04 §1 mandates tool-call mode with a prompt+extract+validate fallback |
| Datastores | Host `latitude` (LAN). **Postgres 18.6**, db `caracal`, pgvector 0.8.1 available, user has CREATE. **Redis 8.0.5**, DB 0 **shared with other apps (~40 k keys)** → every key uses `REDIS_PREFIX` (docs/05 §2). Credentials only in `.env` (mode 600). `redis-cli -u` fails on the `redis://:pw@` form; redis-py works |
| HF / pyannote | `HF_TOKEN` valid; `pyannote/speaker-diarization-community-1` license accepted (HTTP 200, 2026-11-14) |
| yt-dlp | not installed system-wide → it is a Python dependency of this project (`yt-dlp` on PyPI) |
| Ports in use nearby | 11401–11403, 11410, 11436 taken. **Caracal uses 11404** |

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Interfaces | **REST API and MCP server, equal citizens** | Owner requirement. Parity is binding and tested (docs/03 §0) |
| Language | Python 3.12 + uv | pyannote, yt-dlp, torch are native Python |
| Search | Brave + Exa, pluggable with fallback | owner choice; both keys verified 2026-11-14 |
| Speakers | pyannote community-1 + voiceprints | owner choice; HF token valid, model license must be accepted by the owner |
| Vector store | **pgvector in the latitude Postgres, not Qdrant** | Voiceprints are small (≪ 1 M vectors) and must be deleted atomically with their `known_speakers` row (biometric data); one transactional store, joins with jobs. The Qdrant on this host belongs to `chro-as-a-service` and is only on that project's Docker network (no host port). Revisit only for a future semantic-search index over millions of transcript segments — behind the same repo interface |

## Roles

- **Owner** (Sameer): credentials, HF token + pyannote license acceptance, systemd install, final say.
- **Boss / Judge** (Claude Code): writes and refines the docs and cards, launches Kilo, reviews every round
  (gate, diff, non-vacuity, live checks), writes verdicts, resolves errata.
- **Worker** (Kilo, `caracal-worker` agent): implements exactly one card per session, never edits `docs/`.

## Glossary

- **Job** — one analysis request; has stages; produces a Report.
- **Stage** — a pipeline step with a checkpointed output file (docs/01 §4).
- **Evidence** — an addressable fact the report may cite: `seg:<n>`, `meta:<field>`, `web:<n>`, `vp:<speaker>`,
  `caption:<n>`. Every claim in a report cites ≥1 evidence id (docs/04 §9).
- **Speaker label** — diarization label `S1, S2 …` (stable within a job).
- **Voiceprint** — speaker embedding stored in pgvector; links labels across jobs to a known person.
