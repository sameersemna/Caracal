# Caracal

Caracal turns an audio/video file **or a link** (YouTube or any site yt-dlp supports) into a complete,
evidence-backed **audio intelligence report** — not just a transcript:

- transcript with word timestamps and language(s), including code-switching
- subtitles/captions (SRT, VTT) in the source language and any number of translations
- **who is speaking** — diarization, voiceprint matching against known speakers and names from
  self-introductions, metadata and the web, each with a confidence and the evidence behind it
- context: genre, setting, audience, purpose, tone; summary, chapters, topics, entities, quotes, claims
- **cross-verification** of claims and identity hypotheses against Brave/Exa and the platform's own
  captions, each marked supported / contradicted / unverified with citations
- a self-review pass that flags contradictions and low-confidence fields

The REST API and the MCP server are equal citizens over one service layer: every capability exists on
both sides and returns the same data.

Runs as three processes sharing one package:

| Process | Command | Role |
|---|---|---|
| API | `caracal serve` | FastAPI `/v1`, MCP streamable HTTP at `/mcp`, Prometheus `/metrics` |
| Worker | `caracal worker` | Redis Streams consumer, pipeline, daily retention task |
| MCP stdio | `caracal mcp-stdio` | the same MCP tools over stdin/stdout |

## Install

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), ffmpeg/ffprobe, Node 24 (for yt-dlp's
YouTube challenge) and a whisper.cpp server. On the reference host (`promaxgb10-6116`, aarch64,
NVIDIA GB10, ~121 GB unified memory) these are installed already.

```bash
git clone <this repo> caracal
cd caracal
uv sync --extra speakers
```

GPU notes (measured 2026-11-14):

- torch/torchaudio **2.11.0+cu130** come from the `pytorch-cu130` index pinned in `pyproject.toml`.
  torchaudio stops at 2.11.0, so torch is pinned to it; a PyPI torch (CUDA 13.2) is incompatible.
- GB10 is compute capability (12, 1). `uv sync` (without `--extra speakers`) skips pyannote/torch and
  the diarize stage reports `skipped: reason=unavailable`; every other stage still runs.
- pyannote `speaker-diarization-community-1` needs `HF_TOKEN` and a one-time licence acceptance on
  Hugging Face.
- whisper.cpp is a separate server (`WHISPER_URL`, default `http://localhost:8768`) started without
  `--convert`, `--vad` or `--diarize`; Caracal always sends `language` explicitly.

## Configuration

Copy `.env.example` to `.env` (mode 600) and fill in the placeholders. Every variable is documented
in `docs/02-configuration.md`; the important ones:

```ini
CARACAL_API_TOKEN=<at least 32 characters>       # bearer auth for REST and MCP HTTP
DATABASE_URL=postgresql+psycopg://user:pw@latitude:5432/caracal
REDIS_URL=redis://:pw@latitude:6379/0             # DB 0 is shared; every key uses REDIS_PREFIX
CARACAL_DATA_DIR=./data
OLLAMA_CLOUD_API_KEY=<key>                        # optional: local gemma4:31b-cloud proxy otherwise
LLM_ALLOWED_MODELS=gemma4:31b-cloud,local:gemma4:31b   # every model a route may use
LLM_ROUTE_DEFAULT=gemma4:31b-cloud,local:gemma4:31b    # ordered chain; first = preferred
HF_TOKEN=<hf_...>                                 # pyannote model download
BRAVE_API_KEY=<key>
EXA_API_KEY=<key>
```

Every LLM call names a pipeline task (`plan`, `content`, `identity`, `verify`, `translate`,
`review`, `qa`) and runs on that task's `LLM_ROUTE_*` chain; a task without its own route uses
`LLM_ROUTE_DEFAULT`. A ref is `cloud:<name>` (ollama.com with `OLLAMA_CLOUD_API_KEY`),
`local:<name>`, or a bare `<name>` (the local daemon, which proxies `-cloud` tags). Every model a
route uses must be listed in `LLM_ALLOWED_MODELS`, and the chain is tried in order until a model
answers — the next ref takes over after two failed attempts, an auth rejection, a missing model,
or unusable output. `caracal doctor` prints one `/api/show` line per allowed model and fails when a
route has no working model. With no `LLM_ROUTE_*` and no `LLM_ALLOWED_MODELS` the chain is exactly
the pre-routing behaviour (cloud or local proxy, then `LLM_FALLBACK_MODEL`).

Only `src/caracal/config.py` reads the environment. Secrets are `SecretStr` and are redacted from logs.

## First run

```bash
uv run caracal doctor          # one line per check; exit 1 if a required check fails
uv run caracal doctor --json   # one JSON object per check
uv run caracal doctor --deep   # additionally runs one real Brave and Exa query
uv run caracal migrate         # alembic upgrade head (creates the pgvector extension)
uv run caracal serve           # API on CARACAL_HOST:CARACAL_PORT (default 127.0.0.1:11404)
uv run caracal worker          # pipeline worker (run it as a systemd service, see deploy/)
```

Required doctor checks: config, Postgres (+ pgvector and alembic at head), Redis
(write/read/delete under `REDIS_PREFIX`), whisper, Ollama (cloud if the key is set, and the local
fallback model), ffmpeg/ffprobe with `-format_whitelist`. yt-dlp/node, pyannote/CUDA/HF, search keys
and free disk are reported but only warn.

`caracal retention [--dry-run]` deletes `audio.wav`, `source/` and `chunks/` for jobs finished more
than `CARACAL_RETENTION_DAYS` ago, plus unconsumed uploads older than 24 h. Transcripts, subtitles,
reports and stage checkpoints are never deleted. The worker runs the same pass daily under a Redis
singleton lock.

## REST quick start

```bash
TOK=$CARACAL_API_TOKEN
B=http://127.0.0.1:11404
A="Authorization: Bearer $TOK"
```

Upload a file, then create a job from the upload (or pass `{"source":{"url":"https://..."}}`):

```bash
curl -s -H "$A" -F file=@talk.mp3 $B/v1/uploads
# {"upload_id":"...","filename":"talk.mp3","size_bytes":12345,"sha256":"..."}

curl -s -H "$A" -H 'content-type: application/json' \
  -d '{"source":{"upload_id":"<upload_id>"},"options":{"depth":"standard"}}' \
  $B/v1/jobs
# {"job_id":"...","status":"queued",...}
```

Fetch status, stream progress events, then read the artifacts:

```bash
J=<job_id>
curl -s -H "$A" $B/v1/jobs/$J
curl -sN -H "$A" $B/v1/jobs/$J/events                  # SSE, ends with `event: end`
curl -s -H "$A" "$B/v1/jobs/$J/report"                 # JSON report
curl -s -H "$A" "$B/v1/jobs/$J/report?format=md"       # Markdown report
curl -s -H "$A" "$B/v1/jobs/$J/transcript?format=json"
curl -s -H "$A" "$B/v1/jobs/$J/transcript?format=txt&speakers=true"
curl -s -H "$A" -o source.srt "$B/v1/jobs/$J/subtitles/source.srt"
curl -s -H "$A" -o de.vtt    "$B/v1/jobs/$J/subtitles/de.vtt"
```

More capabilities:

```bash
# translate the transcript + subtitles (runs only the translate/subtitle stages)
curl -s -H "$A" -H 'content-type: application/json' -d '{"target_langs":["de"]}' \
  $B/v1/jobs/$J/translate

# ask a question about the recording
curl -s -H "$A" -H 'content-type: application/json' -d '{"question":"What did they claim about pricing?"}' \
  $B/v1/jobs/$J/ask

# full-text search across finished transcripts
curl -s -H "$A" "$B/v1/search?q=pricing&limit=10"

# speakers (voiceprints are biometric data)
curl -s -H "$A" $B/v1/speakers
curl -s -H "$A" -H 'content-type: application/json' \
  -d '{"job_id":"'$J'","speaker_label":"S1","name":"Jane Doe"}' $B/v1/speakers/enroll
curl -s -X DELETE -H "$A" $B/v1/speakers/<speaker_id>
```

`GET /metrics` (bearer-protected) exposes Prometheus counters shared by the API and worker:
`caracal_jobs_total`, `caracal_stage_seconds_sum/_count`, `caracal_stage_failures_total`,
`caracal_llm_calls_total`, `caracal_search_queries_total`, `caracal_search_cost_usd_total`,
`caracal_whisper_audio_seconds_total`.

## MCP

### Streamable HTTP

The MCP endpoint is mounted at `/mcp/` behind the same bearer token. Point a client at
`http://<host>:11404/mcp/` with `Authorization: Bearer $CARACAL_API_TOKEN`. If the host is not
localhost, add it to `CARACAL_PUBLIC_BASE_URL` or `CARACAL_ALLOWED_HOSTS` (the SDK rejects a foreign
`Host` header with 421).

Claude Code (`.mcp.json` in the project):

```json
{
  "mcpServers": {
    "caracal": {
      "type": "http",
      "url": "http://127.0.0.1:11404/mcp/",
      "headers": { "Authorization": "Bearer ${CARACAL_API_TOKEN}" }
    }
  }
}
```

Claude Desktop (`claude_desktop_config.json`) for stdio:

```json
{
  "mcpServers": {
    "caracal": {
      "command": "/path/to/caracal/.venv/bin/caracal",
      "args": ["mcp-stdio"],
      "env": {
        "DATABASE_URL": "postgresql+psycopg://...",
        "REDIS_URL": "redis://:...@latitude:6379/0",
        "CARACAL_API_TOKEN": "<at least 32 characters>",
        "CARACAL_DATA_DIR": "/path/to/caracal/data"
      }
    }
  }
}
```

`caracal mcp-stdio` reads the same `.env` as the other entry points; logs go to stderr and stdout is
reserved for the MCP framing.

### Tools

| Tool | Purpose |
|---|---|
| `analyze_audio` | submit a URL, an upload id or inline base64 audio; waits up to `wait_seconds` |
| `get_job` | status, progress and stages (with MCP progress notifications while it waits) |
| `list_jobs` | recent jobs, filtered by status |
| `get_report` | the report as JSON, optionally filtered to sections, or Markdown |
| `get_transcript` | transcript as text/json/SRT/VTT, source or translated language |
| `translate_job` | translate a finished job into more languages |
| `ask_about_audio` | Q&A over one recording with citations |
| `search_transcripts` | full-text search across finished jobs |
| `cancel_job` | cancel a running job or delete a terminal one (optionally with artifacts) |
| `list_speakers` / `enroll_speaker` / `delete_speaker` | manage known speakers and voiceprints |

## Privacy

Voiceprints are **biometric data**. Enrollment happens only through an explicit API/tool call, and
`DELETE /v1/speakers/{id}` (or `delete_speaker`) removes every embedding for that person. The
database stores only embeddings, not raw voice samples; media files are deleted by the retention
policy and transcripts/reports stay until the job is deleted. Obtain consent before enrolling
someone. Transcripts are never logged at INFO.

For threshold calibration, `GET /v1/jobs/{id}/voiceprints` (bearer-protected) returns the diarize
stage's raw 256-dimension embeddings as `{"items": [{"label": "S1", "embedding": [...]}]}`. This is
the one place raw embeddings leave the process, so it exists **only** as a REST endpoint — there is
deliberately no MCP tool for biometric embeddings.

## Known limitations

- Voice conversion and good impressionists are split into several speakers: diarization clusters by
  voice, not by identity.
- Word timings from whisper.cpp are model timestamps; without DTW they can drift within a segment.
- Some sites block datacenter/VPN addresses or require cookies (`YTDLP_COOKIES_FILE` exists for
  age-gated content the owner is entitled to).
- Music-only audio yields little or no speech; the planner skips speaker and content stages and the
  report says so.
- Cross-verification is only as good as the search providers' indexes; unverifiable claims are
  reported as `unverified`, never silently dropped.

See `docs/` for the binding specification, `deploy/` for systemd units and `scripts/live-suite.sh`
for the end-to-end live test runner.
