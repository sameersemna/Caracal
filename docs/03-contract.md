# 03 — Contract: REST API, MCP tools, Report schema

Names, fields, defaults and error codes in this file are binding. Contract tests compare against it.

## 0. Two front doors, one core (binding)

Caracal is **both** a REST API and an MCP server, as first-class equals. Both are thin adapters over the same
service layer (`db/repo.py`, `queue.py`, `analysis/*`); neither contains business logic the other lacks.
Every capability must exist on both sides:

| Capability | REST | MCP tool |
|---|---|---|
| submit audio (url / upload / inline) | `POST /v1/uploads` + `POST /v1/jobs` | `analyze_audio` (`url` / `upload_id` / `audio_base64`) |
| job status / progress | `GET /v1/jobs/{id}`, `/events` (SSE) | `get_job` (+ MCP progress notifications while `wait_seconds` blocks) |
| list jobs | `GET /v1/jobs` | `list_jobs` |
| report | `GET /v1/jobs/{id}/report` | `get_report` |
| transcript / subtitles | `/transcript`, `/subtitles/{lang}.{ext}` | `get_transcript` (`format` text/json/srt/vtt, `lang`) |
| translate | `POST /v1/jobs/{id}/translate` | `translate_job` |
| ask | `POST /v1/jobs/{id}/ask` | `ask_about_audio` |
| cancel / delete | `DELETE /v1/jobs/{id}` | `cancel_job` |
| full-text search | `GET /v1/search` | `search_transcripts` |
| speakers | `GET/POST/DELETE /v1/speakers…` | `list_speakers`, `enroll_speaker`, `delete_speaker` |
| voiceprint export (calibration) | `GET /v1/jobs/{id}/voiceprints` | — **REST only by design** (biometric data; not offered to LLM agents) |

A contract test (`tests/contract/test_parity.py`, from T1.6 on) asserts this table: every row's REST route and
MCP tool both exist, and for the same fake job they return equal data (MCP JSON == REST JSON). Adding a capability
to one side without the other fails the gate.

## 1. Common

- Auth: `Authorization: Bearer <CARACAL_API_TOKEN>` on everything except `GET /healthz`. Constant-time compare.
- JSON everywhere, UTF-8. Timestamps ISO-8601 UTC. Durations/times in **seconds (float)**.
- Errors: HTTP status + body `{"error": {"code": "<ErrorCode>", "message": "...", "details": {...}}}`.

## 2. REST (`/v1`)

| Method & path | Body / params | Returns |
|---|---|---|
| `GET /healthz` | — | `{"ok": true, "version": "...", "deps": {"postgres": "ok|down", "redis": ..., "whisper": ..., "ollama": ...}}` (deps cached 10 s) |
| `POST /v1/uploads` | multipart `file` | `201 {"upload_id", "filename", "size_bytes", "sha256"}` |
| `POST /v1/jobs` | `JobCreate` JSON | `202 {"job_id", "status": "queued", "links": {...}}` |
| `GET /v1/jobs` | `?status=&limit=20&cursor=` | `{"items": [JobSummary], "next_cursor"}` |
| `GET /v1/jobs/{id}` | — | `JobStatus` |
| `GET /v1/jobs/{id}/events` | — | SSE stream of `{"stage","status","pct","message","ts"}`; ends with `event: end` |
| `GET /v1/jobs/{id}/report` | `?format=json|md` | Report (§5) or Markdown |
| `GET /v1/jobs/{id}/transcript` | `?format=json|txt&speakers=true` | transcript |
| `GET /v1/jobs/{id}/subtitles/{lang}.{ext}` | ext ∈ `srt, vtt` | file (`lang` = `source` or ISO 639-1) |
| `POST /v1/jobs/{id}/translate` | `{"target_langs": ["de"]}` | `202` — runs translate+subtitles only |
| `POST /v1/jobs/{id}/ask` | `{"question": "..."}` | `Answer` (§5.9) |
| `DELETE /v1/jobs/{id}` | — | job `queued`/`running` → set cancel flag, `202 {"job_id","status":"cancelling"}` (the runner stops at the next stage boundary; call DELETE again later to remove it). Terminal job → delete DB rows (cascade) **and** `CARACAL_DATA_DIR/jobs/<id>/`, `204` |
| `GET /v1/speakers` | `?q=` | `{"items": [KnownSpeaker]}` |
| `POST /v1/speakers/enroll` | `{"job_id","speaker_label","name","notes"?}` | `201 KnownSpeaker` |
| `DELETE /v1/speakers/{id}` | — | `204` |
| `GET /v1/search` | `?q=&limit=20` (≤ 50) | `{"items": [{"job_id","title","seg_id","start","speaker","text","rank"}]}` — done jobs only |

### `JobCreate`
```json
{
  "source": {"url": "https://..."}  |  {"upload_id": "..."},
  "options": {
    "language": "auto",              // or ISO 639-1 hint
    "target_langs": null,            // null → CARACAL_DEFAULT_TARGET_LANGS when source differs; [] → none
    "depth": "standard",             // "quick" | "standard" | "deep"   (docs/04 §2)
    "diarize": "auto",               // "auto" | "on" | "off"
    "verify": "auto",                // "auto" | "on" | "off"
    "speaker_hints": [],             // e.g. ["host: Lex Fridman"]  free-text hints from the caller
    "context_hint": null             // free text, e.g. "Friday sermon in Cairo, 2023"
  }
}
```
### `JobStatus`
`{"job_id","status":"queued|running|done|failed|cancelled","created_at","updated_at","source":{"kind":"upload|direct|platform","url"?,"title"?},"progress":{"pct","stage","message"},"stages":[{"name","status":"pending|running|done|skipped|failed","reason"?,"started_at"?,"ended_at"?}],"error"?:{...},"links":{"report","transcript","subtitles":{"source.srt":url,...}}}`

## 3. MCP tools (FastMCP, streamable HTTP at `/mcp` and stdio)

**Every tool returns a JSON object** (MCP `structuredContent` must be an object — a list result is rejected by the SDK's
serialiser over real transports; measured). Tool descriptions are written for an LLM caller: say what the tool does, when to use it, and that analysis is
asynchronous. Inputs validated with pydantic; errors raised as MCP tool errors carrying the ErrorCode.

| Tool | Inputs | Output |
|---|---|---|
| `analyze_audio` | `url?`, `upload_id?`, `audio_base64?` (≤ 25 MB decoded), `filename?`, `options?` (JobCreate.options), `wait_seconds` (0–600, default 120) | If done within `wait_seconds`: `{"job_id","status":"done","report_summary": ReportSummary,"summary": {title,duration_s,languages,transcript_preview} (kept from M1),"links"}`; else `{"job_id","status","progress","hint":"call get_job / get_report later"}` |
| `get_job` | `job_id` | JobStatus |
| `get_report` | `job_id`, `sections?` (subset of top-level Report keys), `format` = `json|markdown` | Report (filtered) |
| `get_transcript` | `job_id`, `format` = `text|json|srt|vtt`, `lang` = `source` | string or JSON |
| `translate_job` | `job_id`, `target_langs` | `{"job_id","status","links"}` |
| `ask_about_audio` | `job_id`, `question` | Answer |
| `search_transcripts` | `query`, `limit` (≤50) | hits |
| `list_jobs` | `status?`, `limit` (≤50, default 20), `cursor?` | `{"items": [JobSummary], "next_cursor"}` |
| `cancel_job` | `job_id`, `delete_artifacts` (default false) | `{"job_id","status"}` |
| `list_speakers` | `query?` | `{"items": [KnownSpeaker]}` |
| `enroll_speaker` | `job_id`, `speaker_label`, `name`, `notes?` | KnownSpeaker |
| `delete_speaker` | `speaker_id` | `{"deleted": true, "voiceprints_removed": n}` |

`ReportSummary` = `{title, duration_s, languages, speakers:[{label,name|null,confidence}], summary, genre}`.

## 4. Error codes (`ErrorCode` enum, value = string)

| Code | HTTP | Meaning |
|---|---|---|
| `unauthorized` | 401 | |
| `invalid_input` | 422 | |
| `not_found` | 404 | |
| `source_blocked` | 400 | SSRF / private address / disallowed scheme |
| `source_unavailable` | 424 | download failed, private/removed video, geo-blocked |
| `too_large` | 413 | upload size or media duration over limit |
| `unsupported_media` | 415 | ffprobe finds no audio stream |
| `dependency_down` | 503 | whisper / ollama / db / redis unreachable (retryable) |
| `job_not_ready` | 409 | report requested before `done` |
| `rate_limited` | 429 | job creation over `CARACAL_RATE_LIMIT_PER_MIN` |
| `internal` | 500 | |

## 5. Report schema (`report/model.py`, pydantic, `schema_version: "1.0"`)

Every list item that asserts something about the world has `evidence: list[str]` (ids, docs/04 §9) and
`confidence: float` in [0,1]. Uncited items are removed by `validate_citations` and counted in `quality.dropped_uncited`.

```text
Report
  schema_version: "1.0"
  job_id, created_at, pipeline_version
  source:        {kind, url?, title?, uploader?, channel?, upload_date?, platform?, duration_s, description_excerpt?,
                  original_format, sha256}
  audio:         {duration_s, sample_rate, channels, loudness_lufs?, speech_ratio, music_detected: bool|null,
                  quality: "good|fair|poor", notes: [str]}
  languages:     [{code, name, share: float, spans: [{start,end}], confidence}]
  speakers:      [SpeakerIdentity]
  summary:       {one_line, short (≤120 words), detailed (≤600 words), evidence}
  context:       {genre, setting?, audience?, purpose?, tone: [str], recorded_when?, recorded_where?, evidence, confidence}
  chapters:      [{start, end, title, summary, evidence}]
  topics:        [{label, weight, evidence}]
  entities:      [{text, type: person|org|place|date|event|work|other, mentions: [seg ids], canonical?, url?, evidence}]
  key_quotes:    [{seg_id, speaker, text, start, why_notable}]
  claims:        [Claim]
  verification:  [Verification]
  translations:  [{lang, status: done|failed, links: {srt, vtt}}]
  transcript_ref:{segments: int, words: int, links: {json, txt, srt, vtt}}
  quality:       {asr_avg_logprob, low_confidence_segments: int, caption_agreement?: float, dropped_uncited: int,
                  stages: {name: status}, review_flags: [ReviewFlag]}
  evidence_index:{<id>: {kind, ref, excerpt}}       -- every id cited above resolves here

SpeakerIdentity
  label: "S1", talk_time_s, share, first_seen_s
  name: str|null, confidence, role?: "host|guest|interviewer|caller|narrator|preacher|...", 
  candidates: [{name, confidence, sources: ["self_intro"|"addressed_by_name"|"metadata"|"voiceprint"|"web"|"hint"],
                evidence}]
  known_speaker_id?: uuid, voiceprint_similarity?: float
  description: str|null      -- e.g. "adult male, calm, appears to be the channel owner"
  evidence

Claim         {id, text, speaker, seg_ids, kind: fact|opinion|prediction|quote, checkworthy: bool, evidence}
Verification  {target: "claim:<id>" | "speaker:<label>", verdict: supported|contradicted|unverified|mixed,
               confidence, rationale, sources: [{url, title, snippet, evidence_id}]}
ReviewFlag    {severity: info|warn|error, field, message}
Answer (§2 ask) {answer, confidence, citations: [{seg_id, start, text}], evidence}
KnownSpeaker  {id, name, notes?, samples: int, created_at, updated_at}
```
