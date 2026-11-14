# 04 — The Intelligence Pipeline

Principle: **deterministic code does everything it can; the LLM only interprets, and must cite.**
Every LLM output is validated against a pydantic model; anything that cannot be traced to evidence is dropped.

## 1. LLM protocol (`llm/client.py`) — mandatory, because of the measured gemma4 behaviour (docs/00)

```python
class LLMClient:
    async def call_tool(self, *, prompt: str, system: str, model_cls: type[T], tool_name: str,
                        temperature: float = 0.1, max_tokens: int = 4096) -> T
    async def chat(self, *, prompt: str, system: str, temperature: float = 0.3, max_tokens: int = 2048) -> str
```

**Measured 2026-11-14 (cloud `gemma4:31b`, 5 + 12 trials):** with pydantic's default schema (`$defs`/`$ref`) the model invents field
names and upper-cases enums; with the schema **inlined** it uses the right fields and cites evidence, but still emits enum *synonyms*
(`location` for `place`, `PERSON`) — **0 / 17 raw outputs validated**. Inlining + enum lists in field descriptions + a **coercion pass**
(case-insensitive enums + synonym map) → **12 / 12 valid on the first attempt** (2.2 s for 12 parallel calls). Local `gemma4:31b` fallback:
same errors, 42.8 s cold start. Fixtures: `tests/fixtures/ollama/`.

**Measured on a realistic task** (content analysis of 2 min of the TED transcript): a property named **`type`** is omitted by the model
(0/6 valid; renamed → 6/6). Evidence ids are copied with the speaker label (`"seg:2 S1"`, 0/12 clean). Rules: LLM-facing models never use
JSON-Schema keyword names (`type`, `items`, `properties`, `enum`, `format`, `required`, `default`, `const`) as field names — map them with
pydantic aliases; evidence lists are normalised by extracting `(seg|meta|web|vp|caption|hint):[A-Za-z0-9_.-]+` tokens; transcript lines are
rendered `[seg:12] S2 03:14.2 | text` (id alone inside the brackets).

**Measured on the full content schema** (7 sections): tool calls validate but are never complete (0/6 — whole sections omitted) and
sometimes come back **empty** (Ollama drops a malformed tool call). **JSON mode** (`format:"json"`, schema in the system prompt, "fill every
field") → 5/6 complete, 0/87 unknown ids. Nullable float fields (`start`/`end`) trigger empty replies — LLM-facing models reference segment ids
instead of times. Rule: big multi-section outputs use json mode + a completeness re-ask; small schemas may use tool mode, with an automatic
json-mode retry on an empty reply.

`call_tool` algorithm:
0. **Schema preparation** (cached per model class): `model_json_schema()` → inline every `$ref` from `$defs`, drop `title`, append
   `Must be exactly one of: a, b, c.` to the description of every `enum`/`Literal` field.
1. POST `<root>/api/chat` of the primary endpoint (docs/02: cloud direct with bearer key if `OLLAMA_CLOUD_API_KEY`
   is set, else the local daemon with `LLM_LOCAL_PROXY_MODEL`) with `stream:false`, `keep_alive`, `options:{temperature, num_predict}`, and one tool
   `{"type":"function","function":{"name":tool_name,"description":..., "parameters": model_cls.model_json_schema()}}`.
   The system prompt ends with: `You MUST answer by calling the tool <tool_name> exactly once.`
2. If `message.tool_calls[0].function.arguments` exists (a dict, or a JSON string → parse): run the **coercion pass** guided by the
   prepared schema — enums matched case-insensitively and through a synonym map (global defaults + per-call `synonyms`), numeric strings →
   numbers, `"null"`/`""` → `None` for optional fields, a scalar → `[scalar]` where an array is expected, numbers clamped to
   `minimum`/`maximum` — then `model_cls.model_validate`. Success → return (record `coerced: bool` in call metadata).
3. Otherwise (no tool call / validation error): **repair attempt** — append the assistant message and a `role: "tool"` message with the
   first 5 validation errors (`loc`, `msg`, no input values), asking to call the tool again; if the reply has no tool call, take its content parse with `llm/jsonx.extract_json` (strips ``` fences,
   finds the outermost `{...}`), validate. Max 2 repair attempts.
4. Transport errors / 5xx / timeout → retry with backoff (1 s, 4 s). Two consecutive primary failures → use
   `LLM_FALLBACK_MODEL` on `OLLAMA_LOCAL_URL` for this call and log `llm.fallback`. HTTP 401/403 from the cloud is
   not retried: go straight to the fallback and log `llm.auth_failed` once per process.
5. Still failing → raise `CaracalError(dependency_down | internal, retryable=True)`.
6. Never log prompt bodies at INFO (they contain transcripts). DEBUG only, truncated to 500 chars.

Prompts live in `llm/prompts/<stage>.md` with `{placeholders}`; loaded by `load_prompt(name, **vars)`.
Every analysis prompt contains this block verbatim:
> Only state what the evidence supports. Cite evidence ids (like `seg:12`, `meta:title`, `web:3`) for every item.
> If you are not sure, say so with a low confidence instead of guessing. Never invent names, dates or sources.

## 2. Planner (`analysis/planner.py`) — the autonomy layer

Runs after `langid` (+ ingest metadata). **Rule-based first**, LLM only to fill `content_profile`.

Inputs: duration, speech_ratio, music_detected, languages, platform metadata, options. Output `StagePlan`:
`{run: {stage: bool}, reasons: {stage: str}, content_profile: {likely_genre, expected_speakers: int|null,
needs_identity: bool, checkworthy_domain: bool}, budgets: {search_queries: int, llm_calls: int}}`.

Rules (in order; each produces a reason string):
- **music / non-lexical** (measured: an instrumental track gives `🎵🎵🎵` segments and whisper labels it `nn`): if ≥ 70 % of speech time is in
  segments whose text has no letters/digits after removing ♪ 🎵 🎶 `[Music]`-style tags → `music_detected: true`, every language span → `und`,
  skip diarize/identity/verify/translate; the report describes it as music.
- `speech_ratio < 0.05` → skip diarize/identity/verify/translate; content = "music/ambient" description only.
- `options.diarize == "off"` or diarizer unavailable → skip diarize/align/voiceprint.
- `diarize == "auto"` and duration < 20 s → diarize anyway (cheap) but identity uses single-speaker logic.
- `verify`: `off` → skip; `on` → run; `auto` → run if `depth != "quick"` and (content_profile.checkworthy_domain
  or any speaker has an unconfirmed name candidate) and ≥1 search provider is configured.
- `translate`: run for `target_langs` (resolved per docs/03 §2) that differ from the dominant language.
- `depth`: `quick` → summary + languages + subtitles only, search budget 0. `standard` → all, budget 12.
  `deep` → all, budget `SEARCH_MAX_QUERIES_PER_JOB`, chapters at finer granularity, verify up to 15 claims.
The LLM call (prompt `plan.md`) receives title, description excerpt, first ~3 min of transcript, and returns only
`content_profile`. It cannot change `run` flags directly.

## 3. Ingest, normalize, chunk

- `classify_source(url)`: http(s) URL whose path ends in a known media extension (`mp3 m4a aac wav flac ogg opus
  webm mp4 mkv mov`) or whose HEAD `Content-Type` is `audio/*|video/*` → `direct`; anything else → `platform`
  (yt-dlp). Unknown to yt-dlp → `source_unavailable`.
- yt-dlp (Python API, `YoutubeDL`, never shell): `noplaylist=True`, `format="bestaudio/best"`,
  `max_filesize`, `socket_timeout=30`, `retries=3`, no post-processors that run arbitrary commands. Save the
  **info dict subset** to `stages/ingest.json`: `id, title, uploader, channel, channel_url, upload_date, duration,
  description, tags, categories, chapters, language, view_count, webpage_url, extractor`. Also download available
  **platform captions** (manual preferred over automatic) in `vtt` for the original language → `captions/<lang>.vtt`.
  Playlists → `invalid_input` ("one item per job").
- `normalize`: `ffprobe` (JSON) → reject no-audio (`unsupported_media`) and over-duration (`too_large`).
  `ffmpeg -i src -vn -ac 1 -ar 16000 -c:a pcm_s16le audio.wav`. Also `ebur128` loudness and
  `silencedetect=noise=-35dB:d=0.4` → `speech_ratio` = 1 − silence/duration.
- `chunk`: target 600 s, max 900 s, cut at the silence closest to each target boundary, 1.0 s overlap.
  Files ≤ 900 s → one chunk.

## 4. Transcribe and language

- One whisper call per chunk under the Redis lock: `response_format=verbose_json`, `language=<hint or auto>`,
  `temperature=0`. **Always** send `language`.
- Map whisper's full language name ("english") to ISO code using `language_probabilities` argmax (keyed by code).
- `merge_chunks`: shift timestamps by chunk start; in each overlap keep words from the chunk whose centre is nearer;
  renumber segment ids from 0. Keep per-segment `avg_logprob`, `no_speech_prob`; drop segments with
  `no_speech_prob > 0.8 and avg_logprob < -1.0` (hallucinated silence text) and record the count.
- Hallucination filter: drop segments whose normalised text repeats the previous segment ≥3 times in a row.
- `language_spans`: per chunk detected language + per-segment re-check: when a chunk's
  `detected_language_probability < 0.6` and duration > 60 s, re-transcribe 30 s windows to find spans
  (max 6 extra calls per job). Output `languages[]` with shares.
- Platform caption cross-check: if `captions/<lang>.vtt` exists, compute word-level agreement
  (1 − WER on normalised text over the overlapping time range) → `quality.caption_agreement`. Large disagreement
  (< 0.6) adds a ReviewFlag; it does not replace our transcript.

## 5. Speakers

- `diarize`: pyannote pipeline loaded once per worker process (lazy, `HF_TOKEN`), on `DIARIZATION_DEVICE`.
  Input `audio.wav`. Output turns `[{start,end,label}]` with labels renamed `S1..Sn` in order of first appearance,
  plus one embedding per label — pyannote's own `speaker_embeddings` (256-d, rows in `labels()` order), L2-normalised. Measured: same
  speaker 0.94–0.99, different speakers 0.29–0.31 (cosine). Pass `num_speakers` only if the
  caller hinted it.
- `align`: each word gets the speaker whose turn overlaps its midpoint most; segments are split at speaker changes.
- `voiceprint`: cosine search in `voiceprints` (pgvector) per label → `VoiceMatch{known_speaker_id, name,
  similarity}`; ≥ MATCH_THRESHOLD is a strong source, ≥ CANDIDATE_THRESHOLD a weak one.
- `enroll_speaker` stores the label's embedding with the given name (and appends to an existing KnownSpeaker
  with the same normalised name). Enrollment is the only way a name is attached to a voiceprint.
  **Never auto-enroll.**

## 6. Content analysis (`analysis/content.py`)

Long transcripts: if the speaker-tagged transcript text exceeds `LLM_CONTEXT_CHARS`, split into windows at segment
boundaries, run `content_window.md` per window (map) producing partial chapters/entities/claims/quotes, then
`content_reduce.md` (reduce) to merge and dedupe. Otherwise one call to `content.md`.

The transcript is passed as lines `[seg:12] S2 03:14.2 | text` so the model can cite. Metadata as
`[meta:title] ...`, `[meta:description] ...`, `[meta:channel] ...`. Caller hints as `[hint:0] ...`.

Outputs (validated pydantic): summary, context, chapters, topics, entities, key_quotes, claims (with `checkworthy`).
Deterministic post-checks: chapter times within duration and non-overlapping (fix by clamping/sorting); quote text
must fuzzy-match (ratio ≥ 0.85) the cited segment, otherwise the quote is dropped.

## 7. Identity resolution (`analysis/identity.py`)

Per speaker label, collect candidates from these **sources**, each producing `(name, score, evidence)`:

| Source | How | Base score |
|---|---|---|
| `self_intro` | regex + LLM over that speaker's segments: "I'm X", "my name is X", "ich bin X", "je m'appelle X", … (LLM handles all languages) | 0.85 |
| `addressed_by_name` | another speaker says "thanks, X" / "welcome X" immediately before this speaker's turn | 0.6 |
| `metadata` | name appears in title/channel/uploader/description; channel owner is likely the host (most talk time or first speaker) | 0.5 |
| `voiceprint` | strong match → similarity; weak → similarity × 0.6 | sim |
| `hint` | caller's `speaker_hints` | 0.7 |
| `web` | verify stage (§8) confirms the name is associated with the channel/event/voice context | +0.15 |

Fusion: `confidence = 1 − Π(1 − score_i)` over independent sources for the same normalised name, capped at 0.98;
conflicting names → keep all as candidates, `name` = top candidate only if it leads the second by ≥ 0.2,
otherwise `name: null` with a ReviewFlag "ambiguous identity". A name with only one `metadata` source is never
promoted to `name` (stays a candidate). Implemented deterministically; the LLM only extracts mentions
(`identity_mentions.md`) and writes the `description`/`role` fields.

## 8. Cross-verification (`analysis/verify.py`)

1. Select targets: up to N checkworthy claims (N = 6 standard / 15 deep; prefer `fact` with numbers, names, dates)
   and every speaker whose top candidate confidence is in [0.4, 0.9).
2. For each target the LLM writes 1–2 search queries (`verify_queries.md`); queries count against the budget.
3. `search/router.py`: try providers in `SEARCH_PROVIDERS` order; on error/empty fall through; cache results.
   Each hit becomes evidence `web:<n>` (url, title, snippet ≤ 500 chars). No page fetching beyond the snippet in v1
   except Exa's `text` contents (≤ 2000 chars).
4. Judge (`verify_judge.md`) gets the target, its evidence, and the hits; returns verdict + rationale + which `web:` ids
   support it. A verdict other than `unverified` must cite ≥1 `web:` id; otherwise it is forced to `unverified`.
5. Identity results feed back into §7 as the `web` source.

## 9. Evidence rules (`report/evidence.py`)

- `EvidenceIndex` is built deterministically from: segments (`seg:<id>`), metadata fields (`meta:<field>`),
  hints (`hint:<n>`), voiceprint matches (`vp:<label>`), platform captions cues (`caption:<n>`), search hits (`web:<n>`).
- `validate_citations(report)`: every cited id must exist in the index; unknown ids are removed from the list; an
  item whose evidence list becomes empty is dropped and counted. Run as the last step of `assemble`.
- `evidence_index` in the report contains only ids that are cited.

## 10. Translation and subtitles

- Translate **cues**, not free text: batch ≈ 40 cues per call (`translate.md`), input `[{i, text}]`, output
  `[{i, text}]` via `call_tool`. Validate that the set of `i` matches exactly; on mismatch retry that batch once,
  then fall back to per-cue calls. Pass the summary + glossary of entity names to keep names consistent.
- Cue segmentation (`subtitles/segmenter.py`): from word timestamps; break at sentence end, then comma, then speaker
  change (always break on speaker change), max 42 chars/line, 2 lines, 1.0–7.0 s, ≥ 0.08 s gap. Optional speaker
  prefix `"- "` when two speakers share a cue is not allowed — never mix speakers in one cue.
- SRT: `HH:MM:SS,mmm`; VTT: `WEBVTT` header, `HH:MM:SS.mmm`, optional `<v S1>` voice tags.

## 11. Self-review (`analysis/review.py`)

Deterministic checks first (each → ReviewFlag): speaker shares sum ≈ 1; chapters cover ≥ 80 % of duration;
`caption_agreement < 0.6`; > 10 % low-confidence segments; identity conflicts; translations failed.
Then one LLM call (`review.md`) with the assembled report (without transcript) + 30 sampled segments, asking only
for inconsistencies as ReviewFlags (it cannot edit the report).

## 12. Q&A (`analysis/qa.py`)

Retrieve top 12 segments by Postgres full-text (`websearch_to_tsquery`, 'simple' config) + always include the summary;
`qa.md` answers with citations (`seg:` ids). Answer without a valid citation → "Not found in the audio."
