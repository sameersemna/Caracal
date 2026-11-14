# Caracal evaluation harness (T4.2)

An external client that measures a **running** Caracal server against a small labelled case
set and writes `docs/findings/eval-<date>.md`. Nothing is mocked: each case goes through the
REST API (`POST /v1/jobs`, uploads for local fixtures), so the numbers include ingest, ASR,
diarization, identity/verification and report assembly.

```
scripts/eval/
├── cases.json     the labelled cases and their expectations
├── metrics.py     pure WER / DER / identity / ROC (unit-tested)
└── run_eval.py    the live runner and Markdown report
```

## Quick start against a throwaway environment

The script mirrors `scripts/live-suite.sh`: a throwaway Postgres schema, a Redis prefix and a
data directory, `caracal serve` + `caracal worker` by PID, then the eval cases. Run from the
repository root; credentials come from `.env` and stay out of the process output.

```bash
set -a; . ./.env; set +a

TAG="caracal_eval_$(openssl rand -hex 4)"
SCRATCH="$(mktemp -d)"
export CARACAL_DB_SCHEMA="$TAG"
export REDIS_PREFIX="$TAG:"
export CARACAL_DATA_DIR="$SCRATCH/data"
export CARACAL_PORT=11497
export CARACAL_PUBLIC_BASE_URL="http://127.0.0.1:11497"

psql "${DATABASE_URL/+psycopg/}" -qAtc "CREATE SCHEMA \"$TAG\""
.venv/bin/caracal migrate
.venv/bin/caracal serve  >"$SCRATCH/serve.log"  2>&1 &
.venv/bin/caracal worker >"$SCRATCH/worker.log" 2>&1 &
# wait for /healthz, then:
uv run python -m scripts.eval.run_eval \
  --base-url http://127.0.0.1:11497 \
  --token-env CARACAL_API_TOKEN \
  --out docs/findings/eval-"$(date +%F)".md

# cleanup
kill %1 %2
psql "${DATABASE_URL/+psycopg/}" -qAtc "DROP SCHEMA IF EXISTS \"$TAG\" CASCADE"
rm -rf "$SCRATCH"
```

`--token-env CARACAL_API_TOKEN` is the **name** of the environment variable holding the
bearer token; the token itself is never printed or written to the report.

## Against the production service

```bash
CARACAL_API_TOKEN=... uv run python -m scripts.eval.run_eval \
  --base-url https://caracal.example.com \
  --token-env CARACAL_API_TOKEN
```

A production run can be deduplicated against an earlier identical request (the service
returns the finished job for the same source and options). Use the throwaway environment
when the timings matter.

## Options

| Flag | Meaning |
|---|---|
| `--base-url` | server root, e.g. `http://127.0.0.1:11497` (required) |
| `--token-env` | env var holding the bearer token (default `CARACAL_API_TOKEN`) |
| `--cases` | comma-separated case ids; default all |
| `--cases-file` | alternate `cases.json` |
| `--out` | Markdown path; default `docs/findings/eval-<UTC date>.md` |
| `--strict` | exit 1 when an expectation is unmet (otherwise the exit code is always 0) |
| `--timeout` | per-job wait in seconds (default 900) |
| `--poll` | status poll interval (default 3 s) |
| `--keep-work` | keep the temp directory with downloaded media |

The script prints progress and the report path to stderr; the report is the only artifact
it writes. Media, caption tracks and the splice live in a temporary directory and are
removed on exit (unless `--keep-work`).

## Cases and what they measure

| id | source | expectation |
|---|---|---|
| `ted-levitin` | TED `8jPQjjsBbIc` | speaker **Daniel Levitin**; language `en`; WER ≤ 0.35 vs YouTube **manual** captions |
| `all-ears-english` | YouTube `g-l96_aCF-0` | speaker **Lindsay McMahon**; language `en` |
| `jfk-ted-splice` | `tests/fixtures/audio/jfk.wav` + TED 20–40 s and 60–75 s, concatenated | S1/S2 unnamed (a JFK *candidate* is allowed); DER ≤ 0.35 against the turns known from the splice |
| `caliendo-impressions` | YouTube `2zxPbb-z5rk` | no label has a promoted name with confidence ≥ 0.6 |
| `soundcloud-instrumental` | `https://soundcloud.com/forss/flickermood` | `audio.music_detected == true` |
| `german-news` | YouTube `OgDZqde_My4` | language `de`; English subtitle file served |

The splice is built on the fly with ffmpeg (16 kHz mono WAV) and uploaded through
`POST /v1/uploads`; the reference turns are `[0, jfk_duration] = JFK` and
`[jfk_duration, end] = TED`. No media is committed to the repository.

## Metric definitions (`metrics.py`)

- **WER** — `wer(ref_words, hyp_words) -> (errors, ref_len)`. Both sides use
  `caracal.asr.captions.normalise_word` (NFKC, lower-case, bracket tags/punctuation removed),
  i.e. exactly the normalisation behind `caption_agreement`. The run restricts both streams
  to their overlapping time range before diffing.
- **DER** — `der(ref_turns, hyp_turns, collar=0.25)`. Intervals within `collar` seconds of a
  reference boundary are not scored. Hypothesis labels are mapped to reference labels with
  an optimal one-to-one assignment (brute force up to 8 labels, greedy above). Returns
  `{der, miss, false_alarm, confusion}`, each a fraction of the scored reference speech, so
  `der == miss + false_alarm + confusion`.
- **Identity** — `identity_score(expected, actual)` gives `correct | wrong | missing |
  abstained` per label. `name: null` is **correct** when the expected answer is unknown.
- **ROC** — `roc(pairs)` returns `(threshold, tpr, fpr)` points in descending threshold
  order; `best_threshold(pairs, max_fpr=0.01)` returns the threshold with the highest TPR
  whose FPR is within budget (highest threshold breaks ties).

## Voiceprint calibration

Skipped, with a note in the report: speaker embeddings are not exposed over the REST API.
`GET /v1/jobs/{id}/report` exposes only a label's `voiceprint_similarity` to an *enrolled*
speaker, not the 256-d label embeddings, so same/different pairs cannot be formed. `roc()`
and `best_threshold()` are implemented and unit-tested; wire them to an embedding export
when one exists.

## LLM call and cost figures

The report reads `caracal_llm_calls_total`, `caracal_search_cost_usd_total` and
`caracal_whisper_audio_seconds_total` from `GET /metrics`. `/metrics` does not track token
usage or LLM spend. As of T4.1 the registry defines `incr_llm_call` and
`add_whisper_seconds`, but no pipeline client calls them, so those series are absent or
zero; the report says "not recorded" rather than inventing a number. Search cost is real
(the search router increments it).

## Tests

```bash
uv run pytest tests/unit/test_eval_metrics.py -q
```

The live runner itself is not a test; the gate never starts a server.
