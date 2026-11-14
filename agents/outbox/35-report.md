STATUS: DONE
## What changed
- src/caracal/analysis/translate.py: sentence-unit grouping, JSON-mode batched translation with index validation/retry/per-unit fallback, proportional cue redistribution (CJK characters, RTL untouched), target-language resolution and ISO code validation.
- src/caracal/llm/prompts/translate.md: translation prompt naming the target language in English + native name, keeping glossary names/numbers and the subtitle register.
- src/caracal/subtitles/segmenter.py: `wrap_lines` helper for re-wrapping translated text to <=42 chars x 2 lines (overlong flag).
- src/caracal/pipeline/stages.py: real `stage_translate` (plan-gated, on-demand override, merge with previous languages), `subtitles` renders every translated language, `JobOptions.target_langs` validation.
- src/caracal/pipeline/runner.py: subset/forced stage runs, `translate_targets` context, `run_translate` for the `translate` queue kind on a finished job.
- src/caracal/worker.py: routes `kind == "translate"` deliveries to `run_translate`.
- src/caracal/services.py: `JobService.translate_job` (done-only, validates codes, writes the translate request, enqueues `translate`).
- src/caracal/api/schemas.py, src/caracal/api/routes.py: `POST /v1/jobs/{id}/translate` -> 202 queued; `TranslateIn` validation.
- src/caracal/mcp/server.py: `translate_job(job_id, target_langs)` tool.
- src/caracal/report/assemble.py: `translations[]` from the translate checkpoint plus a "translations failed" review flag.
- tests/unit/test_translate.py, tests/unit/test_stages.py, tests/unit/test_worker.py, tests/unit/test_assemble.py, tests/contract/test_api_translate.py, tests/contract/test_mcp_tools.py, tests/contract/test_mcp_wire.py, tests/contract/test_parity.py, tests/live/test_translate_live.py: coverage listed below.

## Tests added
- units_from_cues groups same-speaker cues to sentence punctuation, breaks on speaker change and enforces the 4-cue / 200-char caps.
- translate_units sends 25/5-unit JSON-mode batches, names German (Deutsch) + glossary + summary, retries an index mismatch once, falls back per unit, keeps source text and counts failed units.
- redistribute splits proportionally by source character count (2/5/3 tokens in the fixture), keeps source timings, fills every cue, splits CJK by characters, keeps RTL text without bidi marks, and counts cues needing >2/42 lines.
- target-language resolution covers null -> defaults minus source, [] -> none, de-duplication and source exclusion; JobOptions rejects unknown/ill-shaped codes.
- stage_translate writes units/failed_units/overlong_cues/duration_s + cue timings; on-demand mode ignores the plan skip; existing languages survive a re-translation of another language; a skipped on-demand run preserves the previous checkpoint.
- stage_subtitles renders source and every translated language with source timings.
- REST/MCP translate on a done job returns 202/queued and enqueues kind `translate`; 409 when not done; 422 for bad codes; 404 for an unknown job; parity row is available and REST/MCP payloads match.
- worker `translate` delivery runs only translate -> subtitles -> assemble, leaves the job `done` and consumes the request.
- assemble maps the translate checkpoint to `translations[]` links/status and raises the translations-failed flag.

## Measured
- gate exit code: 0
- ........................................................................ [ 25%]
- ........................................................................ [ 32%]
- ........................................................................ [ 38%]
- ........................................................................ [ 45%]
- ........................................................................ [ 51%]
- ........................................................................ [ 58%]
- ........................................................................ [ 64%]
- ........................................................................ [ 70%]
- ........................................................................ [ 77%]
- ........................................................................ [ 83%]
- ........................................................................ [ 90%]
- ........................................................................ [ 96%]
- .....................................                                    [100%]
- 1117 passed in 10.15s
- GATE: PASS

## Deviations
- The translate checkpoint keeps the documented `{langs: {lang: {units, failed_units, overlong_cues, duration_s}}}` fields and adds a per-language `cues` array; the `subtitles` stage needs the redistributed cue text and timings to render `subtitles/<lang>.srt|vtt` (docs/04 §10, card req. 3/5).
- On-demand languages reach the worker through `translate-request.json` in the job directory, because `queue.py` is outside this card's Files list and the repo forbids moving a `done` job back to `queued`. The job therefore keeps `done`; the API/MCP response reports the translation request as `queued`.
- `translate_units` returns a `list[str]` subclass (`TranslatedUnits`) so the stage can read `failed_units` while the public return remains the documented list of strings.

## NOT proven
- `tests/live/test_translate_live.py` was written but not run (the card reserves live tests for the judge); real gemma4 German/Arabic quality, latency and the German "Vor ... Jahren" assertion are therefore unmeasured here.
- The judge's full-pipeline TED de/ar and German-news-to-English runs were not executed; only the hermetic unit/contract layers were.
- LLM batch concurrency was not timed; it relies on the existing `LLMClient` semaphore and `asyncio.gather`.

## Open
- The on-demand request artifact is an extra runtime file at the job root (`translate-request.json`); docs/05 §3 does not list it. If the owner wants the queue entry itself to carry languages, `queue.py`/`docs/05` would need to change in a later card.
