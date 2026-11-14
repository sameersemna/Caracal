STATUS: DONE
## What changed
- `src/caracal/analysis/__init__.py`: `Depth` and the `AnalysisOptions` protocol that keeps analysis free of a `pipeline.stages` import cycle.
- `src/caracal/analysis/planner.py`: `PlanInputs`, `StagePlan`, `ContentProfile`, deterministic rules (music/ambient, diarize option/availability, short clip, quick depth, verify, translate), budgets, and the LLM profile fallback.
- `src/caracal/analysis/content.py`: `ContentAnalysis`, LLM-facing models (aliased `type` → `entity_type`), single-call and map-reduce paths, deterministic chapter/quote/entity/evidence post-checks, depth variants and `llm_meta`.
- `src/caracal/llm/prompts/plan.md`, `content.md`, `content_window.md`, `content_reduce.md`: guard-loaded prompts with DATA blocks, typed evidence ids, report language and no-invention rules.
- `src/caracal/pipeline/stages.py`: `PLAN` moved directly after `LANGID`; `stage_plan`, `stage_content`, plan-gated skips on `diarize`/`align`/`voiceprint`/`content`/`identity`/`verify`/`translate`, and M2 registry wiring.
- `src/caracal/config.py`, `.env.example`: `CARACAL_REPORT_LANGUAGE` (default `en`).
- `tests/unit/test_planner.py`, `tests/unit/test_content.py`, `tests/unit/test_stages.py`, `tests/live/test_content_live.py`.
## Tests added
- `test_music_rule_skips_speakers_and_rewrites_languages_without_an_llm_call`: `🎵`-only transcript → `music_detected`, `und`, all speaker/identity/verify/translate skips, zero LLM requests.
- `test_music_threshold_is_seventy_percent_of_speech_time`: exact 70 % boundary; word-bearing lyrics are not music.
- `test_lyrics_with_words_are_not_music_and_the_profile_is_used`: one real tool-call request, citable prompt lines, profile fields.
- `test_quick_depth_makes_no_llm_call_and_skips_identity_work`: quick depth has zero LLM requests and a zero search budget.
- `test_llm_failure_falls_back_to_the_default_profile`: `CaracalError` → default profile + `llm unavailable` reason.
- `test_llm_output_cannot_flip_run_flags`: a profile containing `run` is ignored; deterministic flags stand.
- Rule tests for diarize off/unavailable, low speech ratio, short clips, verify modes, translate targets and deep budgets.
- Content tests: single call values/ids; map-reduce window count, one-segment overlap and reduce input; chapter clamp/re-derive/overlap; quote fuzzy keep/drop; invalid-id drops; quick/deep/standard variants and claim caps.
- Stage tests: exact stage order; skip-by-plan for all seven gated stages; `stage_plan` music output; `stage_content` checkpoint with `llm_meta`.
- `tests/live/test_content_live.py`: real gemma4 planner+content on `talk_c0`+`talk_c1` (valid analysis, ≥3 entities, ≥1 checkworthy claim, resolving ids) and the synthetic music rule.
## Measured
- gate exit code: 0
```
........................................................................ [  7%]
........................................................................ [ 15%]
........................................................................ [ 23%]
........................................................................ [ 30%]
........................................................................ [ 38%]
........................................................................ [ 46%]
........................................................................ [ 54%]
........................................................................ [ 61%]
........................................................................ [ 69%]
........................................................................ [ 77%]
........................................................................ [ 84%]
........................................................................ [ 92%]
....................................................................     [100%]
932 passed in 8.98s
GATE: PASS
```
## Deviations
- `tests/unit/test_runner.py`: changed the declared stage-order expectation from `["ok", "diarize", "plan"]` to `["ok", "plan", "diarize"]`, because the card requires `PLAN` immediately after `LANGID`.
- `StageContext` gained an optional `llm` field; the runner/worker are outside this card's Files, so they do not pass it yet. Planner records `llm unavailable` and content skips cleanly when it is absent.
- Deep chapter granularity is enforced through the prompt guidance (`2-4` vs `5-10` minutes) plus the deterministic clamp/sort/overlap pass; no splitter invents chapter titles.
## NOT proven
- `tests/live/test_content_live.py` was not executed here (live tests are judge-run).
- The production worker path does not yet construct an `LLMClient` (runner/worker not in this card's Files); it will fall back to the default profile and skip content until a later card injects the client.
- `verify` auto mode cannot see identity candidates at plan time (they do not exist until identity resolution), so it uses the documented `checkworthy_domain` + configured-search condition only.
## Open
- Should the next card wire one process-wide `LLMClient` through `RunnerDeps`/`Worker` into `StageContext.llm`, so plan/content run in `caracal worker` (not only in direct stage tests)?
- Should quick depth keep diarization (speaker labels) or remain summary/languages/subtitles only as currently planned per docs/04 §2?
