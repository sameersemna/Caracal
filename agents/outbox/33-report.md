STATUS: DONE
## What changed
- src/caracal/analysis/planner.py: `verify:auto` now runs whenever a search provider is configured; it no longer reads `profile.checkworthy_domain` (which the planner cannot relate to identity). Quick and music keep their earlier skip reasons; the only new auto skip is `no search provider configured`.
- src/caracal/analysis/verify.py: quote-attribution floor lowered from 30 s to 10 s; `VerifyResult` carries `target_count`; `verify()` returns immediately (no query LLM call, no search, no flag rewrite) when `select_targets` finds nothing, logging `nothing to verify`.
- src/caracal/pipeline/stages.py: `stage_verify` returns `skipped` with reason `nothing to verify` when the verify result has zero targets (runtime decision, verdict 32).
- tests/unit/test_planner.py: replaced the old plan-time verify table with a six-row runtime-gating table.
- tests/unit/test_verify.py: added the 22 s / 9.9 s quote-eligibility boundary test and a no-target no-work test.
- tests/unit/test_stages.py: added the stage-level `nothing to verify` skip test.

## Tests added
- `test_verify_gating_moves_to_run_time` (parametrised table over off/on/auto, checkworthy true/false, search configured true/false, quick, music): proves `auto` no longer skips on an uncheckworthy profile, still skips with `QUICK_REASON`/`MUSIC_REASON`, and skips only for a missing search provider; asserts the exact reason per row.
- `test_quote_attribution_accepts_a_twenty_two_second_speaker`: a 22 s candidate-less speaker with an 8+ word segment becomes a `quote_attribution` target; 9.9 s does not.
- `test_verify_with_no_targets_does_no_work`: zero targets yields `target_count == 0`, no verifications/hits/queries/cost, and no LLM or router calls.
- `test_verify_stage_skips_when_there_is_nothing_to_verify`: with a configured router and LLM but no targets, `stage_verify` returns `skipped` / `nothing to verify` and makes no LLM call.

## Measured
- gate exit code: 0
- last 15 lines of `bash scripts/gate.sh`:
```
........................................................................ [ 26%]
........................................................................ [ 33%]
........................................................................ [ 39%]
........................................................................ [ 46%]
........................................................................ [ 53%]
........................................................................ [ 59%]
........................................................................ [ 66%]
........................................................................ [ 72%]
........................................................................ [ 79%]
........................................................................ [ 86%]
........................................................................ [ 92%]
........................................................................ [ 99%]
.....                                                                    [100%]
1085 passed in 9.72s
GATE: PASS
```

## Deviations
- docs/04 §2 describes the auto rule correctly (quick/music/no provider), but the old reason strings still said "checkworthy domain"; those strings are code constants, not docs. I changed them to `auto verification: search configured` and `auto verification: no search provider configured` so the checkpoint text matches the actual rule. No doc file was edited.
- docs/04 §8 step 1 does not state a talk-time number at all; the card and verdict specify 10 s, so the code uses 10 s (`QUOTE_MIN_TALK_TIME_S = 10.0`) and the function docstring states it.

## NOT proven
- The three real jobs (All Ears English, JFK splice, TED) were not re-run: this worker is not allowed to run `tests/live/` or make live network/LLM calls. The unit coverage targets exactly the three changed decisions the judge will re-run, but end-to-end promotion of Lindsay McMahon and the JFK info flag are unverified here.
- No live search/provider behaviour was exercised; search tests use the existing recorded fixtures and scripted router.

## Open
- The docs gap above: should docs/04 §8 step 1 be amended to say "talk time ≥ 10 s" so the threshold is discoverable? The card's Files list excludes `docs/`, so I did not edit it.
