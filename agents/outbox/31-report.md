STATUS: DONE
## What changed
- `src/caracal/analysis/verify.py`: target selection, LLM query/claim-judge/identity-web steps, global `web:<n>` hit numbering, deterministic verdict/domain/name rules, and the stage-facing `verify()` result.
- `src/caracal/llm/prompts/verify_queries.md`: JSON-mode query prompt (claims, candidate+context identity queries, double-quoted quote queries).
- `src/caracal/llm/prompts/verify_judge.md`: JSON-mode claim judge prompt with claim/transcript/web DATA blocks.
- `src/caracal/llm/prompts/identity_web.md`: JSON-mode identity/quote web-extraction prompt.
- `src/caracal/analysis/identity.py`: exposed `WebSource` + `apply_web_sources()` and factored the existing promotion rule into `_promote_speaker()` so verify reuses (does not duplicate) the fusion; web confirm 0.15, attribution 0.5.
- `src/caracal/pipeline/stages.py`: `SearchRouter` on `StageContext`, checkpoint parsers, `_budget_exhausted`, and the real `stage_verify` (plan skip / `no search provider` / refined speakers / hits / flags / cost).
- `src/caracal/pipeline/runner.py`: carries `search` through `RunnerDeps` into the stage context.
- `src/caracal/worker.py`: builds one `SearchRouter` per process from settings and passes it to every job.
- `src/caracal/report/assemble.py`: `report.verification`, verify speakers preferred over identity, verify flags replace (not duplicate) identity flags, and verify hits enter the evidence index with 1-based `web:<n>` ids.
- `tests/unit/test_verify.py`, `tests/unit/test_identity.py`, `tests/unit/test_assemble.py`, `tests/unit/test_stages.py`: unit coverage.
- `tests/live/test_verify_live.py`: judge-run live checks for the two M3 acceptance cases.

## Tests added
- `test_select_targets_prefers_specific_checkworthy_facts_and_caps_the_list`: 6/15 caps, checkworthy filtering, facts with digits/proper names/months first.
- `test_select_targets_confirms_mid_confidence_names_and_attributes_long_quotes`: [0.4,0.9) identity targets, >=30 s no-candidate quote targets, key quote used.
- `test_select_targets_falls_back_to_the_longest_eight_word_segment`: quote falls back to the longest >=8-word segment.
- `test_lindsay_confirmation_expands_the_name_and_promotes`: query gets title context, candidate becomes "Lindsay McMahon" at 0.6175, sources include `web`, verification supported with web:1/web:4.
- `test_jfk_quote_attribution_is_a_half_confidence_candidate_only`: quoted query, "John F. Kennedy" candidate at 0.5, name stays null, info flag, mixed verification.
- `test_judge_citing_a_missing_web_id_is_forced_unverified`: `web:99` dropped and the supported verdict forced unverified.
- `test_high_confidence_needs_two_registrable_domains[single/double]`: same-domain 0.95 capped to 0.6; two domains keep 0.95.
- `test_budget_exhaustion_marks_the_remaining_targets_unverified`: refused query stops further searches; remaining targets say "search budget exhausted".
- `test_no_results_is_unverified_without_a_judge_call`: empty hits produce a deterministic unverified verdict and no judge call.
- `test_prompt_injection_hit_stays_inside_a_data_block`: hostile snippet sits between `<<<DATA:web>>>` and `<<<END:web>>>`; judge system prompt labels web text untrusted.
- `test_apply_web_sources_confirms_and_expands_a_given_name`: direct fusion of a full-name web source gives 0.6175 and web evidence.
- `test_apply_web_sources_attribution_is_a_candidate_only_and_flags_info`: attribution is a 0.5 web candidate, never promoted, with the info flag.
- `test_apply_web_sources_preserves_performance_and_recomputes_ambiguity_once`: unrelated flags survive and ambiguity is recomputed once.
- `test_verify_speakers_win_and_web_hits_become_evidence`: verify speakers preferred over identity; `web:1` resolves in the evidence index; no `web:0`.
- `test_unknown_web_source_is_dropped_and_the_verdict_forced_unverified`: the existing validator drops an unknown source and forces unverified.
- `test_skipped_verify_keeps_the_identity_flags`: a skipped verify checkpoint does not swallow identity flags.
- `test_verify_stage_refines_speakers_and_checkpoints_the_web_hits`: real router + fake LLM stage output has web:1/web:2, the refined name/confidence, and a supported verification.
- `test_verify_stage_skips_without_a_search_provider`: exact `no search provider` skip.
- `test_live_all_ears_english_resolves_lindsay_mcmahon` (live): real search + LLM resolve S1 to "Lindsay McMahon" and promote.
- `test_live_ted_claim_about_minus_forty_is_supported` (live): real search + LLM return supported with real web sources.

## Measured
- gate exit code: 0
- last 15 lines of `bash scripts/gate.sh`:
  ```
  ........................................................................ [ 20%]
  ........................................................................ [ 26%]
  ........................................................................ [ 33%]
  ........................................................................ [ 40%]
  ........................................................................ [ 46%]
  ........................................................................ [ 53%]
  ........................................................................ [ 60%]
  ........................................................................ [ 66%]
  ........................................................................ [ 73%]
  ........................................................................ [ 80%]
  ........................................................................ [ 86%]
  ........................................................................ [ 93%]
  ......................................................................   [100%]
  1078 passed in 9.79s
  GATE: PASS
  ```

## Deviations
- none

## NOT proven
- `tests/live/test_verify_live.py` was written but not executed, per the card; real Brave/Exa latency, the real gemma4 query/identity answers and the 0.6175 promotion are therefore not measured in this session.
- The worker's provider construction is not directly asserted (the stage test uses a real `SearchRouter`; `test_worker.py` is outside this card's file list).
- The identity-web confirmation deliberately requires the model's `supported` flag as well as a deterministic hit match; a model that marks a valid hit unsupported will not add the 0.15 source.
- The 1-based `web:<n>` contract required a re-key in `assemble.py` because `build_evidence_index` numbers the list it receives from 0.

## Open
- Should `apply_web_sources` re-run promotion for every speaker even when there are no web sources, or leave untouched speakers exactly as the identity checkpoint wrote them? The current code re-runs the same documented rule, which is idempotent for the existing cases and lets a later web confirmation resolve ambiguity.
