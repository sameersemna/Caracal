STATUS: DONE

## What changed
- `src/caracal/config.py`: `Route` StrEnum (default + seven tasks), `ModelRef(endpoint, name, raw)` with `normalised`, `parse_model_ref`, `parse_ref_csv`, `resolve_llm_routing` (pure resolver) and `ROUTE_FIELDS`; nine new fields `LLM_ALLOWED_MODELS` + `LLM_ROUTE_*` (case-preserving CSV); a `model_validator(mode="after")` that rejects a route ref outside the allow-list (naming the variable and the normalised model) and a cloud ref without `OLLAMA_CLOUD_API_KEY`; `Settings.llm_allowed_models()` / `Settings.llm_route_chain(route)`.
- `src/caracal/llm/client.py`: `call_tool`/`chat` take `route: Route`; each call resolves the route's chain into `_Endpoint`s (cloud URL+bearer / local URL, first ref `LLM_TIMEOUT_S`, later local refs `max(LLM_TIMEOUT_S, 240)`); per-ref transport retries (1 s, then 4 s) and failover to the next ref after 2 consecutive transport/5xx/timeout failures, a 401/403 (`llm.auth_failed` once per ref), a 404 model-not-found, or invalid output after the two repairs; chain exhaustion re-raises the last failure labelled with the route. `CallMeta` gained `route`, `model_ref`, `refs_tried` (attempts stays the total request count of the call).
- `src/caracal/ops.py`: `MetricsRegistry.incr_llm_call` takes keyword-only `route`/`model` and stores `{route,model,endpoint,mode,fallback}` (entries written without them still render in the legacy three-label shape); doctor now emits one `POST /api/show` check per allowed model (with the routes that use it) plus a required aggregate `ollama` line: fail when a route's every model fails, warn when only part of a route's chain fails.
- `src/caracal/analysis/{planner,content,identity,verify,translate,review,qa}.py`: every `call_tool` passes its task route (PLAN, CONTENT, IDENTITY, VERIFY ×3, TRANSLATE, REVIEW, QA ×2) — nothing else changed in those modules.
- `src/caracal/analysis/translate.py`: translation units also end at a pause > 1.0 s between cues (all scripts) and at a whisper segment boundary when the previous cue is Han/Kana/Hangul/Thai text without sentence punctuation; new `segment_token_starts(transcript)` helper; `units_from_cues(..., segment_starts=())` (default keeps the old behaviour).
- `src/caracal/pipeline/stages.py`: `stage_translate` passes the transcript's whisper segment starts into `units_from_cues`.
- `.env.example`, `README.md`: the owner's `LLM_ALLOWED_MODELS` / `LLM_ROUTE_DEFAULT` configuration, the ref syntax (`cloud:<name>` / `local:<name>` / bare `<name>`) and the failover/doctor behaviour.
- `tests/unit/test_config.py`, `test_llm_client.py`, `test_ops.py`, `test_translate.py`, `tests/live/test_llm_live.py`, `tests/unit/test_stages.py`: new/updated tests (details below).

## Tests added
- `test_model_refs_parse_into_an_endpoint_and_a_name` (+ `test_a_model_ref_without_a_name_is_rejected`): the ref parsing table (`cloud:`/`local:`/bare/whitespace, normalised form; empty names rejected).
- `test_backward_compatible_chain_without_a_cloud_key` / `test_backward_compatible_chain_with_a_cloud_key`: no routes/allow-list ⇒ exactly today's chain (local proxy then local fallback; or `cloud:LLM_MODEL` then `local:LLM_FALLBACK_MODEL`) for every route.
- `test_explicit_routes_use_the_default_for_unconfigured_tasks`: per-task route wins, unset task and **empty** route variable both use DEFAULT.
- `test_a_route_ref_outside_the_allow_list_names_the_route_and_model`: `ValidationError` naming `LLM_ROUTE_QA` and `local:not-allowed:1b`.
- `test_a_route_ref_is_matched_against_the_allow_list_by_normalised_ref`, `test_a_cloud_ref_without_a_cloud_key_is_rejected`, `test_an_allow_list_without_routes_becomes_the_default_chain`, `test_routes_without_an_allow_list_allow_exactly_their_own_models`, `test_route_refs_are_read_from_the_environment`.
- `test_a_task_route_uses_its_own_chain_and_reports_the_route`, `test_an_unconfigured_task_route_uses_the_default_chain`, `test_a_cloud_ref_calls_the_cloud_endpoint_with_the_bearer_key`: chain order, URL/key/model of the ref and `CallMeta.route/model_ref/refs_tried`.
- `test_two_failures_on_a_ref_move_to_the_next_ref_of_the_chain`: 2×500 ⇒ third request on ref 2, waits `[1.0, 4.0]`, `attempts == 3`, `refs_tried == 2`.
- `test_a_404_model_not_found_moves_straight_to_the_next_ref`: 404 ⇒ immediate second ref, no sleep.
- `test_invalid_output_after_the_repairs_moves_to_the_next_ref`: initial + 2 repairs on ref 1, then ref 2 answers.
- `test_a_local_ref_after_the_first_gets_the_cold_start_timeout`: request timeout 180 s on ref 1, `max(LLM_TIMEOUT_S, 240)` on the local ref (180⇒240, 300⇒300).
- `test_the_metric_carries_the_route_model_endpoint_mode_and_fallback_labels`, `test_chat_follows_the_route_chain`.
- `test_auth_failures_are_logged_once_per_model_ref`: 401 on ref 1 and 403 on ref 2 produce exactly one `llm.auth_failed` record per ref (`model_ref` + status asserted).
- `test_every_analysis_module_passes_its_route`: AST check of every `llm.call_tool` in the seven analysis modules — exact route literal and exact call-site count (1/1/1/3/1/1/2).
- `test_probe_ollama_model_posts_api_show_to_the_ref_endpoint`, `test_probe_ollama_model_reports_a_404_as_fail`, `test_probe_ollama_model_fails_a_cloud_ref_without_a_key`: `POST /api/show` URL, `{"model": …}` body, bearer only for cloud, 404 ⇒ fail.
- `test_one_line_per_allowed_model_names_the_routes_that_use_it`: per-model line detail lists the routes using the ref.
- `test_a_route_whose_every_model_fails_fails_the_required_ollama_check`: required aggregate `ollama` fails, exit code 1.
- `test_a_failing_fallback_only_warns_and_keeps_the_exit_code_zero`: warn + exit code 0.
- `test_metrics_render_the_routed_llm_label_set`: exact 5-label Prometheus line (the legacy 3-label line is still asserted by the existing tests).
- `test_a_long_pause_ends_a_unit_so_separate_news_items_are_not_merged`: the judge's zh case (4.8 s gap) ⇒ two units.
- `test_separate_units_are_translated_as_separate_batch_items`: the two items travel as two independent units, so the first English output cannot mention Tanzania.
- `test_a_short_pause_does_not_end_a_unit`, `test_a_whisper_segment_boundary_ends_a_cjk_unit_without_punctuation`, `test_a_segment_boundary_does_not_end_a_latin_unit`, `test_a_segment_boundary_does_not_end_a_cjk_unit_that_ends_a_sentence`, `test_segment_token_starts_counts_words_not_cues`.
- `test_translate_calls_use_the_translate_route`: a real `LLMClient` + `FakeOllama` shows the TRANSLATE route's model on the wire.
- `test_stage_translate_ends_a_unit_at_a_whisper_segment_boundary_for_cjk` (in `tests/unit/test_stages.py`, see Deviations): two adjacent CJK whisper segments with no pause ⇒ `units == 2`; verified to fail when the `segment_starts` wiring is removed.
- Live (judge-run, not executed here): `test_live_every_route_answers_with_its_preferred_model` (each route answered by the first ref of its chain, `route`/`model_ref`/`refs_tried` asserted) and `test_live_qa_route_falls_back_to_the_local_model` (`LLM_ROUTE_QA=local:nonexistent-model:1b,local:gemma4:31b` ⇒ `refs_tried == 2`, `model_ref == local:gemma4:31b`).

Non-vacuity spot checks (mutation, then restored): removing the 404 branch fails the 404 test; removing the startup route validation fails the allow-list test; removing the `segment_starts` wiring fails the CJK stage test.

## Measured
- gate exit code: 0
- ```
........................................................................ [ 40%]
........................................................................ [ 45%]
........................................................................ [ 51%]
........................................................................ [ 56%]
........................................................................ [ 61%]
........................................................................ [ 66%]
........................................................................ [ 71%]
........................................................................ [ 76%]
........................................................................ [ 81%]
........................................................................ [ 86%]
........................................................................ [ 91%]
........................................................................ [ 97%]
.........................................                                [100%]
1409 passed in 12.12s
GATE: PASS
```

## Deviations
- `tests/unit/test_stages.py` is **not** in the card's Files list, yet two edits were unavoidable and are reported plainly:
  1. `test_stage_review_skips_quick_music_and_missing_llm` scripted three invalid replies ("initial + 2 repairs") while the card's new contract retries the repairs on **each** ref of the chain; with the derived two-ref chain the call now makes six requests. The fake script grew from 3 to 6 invalid responses and the test now also asserts `len(fake.requests) == 6`. No assertion was removed or loosened — the test now covers both refs; leaving it untouched keeps the gate red by construction. (Same class of judge-side omission recorded in outbox 36 and 43; card 46 requirement 3 makes the old count impossible.)
  2. `test_stage_translate_ends_a_unit_at_a_whisper_segment_boundary_for_cjk` is new: it is the only test that proves `stage_translate` actually passes the whisper segment starts to the unit builder (item 7's requirement). No other card-listed file can exercise the stage wiring.
- Timeout reading (requirement 3): "`LLM_TIMEOUT_S` for the first ref, `max(LLM_TIMEOUT_S, 240)` for local refs" is implemented as *first ref = `LLM_TIMEOUT_S`* and *every later local ref = max(…, 240)* — a local ref that happens to be first keeps today's timeout. Rationale: the measured cold-start risk is the *fallback* ("the fallback must not be used with the primary's short timeout"), and the older local-proxy primary behaviour must not change.
- `CallMeta.model_ref` is the answered ref as its normalised string (`cloud:gemma4:31b` / `local:gemma4:31b`), not the `ModelRef` object, because `CallMeta` is carried into report metadata; `attempts` keeps its old meaning (total requests of the whole call, across refs), `refs_tried` counts the models.
- `MetricsRegistry.incr_llm_call` keeps its three positional arguments and takes `route`/`model` as keyword-only; entries written with them render the required `{route,model,endpoint,mode,fallback}` line, entries written without them keep rendering the pre-T4.5 `{endpoint,mode,fallback}` line. This keeps the `tests/contract/test_metrics.py` contract (a file outside the Files list) valid without editing it, and it is also what a running deployment's existing Redis counters need.
- The doctor's per-model probe is a new `DoctorProbes.ollama_model(settings, ref)` field and the old `ollama`/`/api/tags` probe was removed: `/api/tags` cannot answer availability for cloud tags (card 46, measured facts), and `run_doctor` needs per-model results to aggregate routes. Existing doctor tests were adapted to the new injectable probe (allowed file), not weakened.

## NOT proven
- The two live routing cases were written but not executed (no live network/model calls in this session; `tests/live/` is judge-run).
- The real gemma4 behaviour under a 240 s cold start and the owner's exact `-cloud` proxy tag are described from docs/00 and the card's measured facts, not re-measured here.
- I did not measure doctor output against the real cloud endpoint or the local daemon (unit tests use `httpx.MockTransport`); the `/api/show` 200/404 semantics come from the card's measured facts.
- The item-7 fix is proven by unit tests on units and by the stage-level wiring test; no real translation of the judge's clip was run (the "English of the first does not mention Tanzania" result is a live-model property).

## Open
- Should `tests/unit/test_stages.py` be added to the Files list for card 46 (the review-stage request count and the stage wiring are stage-level facts), or does the boss prefer a card-46 addendum for it?
- If the boss wants the *first* local ref to get the 240 s cold-start timeout as well, that is a one-line change in `LLMClient._chain` (index condition) — the current reading follows the requirement's wording.
