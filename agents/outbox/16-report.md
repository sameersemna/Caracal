STATUS: DONE
## What changed
- src/caracal/llm/schema.py: cached `prepare_schema` (inline `$ref`, cycle `ValueError`, drop `title`, enum hints) and schema-guided `coerce` with `DEFAULT_SYNONYMS`.
- src/caracal/llm/jsonx.py: fence/prose-tolerant, string-aware outermost-object `extract_json`.
- src/caracal/llm/client.py: `LLMClient.call_tool`/`chat`, cloud/local endpoint selection, retries 1 s/4 s, auth + two-failure fallback, max-2 repair loop, per-client semaphore, INFO/DEBUG logging.
- src/caracal/llm/prompts/__init__.py: `load_prompt` (guard prepended, safe two-phase `str.replace`) and `wrap_data`.
- src/caracal/llm/prompts/_guard.md: docs/04 §1 evidence block verbatim + docs/06 S9 DATA-block warning.
- src/caracal/llm/prompts/probe.md: tiny live-test prompt.
- src/caracal/llm/__init__.py: package exports.
- tests/fakes/ollama.py: `FakeOllama` on `httpx.MockTransport`, fixture/script helpers, `AsyncProbeTransport` for concurrency.
- tests/unit/test_llm_schema.py: schema prep, cycle, coercion table, fixture replay (inlined valid, `$defs` invalid).
- tests/unit/test_llm_client.py: request body, coercion, repair, fenced JSON, fallback, 401 logging once, exhaustion, semaphore, log hygiene, chat, prompt loader.
- tests/unit/test_jsonx.py: fences, prose, braces in strings, nested, errors.
- tests/live/test_llm_live.py: real cloud call with README transcript, forced fallback to local Ollama.
- tests/live/conftest.py: single `isolated_environment` override for all live tests.
- tests/live/test_db.py, test_pipeline_live.py, test_queue_live.py, test_ytdlp_live.py, test_whisper_live.py: removed per-file overrides (carry-forward 15).

## Tests added
- test_prepare_schema_inlines_refs_in_three_level_model: no `$ref`/`$defs` survive inlining of nested models.
- test_prepare_schema_drops_titles_without_dropping_a_field_named_title: title stripping is structural, not a blind key pop.
- test_prepare_schema_adds_enum_hints: every enum description ends with the exact `Must be exactly one of: …` hint.
- test_prepare_schema_is_cached / rejects_recursive_models: caching identity and cycle → `ValueError`.
- test_coerce_table: PERSON→person, location→place, ORGANIZATION→org, "0.9"→0.9, 1.4→1.0, "null"/""→None, "x"→["x"], clamping.
- test_coerce_merges_per_call_synonyms / leaves_unknown_object_keys: per-call synonyms and extra keys preserved.
- test_inlined_tool_call_fixture_coerces_to_valid_model: real inlined fixture validates with person/place/org and evidence lists.
- test_defs_tool_call_fixture_does_not_validate_after_coercion: real `$defs` fixture is missing `evidence`, proving inlining is required.
- test_request_body_is_cloud_tool_call_with_inlined_schema: cloud URL, bearer, tools, inlined parameters, system suffix, meta.
- test_valid_arguments_without_coercion_changes_report_clean_meta: `coerced` is false when values already conform.
- test_invalid_first_reply_triggers_repair_without_echoing_values: assistant+tool repair message carries loc/msg and no input values.
- test_call_tool_merges_per_call_synonyms: client forwards per-call synonym map into coercion.
- test_no_tool_call_fenced_json_is_extracted_and_coerced: measured no-tool fixture is extracted and repaired into a valid model.
- test_two_primary_failures_switch_to_local_fallback: two cloud 500s, 1 s/4 s waits, local URL/model with no auth header.
- test_401_falls_back_immediately_and_logs_auth_failed_once: per call immediate fallback, one `llm.auth_failed` record across two calls.
- test_all_endpoints_failing_raises_retryable_dependency_down: exhaustion raises retryable `dependency_down`.
- test_semaphore_limits_concurrent_requests: three concurrent calls with limit two never exceed two in flight.
- test_info_logs_never_contain_prompt_text: unique prompt marker absent from INFO records/extras.
- test_chat_returns_text_without_tools: chat omits tools and uses its own temperature/num_predict.
- test_load_prompt_* / test_wrap_data_cannot_be_closed_by_the_data: guard prepended, missing var KeyError, DATA braces literal, delimiters stripped.
- test_jsonx.py suite: fence (json and bare), prose, string-aware braces, escapes, nested outermost, no-JSON error.

## Measured
- gate exit code: 0
- Success: no issues found in 47 source files
  ==> repository rules
  ==> pytest (unit + contract, live excluded)
  ........................................................................ [ 10%]
  ........................................................................ [ 20%]
  ........................................................................ [ 30%]
  ........................................................................ [ 40%]
  ........................................................................ [ 50%]
  ........................................................................ [ 60%]
  ........................................................................ [ 70%]
  ........................................................................ [ 80%]
  ........................................................................ [ 90%]
  .................................................................        [100%]
  713 passed in 7.26s
  GATE: PASS
- live LLM tests were written but not run by the worker (per docs/07 §3; the judge runs them).

## Deviations
- Docs/01 §3 already lists `llm/schema.py`, `llm/jsonx.py`, `llm/client.py` and `llm/prompts/*.md`, so no doc edit was needed for the schema module.
- Attempt semantics: `CallMeta.attempts` counts every HTTP request made for the call, including failed primary attempts and fallback requests (so the 2×500→local path reports 3).
- Backoff mapping: the primary gets failures 1 s apart; the 4 s step is applied before the fallback request, and between fallback retries. This honours both "1 s, 4 s" and "fallback after 2 consecutive primary failures".
- `LLMClient.chat` shares the retry/fallback state machine with `call_tool` but does not append the tool suffix or run repair (it has no tool/model to validate).

## NOT proven
- The live cloud and forced-local-fallback tests were not executed here (worker does not run `tests/live/`). They assert Merkel/Berlin and `fallback_used=True` so the judge can verify against the measured services.
- A real 401 from ollama.com was simulated with `httpx.MockTransport`; only the fallback/logging behaviour is proven, not live auth failure.
- `asyncio.sleep` was injected in tests; real wall-clock backoff timing was not measured.

## Open
- None.
