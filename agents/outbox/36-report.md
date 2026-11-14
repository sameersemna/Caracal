STATUS: DONE
## What changed
- src/caracal/queue.py: `Delivery.payload` decodes an optional JSON `payload` stream field; `JobQueue.enqueue(..., payload=...)` writes it; `Keys.lock_translate(job_id)`.
- src/caracal/worker.py: a `translate` delivery runs inside `redis_lock(keys.lock_translate(job_id))` and passes `delivery.payload` to `run_translate`.
- src/caracal/pipeline/runner.py: `run_translate` takes the requested languages from the entry payload; the `translate-request.json` reader is gone; payload-less deliveries still fall back to the job options.
- src/caracal/services.py: `translate_job` enqueues `payload={"target_langs": [...]}` and no longer writes a side file.
- src/caracal/analysis/translate.py: removed `TRANSLATE_REQUEST_FILENAME`; added the non-Latin-script target set and `name_rule()`; the prompt receives `{name_rule}`.
- src/caracal/llm/prompts/translate.md: names stay verbatim for Latin-script targets; non-Latin-script targets use the conventional target-script form; the rule also governs the glossary.
- src/caracal/report/assemble.py: `translations` lists only languages with a translate entry *and* both rendered `srt`/`vtt` files in the subtitles checkpoint.
- tests/unit/test_queue.py: payload round-trip, malformed payload, `lock_translate` key.
- tests/unit/test_worker.py: two payload deliveries end-to-end through the real translate/subtitles/assemble stages; per-job lock serialisation.
- tests/unit/test_translate.py: prompt name rule for Latin (`de`) and non-Latin (`ar`) targets.
- tests/unit/test_assemble.py: translate entries without subtitle files are not reported.
- tests/contract/test_api_translate.py: entry payload assertions; two back-to-back REST requests keep their own languages.
- tests/contract/test_mcp_tools.py: payload assertion replaces the side-file assertion.
- tests/live/test_queue_live.py: payload (incl. nested object) round-trip on real Redis.
## Tests added
- test_enqueue_carries_a_payload_round_trip: the stream entry carries its own JSON payload, absent payload is `{}`, raw XADD field is exact.
- test_malformed_entry_payload_is_a_caracal_error: invalid payload JSON raises `CaracalError(INTERNAL, "malformed queue entry")`.
- test_two_translate_requests_each_carry_their_own_languages: REST back-to-back `de` then `ar` produce two entries with distinct payloads.
- test_back_to_back_translate_deliveries_both_reach_the_report: two deliveries for one job, real translate+subtitles+assemble, report lists `de` and `ar` and both file pairs exist.
- test_translate_deliveries_for_one_job_are_serialised: with worker concurrency 2 the translate stage is never entered twice at once (max concurrency 1).
- test_translate_prompt_keeps_names_verbatim_for_a_latin_target / ..._uses_target_script_names_for_arabic: the system prompt carries the correct name rule for `de` and `ar`.
- test_translate_entry_without_subtitle_files_is_not_reported: a `fr` translate entry with no rendered files is absent from `report.translations`.
- live test_queue_live.py: entry payload survives real Redis XADD/XREADGROUP, including a nested object.
## Measured
- gate exit code: 0
- ........................................................................ [ 31%]
  ........................................................................ [ 38%]
  ........................................................................ [ 44%]
  ........................................................................ [ 51%]
  ........................................................................ [ 57%]
  ........................................................................ [ 63%]
  ........................................................................ [ 70%]
  ........................................................................ [ 76%]
  ........................................................................ [ 83%]
  ........................................................................ [ 89%]
  ........................................................................ [ 95%]
  ..............................................                           [100%]
  1126 passed in 10.69s
  GATE: PASS
## Deviations
- docs/05 §2 lists queue entry fields `{job_id, kind, enqueued_at}`; the entry now also carries an optional `payload` JSON field. Card 36 fix 1 explicitly mandates it (`enqueue_job(job_id, "translate", payload={"target_langs": [...]})`), so it is implemented as specified.
- `tests/contract/test_mcp_tools.py` was edited although it is not in the card's Files list: it asserted the now-deleted `translate-request.json`. Leaving it untouched would keep the gate red; only that assertion changed (side file -> entry payload).
- `tests/unit/test_worker.py` (existing translate test) and `tests/unit/test_assemble.py` (existing translate report test) were adapted where they asserted the side file / listed languages without subtitle files, as the card allows; both adaptations strengthen the assertion.
- `TRANSLATE_REQUEST_FILENAME` was removed from `analysis/translate.py` (and its `__all__`) together with the side file, per fix 1.
## NOT proven
- `tests/live/test_queue_live.py` was written but not run (the gate excludes live tests; the judge runs it).
- The real REST(`de`)+MCP(`ar`) back-to-back race against live services was not executed here; it is covered by the worker end-to-end test with the same three real stages and fake Redis/LLM.
- Translation quality for target-script names was not checked against the real model; only the prompt text that requests it is asserted.
## Open
- none
