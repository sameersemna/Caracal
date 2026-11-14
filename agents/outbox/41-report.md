STATUS: DONE

## What changed
- `src/caracal/analysis/qa.py`: `Answer.answer` prose is passed through `strip_prose_citations`; the ids stay in `citations`.
- `src/caracal/ops.py`: migration probe returns one `fail - database not migrated — run 'caracal migrate'` line for a missing/empty/stale `alembic_version`; added `incr_stage_failure` and `caracal_stage_failures_total` rendering.
- `src/caracal/cli.py`: doctor silences `httpx`/`httpcore` INFO (restoring levels afterwards); worker gained `--worker-id`, `--reclaim-after-s`, `--reclaim-every-s` (defaulting to production cadence).
- `src/caracal/llm/client.py`: injectable `LlmMetrics`; one `caracal_llm_calls_total{endpoint,mode,fallback}` increment per successful `call_tool`/`chat`.
- `src/caracal/pipeline/stages.py`: `MetricsSink` injected via `StageContext`; transcribe adds each chunk duration to `caracal_whisper_audio_seconds_total`.
- `src/caracal/pipeline/runner.py`: carries the metrics sink and increments `caracal_stage_failures_total{stage}` on a failed stage.
- `src/caracal/worker.py`: wires the shared `MetricsRegistry` into `LLMClient`, `RunnerDeps` and every stage context.
- `src/caracal/api/routes.py` + `schemas.py` + `services.py`: bearer-protected `GET /v1/jobs/{id}/voiceprints` returning `{"items":[{label, embedding:[256 floats]}]}` from the diarize checkpoint (REST only).
- `README.md`: documented the voiceprint calibration export in Privacy and added the stage-failure counter.
- `pyproject.toml` / `uv.lock`: dev dependency `hypothesis` (6.170.15 pinned by the lock).

## Tests added
- `tests/unit/test_qa.py::test_inline_citation_tokens_are_stripped_from_the_answer_prose`: inline `[seg:n]` disappears from prose while citations/evidence keep the segment.
- `tests/unit/test_ops.py`: missing/stale revision mapping, DBAPI "relation alembic_version does not exist" mapping (and unrelated errors not hidden), stage-failure metric rendering.
- `tests/unit/test_cli.py`: `_quiet_http_logs` suppresses and restores levels; worker recovery flags build intervals.
- `tests/unit/test_llm_client.py`: one metric per successful call, fallback endpoint/mode labels, metric failures never fail a call.
- `tests/unit/test_stages.py::test_transcribe_adds_each_chunk_duration_to_the_whisper_metric`: adds 1.0 s for a one-second chunk (not the fixture's ~11 s).
- `tests/unit/test_runner.py::test_failed_stage_increments_the_stage_failures_metric`: a failed non-critical stage increments the counter.
- `tests/unit/test_worker.py::test_worker_wires_the_metrics_registry_into_stages_and_the_llm`: production wiring, not just the classes.
- `tests/contract/test_voiceprint_export.py`: shape/sorted labels/256 floats, empty skipped checkpoint, 401/404/409/500, no-MCP parity via `test_parity`.
- `tests/unit/test_fuzz_inputs.py`: bounded deterministic property suites for `resolve_and_check`, `sanitize_filename`, `extract_json`, `coerce`+`model_validate`, `md_escape`, `parse_vtt` and the segmenter invariants.
- `tests/unit/test_concurrency.py`: two pipelines + one translate at concurrency 2, real Redis whisper lock recorded at peak 1, mid-run cancel leaves one `cancelled` and the others `done`.
- `tests/live/test_resilience_live.py` (judge-run): two fast-recovery workers, `SIGKILL` of the worker owning the entry during transcribe, recovery to `done` with a valid report.
- `tests/live/test_mcp_stdio_live.py` (judge-run): real `caracal mcp-stdio`, initialize/tools/list calls, stdout is only JSON-RPC.

## Measured
- gate exit code: 0
- ```
  ........................................................................ [ 34%]
  ........................................................................ [ 39%]
  ........................................................................ [ 45%]
  ........................................................................ [ 51%]
  ........................................................................ [ 57%]
  ........................................................................ [ 62%]
  ........................................................................ [ 68%]
  ........................................................................ [ 74%]
  ........................................................................ [ 79%]
  ........................................................................ [ 85%]
  ........................................................................ [ 91%]
  ........................................................................ [ 96%]
  .......................................                                  [100%]
  1263 passed in 11.64s
  GATE: PASS
  ```

## Deviations
- `cli.py`/`worker.py`: added `--worker-id`, `--reclaim-after-s`, `--reclaim-every-s` so the live resilience test can use a 5 s reclaim instead of the production 300 s. Defaults are unchanged; card listed both files.
- `hypothesis` is declared as `>=6.170.15` with the exact version pinned in `uv.lock`, matching the card's "latest, pinned by uv.lock" wording.
- The REST-only voiceprint parity exception is recorded in `tests/contract/test_parity.py` (`rest_only=True`, reason) and README; docs/03 §0 was not edited (docs are immutable for workers).

## NOT proven
- The live tests were written but not executed here (no live network/DB/Redis calls allowed in this session); the judge runs them.
- Crash recovery while the killed worker is *inside* the whisper HTTP call is not covered: a dead process leaves the documented 16-minute `lock:whisper` TTL, which would stall takeover. The live test makes transcribe block on a test-held lock, so the killed worker holds no Redis lock; this still proves queue recovery, but not stale-lock expiry.
- `/metrics` after a real pipeline run was not measured; wiring is proven by unit tests and will show in the judge's eval.

## Open
- Should the whisper single-slot lock get a shorter TTL or a dead-owner takeover, so a crash inside the request recovers faster? This would require changing `queue.py`, which this card did not list, and docs/05 §2 documents `PX 960000`.
- `docs/03` is untouched; the parity exception lives in the test table and README until the judge promotes it into the doc.
