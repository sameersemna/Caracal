STATUS: DONE

## What changed
- `src/caracal/queue.py`: `Keys` (one prefix helper, glob/whitespace rejection), `JobQueue` (MKSTREAM group at id 0, XADD MAXLEN ~10000, XREADGROUP, XCLAIM touch, XACK+XDEL, XAUTOCLAIM reclaim with XPENDING delivery counts), owner-token Redis lock (Lua renew/release), whisper/retention lock helper, event publish/subscribe, cancel flag (1 d), INCR+EXPIRE NX rate limit.
- `src/caracal/storage.py`: `JobFiles(data_dir, job_id)` for every docs/05 §3 path, lang/ext/stage-name validation (`invalid_input`), `ensure()`, atomic `write_json`/`write_text` (same-dir temp + fsync + `os.replace`), `read_json`.
- `src/caracal/pipeline/stages.py`: `Stage` (16 values, canonical order), `CRITICAL_STAGES`, `JobOptions` (pydantic, `extra="forbid"`, 20×200 hint limits, 2000-char context), plain `StageContext`, frozen `StageResult`, `StageFn`, `default_registry()` (all stages `skipped: not implemented in M0`).
- `src/caracal/pipeline/runner.py`: `RunnerDeps` + `run_pipeline()` with terminal no-op, queued→running, per-stage running/done/failed rows and events, checkpoint-before-row, resume (checkpoint **and** done|skipped row), retryable-only retries with sleeps `[2, 8]`, critical failure policy, non-critical continue, cancel between stages, `asyncio.timeout` → `internal: job timeout`, exact `ts`/`pct` on every event, progress callback.
- `src/caracal/worker.py`: `Worker` (bounded read loop, heartbeat `SET EX 30`, per-delivery touch task, ack after return, XAUTOCLAIM loop, max-deliveries failure, grace-period shutdown that leaves running jobs unacked), injectable `WorkerIntervals`, `RepoBundle`.
- `src/caracal/cli.py`: real `caracal worker` (asyncio redis client, engine, Postgres repos, `Worker`, SIGINT/SIGTERM → stop, clean client/engine close, exit 0).
- `src/caracal/config.py`: `CARACAL_RATE_LIMIT_PER_MIN` (default 30).
- `.env.example`: added `CARACAL_DB_SCHEMA` and `CARACAL_RATE_LIMIT_PER_MIN`.
- `pyproject.toml` / `uv.lock`: dev dependency `fakeredis[lua]==2.40.0`.
- `tests/fakes/redis.py`: fakeredis subclass recording every command and every `PUBLISH` channel.
- `tests/unit/test_{queue,storage,runner,worker}.py`, `tests/unit/test_config.py` (additions), `tests/unit/test_cli.py` (worker un-stubbed), `tests/live/test_queue_live.py`.

## Tests added
- `test_storage.py::test_every_path_derives_from_the_job_uuid`: exact docs/05 §3 paths.
- `test_storage.py::test_invalid_language_tags_raise_invalid_input` / `test_invalid_subtitle_extensions_raise_invalid_input` / `test_stage_names_must_be_stage_values`: path-parameter validation.
- `test_storage.py::test_write_json_is_atomic_and_round_trips` / `test_write_text_is_atomic_and_round_trips`: fsync is called, no temp file survives, content round-trips.
- `test_storage.py::test_write_json_keeps_previous_content_when_serialisation_fails`: failed serialisation cannot corrupt an existing artifact.
- `test_queue.py::test_key_prefixes_and_joins_parts` / `test_key_rejects_empty_glob_and_whitespace_parts`: every key is prefixed; `* ? [ ]` whitespace and empty parts rejected.
- `test_queue.py::test_enqueue_read_and_ack`: enqueue → read → touch → ack deletes the stream entry and the PEL entry.
- `test_queue.py::test_reclaim_reports_delivery_count`: XAUTOCLAIM delivery count comes from XPENDING (1 → 2).
- `test_queue.py::test_publish_and_subscribe_events`: both directions of the events channel.
- `test_queue.py::test_cancel_flag_round_trip`: TTL ≈ 1 day.
- `test_queue.py::test_rate_limit_allows_then_blocks`: counter allows `limit` requests and returns retry_after.
- `test_queue.py::test_whisper_lock_excludes_a_second_acquirer`: second acquirer waits and times out with `dependency_down`.
- `test_queue.py::test_lock_release_never_deletes_another_owners_token`: compare-and-DEL (fails if the Lua guard is removed).
- `test_queue.py::test_lock_renewal_extends_the_ttl`: background renewal keeps the TTL high.
- `test_queue.py::test_every_key_and_channel_uses_the_prefix`: `SCAN *` and recorded channels contain only `t3st:` names.
- `test_runner.py::test_stage_order_events_and_progress_percentage`: exact 3-stage event sequence and `pct` values.
- `test_runner.py::test_checkpoint_is_written_before_the_stage_is_marked_done`: observes the file at `upsert_stage(done)` time.
- `test_runner.py::test_resume_skips_done_stages_without_calling_them`: resumed stages never invoke their function.
- `test_runner.py::test_done_row_without_a_checkpoint_is_rerun` / `test_checkpoint_without_a_done_row_is_rerun`: resume requires both conditions.
- `test_runner.py::test_retryable_failure_is_retried_with_backoff`: 3 attempts, sleeps exactly `[2.0, 8.0]`.
- `test_runner.py::test_critical_stage_exhausted_retries_fail_the_job` / `test_non_retryable_failure_of_critical_stage_stops_immediately`: critical failure stops later stages and stores `{code,message,stage}`.
- `test_runner.py::test_non_critical_failure_continues_and_job_finishes_done`: failed non-critical row, job still `done`.
- `test_runner.py::test_cancel_between_stages_settles_cancelled`: cancel flag between stages → `cancelled` end event.
- `test_runner.py::test_job_timeout_fails_the_job`: `internal` / `job timeout`.
- `test_runner.py::test_terminal_job_is_a_noop` / `test_missing_job_raises_not_found`.
- `test_runner.py::test_progress_callback_updates_job_and_publishes_event`.
- `test_runner.py::test_default_registry_skips_all_stages_and_finishes_done`: all 16 stubs checkpoint `skipped`.
- `test_runner.py::test_job_options_*`: contract defaults, `extra="forbid"`, hint/context limits.
- `test_worker.py::test_worker_processes_a_job_end_to_end_and_cleans_the_stream`: job `done`, all stages `skipped`, stream entry deleted, heartbeat key present, prefix-only keys/channels.
- `test_worker.py::test_job_enqueued_before_the_worker_starts_is_delivered`: group created at id 0 delivers pre-existing jobs.
- `test_worker.py::test_dead_consumer_entry_is_reclaimed_and_resumed`: A reads and dies, B reclaims and resumes from A's checkpoints without re-running them.
- `test_worker.py::test_entry_over_max_deliveries_fails_the_job`: `internal: worker crashed repeatedly`.
- `test_worker.py::test_concurrency_limit_is_respected`: four jobs, two slots, observed peak exactly 2.
- `test_worker.py::test_shutdown_leaves_a_running_job_unacked`: grace period leaves the entry pending and the job running.
- `test_cli.py::test_worker_command_runs_the_worker` / `test_run_worker_builds_datastores_and_stops_cleanly`: CLI builds redis client, engine, Postgres repos and closes both.
- `tests/live/test_queue_live.py`: real Redis under `caracal_t_<8 hex>:` with SCAN cleanup of exactly that prefix (not run here).

## Measured
- gate exit code: 0
- ```
  ==> ruff check
  All checks passed!
  ==> ruff format --check
  39 files already formatted
  ==> mypy (strict)
  Success: no issues found in 17 source files
  ==> repository rules
  ==> pytest (unit + contract, live excluded)
  ........................................................................ [ 26%]
  ........................................................................ [ 52%]
  ........................................................................ [ 78%]
  ...........................................................              [100%]
  275 passed in 2.72s
  GATE: PASS
  ```

## Deviations
- `tests/unit/test_cli.py` was adjusted as the card allows: `worker` was removed from the "unimplemented subcommands" parametrization and two worker tests were added.
- `agents/outbox/04-report.md` is the only agents file committed. The pre-existing untracked `NOTES.md` (owner's local DB bootstrap notes, contains a password) was deliberately **not** committed, so `git add -A` was not used.
- Progress events carry the card's computed `pct` (finished stages / total × 100, 1 decimal) like every other event; the finer stage-supplied percentage is persisted through `repos.jobs.set_progress` as `{"pct","stage","message"}` (docs/03 `JobStatus.progress`).
- `JobQueue.ensure_group()` creates the group at id `0`, not `$`, so a job enqueued by the API before the first worker exists is still delivered. Docs/05 §2 requires MKSTREAM + BUSYGROUP handling but does not pin the id.
- `storage.JobFiles` exposes a few extra read-only directory properties (`chunks_dir`, `captions_dir`, `stages_dir`, `subtitles_dir`) in addition to the required paths; purely additive.
- `Worker.__init__` uses a module-level frozen `WorkerIntervals()` instance as the default for `intervals` (the card's `intervals: WorkerIntervals = default`), so tests can inject timings without mutating shared state.
- `cli._run_worker(settings)` is a small testable helper behind `_worker()`; the public CLI surface is unchanged.

## NOT proven
- `tests/live/test_queue_live.py` was written but **not run** (the card forbids running live tests). Real Redis behaviour (redis 8.1 XAUTOCLAIM/XPENDING/Lua) is therefore unmeasured here; fakeredis 2.40.0 with Lua backed every queue test.
- Crash recovery is simulated by leaving a fakeredis entry pending for consumer A and reclaiming it with worker B; no real `kill -9`, Postgres, signal handler, or multi-process interaction was exercised.
- The worker end-to-end path uses the M0 stub registry, so no real ingest/whisper/pyannote stage was executed.
- Retention (T4.1) is intentionally absent.

## Open
- Should the progress event's `pct` be the stage-supplied percentage rather than finished-stages/total? I followed the literal "every event ... pct (stages finished / total × 100)" sentence and persisted the finer value separately.
- Should `run_pipeline` treat a job already in `running` (crash resume) as needing a fresh `set_status(running)`? I only call it when the job was `queued`, then publish the running event; the row is already `running` after a crash.
