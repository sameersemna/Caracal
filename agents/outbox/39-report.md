STATUS: DONE
## What changed
- `src/caracal/ops.py`: new module — injectable doctor probes (config, Postgres/pgvector/alembic head, Redis, whisper, Ollama, ffmpeg, yt-dlp/node/ejs, pyannote/CUDA/HF, search keys/`--deep`, disk), retention policy, Prometheus `MetricsRegistry` over `REDIS_PREFIX` hash keys.
- `src/caracal/cli.py`: `caracal doctor [--json] [--deep]` (required-vs-warn exit codes, redacted one-line-per-check stderr output) and `caracal retention [--dry-run]` (counts + bytes).
- `src/caracal/worker.py`: immediate-then-daily retention loop under `Keys.lock_retention()`; best-effort job/stage metrics after each pipeline delivery.
- `src/caracal/app.py`: bearer-protected `GET /metrics` (Prometheus text, no third-party library); `/metrics` added to the auth middleware path set.
- `src/caracal/db/repo.py`: `UploadRepo.delete(id)` protocol + idempotent Postgres implementation.
- `tests/fakes/repo.py`: fake upload delete, contract case support, and time-backdating helpers.
- `tests/repo_contract.py`: `case_upload_delete` (row gone, second delete idempotent).
- `tests/unit/test_ops.py`: doctor matrix, redaction, retention age/dry-run/upload/path-escape, metrics text, worker retention lock and metrics recording.
- `tests/unit/test_cli.py`: doctor expectations adapted to line output/required exit codes; retention CLI output.
- `tests/contract/test_metrics.py`: `/metrics` 401/200 and parsed counter values.
- `tests/live/test_pipeline_live.py`: repo bundle now passes `speakers=PostgresSpeakerRepo(...)` (stale since T2.3).
- `deploy/caracal-api.service`, `deploy/caracal-worker.service`, `deploy/README.md`: hardened systemd units and owner-run install/log commands.
- `scripts/live-suite.sh`: one-command throwaway-schema/prefix/data-dir runner for `pytest -m live tests/live`, PID-based `.venv/bin/caracal`, `trap` cleanup, pytest exit code.
- `README.md`, `.env.example`: full usage/install/MCP/privacy/limitations docs and the missing `CARACAL_ALLOWED_HOSTS` variable.
## Tests added
- `test_doctor_all_ok_sets_required_flags_and_exit_zero`: check order, required set, exit 0.
- `test_doctor_required_failure_exits_one` / `test_each_probe_failure_is_reported_with_its_required_flag`: each probe fail path and required-vs-optional exit semantics.
- `test_doctor_passes_the_deep_flag_to_the_search_probe`: `--deep` reaches the search probe.
- `test_emit_checks_redacts_secrets` / `test_emit_checks_json_is_one_object_per_line`: no DSN password/token in output; JSONL shape `{check,status,detail}`.
- `test_retention_deletes_only_old_job_media`: deletes only `audio.wav`/`source/`/`chunks/` of expired jobs, keeps transcript/subtitles/report/stages, respects age.
- `test_retention_dry_run_counts_but_deletes_nothing`: dry run reports 3 files/18 bytes and leaves them.
- `test_retention_deletes_only_stale_unconsumed_uploads`: 25 h unconsumed upload file+row removed; fresh and consumed kept.
- `test_retention_refuses_a_symlink_that_escapes_the_data_dir`: raises `internal`, target outside survives.
- `test_worker_retention_pass_holds_the_singleton_lock`: retention runs with the lock value visible.
- `test_worker_records_job_and_stage_metrics`: job total and stage count land in Redis.
- `test_metrics_render_is_prometheus_text` / `test_metrics_empty_registry_still_reports_global_totals`: every line `name{labels} value`, exact expected values.
- `test_retention_dry_run_prints_counts_and_bytes` / `test_retention_reports_deletions_without_dry_run`: CLI wording and `--dry-run` propagation.
- Contract `test_metrics_requires_a_bearer_token` / `test_metrics_renders_shared_counters`: 401 without bearer, 200 and seeded values with bearer.
## Measured
- gate exit code: 0
```
........................................................................ [ 29%]
........................................................................ [ 35%]
........................................................................ [ 41%]
........................................................................ [ 47%]
........................................................................ [ 53%]
........................................................................ [ 59%]
........................................................................ [ 65%]
........................................................................ [ 71%]
........................................................................ [ 77%]
........................................................................ [ 83%]
........................................................................ [ 89%]
........................................................................ [ 94%]
.............................................................            [100%]
1213 passed in 10.81s
GATE: PASS
```
- Non-vacuity spot checks: `doctor_exit_code` forced to 0 → 6 doctor tests failed; retention age comparison removed → the age test failed; `/metrics` removed from the auth middleware → the 401 contract test failed. All restored.
- Manual `caracal doctor` run (real shell env): config/ffmpeg/ytdlp/disk ok, pyannote warn (HF_TOKEN unset), search warn (no keys), Postgres/Redis/whisper/Ollama fail (that shell points at `127.0.0.1:9` / an unreachable whisper/Ollama); exit 1.
## Deviations
- `run_retention` takes injected dependencies: `run_retention(settings, repos, *, now=None, dry_run=False)` instead of a bare `run_retention(now, *, dry_run)`. Reason: the worker/CLI need the repos and `docs/01` does not fix the signature; behaviour matches docs/05 §3.
- Doctor and retention output goes to **stderr** (default human line or JSONL with `--json`), not stdout: `docs/01`/`docs/07` reserve stdout for MCP stdio and `check_rules.py` forbids `print` in `src/`.
- `UploadRepo.delete` is idempotent (missing id is not an error), unlike `JobRepo.delete`. Reason: two workers racing the daily retention singleton must not fail the pass; no doc fixes the error behaviour.
- Ollama is one required check that passes when **either** cloud or local works (warn when only one path works), matching "Ollama (cloud or local)" in the card; the local `LLM_FALLBACK_MODEL` and cloud `GET /api/tags` are both probed.
- `format_whitelist` support is detected from `ffmpeg -h full` (read-only) rather than invoking the flag.
- `docs/01 §3` already lists `ops.py` (line 58), so no docs edit was needed.
## NOT proven
- `scripts/live-suite.sh` was syntax-checked (`bash -n`) but not executed: the card says the live runner is not run in the gate and this shell points at unreachable datastores/servers. `pytest -m live tests/live` was not run.
- Success branches of the Postgres/Redis/whisper/Ollama doctor probes were not measured here because those services are unreachable from this environment; only their failure branches and the ffmpeg/yt-dlp/pyannote/disk/config paths ran.
- LLM/search/whisper counters are rendered by `/metrics` and have registry methods, but their production call sites (`llm/client.py`, `search/router.py`, `asr/whisper.py`) are outside this card's Files list, so only job and stage counters are incremented by the worker.
- Worker retention cadence: the lock + one pass are unit-tested; the 24 h sleep was not waited for.
## Open
- Confirm stderr is the intended doctor/retention output stream for the judge's tooling (stdout would be available for `doctor`/`retention`, but conflicts with the repository's stdout rule).
- Should `/metrics` also expose build/version or worker liveness gauges? Not in the card; not added.
