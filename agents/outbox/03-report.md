STATUS: DONE

## What changed
- `src/caracal/config.py`: added `CARACAL_DB_SCHEMA` (default `public`) with the `^[a-z_][a-z0-9_]{0,62}$` identifier validator.
- `src/caracal/db/tables.py`: schema-less Core `Table`s `jobs`, `job_stages`, `uploads`, `reports`, `segments` with enum CHECKs, cascading FKs, `segments.tsv` stored generated column + GIN index.
- `src/caracal/db/engine.py`: `make_engine(settings)` (asyncpg, `pool_pre_ping`, `search_path`) and transactional `session_scope(engine)`.
- `src/caracal/db/repo.py`: `JobRepo`/`UploadRepo`/`ReportRepo`/`SegmentRepo` protocols, frozen records, opaque base64 keyset cursors, and the Postgres implementations.
- `src/caracal/db/__init__.py`: package marker.
- `alembic.ini`, `src/caracal/migrations/{env.py,script.py.mako,versions/0001_initial.py}`: async alembic env that reads URL/schema from `get_settings()`, handwritten initial migration (starts with `CREATE EXTENSION IF NOT EXISTS vector`, `downgrade()` drops the five tables only).
- `src/caracal/cli.py`: implemented `migrate` (`alembic.command.upgrade(config, "head")`), 0/1 exit codes, JSON logging with redaction; factored `_invalid_config_fields`.
- `pyproject.toml`: `[tool.pytest.ini_options] pythonpath = ["."]` so `tests.fakes` / `tests.repo_contract` import.
- `tests/fakes/repo.py`: in-memory implementations of all four protocols over one shared store (cursor/transition/consume semantics identical; search ranking approximated by term-count, documented).
- `tests/repo_contract.py`: 13 plain async `case_*` functions covering every method and every required error path, plus a `Repos` bundle with a cleanup hook.
- `tests/unit/test_repo_fake.py`: runs every contract case against the fakes.
- `tests/unit/test_db_tables.py`: compiles PG DDL and asserts columns, types, CHECK sets, computed `tsv`, GIN index, cascades, no `voiceprints`, no `start`/`end` columns.
- `tests/unit/test_config.py`, `tests/unit/test_cli.py`: added schema-validation and migrate tests.
- `tests/live/test_db.py`: `@pytest.mark.live`, throwaway `caracal_t_<8 hex>` schema, programmatic `upgrade head` → same contract cases → `downgrade base` → `DROP SCHEMA … CASCADE`; separate upgrade/downgrade round-trip test.

## Tests added
- `case_job_create_and_get`: created record fields/defaults, read-back equality, unknown id `None`.
- `case_job_create_validates_source_kind`: `source_kind` enum rejected as `invalid_input`.
- `case_job_list_pagination`: keyset pages are disjoint/complete, cursor ends, limit clamped to 1..100.
- `case_job_list_status_filter`: status filter includes/excludes correctly.
- `case_job_list_invalid_cursor`: five malformed cursors (empty, non-base64, `{}`, JSON list, bad values) all `invalid_input`.
- `case_job_status_transitions`: running→done sets `finished_at`; every terminal transition raises `invalid_input`; unknown id `not_found`; failure stores the error object.
- `case_job_progress_and_title`: progress/title updates persist; unknown ids `not_found`.
- `case_job_stage_lifecycle`: pending→running (attempt 1)→running (attempt 2)→failed (`ended_at`)→running (attempt 3, `ended_at` cleared)→done; skipped timestamps; unknown job/invalid status errors.
- `case_find_recent_done`: only done jobs, 7-day window (0 excludes), identical options, sha256 or source_url; no-key call returns `None`.
- `case_delete_cascades`: delete removes job, stages, report and segments; second delete `not_found`.
- `case_upload_lifecycle`: create/get; consume sets `consumed_by`; double consume `invalid_input`; unknown `not_found`; unconsumed-older-than filtering.
- `case_report_upsert`: insert, overwrite (schema_version + body), unknown job `not_found`, missing `None`.
- `case_segments_replace_and_search`: FTS hit joins title and returns rank/`start_s`; multi-term query; empty/no-match query; replace removes old rows; limit; mismatched row job id `invalid_input`; unknown job `not_found`.
- `test_db_schema_*`: default/valid/invalid/env parsing for `CARACAL_DB_SCHEMA`.
- `test_migrate_*`: programmatic head upgrade, 0 on success, 1 with a redacted error on failure, 1 without running alembic when config is invalid.
- (manual non-vacuity probe, restored afterwards) removing the fake's attempts increment or terminal guard makes the relevant contract case fail.

## Measured
- gate exit code: 0
```
==> ruff check
All checks passed!
==> ruff format --check
27 files already formatted
==> mypy (strict)
Success: no issues found in 11 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 36%]
........................................................................ [ 72%]
......................................................                   [100%]
198 passed in 0.35s
GATE: PASS
```
- `alembic upgrade head --sql` (offline, no DB) renders `CREATE EXTENSION IF NOT EXISTS vector`, all five tables, indexes, and a schema-qualified `alembic_version` insert; this exercised `env.py` and `0001_initial.py`.

## Deviations
- `tests/unit/test_cli.py` had to be changed, not only extended: the pre-existing `test_unimplemented_subcommands_exit_two_with_stderr_message` parametrized `migrate` as an unimplemented stub (`exit 2`, "available from a later milestone"). Card requirement 7 makes that impossible: `migrate` now runs alembic and returns 0/1. I removed only `"migrate"` from that list; the test still covers `serve`, `worker`, `mcp-stdio`, and three new migrate tests were added. This is the only existing test assertion I touched.
- Fake search approximates ranking by term-match count (allowed and documented in `tests/fakes/repo.py`); membership/filtering/limit are identical to `websearch_to_tsquery` for the simple term queries in the contract.
- `SegmentRow` omits the generated `tsv` column (it cannot be written); every writable table column is present.
- `upsert_stage` clears `ended_at` when a stage re-enters `running` (a new attempt is not "ended"); requirement 1 only says `ended_at` is set on done/skipped/failed. Locked by `case_job_stage_lifecycle`.
- `list_stages` for an unknown job returns `[]` rather than raising: it is a list getter, and docs/05/03 do not define an error for that case.
- `find_recent_done` uses the DB clock (`now() - make_interval(...)`) and JSONB equality for "identical options" (docs/05 §1 dedup).

## NOT proven
- The live Postgres path was not executed by me (the card says live tests are run by the judge). It is exercised only by `tests/live/test_db.py`; I validated the migration/env path offline via `alembic upgrade head --sql`.
- The Postgres repository implementations have no unit-level execution path (by card design); the shared cases run against the fake in the gate and against Postgres only in the live suite.
- `.env.example` still lacks `CARACAL_DB_SCHEMA` (that file is not in this card's Files list).

## Open
- Should a live/enrollment card add `CARACAL_DB_SCHEMA` to `.env.example` so docs/02's "lists every variable" holds? Owner/boss call.
- Confirm that re-entering `running` must reset `ended_at` (I chose yes, see Deviations).
- Confirm `list_stages(unknown_job)` should stay `[]` rather than `not_found`.
