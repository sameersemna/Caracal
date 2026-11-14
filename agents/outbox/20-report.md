STATUS: DONE

## What changed
- `src/caracal/db/engine.py`: added `search_path()` / `server_settings()` so every engine connection uses `"public"` or `"<schema>, public"`; `make_engine` now uses them.
- `src/caracal/migrations/env.py`: the alembic async engine uses the same `server_settings()` helper, so migrations in any non-public schema keep pgvector visible.
- `src/caracal/migrations/versions/0002_voiceprints.py`: `_require_vector_type()` runs `CREATE EXTENSION IF NOT EXISTS vector` and raises `RuntimeError("pgvector not visible on search_path: ...")` when `to_regtype('vector')` is NULL; skipped for offline `--sql` generation.
- `tests/unit/test_db_engine.py`: new tests for exact search-path values in the helper, `make_engine`, alembic `env.py`, and the migration guard.

## Tests added
- `test_search_path_keeps_public_visible[public|caracal_t_x]`: exact `"public"` / `"caracal_t_x, public"` strings.
- `test_server_settings_are_exact[public|caracal_t_x]`: exact asyncpg `server_settings` dict for both schemas.
- `test_make_engine_passes_exact_server_settings[public|caracal_t_x]`: `make_engine` forwards URL, `pool_pre_ping=True` and the exact `connect_args`.
- `test_migrations_env_pins_the_same_search_path[public|caracal_t_x]`: `env.run_async_migrations()` builds the engine with `poolclass=NullPool` and the same exact `search_path`, then disposes it.
- `test_require_vector_type_raises_when_the_type_is_invisible`: guard executes `CREATE EXTENSION` and raises on NULL `to_regtype`.
- `test_require_vector_type_accepts_a_resolvable_vector_type`: guard runs the regtype query and passes when the type resolves.
- `test_require_vector_type_skips_the_check_for_offline_sql`: offline `--sql` mode emits the extension DDL without requiring a bind.

## Measured
- gate exit code: 0
```
==> pytest (unit + contract, live excluded)
........................................................................ [  8%]
........................................................................ [ 17%]
........................................................................ [ 26%]
........................................................................ [ 35%]
........................................................................ [ 43%]
........................................................................ [ 52%]
........................................................................ [ 61%]
........................................................................ [ 70%]
........................................................................ [ 78%]
........................................................................ [ 87%]
........................................................................ [ 96%]
.............................                                            [100%]
821 passed in 8.60s
GATE: PASS
```

## Deviations
- none

## NOT proven
- `tests/live/test_db.py` was not run: per the worker rules the live layer belongs to the judge. The fix is covered by unit tests plus static SQL checks (unqualified `vector`/`vector_cosine_ops`/`<=>` now resolve through `public` on the search path).

## Open
- none
