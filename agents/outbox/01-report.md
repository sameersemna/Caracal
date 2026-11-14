STATUS: DONE

## What changed
- `pyproject.toml`: uv/hatchling project, exact-pinned runtime deps, dev group, ruff/mypy/pytest config (docs/02, docs/07 §1).
- `uv.lock`: resolved lockfile for all runtime and dev dependencies.
- `src/caracal/__init__.py`: `__version__ = "0.1.0"`.
- `src/caracal/config.py`: `Settings(BaseSettings)` with every docs/02 variable, required token (>=32), CSV parsing via `NoDecode`, optional-secret empty->None, Ollama URL / `DATABASE_URL` / `REDIS_PREFIX` validators, cached `get_settings()`.
- `src/caracal/log.py`: JSON-lines stderr formatter (`ts, level, logger, msg` + extras), `redact()` for bearer/`api_key=`/`token=`/`key=`/DSN passwords, idempotent `configure_logging()`.
- `src/caracal/errors.py`: `ErrorCode` StrEnum, full `HTTP_STATUS` table, `CaracalError` with `code/message/details/retryable` and `to_dict()`.
- `src/caracal/cli.py`: argparse `serve|worker|mcp-stdio|migrate|doctor`; stubs exit 2 with the required stderr line; `doctor` validates settings via the JSON logger and returns 0/1 without network calls.
- `scripts/gate.sh`: docs/07 §1 steps in order, fail-fast, prints `GATE: PASS`.
- `scripts/check_rules.py`: all docs/07 §2 rules including the docs/01 §3 layout check (dirs, `dir/**`, `{a,b}`, globs, `__init__.py` exception, `.py` only).
- `.env.example`: every docs/02 variable with defaults/placeholders; parses cleanly in an empty environment.
- `.gitignore`: kept `.env`, appended `.venv/`, `data/`, `agents/logs/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`.
- `README.md`: short Caracal description + how to run the gate.
- `tests/conftest.py`: autouse env/cwd isolation, settings cache reset, `make_settings` factory.
- `tests/unit/test_config.py`, `test_log.py`, `test_errors.py`, `test_check_rules.py`, `test_cli.py`: the tests required by the card's Accept section.

## Tests added
- `test_config.py`: missing/31-char/empty token rejected, 32 accepted; CSV parsing from kwargs and from the environment (proves `NoDecode`); empty optional secrets -> None; >=30 docs/02 defaults incl. `CARACAL_PORT == 11404` and `LLM_MODEL == "gemma4:31b"`; Ollama cloud/local normalisation; Postgres scheme normalisation preserving credentials/query; non-Postgres URL rejected; `REDIS_PREFIX` append/empty rules; `get_settings` caching.
- `test_log.py`: every redact pattern (bearer, `api_key=`, `token=`, `key=`, `api-key=`, DSN), idempotence; one JSON line with exact `level/logger/msg/ts` and redacted extras; stdout stays empty; duplicate `configure_logging` does not duplicate output; level filtering; unknown level rejected.
- `test_errors.py`: exact `ErrorCode` value set; every code -> docs/03 §4 HTTP status; `to_dict()` envelope; default `retryable`; details copied; `retryable` is keyword-only.
- `test_check_rules.py`: each of the 19 line rules reported at the exact line (incl. `run("...")` string arg, all queue imports, Redis safety); clean file passes; unlisted module reported; `api/routes.py` and `migrations/**` pass; `__init__.py` allowed/denied correctly; non-`.py` ignored; `config.py` exemption; brace expansion, `dir/**`, trailing-slash dirs; CLI exit 0/1/2 and output format; own `src/caracal` is clean.
- `test_cli.py`: `--help` lists all five subcommands; four stubs return 2 with exactly the required stderr text and empty stdout; no command -> usage + 2; `doctor` returns 0 with "config valid" JSON, 1 with "config invalid" naming the field, and never leaks the token.

## Measured
- gate exit code: 0
- clean clone of `task/T0.1-scaffold` in `/tmp`, `uv sync`, then gate:
```
==> ruff check
All checks passed!
==> ruff format --check
13 files already formatted
==> mypy (strict)
Success: no issues found in 5 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 60%]
................................................                         [100%]
120 passed in 0.14s
GATE: PASS
```
- `git check-ignore .env` succeeds; `.env` is absent from both commits.
- Mutation spot-checks in the throwaway clone: removing DSN redaction fails 3 log tests; removing the `print(` rule fails 2 checker tests; removing the `REDIS_PREFIX` colon append fails 3 config tests.

## Deviations
- docs/07 §2 says a token is a path if it "contains `*`". Taken literally this makes the description token `key(*parts),` in docs/01 a glob. `_is_path_token` additionally requires a path-like shape for `*` tokens (letters, digits, `. / { } , - *`), so real patterns still work and descriptions do not become patterns. No doc change needed.
- Dev tools were added with `uv add --dev` (lower bounds in `pyproject.toml`); the card says they are pinned exactly by `uv.lock`, which they are. `python-multipart` is unpinned per the card's "latest".
- `scripts/gate.sh` runs `mkdir -p tests/unit tests/contract` because docs/07 §1 names `tests/contract` but this card's Files list has no file there yet; `pytest` needs the path to exist.
- Added `asyncio_default_fixture_loop_scope = "function"` to `[tool.pytest.ini_options]` to silence pytest-asyncio's deprecation warning; it does not change test behaviour.

## NOT proven
- Real reachability checks in `caracal doctor` (Postgres, Redis, whisper, Ollama, HF, CUDA, search keys) are intentionally absent: the card scopes doctor to config validation only and forbids network calls in this round.
- No live-service or MCP/FastAPI contract tests; those belong to later cards. The gate touches no network.
- `uv sync` was verified from a local clone with the uv cache warm; no cold-cache install was measured.

## Open
- None.
