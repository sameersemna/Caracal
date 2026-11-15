STATUS: DONE
## What changed
- `src/caracal/config.py`: `Settings.model_config` gains `hide_input_in_errors=True` (validation errors cannot echo settings input, which may be a secret).
- `tests/live/test_llm_live.py`: new `_settings()` helper reconstructs a validated `Settings` from `get_settings().model_dump()` with `_env_file=None` and overwrites `LLM_ALLOWED_MODELS` plus **every** `LLM_ROUTE_*` explicitly; all five live tests use it, and the fallback test configures `[local:nonexistent-model:1b, local:gemma4:31b]` as both its chain and allow-list.
- `tests/unit/test_config.py`: new test plants a secret-looking value in a mistyped field and asserts it is absent from `str`/`repr` of the `ValidationError` while the field name is still reported.
## Tests added
- `test_validation_errors_never_echo_the_settings_input`: a planted secret (`sk-live-planted-secret-...`) used as an invalid `CARACAL_PORT` does not appear in the error text; verified non-vacuous by removing `hide_input_in_errors` (test then fails with the secret quoted in `input_value`).
- Live isolation is exercised by the existing five `tests/live/test_llm_live.py` tests, now routing-explicit.
## Measured
- gate exit code: 0
```
........................................................................ [ 91%]
........................................................................ [ 97%]
..........................................                               [100%]
1410 passed in 12.25s
GATE: PASS
```
- `tests/live/test_llm_live.py -m live` with the owner's `.env` routing (`LLM_ALLOWED_MODELS=gemma4:31b-cloud,local:gemma4:31b`, `LLM_ROUTE_DEFAULT` same): **5 passed in 76.89s**.
- Same file with routing removed (process env `LLM_ALLOWED_MODELS=` / `LLM_ROUTE_DEFAULT=` blank, so legacy derivation applies): **5 passed in 128.99s**.
- Non-vacuity probe: without `hide_input_in_errors`, `pytest -k never_echo` fails quoting `'sk-live-planted-secret-0123456789' is contained here`; restored afterwards.
## Deviations
- none (only the three Files entries plus this report were touched).
## NOT proven
- The live runs above had to unset `OLLAMA_CLOUD_URL`, `OLLAMA_LOCAL_URL` and `OLLAMA_CLOUD_API_KEY` in the test process because this agent session exports both Ollama URLs as `http://127.0.0.1:9` (env beats `.env`). The judge's shell may or may not carry that leftover; the tests themselves read the real `.env` once those are unset.
- The live runs only had `LLM_ROUTE_DEFAULT` set in `.env`; a `.env` with a distinct per-task route (e.g. `LLM_ROUTE_QA`) was not exercised live. Isolation is structural (the helper overwrites all eight route fields and bypasses the env file), but it is argued, not measured.
## Open
- none.
