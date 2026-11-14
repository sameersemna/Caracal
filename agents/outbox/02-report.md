STATUS: DONE
## What changed
- `scripts/check_rules.py`: code rules now run on an AST `NodeVisitor` with import-alias resolution; raw-line regexes kept only for TODO/FIXME/XXX, `pass  # stub` and the `"caracal:"` literal; parse failures report `syntax-error`.
- `src/caracal/log.py`: added the three S8 masking paths (credential key-names, text `name=value` / `name: value` / `"name": "value"` + Bearer/Basic/Digest, token shapes + registered Settings values); `configure_logging(level, secrets=())` registers exact values ≥ 8 chars.
- `src/caracal/cli.py`: `_doctor` passes every loaded `Settings` `SecretStr` value to `configure_logging`.
- `src/caracal/config.py`: `DATABASE_URL` scheme matching is case-insensitive; output is always lower-case `postgresql+asyncpg://`.
- `pyproject.toml`, `uv.lock`: pinned `python-multipart==0.0.33`.
- `tests/unit/test_check_rules.py`, `tests/unit/test_log.py`, `tests/unit/test_config.py`, `tests/unit/test_cli.py`: appended new tests; no existing test was changed.
## Tests added
- `test_a1_from_os_import_environ_and_getenv_is_reported`: `from os import environ, getenv` alias import is a hit (A1 case 1).
- `test_a1_getenv_call_is_reported`: bare `getenv(...)` use is a hit (A1 case 2).
- `test_a1_asyncio_create_subprocess_shell_is_reported`: `asyncio.create_subprocess_shell` is a hit (A1 case 3).
- `test_a1_subprocess_run_with_string_on_the_next_line_is_reported`: multi-line `subprocess.run("ls -la")` is a hit (A1 case 4).
- `test_from_os_import_alias_is_reported`, `test_from_subprocess_import_alias_is_reported`, `test_subprocess_module_alias_is_reported`, `test_os_module_alias_environment_access_is_reported`, `test_asyncio_module_alias_create_subprocess_shell_is_reported`: `as` aliases are resolved.
- `test_os_popen_is_reported`, `test_os_putenv_call_is_reported`, `test_from_os_import_putenv_is_reported`: `os.popen` / `os.putenv` coverage.
- `test_file_that_does_not_parse_is_a_syntax_error`: unparsable file yields `syntax-error`.
- `test_legal_snippets_are_clean`: `asyncio.create_subprocess_exec("ffmpeg", "-i", path)` and `settings.HF_TOKEN` produce no hits.
- `test_a2_extra_api_key_is_masked`, `test_a2_extra_token_is_masked`, `test_a2_extra_password_is_masked`: key-name masking on top-level extras.
- `test_a2_message_access_token_is_masked`, `test_a2_message_password_is_masked`: `name=value` text masking with prefixes.
- `test_a2_message_x_api_key_header_is_masked`, `test_a2_message_x_subscription_token_header_is_masked`: Exa/Brave header colon forms.
- `test_a2_hf_token_shape_is_masked`, `test_sk_token_shape_is_masked`: known token shapes.
- `test_a2_nested_x_api_key_header_is_masked`: recursive key-name masking in nested dicts.
- `test_json_name_value_form_is_masked`: `"name": "value"` JSON form.
- `test_registered_setting_secret_value_is_masked`, `test_registered_secret_shorter_than_eight_characters_is_ignored`: exact registered values, 8-char threshold.
- `test_token_count_extra_is_not_masked`, `test_token_metric_suffixes_are_not_masked`: `token`-metric exception (`_count`, `_counts`, `_usage`, `_limit`).
- `test_database_url_scheme_is_case_insensitive`, `test_unknown_database_scheme_is_rejected`: `POSTGRESQL://` / mixed case normalised, unknown schemes still rejected.
- `test_doctor_registers_loaded_secrets_for_exact_match_redaction`: `caracal doctor` masks a loaded `EXA_API_KEY` value in a later log line.
## Measured
- gate exit code: 0
- ```
==> ruff check
All checks passed!
==> ruff format --check
13 files already formatted
==> mypy (strict)
Success: no issues found in 5 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 45%]
........................................................................ [ 91%]
..............                                                           [100%]
158 passed in 0.22s
GATE: PASS
```
## Deviations
- docs/06 S8a says any key containing `token` is masked, but per card 02 `token_count` must stay visible. Implementation exempts keys containing `token` that end in `_count`, `_counts`, `_usage` or `_limit` (only the `token` marker; other credential markers still mask). Please confirm this exception into docs/06 S8a.
## NOT proven
- I did not run the judge's literal A1/A2 scripts (not stored in the repo); I reconstructed both from verdict 01 and reproduced the expected hits, and the new tests encode each case.
- No live-service behavior was exercised (none applies to T0.1; the gate excludes `live`).
- Redaction is pattern-based and does not attempt encoded/obfuscated secrets (base64, secrets split across log calls); docs/06 S8 does not require it.
## Open
- docs/07 §2 does not name rules for `os.popen` and `asyncio.create_subprocess_shell`; I used `os-popen` and `async-subprocess-shell`, and kept `os-environ` for `os.environ`/`getenv`/`putenv`. Confirm or rename.
