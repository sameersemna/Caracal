# 07 — Testing and the Gate

## 1. The gate (`scripts/gate.sh`, exit 0 = green)

```
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src                      # strict = true in pyproject
uv run python scripts/check_rules.py
uv run pytest tests/unit tests/contract -q
```
The gate makes **no network calls** outside localhost fakes. Live tests are excluded (marker `live`).

## 2. `scripts/check_rules.py` (fails with file:line for each hit in `src/`)

Code rules (environment access, subprocess, eval/exec, pickle, forbidden imports) are checked on the **AST**
(`ast.parse` + a `NodeVisitor`), not with line regexes, so multi-line calls and `from x import y` aliases are caught.
Line regexes are only used for comment/text markers (TODO etc.) and string-literal rules. In particular these must be
reported: `from os import environ|getenv|putenv` (any alias), `os.environ`, `os.getenv`, `os.putenv`;
`asyncio.create_subprocess_shell`, `os.system`, `os.popen`, any `subprocess.*` call with `shell=True` or whose first
argument is a string literal / f-string (even on the next line), including `from subprocess import run` aliases.

- `os.environ` / `os.getenv` outside `src/caracal/config.py`
- `print(` anywhere in `src/` (stdout is the MCP stdio channel — use `log.py`)
- `shell=True`, `os.system`, `subprocess.call(`/`run(` with a string first arg, `eval(`, `exec(`, `pickle.load`
- `TODO`, `FIXME`, `XXX`, `NotImplementedError`, `pass  # stub`
- `requests` import (use httpx); `arq`, `rq`, `celery` imports (docs/05 §2)
- Redis safety: `flushdb`, `flushall`, `.keys(` calls anywhere in `src/`; and in `src/` any string literal starting
  with `"caracal:"` outside `config.py` (keys must come from `queue.key()` so `REDIS_PREFIX` is honoured)
- a module under `src/caracal/` that is not listed in docs/01 §3 (keeps the layout binding)

## 3. Layers

| Layer | Location | Rules |
|---|---|---|
| unit | `tests/unit/` | pure logic + fakes; fast (< 1 s each) |
| contract | `tests/contract/` | FastAPI `TestClient`/httpx ASGI for REST; MCP via the SDK's in-memory client session against `build_mcp()`; compares names/fields/error codes with docs/03 |
| live | `tests/live/` (`@pytest.mark.live`) | real whisper, Ollama, Postgres, Redis, pyannote, search; run by the boss, not by the gate |

DB in unit/contract tests: repositories are behind protocols; tests use in-memory fakes (`tests/fakes/`).
A separate `tests/live/test_db.py` runs migrations against a throwaway schema on latitude.

## 4. Fakes (`tests/fakes/`)

- `FakeWhisper` — httpx `MockTransport` returning recorded `verbose_json` (fixture `tests/fixtures/whisper/jfk.json`,
  recorded from the real server in docs/00) and synthetic multi-chunk responses.
- `FakeOllama` — returns scripted tool calls **and** the measured failure modes: markdown instead of a tool call,
  fenced JSON, invalid JSON, schema-invalid arguments, 500, timeout. Tests must cover each mode for `call_tool`.
- `FakeSearch`, `FakeDiarizer` (scripted turns + embeddings), `FakeResolver` (DNS for SSRF tests).

## 5. Test quality rules

- Each test must fail if the behaviour it covers is removed (the judge neuters code to check — non-vacuity).
- No `assert x is not None` as the only assertion. Assert values.
- Do not edit an existing test to make it pass. If you believe a test is wrong, STOP and report.
- Mock only network boundaries, the clock, DNS, and the diarizer model. Never mock the unit under test.

## 6. Fixtures

- `tests/fixtures/audio/jfk.wav` — 11 s, mono 16 kHz, public domain (US Government work).
- Additional fixtures (two-speaker, German, music-only, 25-min long) are produced by the boss in card T1.0/T2.2
  from public-domain/CC sources with attribution in `tests/fixtures/ATTRIBUTION.md`.
