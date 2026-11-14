STATUS: DONE

## What changed
- `src/caracal/services.py`: new `JobService(settings, repos, redis, keys, queue, data_dir)`; moved every business rule out of the routes (`create_job`, upload-from-chunks/base64 with S7 pre-decode check, status/list, transcript/subtitles, cancel/delete, subscribe-then-snapshot `wait_for_job`, `analyze_summary`); raises `CaracalError` only.
- `src/caracal/mcp/__init__.py`, `src/caracal/mcp/server.py`: new `build_mcp(service, settings, *, max_inline_bytes)` with `analyze_audio`, `get_job`, `list_jobs`, `get_transcript`, `cancel_job`; pydantic results, LLM-facing descriptions, `CaracalError → ToolError("<code>: <message>")`.
- `src/caracal/api/schemas.py`: `JobCreated`, `CancelResult`, `AnalyzeSummary`; serializers that keep the REST JSON shape (omit absent `source.url/title`, stage timestamps, and `JobStatus.error`).
- `src/caracal/api/routes.py`: every route is now a thin adapter; multipart parsing stays here (transport concern) but persistence moved to the service; status codes, headers and SSE framing unchanged.
- `src/caracal/app.py`: `/mcp` mounted via `streamable_http_app(stateless_http=True, json_response=True)`, sub-app lifespan inside the FastAPI lifespan; bearer + 1 MiB body cap now cover `/mcp` before any body read; `TransportSecuritySettings` from `CARACAL_PORT`, `CARACAL_PUBLIC_BASE_URL` and `CARACAL_ALLOWED_HOSTS`; Origin guard unchanged; real deps are built at app construction so the MCP app can be mounted and are still closed by the lifespan.
- `src/caracal/config.py`: `CARACAL_ALLOWED_HOSTS` (CSV list via the existing `parse_csv`).
- `src/caracal/cli.py`: real `mcp-stdio` entry point (builds deps, `run_stdio_async()`, closes resources; logs stay on stderr).
- `tests/contract/test_api_auth.py`: adapted (see Deviations) for authenticated `/mcp` and the mount shadowing the old `/mcp/ping` test route.
- `tests/unit/test_cli.py`: adapted; `mcp-stdio` is no longer a stub.

## Tests added
- `tests/unit/test_services.py`: job create/enqueue/dedup/consumed upload, base64 oversize rejected before `b64decode` (spy), invalid/valid base64 storage, transcript JSON/txt/SRT and subtitle semantics, `job_not_ready`, analyze summary, progress + terminal + timeout waits, cancel vs delete, list filtering/validation.
- `tests/contract/test_mcp_tools.py`: each tool happy path; URL / upload_id / tiny-WAV base64 sources; zero and two sources → `invalid_input`; `wait_seconds=0` hint; scripted worker finishes during the wait → `status:"done"` with `report_progress` notifications; oversize inline base64 rejected without decoding (spy); errors carry the code and no traceback; cancel/delete.
- `tests/contract/test_mcp_http.py`: `/mcp/` without bearer → 401 with zero body chunks pulled; real `initialize` + `tools/call get_job`; `Host: promaxgb10-6116:11404` accepted only when listed in `CARACAL_ALLOWED_HOSTS`, 421 otherwise; foreign `Origin` → 403 envelope; >1 MiB body → `too_large`; unknown tool → `isError`.
- `tests/contract/test_parity.py`: docs/03 §0 table as data (5 available rows, 5 pending rows with card ids); both sides exist for available rows; neither side exists for pending rows; `structured_content == REST JSON` for `get_job`, `list_jobs`, `get_transcript(format="json")`.
- `tests/live/test_mcp_live.py` (written, judge-run): uploads `jfk.wav`, `analyze_audio(wait_seconds=120)` → done with "country" in the preview, then non-empty SRT.

## Measured
- gate exit code: 0
```
Success: no issues found in 42 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 10%]
........................................................................ [ 21%]
........................................................................ [ 32%]
........................................................................ [ 43%]
........................................................................ [ 53%]
........................................................................ [ 64%]
........................................................................ [ 75%]
........................................................................ [ 86%]
........................................................................ [ 97%]
....................                                                     [100%]
668 passed in 7.00s
GATE: PASS
```

## Deviations
- `tests/contract/test_api_auth.py` was adapted although the factory signature did not change. Card requirement 3 makes `/mcp` bearer-protected and mounts the real MCP app at `/mcp`, so the old tests' `/mcp/ping` route is shadowed and unauthenticated `/mcp` is a 401 by design. The origin matrix now drives a real MCP `initialize` request; the foreign-origin cases still assert the 403 envelope.
- `tests/unit/test_cli.py` adaptation is the one the card explicitly allows/declares (mcp-stdio is real now).
- docs/03 §3 names the analyze result `report_summary: ReportSummary`; card 15 requires `summary:{title,duration_s,languages,transcript_preview}` for M1, so `summary` is used.
- docs/03 §3 says `get_transcript` returns "string or JSON"; card 15 requires every tool to return a pydantic model and the parity test requires `structured_content == REST JSON`. `TranscriptResult` is a `RootModel[dict]`: for `format="json"` its root is exactly the REST transcript payload; text/SRT/VTT use `{format,lang,content}`.
- `create_app` now builds real dependencies eagerly (needed so the MCP sub-app can be mounted and its lifespan entered); they are still closed in the lifespan. The factory signature is unchanged.
- MCP rate-limit identity is `sha1(token + ":mcp")` because stdio has no client IP (REST keeps `sha1(token + IP)` per S11).
- `_BodyTooLarge` changed from an `HTTPException` subclass to a `BaseException` subclass: FastAPI converts ordinary exceptions during body parsing into 400, and the mounted Starlette sub-app converts `HTTPException` into plain-text 413; only a `BaseException` reaches the outer middleware. Existing `/v1` limit tests still pass.

## NOT proven
- `caracal mcp-stdio` against real Postgres/Redis/whisper: the unit test proves the entry point builds the server and writes nothing to stdout, but the live stdio handshake is the judge's check.
- `tests/live/test_mcp_live.py` was written but not run (no live server/worker in this session).
- Progress notifications over streamable HTTP were not exercised end-to-end; the in-memory client proves notifications are emitted, and the SDK makes `report_progress` a no-op when the HTTP caller sends no progress token.

## Open
- docs/03 §3 should be updated to `summary` (or card 15 changed to `report_summary`) and to describe `get_transcript`'s structured shape.
- `/mcp` shares one rate-limit bucket per API token; if per-client HTTP limits are required, the SDK request context needs to expose the client address.
