STATUS: DONE
## What changed
- `src/caracal/app.py`: new pure-ASGI `AuthBodyLimitMiddleware` (bearer check before any body read, streaming per-route byte cap, `body_limits` mapping for later upload routes); `OriginGuardMiddleware` converted from `BaseHTTPMiddleware` to pure ASGI; `RATE_LIMITED` → `Retry-After` in `_caracal_response`; `_BodyTooLarge` + its 413 `too_large` handler; removed `jobs_body_guard` and `MAX_JOBS_BODY_BYTES`.
- `src/caracal/api/auth.py`: `bearer_from_header` and constant-time `token_matches` shared by `require_auth` and the middleware, so both apply identical syntax rules; `require_auth` stays on the router as defence in depth.
- `src/caracal/api/routes.py`: rate limit raises `CaracalError(ErrorCode.RATE_LIMITED, details={"retry_after_s": n})` instead of a literal envelope; removed the now-redundant route-level `request.body()`/`len(body)` check.
- `src/caracal/errors.py`: `ErrorCode.RATE_LIMITED = "rate_limited"` and `HTTP_STATUS` 429.
- `tests/unit/test_errors.py`: expected code set gains `rate_limited: 429` (the only change to this file; declared).
- `tests/contract/test_api_auth.py`: unauthenticated unknown `/v1` path must be 401 before routing.
- `tests/contract/test_api_limits.py` (new): ASGI-level guard tests with a hand-written `receive` that counts pulled chunks/bytes.

## Tests added
- `test_unauthenticated_chunked_body_is_401_before_reading_any_chunk`: 30 MiB chunked POST without Authorization → 401 + `WWW-Authenticate`, **0** receive calls, 0 bytes.
- `test_unauthenticated_invalid_json_is_401_not_422`: invalid JSON without auth → 401, 0 chunks pulled.
- `test_authenticated_chunked_body_over_the_limit_is_413_mid_stream`: 50 MiB chunked → 413 `too_large` after exactly 2 chunks / 131,072 bytes (= limit + one chunk).
- `test_understated_content_length_cannot_bypass_the_cap`: `Content-Length: 10` with 128 KiB → 413 after limit + one chunk.
- `test_declared_content_length_over_the_limit_is_413_without_reading`: 70,000 declared → 413 with 0 chunks pulled.
- `test_chunked_body_under_the_limit_reaches_the_route`: a valid chunked `JobCreate` still yields 202 (proves the cap did not break chunked bodies).
- `test_per_route_limits_let_an_upload_route_declare_a_larger_cap`: `create_app(..., body_limits={("POST", "/v1/uploads"): 256 KiB})` accepts a 200 KiB chunked body — the T1.1 hook.
- `test_the_default_cap_applies_to_every_v1_route`: a non-jobs `/v1` route also gets the 64 KiB cap.
- `test_rate_limited_uses_the_error_code_and_retry_after`: 429 body uses `ErrorCode.RATE_LIMITED` and `Retry-After` equals `details.retry_after_s`.
- `test_unauthenticated_unknown_v1_path_is_401_before_routing`: auth runs before routing for every `/v1` path.

## Measured
- gate exit code: 0
- ```
==> ruff check
All checks passed!
==> ruff format --check
55 files already formatted
==> mypy (strict)
Success: no issues found in 24 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 19%]
........................................................................ [ 38%]
........................................................................ [ 57%]
........................................................................ [ 77%]
........................................................................ [ 96%]
.............                                                            [100%]
373 passed in 2.96s
GATE: PASS
```
- 373 = 362 previous + 10 new contract tests + the extra `rate_limited` parametrisation in `test_errors.py`.
- Mutation checks (temporary edits, then restored; gate re-run green):
  - auth gate mutated to `if False:` → both unauthenticated tests fail (422, body pulled).
  - streaming counter removed → chunked-413 and default-cap tests fail (200, 131,072 bytes buffered).
  - `ErrorCode.RATE_LIMITED` removed → `test_errors` (2 cases) and the rate-limit test fail.
- Local uvicorn smoke (`uvicorn` 0.54, in-memory fakes, real sockets): unauth chunked 30 MiB → 401; auth chunked 50 MiB → 413; unauth invalid JSON → 401.

## Deviations
- **`OriginGuardMiddleware` is now pure ASGI** (was `@app.middleware("http")` / `BaseHTTPMiddleware`). Verified with Starlette 1.10: `BaseHTTPMiddleware` runs the downstream app's `receive` inside an `anyio` task group, so a signal raised from `receive` arrives as `ExceptionGroup` and defeats the early 413. Removing the only `BaseHTTPMiddleware` makes `AuthBodyLimitMiddleware` literally the outermost user middleware ("outermost after error handling") and keeps its contract (403 envelope on foreign `/mcp` Origin) — existing tests pass unchanged.
- **`_BodyTooLarge` subclasses `StarletteHTTPException`.** FastAPI's body-parse wrapper converts *every* non-HTTP exception into 400 `invalid_input` and only re-raises `HTTPException`. A dedicated handler registered in `create_app` renders `_BodyTooLarge` as the 413 `too_large` envelope; the generic HTTP handler (which would call it `invalid_input`) never sees it. The middleware also keeps a fallback catch for non-FastAPI uses.
- Behaviour change, intended by docs/03 §1: `GET /v1/<unknown>` without a token is now 401, not 404 (auth before routing). Round 06's Open question asked for exactly this.
- The 64 KiB cap now applies to every `/v1` request body (not just `POST /v1/jobs`), per fix 2.

## NOT proven
- The judge's live socket attacks against real Postgres/Redis were not reproduced here; the smoke used in-memory fakes. The precise read bound is proven at the ASGI `receive` layer (0 chunks for unauthenticated, 2 chunks for oversize).
- Client-side "bytes written before the 413" was ~2.5 MiB in the loopback smoke — that is kernel/TCP buffering plus uvicorn's transport buffer, not application reads; the app itself pulled exactly `limit + one chunk` (asserted). Server-side socket buffer sizing is owned by uvicorn.
- Nothing else changed; `tests/live/` untouched.

## Open
- T1.1 must call `create_app(..., body_limits={("POST", "/v1/uploads"): <cap>})` to get its own upload cap; the default 64 KiB applies otherwise.
- `/mcp` is deliberately not covered by this middleware (the MCP card owns its token handling); only `/v1` is pre-body authenticated.
