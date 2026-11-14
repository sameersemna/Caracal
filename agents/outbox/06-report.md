STATUS: DONE
## What changed
- `src/caracal/app.py`: `AppDeps`, `create_app(settings, deps=None)` with real-deps lifespan and disposal, error handlers, Origin guard, 64 KiB job-body guard, cached concurrent `health_payload`.
- `src/caracal/api/auth.py`: constant-time bearer dependency, `WWW-Authenticate` support, validated token on `request.state`.
- `src/caracal/api/schemas.py`: `JobCreate`/`JobSourceIn` (exactly one source, `extra="forbid"`, URL rules) reusing `JobOptions`, plus `JobStatus`/`JobSummary` models.
- `src/caracal/api/routes.py`: `/v1/jobs` create/list/get/delete, SSE `job_event_stream`, `/healthz`, rate limit, dedup, enqueue-failure handling, safe job-directory removal.
- `src/caracal/api/__init__.py`: package marker.
- `src/caracal/ingest/router.py`: `classify_source_static` (media-extension path → `direct`, else `platform`, no network).
- `src/caracal/ingest/__init__.py`: re-export for the future ingest module.
- `src/caracal/cli.py`: implemented `serve` (`uvicorn.run(create_app(settings), workers=1, log_config=None, proxy_headers=False)`), JSON logging, config-error handling.
- `tests/contract/{__init__,conftest}.py`: injected fake-repo / fakeredis / MockTransport harness.
- `tests/contract/test_api_auth.py`: auth, public healthz, Origin policy.
- `tests/contract/test_api_jobs.py`: job creation, classification, upload consumption, dedup, rate limit, enqueue failure, body limit, validation, listing, status, DELETE.
- `tests/contract/test_api_events.py`: SSE generator, subscribe/snapshot race hook, keep-alive, terminal HTTP stream.
- `tests/contract/test_healthz.py`: probes, down states, timed cache.
- `tests/unit/test_ingest_router.py`: static classifier table.
- `tests/unit/test_cli.py`: declared `serve`, asserted uvicorn options and config failure.
## Tests added
- `test_api_auth.py`: missing/wrong/malformed/trailing-space bearer → 401 + `WWW-Authenticate: Bearer`; `/healthz` public; allowed/denied `/mcp` Origins with the 403 envelope.
- `test_url_job_is_queued_and_enqueued`: 202 shape/links, repo row, stream entry under the configured prefix.
- `test_direct_media_url_is_classified_direct`, `test_upload_job_consumes_the_upload`, `test_unknown_upload_is_404`: source-kind routing and upload consumption.
- `test_dedup_returns_the_existing_done_job`, `test_depth_deep_bypasses_dedup`: 200 dedup id vs `deep` 202.
- `test_rate_limit_returns_429_with_retry_after`, `test_enqueue_failure_marks_job_failed_and_returns_503`: S11 and queue-outage semantics.
- `test_body_over_64_kib_is_413`, invalid-URL and extra-field tests assert 422 and that the submitted value never appears in the response.
- `test_unknown_route_uses_the_error_envelope`, `test_unexpected_exception_is_500_without_details`: 404 envelope and 500 `"internal error"` with no traceback text.
- `test_list_jobs_filters_and_paginates`, `test_list_jobs_rejects_an_unknown_status`, `test_get_job_status_shape`, `test_get_job_progress_and_error`, `test_unknown_and_invalid_job_ids`: list/status contract, Stage ordering, defaults, UUID errors.
- `test_delete_running_job_sets_cancel_flag`, `test_delete_terminal_job_removes_rows_and_directory`, `test_delete_never_follows_a_symlink_outside_the_data_dir`: DELETE semantics and path safety.
- `test_api_events.py`: snapshot first, published stage forwarded, `end` closes; event published between SUBSCRIBE and snapshot delivered; `: ping`; terminal job over HTTP yields snapshot+end.
- `test_healthz.py`: all-up, whisper 500/timeout, postgres/redis down, cache hit at +5 s and refresh at +11 s.
- `test_ingest_router.py`: every media extension → `direct`; playlists, non-media, non-http, trailing slash → `platform`.
- `test_cli.py`: `serve` dispatches, passes exactly the documented uvicorn options, returns 1 on invalid config.
## Measured
- gate exit code: 0
- ```
==> ruff check
All checks passed!
==> ruff format --check
53 files already formatted
==> mypy (strict)
Success: no issues found in 24 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 19%]
........................................................................ [ 39%]
........................................................................ [ 59%]
........................................................................ [ 79%]
........................................................................ [ 99%]
..                                                                       [100%]
362 passed in 3.02s
GATE: PASS
```
## Deviations
- `ErrorCode` has no `RATE_LIMITED` value and `HTTP_STATUS` has no 429 entry, although docs/03 §4 and docs/06 S11 require `rate_limited`. `src/caracal/errors.py` and `tests/unit/test_errors.py` are outside this card's Files list, and the existing test pins the enum set (without `rate_limited`). The route therefore returns a literal docs/03 §1 envelope with `"code": "rate_limited"` and `Retry-After`, rather than raising `CaracalError`.
- URL/source validation lives in `api/schemas.py` (pydantic), so FastAPI's `RequestValidationError` handler produces the 422 envelope; the URL is never echoed because the handler keeps only `loc`/`msg`/`type`.
- The 413 guard is a Content-Length middleware plus a route-level `len(body)` check; a chunked request that sends no Content-Length is rejected by the route check only after body parsing.
- `GET /v1/jobs` treats empty `status=`/`cursor=` as absent (the docs table shows them as optional placeholders).
## NOT proven
- No live run of `caracal serve` against real Postgres/Redis/whisper/Ollama, so the real-deps lifespan and disposal are only type-checked, not executed here.
- Dedup and cursors against real Postgres (the repository contract was covered by T0.2; these tests use the in-memory fakes).
- SSE keep-alive with the production 15 s interval and a real proxy; only an injected 0.05 s interval is tested.
- Chunked job bodies without Content-Length (see Deviations).
- Subtitle-link discovery with actual `.srt`/`.vtt` files (the code path exists but no test asserts a non-empty `links.subtitles`).
## Open
- Should `RATE_LIMITED` be added to `ErrorCode`/`HTTP_STATUS` and `tests/unit/test_errors.py` updated in a follow-up card? That would replace the direct 429 response with `CaracalError`.
- Should unauthenticated requests to an unknown `/v1/*` path return 401 instead of the current 404 envelope? Route-level dependencies cannot run for an unmatched path; this would need a path-based middleware if the stricter reading of docs/03 §1 is intended.
