STATUS: DONE
## What changed
- `src/caracal/ingest/ssrf.py`: `Resolver`/`SystemResolver`, `default_ip_policy` (IPv4-mapped/NAT64/6to4 unwrapping, global-only), `resolve_and_check` pinning the first permitted address and rejecting literal decimal/octal/hex IPs and userinfo.
- `src/caracal/ingest/direct.py`: `PinnedBackend` (connect to the pinned IP, refuse other hosts), private-pool transport swap, manual redirects with per-hop revalidation, content-type/size checks, streamed sha256 with `.part` + atomic rename, `head_content_type`.
- `src/caracal/ingest/upload.py`: `sanitize_filename`, streamed `save_upload` (sha256, cap, partial cleanup, internal error without path) and `MultipartFileStream`/`receive_upload` streaming multipart file parts.
- `src/caracal/ingest/router.py`: platform-host suffix matching and async `classify_source` (static → platform hosts → one HEAD → platform on failure, blocked always propagates).
- `src/caracal/api/routes.py`: `POST /v1/uploads` (multipart-only route, streams to `UploadFiles`, creates the row, 201 payload; 415/422/500 paths).
- `src/caracal/api/schemas.py`: `UploadCreated` response model.
- `src/caracal/app.py`: registers the upload body cap (`CARACAL_MAX_UPLOAD_MB` + 64 KiB multipart overhead) and renders non-multipart `POST /v1/uploads` as 415.
- `src/caracal/storage.py`: `UploadFiles(data_dir, upload_id)` with UUID-keyed root and separator-rejecting `file()`.
- `tests/fakes/resolver.py`: deterministic fake DNS resolver (unregistered hosts raise `gaierror`).
- `docs/findings/errata.md`: records the upload-directory/row-id mismatch (see Deviations).

## Tests added
- `tests/unit/test_ssrf.py`: full blocked-destination table (loopback, mapped/NAT64/6to4 literals, decimal/octal/hex, CGNAT, link-local, userinfo, ftp, mixed resolution), public allow+pin, DNS failure, host-only error details, `inet_aton` forms, policy ranges.
- `tests/unit/test_direct.py`: download + atomic rename + sha256; Content-Disposition sanitisation; content-type rejection/acceptance; declared-length over cap with a body stream that raises if read; streamed over-cap deletes `.part`; redirect to 127.0.0.1 blocked before the second request; redirect limit; non-200; pinned backend records the IP and refuses other hosts; httpx `_pool` seam test; HEAD media type and public redirect.
- `tests/unit/test_upload.py`: sanitisation table incl. `../../etc/passwd → etc_passwd`, length cap, streaming hash/size/rename, over-cap leaves no partial, `UploadFiles` path safety, disk failure → internal without path.
- `tests/unit/test_ingest_router.py` (appended): platform hosts need no network, static extensions need no probe, audio/video HEAD → direct, html/HEAD failure/unavailable → platform, blocked HEAD propagates, lookalike suffix is probed, non-http scheme is platform.
- `tests/contract/test_api_uploads.py`: happy path (sha/size, file on disk, row), traversal filename, chunked oversize 413 with no file, no-file-part 422, two-file-parts 422, text/plain 415, unauthenticated 401 with no file, disk failure 500 without a path in the message.

## Measured
- gate exit code: 0
- ```
  ==> ruff format --check
  63 files already formatted
  ==> mypy (strict)
  Success: no issues found in 27 source files
  ==> repository rules
  ==> pytest (unit + contract, live excluded)
  ........................................................................ [ 15%]
  ........................................................................ [ 31%]
  ........................................................................ [ 47%]
  ........................................................................ [ 62%]
  ........................................................................ [ 78%]
  ........................................................................ [ 94%]
  ..........................                                               [100%]
  458 passed in 3.14s
  GATE: PASS
  ```

## Deviations
- **Upload directory name vs row id:** `UploadRepo.create` has no `upload_id` parameter and generates its own UUID *after* the file must be written, and `src/caracal/db/repo.py` is outside this card's Files list. The API therefore writes to `uploads/<storage id>/<name>` and stores that exact path; the response/row id is the repo-generated id. `record.path` is authoritative and points at the real file (verified). Recorded in `docs/findings/errata.md`.
- **Multipart-only route matching:** the route class only claims `Content-Type: multipart/form-data`; other content types fall through and `create_app` maps a 404 on `POST /v1/uploads` to 415. This is needed because the pre-existing `tests/contract/test_api_limits.py::test_per_route_limits_let_an_upload_route_declare_a_larger_cap` mounts its own `POST /v1/uploads` after `create_app` and expects to handle non-multipart chunked bodies; the real upload route must not shadow it. No existing test was edited.

## NOT proven
- No real HTTP/DNS/TLS traffic was exercised (gate forbids live network): pinning is proven with a recording backend and `httpx.MockTransport`, not a real TLS handshake against a public host.
- The real Postgres `UploadRepo` path/row was not exercised (contract upload tests use the in-memory fake); the directory/id mismatch is therefore demonstrated on the fake and by code inspection for Postgres.
- `classify_source` and `head_content_type` are not wired into the job route or pipeline yet (T1.6); only their unit behaviour is proven.
- Retention/unconsumed-upload cleanup interacting with the stored path was not tested.

## Open
- Decide the upload id/directory contract (add `upload_id` to `UploadRepo.create`, or bless `path` as authoritative). Until then a client reconstructing `uploads/<upload_id>/` from the response will not find the file.
