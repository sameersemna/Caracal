STATUS: DONE
## What changed
- `pyproject.toml`, `uv.lock`: added `curl-cffi==0.17.0` (carry-forward 3).
- `src/caracal/asr/__init__.py`: `join_words` attaches closing punctuation/quote/bracket-only tokens (e.g. `."`) with no space (carry-forward 5).
- `src/caracal/ingest/ytdlp.py`: TLS-fingerprint / impersonation / `HTTP Error 403` map to `source_unavailable` "site blocked automated access", checked after the real sign-in/age/geo categories (carry-forward 4).
- `src/caracal/db/repo.py` + `tests/fakes/repo.py`: `UploadRepo.create(*, upload_id=None, ...)` honours a caller-supplied id (carry-forward 1).
- `src/caracal/api/routes.py`: upload route generates the id before streaming, so `uploads/<id>/` matches the row; removed the multipart-only route filter; added `GET /v1/jobs/{id}/transcript` (json/txt + `job_id`/`languages`) and `GET /v1/jobs/{id}/subtitles/{filename}` (S10 lang validation, SRT/VTT content types, attachment filename).
- `src/caracal/app.py`: deleted the 404→415 mapping; the real upload route answers non-multipart with 415 (carry-forward 2).
- `src/caracal/pipeline/stages.py`: M1 stage implementations (`ingest`, `normalize`, `chunk`, `transcribe`, `langid`, `subtitles`), relative artifact paths, `StageContext` client/cancel fields, `m1_registry()`; later stages are `skipped` with "not implemented in M1".
- `src/caracal/pipeline/runner.py`: passes whisper/resolver/redis/keys and the per-job cancel probe into `StageContext`.
- `src/caracal/worker.py`: builds one `httpx.AsyncClient`, `WhisperClient` (lock = `whisper_lock(redis, keys)`) and `SystemResolver` per process, uses `m1_registry()` by default, closes its own HTTP client.
- Tests: `tests/unit/test_stages.py`, `tests/contract/test_api_transcript.py`, `tests/live/test_pipeline_live.py`; add-only cases for all five carry-forwards.
## Tests added
- `test_ingest_upload_links_the_file_and_sets_the_title`: upload hard-linked into `source/`, output provenance, title.
- `test_ingest_direct_downloads_through_the_pinned_client`: direct output (`final_url`, sha) and `download_direct` destination.
- `test_ingest_platform_stores_metadata_and_ignores_caption_failure`: platform metadata subset persisted; caption error logged/ignored, stage still done.
- `test_normalize_and_chunk_run_real_ffmpeg_and_delete_the_source`: real ffmpeg/lavfi, probe/stats, source deletion, chunk cut on disk.
- `test_transcribe_writes_transcript_segments_and_removes_chunks`: fixture transcript JSON/TXT, repo segments, chunk deletion, one whisper lock acquisition per chunk.
- `test_transcribe_stops_between_chunks_without_writing_a_transcript`: cancel between chunks writes no transcript.
- `test_cancel_between_chunks_settles_the_job_cancelled`: through `run_pipeline`, job status `cancelled` and no `transcript.json`.
- `test_langid_summarises_languages_and_caption_agreement`: languages summary, caption lang and agreement from a real VTT.
- `test_langid_rewrites_the_transcript_when_a_recheck_relabels`: `refine_spans` wiring (chunk durations/probs) and transcript rewrite.
- `test_subtitles_writes_srt_and_vtt`: cues and both rendered files.
- `test_m1_pipeline_runs_end_to_end_on_an_uploaded_tone`: whole M1 registry through `run_pipeline`, job done, artifacts, deleted `chunks/`+`source/`, segments, lock count.
- `tests/contract/test_api_transcript.py` (10 tests): JSON/txt content types and payload, 409 while running, 404 failed, SRT/VTT types+filename, 404 unknown lang, 422 `..%2Fx`, `links.subtitles`.
- `test_upload_directory_is_named_after_the_row_id`; `case_upload_create_honours_the_given_id`; `test_join_words_attaches_a_closing_quote_after_punctuation`; three blocked-access mapping rows plus the 403+sign-in auth row; `test_curl_cffi_is_installed_for_ytdlp_impersonation`.
- `tests/live/test_pipeline_live.py`: real in-process M1 pipeline on `jfk.wav` upload and on `CARACAL_LIVE_YT_URL` (skipped when unset), throwaway schema+prefix.
## Measured
- gate exit code: 0
- ==> mypy (strict)
- Success: no issues found in 39 source files
- ==> repository rules
- ==> pytest (unit + contract, live excluded)
- ........................................................................ [ 11%]
- ........................................................................ [ 22%]
- ........................................................................ [ 34%]
- ........................................................................ [ 45%]
- ........................................................................ [ 57%]
- ........................................................................ [ 68%]
- ........................................................................ [ 79%]
- ........................................................................ [ 91%]
- ....................................................... [100%]
- 631 passed in 5.90s
- GATE: PASS
## Deviations
- `default_registry()` is kept as the M0 stub registry: the existing `test_runner.py::test_default_registry_skips_all_stages_and_finishes_done` pins "not implemented in M0" for all stages and the card allows only add-only test edits. Production uses the new `m1_registry()` (worker default; used by every new test).
- Two existing worker tests (`test_worker_processes_a_job_end_to_end_and_cleans_the_stream`, `test_job_enqueued_before_the_worker_starts_is_delivered`) now pass `registry=default_registry()` explicitly. Required: `Worker`'s default registry changed from M0 stubs to `m1_registry()`; the edits preserve those tests' queue-mechanics intent. Declared.
- `StageContext.whisper/resolver/redis/keys` are typed optional with `None` defaults (and `cancelled` defaults to an always-false probe). Reason: the M0 stub registry must still run in isolated runner tests; every M1 stage raises `internal` when its client is missing. The worker always populates them.
- Direct downloads are stored as `source/download` because the server filename is not known before the request; the checkpoint `filename` is that stored name and `final_url`/sha come from `DirectInfo`.
- `tests/repo_contract.py` received an add-only `case_upload_create_honours_the_given_id` (registered in `CASES`) so the fake and the live Postgres repo prove carry-forward 1 together; it is an existing test helper of `db/repo.py`.
## NOT proven
- `tests/live/test_pipeline_live.py` was written but not run here (live marker; judge runs it against real whisper/ffmpeg/Postgres/Redis and optionally `CARACAL_LIVE_YT_URL`).
- Multi-chunk (>600 s/>900 s) real media was not run; cancel-between-chunks is driven by a synthetic two-chunk checkpoint over cut files.
- Platform ingest network behaviour (real yt-dlp/captions) is exercised only at the `ytdlp` function boundary with fakes; verdicts 08/09 remain the live evidence for downloads.
- `curl-cffi` installation/version is asserted; actual TLS impersonation negotiation is not exercised (no network in the gate).
- `ingest`/`normalize`/`langid` blocking file work is small but synchronous in a few places (`Path.read_text`, `unlink`); `check_rules` does not flag it.
## Open
- Should `default_registry()` be renamed/deleted once the last M0-only test is retired, so there is one registry again?
- Should the ingest download honour `ctx.cancelled()` mid-transfer (currently yt-dlp gets a fresh `threading.Event`; cancel is checked at stage boundaries)?
