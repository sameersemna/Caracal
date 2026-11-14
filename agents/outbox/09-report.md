STATUS: DONE

## What changed
- `src/caracal/ingest/ytdlp.py`: hardened yt-dlp ingest — one options builder, SSRF pre-check, playlist/live/duration guards, metadata mapping, audio download + sha256, platform-caption selection/saving, URL-free error mapping, logger adapter.
- `tests/unit/test_ytdlp.py`: 39 tests with `yt_dlp.YoutubeDL` replaced by a recording fake; an autouse stdout guard fails any test that leaks a byte to stdout.
- `tests/live/test_ytdlp_live.py`: live metadata + captions + audio test gated by `CARACAL_LIVE_YT_URL` (marked `live`).

## Tests added
- `test_metadata_options_are_exactly_the_hardened_set`: asserts every required option by value, the logger type, the fixed `source.%(ext)s` template, `skip_download`, cookiefile, and that `exec`/`postprocessor_args`/`external_downloader` are absent.
- `test_audio_options_do_not_skip_or_write_subtitles` / `test_captions_options_request_base_and_original_track`: per-purpose option deltas (`subtitleslangs == ["en", "en-orig"]`).
- `test_match_filter_returns_reasons_for_live_and_long`: the callable rejects `is_live`, `live_status=is_upcoming`, `duration > max`, and allows `duration == max`/unknown.
- `test_logger_maps_levels_and_redacts`: debug/info → DEBUG, warning/error → WARNING, and token-shaped text is redacted.
- `test_progress_hook_raises_when_the_event_is_set`: the hook raises `_Cancelled`.
- `test_*_runs_ssrf_check_before_constructing_ytdlp` (metadata/audio/captions × `127.0.0.1`, `latitude:6379`): `source_blocked` and **zero** `YoutubeDL` constructions.
- `test_playlist_url_is_rejected_without_contacting_ytdlp`, `test_playlist_info_is_rejected[playlist|multi_video]`: `invalid_input` "one item per job".
- `test_live_stream_is_invalid_input`, `test_duration_over_the_limit_is_too_large`: match-filter rejection surfaces as `invalid_input` / `too_large`.
- `test_fetch_metadata_maps_every_documented_field` / `..._missing_fields_become_none_and_empty`: all 15 fields, 5 000-char description truncation, `None`/`[]` defaults.
- `test_download_audio_keeps_one_source_file_and_hashes_it` / `..._removes_stale_source_files`: single `source.*`, ext, size and sha256; stale files cleaned.
- `test_cancel_aborts_the_download_and_removes_the_partial_file`: progress hook aborts and `source.*.part` is gone.
- `test_manual_captions_beat_automatic`, `test_original_automatic_caption_beats_translated_one`, `test_first_manual_language_is_used_when_metadata_has_none`, `test_no_captions_is_empty_and_does_not_download`: selection rules, `en-orig` over `en-de`, `captions/<lang>.vtt`, leftover intermediate cleanup, `[]` with no second download.
- `test_download_error_mapping_is_short_and_url_free` (11 rows) + `test_extractor_error_is_mapped_too`: every mapping row, retryable flag, and no URL in the message.

## Measured
- gate exit code: 0
- ```
  ==> ruff format --check
  66 files already formatted
  ==> mypy (strict)
  Success: no issues found in 28 source files
  ==> repository rules
  ==> pytest (unit + contract, live excluded)
  ........................................................................ [ 14%]
  ........................................................................ [ 28%]
  ........................................................................ [ 43%]
  ........................................................................ [ 57%]
  ........................................................................ [ 72%]
  ........................................................................ [ 86%]
  .................................................................        [100%]
  497 passed in 3.16s
  GATE: PASS
  ```

## Deviations
- Card T1.2 requirement 1 gives `build_options(settings, *, dest_dir, cancel, purpose)` but also requires `subtitleslangs=[<lang>, f"{<lang}-orig"]`. That signature has no language input, so I added a keyword-only `lang: str | None = None` (required only for `purpose="captions"`). `fetch_platform_captions` probes available tracks first, then calls it with the chosen base language. Everything else in the signature is unchanged.
- The card describes playlist detection "from a flat metadata probe". I inspect the normal processed info (`info["_type"]`) instead of setting `extract_flat=True`, because the same probe must return full metadata and `subtitles`/`automatic_captions` for caption selection. The observable rule (`_type in {"playlist","multi_video"}` or `list=` without `v=` → `invalid_input`) is implemented and tested.

## NOT proven
- No live network call was made: `tests/live/test_ytdlp_live.py` is written but not run here (the judge runs it with `CARACAL_LIVE_YT_URL`).
- The timeout path is implemented (`asyncio.timeout` + `cancel.set()` + mapped `source_unavailable`/retryable) but not unit-tested with a genuinely hanging extractor, to avoid a flaky thread that outlives the test.
- `fetch_platform_captions` makes two yt-dlp calls (probe + captions download), each with its own 60 s timeout; the whole operation can therefore take up to ~120 s. The card's "captions 60 s" is applied per yt-dlp call.
- Real-world yt-dlp behaviour for non-YouTube extractors (SoundCloud/Vimeo) and for YouTube's exact caption-key naming is left to the judge's live check.

## Open
- Whether the judge prefers a single 60 s budget covering both captions calls rather than per-call timeouts.
- Whether the added `build_options(..., lang=...)` keyword is acceptable as the resolution of the card's missing language input.
