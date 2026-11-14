STATUS: DONE
## What changed
- src/caracal/audio/__init__.py: package marker for the audio stage.
- src/caracal/audio/ffmpeg.py: argv-only `_run` runner (which-at-call, 4 KiB stderr tail, timeout/cancel kill), `probe`, `normalize`, `analyse`, `cut` with both demuxer whitelists before every input.
- src/caracal/audio/chunking.py: pure `plan_chunks`/`Chunk` with silence-centred boundaries, hard cuts, overlap and <30 s tail merge.
- tests/unit/test_ffmpeg.py: real-local ffmpeg tests for probe/normalize/analyse/cut, hostile inputs, argv hardening, runner config, timeout and partial-file cleanup.
- tests/unit/test_chunking.py: table, tail-merge, 25-min example and 300-iteration randomised invariant loop.
- tests/fixtures/media/README.md: how every fixture is generated with lavfi (no committed binaries).
## Tests added
- test_probe_reads_first_audio_stream: m4a probes as `mov,mp4,m4a,...` with 44.1 kHz stereo AAC metadata.
- test_probe_keeps_only_allowed_lowercased_tags: title/artist/publisher/comment retained, copyright dropped, keys lowercased.
- test_normalize_produces_16k_mono_pcm_wav: output is wav/pcm_s16le/16 kHz/mono within 0.05 s of source; source ProbeInfo returned.
- test_normalize_rejects_over_duration_before_writing: 3 s source with 2 s cap → `too_large`, no dst or `.part`.
- test_hostile_inputs_are_unsupported_and_leave_nothing_behind: HLS playlist, ffconcat, random bytes and empty file → `unsupported_media`, no output leftovers.
- test_video_without_audio_is_unsupported: testsrc MP4 → `unsupported_media` "no audio stream".
- test_analyse_finds_one_silence_and_speech_ratio: tone(2)+silence(2)+tone(2) → silence (2.0,4.0) ±0.15, ratio 0.67 ±0.05.
- test_analyse_pure_silence_is_poor: ratio < 0.05 and quality `poor`.
- test_cut_extracts_the_requested_range / test_cut_rejects_invalid_ranges: 1.0–2.5 s slice is 1.5 s 16 kHz mono; bad ranges are `invalid_input`.
- test_process_runner_uses_argv_exec_and_devnull_stdin: every spawn uses argv, DEVNULL stdin and PIPE stdout/stderr, no shell.
- test_run_kills_the_process_on_timeout: 0.01 s timeout raises `internal` "ffmpeg timed out" and the reaped process has a returncode.
- test_missing_binary_is_dependency_down: no `which` result → `dependency_down` naming `ffprobe`.
- test_normalize_removes_the_partial_file_when_conversion_fails: a simulated post-probe failure leaves neither dst nor `.part`.
- test_plan_chunks_table / custom target / tail merge / 25-min example: exact boundaries at expected targets and overlaps.
- test_random_silence_sets_satisfy_chunk_invariants: starts increase, cover [0,duration], consecutive overlap == 1.0 s, every chunk ≤ max+overlap+30.
## Measured
- gate exit code: 0
- tail of `bash scripts/gate.sh`:
```
72 files already formatted
==> mypy (strict)
Success: no issues found in 31 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 13%]
........................................................................ [ 27%]
........................................................................ [ 40%]
........................................................................ [ 54%]
........................................................................ [ 67%]
........................................................................ [ 81%]
........................................................................ [ 95%]
..........................                                               [100%]
530 passed in 5.20s
GATE: PASS
```
## Deviations
- `cut` adds `-f wav` before the `<dst>.part` output. Without it ffmpeg cannot choose a muxer for the extension-less `.part` name and exits 234; the card's explicit file name therefore requires the explicit format.
- `_run` has an added optional `stderr_limit` keyword (default 4096). Error tails stay ≤4 KiB; `analyse` passes `None` because it must parse every silence boundary from potentially long `-v info` output in one pass.
## NOT proven
- No live/long-file run: the 25-minute chunking case is arithmetic, not a real 25-minute decode, and `analyse` was exercised on 2–6 s lavfi media.
- The full-stderr path of `analyse` is only proven on short files; memory use on multi-hour `-v info` output was not measured.
- Hostile inputs are proven rejected on this host's ffmpeg 6.1.1; other demuxer builds were not tested.
## Open
- docs/01 §3 still lists `detect_silences(path)` while this card and docs/04 §3 define the single-pass `analyse(wav)`; the card was treated as authoritative, and no docs were edited.
