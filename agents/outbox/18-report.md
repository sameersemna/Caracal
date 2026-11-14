STATUS: DONE
## What changed
- pyproject.toml / uv.lock: optional `speakers` extra (`pyannote.audio==4.0.7`, `torch==2.11.0`, `torchaudio==2.11.0`, `torchcodec`) on the explicit `pytorch-cu130` index; `uv lock --check` passes.
- src/caracal/speakers/__init__.py: package exports for `Diarizer`, `Turn`, `DiarizationResult`, alignment helpers.
- src/caracal/speakers/diarize.py: `Diarizer(settings)` with lazy `asyncio.Lock`-guarded load, `asyncio.to_thread` inference, CUDA→CPU fallback logged once, first-appearance `S1..Sn` rename across turns/exclusive turns/embeddings, L2-normalised embeddings, `num_speakers` hint only when passed; pyannote/torch imported only inside functions via importlib.
- src/caracal/speakers/align.py: `assign_speakers` (midpoint, 1 s nearest fallback, split at speaker changes, ids renumbered, `join_words` text), `speaker_stats`, and `retime_outside_speech` (before-first-turn and >2 s gaps → following turn, even spread to the first in-turn word, ≥0.05 s, max-shift guard, segment bounds recomputed).
- src/caracal/pipeline/stages.py: `stage_diarize`, `stage_align`, `m2_registry`, speaker-aware transcript txt (`[mm:ss] S1: text`), shared transcript/segment/subtitle writers, align re-renders source SRT/VTT after retiming.
- src/caracal/pipeline/runner.py, src/caracal/worker.py: `diarizer` on `RunnerDeps`/`StageContext`, one lazy `Diarizer` per worker, `m2_registry()` as the worker default.
- src/caracal/cli.py: `doctor` reports `{pyannote, torch, cuda}` under `diarization` without importing torch unless present.
- tests/fakes/diarizer.py, tests/unit/test_diarize.py, tests/unit/test_align.py, tests/unit/test_stages.py (added tests only), tests/live/test_diarize_live.py.
- docs/findings/errata.md: recorded that docs/04 §5's "centroid over turns ≥ 1.5 s" wording is superseded by `DiarizeOutput.speaker_embeddings`.

## Tests added
- test_labels_are_renamed_by_first_appearance_across_turns_exclusive_and_embeddings: `SPEAKER_01` at t=0 becomes `S1`; exclusive turns and embedding keys follow the same map.
- test_turn_times_are_rounded_to_milliseconds: 0.123456/1.987654 → 0.123/1.988.
- test_every_embedding_is_l2_normalised + test_zero_embedding_is_reported_as_an_internal_failure: unit-norm rows and an explicit `internal` error for a zero vector.
- test_cpu_fallback_when_cuda_is_unavailable / test_explicit_cpu_device_is_not_probed: `_select_device` picks CPU when CUDA reports unavailable or zero devices.
- test_available_requires_both_modules_and_a_hf_token / test_available_is_false_when_the_modules_are_missing: `available()` is the conjunction of modules and `HF_TOKEN`.
- test_infer_passes_num_speakers_only_when_given_and_renames_the_output: absent hint sends no `num_speakers`; `2` sends it; turns/embeddings are renamed and normalised; sample rate is passed through.
- test_word_midpoint_selects_the_exclusive_turn_that_overlaps_it, test_word_in_no_turn_uses_nearest_turn_within_one_second_else_none: midpoint assignment and the 1 s nearest fallback.
- test_segment_is_split_at_a_mid_segment_speaker_change_without_spacing_artefacts: two segments, ids 0/1, exact `"hello there"` / `"friend"` text.
- test_speaker_stats_shares_sum_to_one_and_first_appearance_is_the_earliest_word: concrete talk times, shares 0.625/0.375 summing to 1, word counts.
- test_retime_moves_intro_words_into_the_first_turn_and_preserves_order: TED-like 0.18–4.2 s words land at 13.2/13.25/13.3/13.35, order kept, in-turn words untouched, segment bounds 13.2–13.75.
- test_retime_moves_words_sitting_in_a_long_gap_into_the_following_turn: gap word moves from 3.0 to 6.0, neighbours untouched.
- test_retime_never_shifts_a_word_further_than_max_shift / test_retime_is_a_no_op_when_every_word_is_already_inside_a_turn.
- test_diarize_skips_when_the_diarizer_is_unavailable / test_diarize_skips_without_a_hf_token / test_diarize_skips_when_the_option_is_off: all yield `skipped`/`unavailable` and no diarizer call.
- test_diarize_checkpoints_turns_exclusive_turns_and_embeddings: exact `{num_speakers, turns, exclusive_turns, embeddings}` output and the normalized audio path.
- test_align_retimes_intro_words_assigns_speakers_and_rewrites_artifacts: `retimed_words=4`, shares sum to 1, transcript JSON/txt with `S1/S2`, segments persisted, `<v S1>`/`<v S2>` in VTT with no mixed cues.
- test_align_skips_and_keeps_the_transcript_when_diarize_was_skipped: align skips and transcript bytes are unchanged.
- test_m2_registry_wires_diarize_and_align_and_keeps_m1_stages.
- Live (not run by worker): jfk → 1 speaker; `CARACAL_LIVE_TWO_SPEAKER_WAV` → 2 speakers; full `m2_registry` chain → words carry S1/S2 consistent with exclusive turns and VTT voice tags.

## Measured
- gate exit code: 0
- ```text
  ==> repository rules
  ==> pytest (unit + contract, live excluded)
  ........................................................................ [  9%]
  ........................................................................ [ 18%]
  ........................................................................ [ 27%]
  ........................................................................ [ 36%]
  ........................................................................ [ 46%]
  ........................................................................ [ 55%]
  ........................................................................ [ 64%]
  ........................................................................ [ 73%]
  ........................................................................ [ 83%]
  ........................................................................ [ 92%]
  ............................................................             [100%]
  780 passed in 7.18s
  GATE: PASS
  ```
- `uv lock --check`: "Resolved 164 packages" (exit 0); `torch==2.11.0+cu130`, `torchaudio==2.11.0+cu130`, `torchcodec==0.17.0+cu130`, `pyannote-audio==4.0.7` locked.
- Gate ran with the extras absent (`find_spec("torch")`/`find_spec("pyannote")` → None).

## Deviations
- Embeddings use pyannote's `DiarizeOutput.speaker_embeddings` (card §1 + docs/00 measured facts) instead of recomputing docs/04 §5's "centroid over turns ≥ 1.5 s"; recorded in `docs/findings/errata.md`.
- `stage_diarize` never passes `num_speakers`: `JobOptions` (docs/03 §2) carries `speaker_hints` (names) but no speaker-count hint, so there is no caller hint to forward. `Diarizer.run(..., num_speakers=...)` implements the plumbing and is unit-tested directly.
- docs/01 §3 writes `assign_speakers(transcript, turns)`; the stage passes the exclusive (non-overlapping) turns as the card requires, because the full annotation can contain overlaps.
- `retime_outside_speech` returns the input transcript object unchanged when nothing needs moving (cheap no-op); behaviourally identical to returning a copy.

## NOT proven
- The live tests were written but not executed (docs/07 §3: the worker never runs `tests/live/`). Real pyannote load time, CUDA execution, the 256-dim embedding shape and jfk=1-speaker behaviour must be verified by the judge on the GB10 after `uv sync --extra speakers`.
- The full `uv sync --extra speakers` install was not run here; only `uv lock` resolution was verified. `uv lock --check` and the extra's pinned versions are the evidence.
- No real two-speaker fixture exists in the repo; the live two-speaker tests skip unless `CARACAL_LIVE_TWO_SPEAKER_WAV` is set.
- The GPU device-fallback path was simulated with a fake torch namespace (`cuda.is_available()` false), not a machine without CUDA.

## Open
- None.
