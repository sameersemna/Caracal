STATUS: DONE
## What changed
- `src/caracal/asr/merge.py`: added pure `stitch_windows` (absolute, whole, ordered, non-overlapping segments; seam = A's last reliable segment end, else B.start; last window gets no end-cut; later segment starts pulled up to the previous end without touching words); `_cut` treats merely touching chunks as non-overlapping; `_kept_segment` gained keyword-only `preserve_bounds`; `merge_chunks` accepts `preserve_bounds` chunk indices.
- `src/caracal/pipeline/stages.py`: `stage_transcribe` stitches each recovered span's filtered windows with `stitch_windows`, converts the result to span-local time (`_span_local`) and merges it as ONE piece whose chunk is exactly the credit span; regions before/after stay adjacent. Budget, log, cancel, metrics, temp-file cleanup and `credit_recovered_spans` unchanged.
- `src/caracal/analysis/translate.py`: `_LOCALISED_LATIN_RULE` now uses neutral exonyms (Москва→Moscow, 東京→Tokyo, القاهرة→Cairo); the `name_rule` docstring no longer quotes the test clip.
- `tests/unit/test_merge.py`: real-window stitching test plus empty-A, A-touches-end and word-preserving clamp cases.
- `tests/unit/test_stages.py`: real-fixture stage test now asserts the exact recovered cue list, no `切換了警方的手机`, and the unchanged post-30 s fixture segments.
- `tests/unit/test_translate.py`: zh→en system prompt contains none of the test clip's names and uses the neutral examples.

## Tests added
- `test_stitch_windows_keeps_the_real_recovered_story_once_in_order`: on the four real `zh_intro_w*.json` (after `drop_window_credits`) the text list is exactly the nine required cues, starts strictly increase, segments never overlap, times/words are absolute and ids renumbered.
- `test_stitch_windows_keeps_b_whole_from_its_start_when_a_is_empty`: empty A makes the seam `B.start` and B is kept whole from 9.0 s.
- `test_stitch_windows_seam_falls_back_to_b_start_when_a_touches_its_end`: a segment touching A's end neither sets the seam nor survives; B is kept from 9.0 s.
- `test_stitch_windows_pulls_a_late_start_forward_without_trimming_words`: overlapping seam times move only B's reported start (9.5 s); its two words stay at 9.0/10.0 and its text stays whole.
- `test_transcribe_recovers_the_bahrain_story_under_the_real_zh_credit` (extended): stored cues from 9–30 s are exactly the nine cues with exact times; `字幕志愿者` and `切换了警方的手机` absent; segments ≥30 s equal the plain-merge reference unchanged.
- `test_zh_to_en_system_prompt_uses_neutral_name_examples`: the zh→en system prompt has none of `坦桑尼亞 坦桑尼亚 达蕾斯萨拉姆 達蕾斯薩拉姆 国际空间站 Tanzania "Dar es Salaam" "International Space Station"` and contains Moscow/Tokyo/Cairo.

## Measured
- gate exit code: 0
```
........................................................................ [ 40%]
........................................................................ [ 45%]
........................................................................ [ 50%]
........................................................................ [ 55%]
........................................................................ [ 60%]
........................................................................ [ 65%]
........................................................................ [ 70%]
........................................................................ [ 75%]
........................................................................ [ 80%]
........................................................................ [ 85%]
........................................................................ [ 90%]
........................................................................ [ 95%]
......................................................................   [100%]
1438 passed in 16.20s
GATE: PASS
```
- Mutation checks (run locally, then reverted): removing the stitch start clamp fails both the real-window merge test and the stage test; dropping `preserve_bounds` from the stage call fails the stage non-overlap/position assertions.

## Deviations
- docs/01 §4 documents `merge_chunks(results, chunks) -> Transcript`; I added a keyword-only `preserve_bounds: Collection[int] = ()` (compatible with every existing call). Reason: `merge_chunks` re-derives a segment's start/end from its word times whenever words exist; that would undo the seam start clamp `stitch_windows` applies (whisper times the same CJK word differently in two windows). The stitched piece's chunk index is listed so its deliberate bounds survive; all other merges keep the old behaviour (the evidence-index test that expects `0.2-15.4s` still passes).
- No round-48 stage expectation encoded the midpoint behaviour, so none was adapted; only new assertions were added.
- Observed in the stage test: `_clip_window_result` rebuilds a window segment's bounds from its last word, so the stored span starts/ends are `11.78`/`15.31` (the stitch function itself, which the card's Accept targets, yields `11.8`/`15.3` on the raw fixtures).

## NOT proven
- I did not run the live clip (`HPVvfcTOSJg`) against real whisper/gemma4; the card says the judge does that on an isolated instance. All stage tests use the recorded fixtures and `FakeWhisper`.
- No live network, Postgres, Redis, Ollama or whisper call was made in this session.

## Open
- Judge to confirm that the keyword-only `preserve_bounds` is acceptable plumbing, or specify the preferred alternative (post-merge boundary reconciliation in the stage).
