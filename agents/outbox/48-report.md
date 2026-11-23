STATUS: DONE
## What changed
- `src/caracal/asr/merge.py`: added `credit_spans` (> `min_s`, absolute, 0.1 s snap), `recovery_windows` (10 s / 1 s overlap, tail window `[max(last, end-size), end]`) and `drop_window_credits` (drops whole credit segments from window results before the overlap trim can cut the credit text).
- `src/caracal/pipeline/stages.py`: `stage_transcribe` recovers up to 6 credit spans per job — cuts every recovery window from `audio_wav` into `transcribe/credit-<chunk>-<start>-<end>.wav`, transcribes with the same client/language, deletes it, counts whisper seconds, checks cancel between windows, logs `transcribe.credit_recovered`, records `dropped.credit_recovered_spans`, and merges the chunk's remaining pieces plus the windows through `merge_chunks` (no second overlap algorithm). `stage_translate` writes `script_leaks` per language.
- `src/caracal/analysis/translate.py`: script detection (`script_of_text`, `target_script`, `has_script_leak`), `name_rule(target_lang, source_script)` replacing the two old rules, batch-level script-leak rejection (same retry path as an index mismatch), per-unit retry with the extra line `Write every name in the Latin script.`, leaked units kept and counted in `TranslatedUnits.script_leaks`.
- `src/caracal/llm/prompts/translate.md`: glossary line now defers to the rule and says the names are in the source script; added a line telling the model to write the whole unit in the target script.
## Tests added
- `tests/unit/test_merge.py::test_credit_spans_of_the_real_zh_intro_is_the_whole_first_window` — `zh_intro_full` → `[(0.0, 30.0)]`.
- `tests/unit/test_merge.py::test_credit_spans_ignores_a_credit_shorter_than_eight_seconds` — a 3 s credit yields no span.
- `tests/unit/test_merge.py::test_credit_spans_shift_to_absolute_time_for_a_later_chunk` — chunk at 60 s → `(60.0, 72.0)`.
- `tests/unit/test_merge.py::test_recovery_windows_cover_the_span_with_ten_second_windows` — table for `(0,30)`, `(5,12)`, `(0,10)`, `(0,19)`.
- `tests/unit/test_merge.py::test_drop_window_credits_drops_a_pure_music_window_whole` — a 10 s credit is removed whole, real speech in the neighbouring window survives.
- `tests/unit/test_merge.py::test_drop_window_credits_drops_a_credit_that_repeats_mid_file` — a recurring credit is dropped even away from the edges.
- `tests/unit/test_stages.py::test_transcribe_drops_a_credit_even_when_every_window_is_pure_music` — every window returns the credit: transcript has only the two real segments, `credit_recovered_spans == 1`, exactly the four window calls, temp files gone.
- `tests/unit/test_stages.py::test_transcribe_recovers_the_bahrain_story_under_the_real_zh_credit` — real `zh_intro_full` + the four `zh_intro_w*.json`: `巴林`/`麦纳麦`/`珍珠廣場` present, no `字幕志愿者`/`中文字幕`, `反對派說` once, first segment ≥ 9 s, ordered and non-overlapping, everything after 30 s identical to a plain merge of the fixture, `credit_recovered_spans == 1`, 4 extra calls named `credit-000-{0,9,18,20}-*.wav`.
- `tests/unit/test_stages.py::test_transcribe_recovers_at_most_six_credit_spans_per_job` — 7 mid-file credits → `credit_recovered_spans == 6`, six recovered texts, the seventh credit left (and kept by the normal filter), 7 whisper calls.
- `tests/unit/test_stages.py::test_stage_translate_counts_a_script_leak_that_survives_the_retry` — `stage_translate` output carries `script_leaks == 1` and keeps the leaked German translation.
- `tests/unit/test_translate.py::test_name_rule_is_chosen_from_the_source_and_target_script` — table `zh→en`, `ar→en`, `en→de`, `en→ar`, `de→en`.
- `tests/unit/test_translate.py::test_script_of_text_reads_the_units_not_a_language_code` and `::test_digits_punctuation_and_symbols_are_never_a_script_leak` — Han/Arabic/Latin detection; `12 °C`/`€5` clean, Han and Cyrillic leak.
- `tests/unit/test_translate.py::test_a_leaked_name_is_retried_and_the_latin_translation_wins` — first `In 坦桑尼亞, police said`, retry `In Tanzania, police said` → Latin text, `script_leaks == 0`.
- `tests/unit/test_translate.py::test_a_unit_that_keeps_leaking_is_kept_and_counted` — always leaking → translation kept (not source), `script_leaks == 1`, 4 calls, last system has the extra Latin line.
- `tests/unit/test_translate.py::test_a_non_latin_target_is_not_checked_for_script_leaks` — `zh` target with Han output is clean, one call.
## Measured
- gate exit code: 0
- ```
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
.................................................................        [100%]
1433 passed in 15.10s
GATE: PASS
```
## Deviations
- `credit_spans` snaps bounds to 0.1 s. The fixture's credit segment is `0.0-29.98`, but the card's Accept wants `(0.0, 30.0)` and `recovery_windows(0, 30)` is the only span that produces the measured `(20, 30)` tail window; whisper reports segment ends a hair short of its own 30 s decode window. Documented in the helper's docstring.
- Credit windows are filtered *whole* by `drop_window_credits` before the merge (docs/04 §4 says "merged … and filtered like everything else"). Doing it after the merge lets the `(0,10)`/`(9,19)` cut at 9.5 s trim `中文字幕志愿者 李宗盛` to `中文`, which no longer matches a credit, so it would survive at 0.08 s and break the card's "first kept segment ≥ 9 s". The helper uses the same `is_hallucinated_credit` / loop / edge / silence predicates, it is not a second overlap algorithm, and the surviving results still go through `merge_chunks`.
- `name_rule` gained `source_script: Script = "latin"` with a default so the existing Latin-source prompt tests keep their meaning; `translate_units` always passes the script detected from the units.
- `tests/unit/test_stages.py::test_transcribe_writes_transcript_segments_and_removes_chunks` now expects the fifth `dropped` key `credit_recovered_spans: 0`; the card mandates that field.
## NOT proven
- No live run of `scripts/caracal-youtube.sh` on HPVvfcTOSJg `DEPTH=deep` (worker sessions make no live network calls). The `.srt`/`.en.srt` acceptance is the judge's live check.
- That the real gemma4 follows the new name rule; only the detection, retry and counting paths are proven with scripted fakes.
- Credit recovery on real multi-chunk audio: the over-budget case uses a synthetic 620 s tone with fabricated credits, not a recorded multi-chunk fixture.
## Open
- None.
