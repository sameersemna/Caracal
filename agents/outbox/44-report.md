STATUS: DONE
## What changed
- `src/caracal/subtitles/segmenter.py`: per-cue line limit by script — ≥50 % Han/Kana/Hangul/Thai characters ⇒ 16 per line (still 2 lines, still 42 for other scripts); a whisper segment boundary is now a preferred cue break in every script and a hard break for CJK/Thai text whose cue-so-far does not end in sentence punctuation; word grouping still spans segments so a word whisper cut at a chunk boundary (`couldn'` | `'t`) is never split.
- `src/caracal/pipeline/stages.py`: `stage_transcribe` reads `duration_s` and the normalize `silences` from the normalize checkpoint and passes both to `merge_chunks`.
- `tests/unit/test_segmenter.py`: new fix-1/fix-2 tests plus the adapted real-fixture assertions (declared below).
- `tests/unit/test_stages.py`: `_normalize_checkpoint` helper, normalize checkpoints in the three existing transcribe setups, and the new duration/silence wiring test.

## Tests added
- `test_cjk_line_limit_is_sixteen_characters_while_latin_keeps_forty_two`: a 29-character Han cue wraps as 14/15 characters per line, a 65-character Latin cue keeps two lines within 42 and wider than 16.
- `test_segment_boundary_is_a_preferred_break_for_latin_text`: a punctuation-free 17-character first segment breaks at the segment boundary (two cues, one segment each), while an 8-character first segment keeps spanning both segments — preferred, not hard.
- `test_segment_boundary_is_a_hard_break_for_cjk_without_sentence_punctuation[zh,th]`: two short CJK/Thai segments (19 and 22 characters, together well under 2×16) still produce two cues, each wholly inside one segment.
- `test_a_word_cut_across_a_segment_boundary_stays_one_word`: with `max_chars=10` the wrap keeps `couldn't` intact (`["I couldn't", "go back"]`), so per-segment grouping cannot silently reappear.
- `test_transcribe_passes_duration_and_silences_to_the_credit_filter`: on a 600 s timeline a credit at 100 s inside the measured silence is dropped while a credit at 200 s in speech is kept — removing either `duration_s` or `silences` changes the result (old defaults drop both).
- Adapted real fixtures (fix 1/2): `test_real_chinese_cues_break_between_characters_without_a_space_signal` now asserts every line ≤ 16, exactly 6 cues (was 5), no cue spanning two segments, and full coverage/rendering; `test_real_merged_transcript_satisfies_every_cue_invariant` asserts 43 cues (was 36) and 10 cues ending without punctuation at a segment boundary (was 1). The test's independent break helper now accepts a segment boundary as a legitimate break point, as the new rule documents.

## Measured
- gate exit code: 0
```
........................................................................ [ 37%]
........................................................................ [ 42%]
........................................................................ [ 48%]
........................................................................ [ 53%]
........................................................................ [ 58%]
........................................................................ [ 64%]
........................................................................ [ 69%]
........................................................................ [ 74%]
........................................................................ [ 80%]
........................................................................ [ 85%]
........................................................................ [ 90%]
........................................................................ [ 96%]
.....................................................                    [100%]
1349 passed in 12.47s
GATE: PASS
```
- Real fixtures: `zh_speech40` 6 cues, max line 14, 0 cues spanning two source segments, 0 U+FFFD; TED pair 43 cues (36 before), 8 cues still span segments only because the preferred rule does not fire on a short cue or the boundary sits inside the chunk-cut word; Tagesschau test unchanged and green.
- Mutation checks (each fix removed, targeted tests rerun): no CJK limit → CJK tests fail; no hard CJK break → zh real + both hard-break cases fail; no preferred segment break → TED + Latin preferred test fail; no `duration_s`/`silences` forwarded → stage wiring test fails (`2 != 1`); per-segment grouping → TED count and word-cut test fail.

## Deviations
- `stage_transcribe` treats a **missing** normalize checkpoint as "duration and silences unknown" and lets `merge_chunks` fall back to the chunk timeline (its documented `duration_s is None` behaviour). Verdict 43 fix 3 requires the values when they exist; the concurrency test's partial registry (`tests/unit/test_concurrency.py`, outside this card's Files) runs transcribe without normalize, so raising would require editing a file the card forbids. In the real pipeline normalize always precedes transcribe.
- Adapted expectations (declared, per the card): the zh cue count/line limit changed by fix 1 (42 → 16, `>= 5` → `6`), and the TED cue count changed by fix 2 (36 → 43); the test helper `_ends_at_punctuation_or_hard_break` now counts a segment boundary as a valid cue ending. Three existing transcribe tests gained a normalize checkpoint because the stage now reads it — a required input, not a weakened assertion.
- `wrap_lines` (translated text) is unchanged: `src/caracal/analysis/translate.py` passes `max_chars=42` explicitly for CJK targets and is outside this card's Files; only source cues from `cues_from_transcript` got the 16-character rule.

## NOT proven
- The live Chinese and Tagesschau clips were not run (no live network/subprocess runs in this session); the rules are verified on the recorded real fixtures and synthetic Han/Thai text.
- No real Kana/Hangul/Thai whisper fixture exists; those scripts are covered by synthetic segmented tests and codepoint-range unit coverage only.
- Translated CJK subtitle line length (`.en.srt` is Latin; other targets) is unchanged and untested against the 16-character guideline.

## Open
- Should translated CJK subtitles also use 16 characters per line? That needs `src/caracal/analysis/translate.py` (`_MAX_LINE_CHARS`), outside this card's Files.
- Should a missing normalize checkpoint be a hard error for transcribe in production? Making it one requires editing `tests/unit/test_concurrency.py` (outside this card's Files).
