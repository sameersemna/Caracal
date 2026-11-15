STATUS: DONE
## What changed
- `src/caracal/asr/whisper.py`: every `POST /inference` multipart body now sends `max_context=0` (and the module docstring says why).
- `src/caracal/asr/merge.py`: added `credit_matches()`; a credit phrase that recurs across segments is treated as a decoding loop and **all** matching segments are dropped wherever they sit, while a lone credit keeps the existing edge/silence rule.
- `src/caracal/subtitles/segmenter.py`: exported the script-aware limit as `line_chars` (was `_line_chars`); no behaviour change.
- `src/caracal/analysis/translate.py`: translated cue text wraps at `line_chars(chunk, 42)` (16 for Han/Kana/Hangul/Thai) instead of a fixed 42, and Thai joins the character-break targets.
- `tests/unit/test_whisper_client.py`, `tests/unit/test_merge.py`, `tests/unit/test_translate.py`: new assertions and tests.

## Tests added
- `test_request_is_explicit_multipart_and_parses_real_fixture`: the request form carries `max_context == "0"`.
- `test_forced_language_uses_real_detection_not_the_echo`: `max_context == "0"` is also sent with a forced language.
- `test_real_chinese_context_loop_is_dropped_entirely`: the judge's `zh_full_loop.json` (4 segments: credit at 0/30/60/90 s) yields 0 kept segments and 4 credit drops.
- `test_a_recurring_credit_is_dropped_even_in_the_middle_of_a_talk`: two mid-audio occurrences of the same phrase are both dropped (position/silence would have kept them).
- `test_two_different_one_off_credits_are_not_a_loop`: two different lone credit phrases are judged individually — only the one inside measured silence is dropped.
- `test_a_looped_credit_does_not_take_a_real_sentence_with_it`: a real Chinese sentence between two looped credits survives.
- `test_translated_cjk_and_thai_lines_stay_within_sixteen_characters[zh|ja|th]`: every translated cue line is ≤ 16 characters, there are ≥ 2 lines, and the characters are preserved.

Mutation checks (run, then reverted): removing `max_context` fails both whisper request tests; restoring `max_chars=_MAX_LINE_CHARS` fails all three zh/ja/th wrapping cases; forcing `_is_looped_credit` to `False` fails the three loop tests (real fixture + synthetic + real-sentence-survives).

## Measured
- gate exit code: 0
```
........................................................................ [ 95%]
............................................................             [100%]
1356 passed in 11.89s
GATE: PASS
```

## Deviations
- Requirement 2 is implemented as a **recurring credit phrase** (the same phrase in ≥2 segments), not "any two credit-matching segments". Reason: `tests/unit/test_stages.py::test_transcribe_passes_duration_and_silences_to_the_credit_filter` (outside this card's Files list, and existing tests may not be edited) contains two *different* one-off credits — "Thanks for watching" inside measured silence and "Subtitles by the Amara.org community" mid-speech — and requires only the first to be dropped. Reading "a single match keeps the current position/silence rule" per phrase satisfies both: on the judge's fixture `字幕志愿者` recurs 3× and `中文字幕` 2×, so all 4 segments still drop. Card 45 requirement 2, docs/04 §4.
- `th` was added to the character-break target set, since requirement 3 says "CJK/Thai targets"; docs/04 §10 only names CJK for the 16-character guideline. Card 45 requirement 3 and docs/04 §10.
- No docs were edited; `docs/findings/errata.md` was not needed.

## NOT proven
- The Chinese clip was not re-run end to end against the live whisper.cpp server (the card says the judge does this); worker tests make no live network calls.
- Only the mock asserts the request body, so I did not verify that the deployed whisper.cpp build accepts the `max_context` multipart field; docs/00 records that field name as measured.
- Thai wrapping is shown on a synthetic Thai string, not a real Thai translation from the LLM.

## Open
- If the judge intends "≥2 credit-matching segments of *any* phrase" literally, the per-phrase reading above must be revisited together with that pre-existing stage test.
- A talk that legitimately repeats a credit-like sentence twice (e.g. a host saying "thanks for watching" at two breaks) will now lose both; this is the card's stated rule, recorded here in case the owner wants a duration/chunk guard later.
