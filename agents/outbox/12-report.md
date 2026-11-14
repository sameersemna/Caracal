STATUS: DONE
## What changed
- `src/caracal/asr/__init__.py`: `Word.space_before: bool` (after `speaker`, default `False`) records whisper's leading-whitespace signal; new `join_words(words)` inserts a space only where `space_before` is True; `Word.to_dict` serialises the flag.
- `src/caracal/asr/whisper.py`: `parse_whisper_response` keeps the raw token, strips only the stored text, and sets `space_before` from the token's leading whitespace (first word of a segment → `False`).
- `src/caracal/asr/merge.py`: `_kept_segment` leaves an untrimmed segment's `text.strip()` verbatim; only a trimmed segment is rebuilt with `join_words`.
- `tests/unit/test_merge.py`: real-pair verbatim/artefact tests, synthetic trimmed and untrimmed tests, `join_words` test; `_word` helper gained `space_before` (default `True` for synthetic space-separated fixtures).
- `tests/unit/test_whisper_client.py`: parser test for `space_before` on real `jfk` tokens (including attached `","`).

## Tests added
- `test_untrimmed_segments_keep_whisper_text_verbatim`: with the real `talk_c0`/`talk_c1` pair, 47 of 49 raw segments lose no words and every one of their texts appears verbatim in the merged transcript; only the two segments at the 60.5 s seam are rebuilt (49 merged segments, no drops). Fails when the verbatim-text branch is reverted (verified: 4 merge tests fail).
- `test_merge_does_not_add_punctuation_spacing_artefacts`: `r" '(re|s|t|ve|ll|d|m)\b| [,.?!;:]"` hits exactly the one raw `"and ..."` segment and no merged contraction/punctuation artefact; merged hits ≤ raw hits.
- `test_trimmed_segment_is_rebuilt_with_whisper_spacing`: synthetic `[" I", "'m", " here", ","]` cut after `"'m"` merges to `text == "I'm"` (5.0–7.0 s) with words `["I", "'m"]`; also fails on the verbatim-branch revert.
- `test_untrimmed_synthetic_segment_keeps_its_original_text`: no cut → `"I'm here,"` is not rerendered.
- `test_join_words_keeps_contractions_and_attached_punctuation`: `join_words` → `"I'm here,"`, empty input → `""`.
- `test_word_space_before_records_whisper_whitespace`: real `jfk` segment yields `("And", False), ("so", True), …, (",", False)`, first word of every segment `False`, and `space_before` is serialised in `to_dict`. Fails when the parser drops the signal (verified: 1 failure).

## Measured
- gate exit code: 0
- ```
==> mypy (strict)
Success: no issues found in 36 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 12%]
........................................................................ [ 24%]
........................................................................ [ 37%]
........................................................................ [ 49%]
........................................................................ [ 61%]
........................................................................ [ 74%]
........................................................................ [ 86%]
........................................................................ [ 99%]
.....                                                                    [100%]
581 passed in 4.99s
GATE: PASS
```

## Deviations
- `Word.space_before` was appended after `speaker` (default `False`) rather than before it, so no existing positional construction shifts. Verdict 11 required the field but not its position.
- `Word.to_dict` now emits `space_before`; the verdict did not ask for serialisation, but checkpoints must round-trip the signal that `join_words` depends on.
- `tests/unit/test_merge.py::_word` now defaults `space_before=True` (previously implicit space-separated words). This is the declared constructor adaptation the card permits; existing synthetic expectations (`"beta gamma"`) are unchanged. Real-fixture tests use parser-set values.
- `src/caracal/asr/captions.py` was not touched: it builds `CaptionWord`, not `Word` (the card allows it only if it builds `Word`s).

## NOT proven
- Did not run the judge's live 12-min whisper + YouTube-caption check (no live network/whisper calls from the worker per docs/00 and docs/07). The unit evidence uses the recorded real `talk_c0`/`talk_c1` pair; the judge's artefact-count ≤ raw and agreement ≥ 0.80 checks remain to be re-run by the judge.
- The recorded pair's synthetic seam cuts mid-contraction at 60.5 s, so the two trimmed segments rebuild as `"I couldn"` and `"'t go back …"`. That is faithful to word ownership at an artificial seam; production `plan_chunks` places seams at detected silences (T1.3), and the round-11 live run reported natural seams.

## Open
- none
