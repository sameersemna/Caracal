STATUS: DONE

## What changed
- `src/caracal/asr/__init__.py`: added the no-space script ranges (Han, Hiragana, Katakana, Thai, Lao, Khmer, Myanmar,
  Tibetan, halfwidth Katakana) and `word_groups` now starts a new group at every token whose first character is one of
  them; punctuation tokens, combining marks (Thai tone, Tibetan vowel) and empty/U+FFFD fragments still attach to the
  previous token, so Latin behaviour is untouched.
- `src/caracal/asr/whisper.py`: the parser turns a whitespace-only token into `space_before` on the next token (the zh
  fixtures carry the word separator as its own `" "` token) and repairs U+FFFD token texts from `segment.text`: intact
  tokens are anchors inside the whitespace-free text, the characters between two anchors are distributed over the
  fragment tokens between them (`floor(j * tokens / chars)`), continuation fragments become empty and are merged into the
  preceding token's time span; when the anchors do not line up, one token per segment character is spread over the
  segment span.
- `src/caracal/asr/merge.py`: credit-phrase filter — 12 normalised phrases (`字幕志愿者`, `字幕由`, `请不吝点赞`,
  `ご視聴ありがとうございました`, the Amara/ZDF/Sous-titres English/German/French lines, `Thanks for watching`,
  `Thank you for watching`) plus the `订阅`+`转发`+`打赏` conjunction, counted in `dropped["hallucinated_credit"]`;
  applied only when the segment lies wholly in the first/last 30 s or its midpoint sits in a measured silence span.
  `merge_chunks` gained keyword-only `duration_s=None` (default: end of the last chunk) and `silences=()`.
- `src/caracal/subtitles/segmenter.py`: no change needed (see Deviations) — with per-character groups the real Chinese
  cues already break between characters and no line exceeds 42 chars.
- tests: `tests/unit/test_word_groups.py`, `test_whisper_client.py`, `test_merge.py`, `test_segmenter.py` extended;
  one exact-dict assertion in `tests/unit/test_stages.py` (see Deviations).

## Tests added
- `test_multibyte_byte_fragments_are_repaired_from_the_segment_text[zh_news40|zh_speech40]`: every token text is a
  piece of whisper's own text, concatenation (minus spaces) equals the segment text, no token contains U+FFFD.
- `test_a_split_character_becomes_one_token_spanning_its_fragments`: `志 � � 者` becomes `志 愿 者`; `愿` carries the
  joint time span (21.4 → 25.67) and the whitespace-only token gives `李` its `space_before`.
- `test_chinese_response_through_the_client_has_no_replacement_character`: whole zh fixture through the HTTP client —
  103 words, none with U+FFFD, first characters `欧 洲 阿 里`.
- `test_unrepairable_alignment_falls_back_to_one_token_per_character`,
  `test_all_fragment_segment_is_spread_over_the_segment_span`: the two fallback paths (anchors unusable / no anchors),
  with exact per-character timings and mean probability.
- `test_latin_tokens_are_not_touched_by_the_repair`: jfk tokens unchanged.
- `test_real_chinese_tokens_without_spaces_are_each_breakable`: zh_speech40 has 0 space-led tokens, yet the first eight
  groups are `欧 洲 阿 里 亚 纳 火 箭`, the mean group length is < 2 chars and every token is covered exactly once.
- `test_chinese_punctuation_attaches_to_the_previous_character`: `水、食、气、` stay attached to their character.
- `test_other_no_space_scripts_break_between_characters[Thai|Hiragana|Katakana|Lao|Khmer|Myanmar|Tibetan]` and
  `test_a_combining_mark_stays_with_its_base_character`.
- `test_real_chinese_music_intro_credits_are_dropped`: zh_news40 → 0 segments, `hallucinated_credit == 2`.
- `test_real_chinese_speech_keeps_every_segment_but_the_trailing_credit`: the four real segments survive verbatim,
  `中文字幕:CJ` at 30 s is dropped.
- `test_credit_phrase_in_the_intro_window_is_dropped[...]` / `test_the_same_phrase_in_the_middle_of_a_talk_is_kept[...]`
  (12 phrases each): dropped at 0–4 s, kept at 300–304 s of a 600 s file.
- `test_credit_in_the_outro_window_is_dropped`, `test_credit_inside_a_measured_silence_span_is_dropped_mid_file`,
  `test_chinese_subscribe_begging_needs_all_three_verbs`, `test_a_plain_thank_you_is_not_a_credit`.
- `test_real_chinese_cues_break_between_characters_without_a_space_signal` (replaces the adapted synthetic CJK test):
  real zh_speech40 → ≥5 cues, `cue 1 == ["欧洲阿里亚纳火箭"]`, every line ≤ 42 chars and free of U+FFFD, rendered text
  equals the transcript text, no `字幕`, cue spans cover every word once.
- `test_a_latin_sub_word_token_still_continues_its_word`: `Tagesschau` is still one group.

## Measured
- gate exit code: 0
- ```
==> ruff check
==> ruff format --check
==> mypy (strict)
Success: no issues found in 71 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 37%]
........................................................................ [ 42%]
........................................................................ [ 48%]
........................................................................ [ 53%]
........................................................................ [ 58%]
........................................................................ [ 64%]
........................................................................ [ 69%]
........................................................................ [ 75%]
........................................................................ [ 80%]
........................................................................ [ 85%]
........................................................................ [ 91%]
........................................................................ [ 96%]
...............................................                          [100%]
1343 passed in 12.48s
GATE: PASS
```
- zh_news40: 2/2 segments dropped as `hallucinated_credit`; zh_speech40: 4 segments kept, `中文字幕:CJ` dropped,
  96 words, 5 cues, longest line 31 chars.
- Latin fixtures unchanged: talk_c0+c1 still 404 words / 49 segments with every untrimmed text verbatim; jfk tokens
  identical; `tagesschau_seg0` groups identical.
- Mutation checks run by hand (each reverted): removing the no-space rule → 11 test failures; removing the repair → 7;
  removing the credit filter → 16; applying the filter everywhere (no edge/spans) → 12.

## Deviations
- `tests/unit/test_stages.py` (not in the Files list): one exact-dict assertion now includes
  `"hallucinated_credit": 0`, because `Transcript.dropped` gained the counter the verdict requires and that test
  compares the whole dict. This is the same kind of judge-side omission accepted in verdict 18 ("not in the Files list
  — the judge's omission … accepted"); the assertion is not weakened, the new key is asserted explicitly.
- `src/caracal/asr/merge.py` signature: `merge_chunks(results, chunks)` (docs/01 §3) gained keyword-only
  `duration_s` and `silences`. The verdict's rule needs the file duration and the normalize silence/music spans; the
  extension is backwards compatible (all existing callers and doc signatures still work).
- Fix 3's silence/music branch is implemented and unit-tested but not wired into the pipeline: `stage_transcribe`
  (`src/caracal/pipeline/stages.py`, outside this card's Files) calls `merge_chunks(results, chunks)` and never reads
  `normalize`'s `stats.silences`, so in a real job only the "first/last 30 s" rule is active. Wiring it is a ~5-line
  change in `stages.py`.
- The phrase list goes beyond the verdict's "at least" list with `中文字幕`: the real zh_speech40 hallucination is
  `中文字幕:CJ` (30.0–31.0 s of a 33.5 s file), which contains `字幕` but neither `字幕志愿者` nor `字幕由`. It is
  still edge-restricted, so a mid-talk mention is kept.
- `src/caracal/subtitles/segmenter.py` is listed in Files but needed no edit: cue/line breaking already works on word
  groups, and the measured output for the real Chinese transcript is 5 cues with a longest line of 31 chars. Editing
  it would have been a no-op change.
- The adapted synthetic CJK test `test_cjk_punctuation_breaks_between_word_groups` was replaced by real behaviour
  (`test_real_chinese_cues_break_between_characters_without_a_space_signal`), as fix 1 and the card require.

## NOT proven
- The live Chinese (HPVvfcTOSJg) and Tagesschau re-runs: no live whisper/network calls are allowed from this session;
  the judge runs them. The pipeline will run them with only the edge rule of the credit filter (see Deviations).
- The silence/music branch end to end (unit-tested through `merge_chunks(..., silences=…)` only).
- Thai/Lao/Khmer/Myanmar/Tibetan behaviour on real whisper output (no fixtures): only synthetic two-token cases and a
  combining-mark case.
- Repair when `segment.text` itself contains U+FFFD: nothing better exists, the fallback spreads the broken characters
  and would still emit U+FFFD; the measured fixtures all have correct segment text.
- A judge-side "unchanged Latin" diff beyond the fixtures: tagesschau_seg0 is a 14-token prefix, so the full German
  re-run is only covered by its live check.

## Open
- Should the next card add `silences=normalize.stats.silences` (and the plan's `music_detected` as a whole-file
  non-speech span) to the `stage_transcribe` call? That file is outside this card's Files list.
- Keep `中文字幕` in the credit list? It can match a genuine opening mention of Chinese subtitles, but only within the
  first/last 30 s.
