STATUS: DONE
## What changed
- `src/caracal/asr/whisper.py`: `space_before` now records whisper's raw leading whitespace for **every** word, including each segment's first (carry-forward).
- `src/caracal/asr/__init__.py`: `Word.space_before` docstring corrected; `join_words` still never emits a leading space and now attaches an isolated sentence-punctuation token (whisper's `"..."`) so no `"and ..."` artefact survives.
- `src/caracal/subtitles/__init__.py`: exports `Cue`, `cues_from_transcript`, `to_srt`, `to_vtt`.
- `src/caracal/subtitles/segmenter.py`: `Cue` + `cues_from_transcript` — flat-word cue building with speaker/pause/size hard breaks, sentence-then-clause preferred breaks, balanced 2-line wrapping, min/max duration, min gap and inclusive `word_span`.
- `src/caracal/subtitles/formats.py`: `to_srt`, `to_vtt` (ms rounding, 2+ digit hours, VTT escaping, `<v S1>` tags, single trailing newline).
- `tests/unit/test_segmenter.py`: real-transcript invariants plus targeted break/timing/CJK/line-split tests.
- `tests/unit/test_subtitle_formats.py`: exact SRT/VTT, escaping, 100-hour timestamps, round-trip through `asr.captions.parse_vtt`, ffmpeg validation.
- `tests/unit/test_merge.py`: cross-segment seam spacing on the real `talk_c0` fixture and isolated-punctuation joining.
- `tests/unit/test_whisper_client.py`: expectations updated for the new first-word rule (declared under Deviations).

## Tests added
- `test_real_merged_transcript_satisfies_every_cue_invariant`: on merged `talk_c0`+`talk_c1` (404 words, 35 cues) every line ≤42 (or one long word), ≤2 lines, cue ≤7.0 s, strictly ordered, gap ≥0.08 s, contiguous word spans cover 0..403 exactly once, ≥80 % of cues end at punctuation or a rule-forced hard break, and the artefact regex matches nothing.
- `test_speaker_change_always_breaks_and_never_mixes_speakers`: S1/S2 alternation yields `["S1","S2","S1"]` cues, each single-speaker.
- `test_pause_longer_than_one_second_starts_a_new_cue`: 1.5 s pause splits, and the short tail is extended to `min_dur`.
- `test_max_duration_splits_a_long_run_of_words`: 9 s of contiguous words becomes ≥2 cues, none >7 s, all words covered.
- `test_sentence_punctuation_ends_a_long_enough_cue` / `test_clause_punctuation_ends_a_cue_near_the_line_limit`: exact preferred-break placements.
- `test_two_line_split_balances_and_prefers_a_punctuation_boundary`: 52-char cue splits at `jumps,` into two balanced ≤42 lines.
- `test_a_single_word_longer_than_a_line_stays_on_its_own_line`: 64-char word gets its own line.
- `test_cjk_punctuation_breaks_without_inserting_spaces`: `。` ends a 15-char CJK cue; no spaces inserted.
- `test_min_duration_is_extended_but_never_up_to_the_next_cue` / `test_a_gap_smaller_than_min_gap_pulls_the_earlier_end_back`: exact timings (1.0 s extension; 0.08 s pull-back).
- `test_srt_exact_text_for_two_cues` / `test_vtt_exact_text_for_two_cues`: exact full strings, including `<v S1>` and ms rounding.
- `test_vtt_voice_tags_can_be_disabled`, `test_vtt_escapes_markup_and_ampersands_while_srt_keeps_the_text`, `test_hours_may_exceed_two_digits`, `test_both_formats_end_with_exactly_one_newline`: format details.
- `test_vtt_round_trips_through_the_platform_caption_parser`: our VTT parsed by `parse_vtt` yields the same token sequence.
- `test_ffmpeg_accepts_the_rendered_files`: ffmpeg demuxes both real rendered files with an empty error log.
- `test_join_words_keeps_the_space_across_a_real_segment_seam`: every alnum/alnum seam equals `"<left> <right>"` (`. I`, `40 degrees`), so no `totalcatastrophe`/`effects.That's`.
- `test_join_words_attaches_a_detached_punctuation_token`: `"..."` joins as `and...`.
- Updated `test_word_space_before_records_whisper_whitespace`: first `jfk` word now `("And", True)` and every segment's first word is `True`.

## Measured
- gate exit code: 0
```
==> mypy (strict)
Success: no issues found in 39 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 11%]
........................................................................ [ 23%]
........................................................................ [ 35%]
........................................................................ [ 47%]
........................................................................ [ 59%]
........................................................................ [ 71%]
........................................................................ [ 83%]
........................................................................ [ 95%]
..........................                                               [100%]
602 passed in 5.17s
GATE: PASS
```

## Deviations
- **`join_words` attaches isolated sentence punctuation** (`asr/__init__.py`, allowed by the card "only if `join_words` needs it"): the real fixture contains word `"..."` with `space_before=True`, so plain joining produces `"and ..."`, which the card's mandatory regex `r" [,.?!;:]"` matches. Attaching tokens made only of `.,?!;:` removes the artefact without changing any untrimmed segment text (the only such token is in an untrimmed segment); `"I'm"`/`","` behaviour is unchanged.
- **Hard break also enforces renderability**: a cue is closed before a word if the resulting text would need more than `max_lines` lines of `max_chars` (greedy word wrap), not only if `join_words` text exceeds `max_lines × max_chars`. With the literal rule alone, three real cues produced 43–48-char lines, failing the card's line invariant; the extra check only breaks earlier.
- **"near the limit" (clause break) = cue length ≥ `max_chars − 8` (34)**; the card leaves the threshold unspecified.
- **ffmpeg validation command**: the card's literal `ffmpeg -v error -i out.srt -f null -` exits **234** on this host even for a hand-written valid SRT (`Output file does not contain any stream`; ffmpeg 6.1.1, null muxer has no subtitle encoder). The test uses `ffmpeg -v error -i <file> -c:s copy -f null -`, which demuxes/stream-copies the subtitle file and exits 0 for both SRT and VTT; stderr must be empty.
- **Updated old first-word expectations** in `tests/unit/test_whisper_client.py` as the card allows: `jfk` segment 0 first word is now `("And", True)`, and `all(segment.words[0].space_before is True)`; the docstring/name were changed to match the carry-forward.

## NOT proven
- No live pipeline run: no stage consumes `subtitles/` yet (out of scope for this card), and no real whisper/ffmpeg end-to-end invocation was made.
- Speaker cues were exercised with synthetic `S1`/`S2` labels only; diarization is a later milestone.
- The 100-hour timestamp is asserted as text only; ffmpeg validation was run on the real ~2-minute transcript, not on a 100-hour file.
- ffmpeg acceptance was measured with `-c:s copy` (see Deviations), not the card's literal command.

## Open
- Judge to confirm the ffmpeg command substitution and the ellipsis-attachment rule in `join_words`.
