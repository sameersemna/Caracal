STATUS: DONE
## What changed
- `src/caracal/asr/__init__.py`: added `WordGroup` and `word_groups` (tokens continue a group while `space_before` is false; punctuation attaches; text/start/end/token span; majority speaker with first-token tie-break; first token always starts a group).
- `src/caracal/subtitles/segmenter.py`: cue spans, break rules, 42-char limit, line wrapping and `Cue.word_span` now work on word groups; `word_span` still holds token indices and covers whole groups; a group longer than `max_chars` stays whole on its own line.
- `src/caracal/asr/captions.py`: `caption_agreement` builds the transcript side from `word_groups(segment.words)` (`group.text`, group midpoint) instead of raw tokens.
- `src/caracal/speakers/align.py`: `assign_speakers` picks one speaker per group from the group midpoint and writes it to every token; segments still split only at group-level speaker changes.
- `src/caracal/ingest/router.py`: added pure `canonical_url` — lower-case scheme/host, fragment and tracking params dropped; YouTube `watch?v=`, `youtu.be/<id>`, `shorts/`, `embed/`, `live/`, `v/` under `youtube.com`/`www.`/`m.`/`music.` with a valid 11-char id → `https://www.youtube.com/watch?v=<id>`; otherwise cleaned URL; invalid id left cleaned but unmapped.
- `src/caracal/services.py`: `create_job` stores/uses the canonical URL for dedup, falls back to the raw URL for pre-change rows; uploads still dedup by sha256.
- `tests/unit/test_word_groups.py`: new file.
- `tests/unit/test_segmenter.py`, `test_captions.py`, `test_align.py`, `test_ingest_router.py`, `tests/contract/test_api_jobs.py`: new coverage.
## Tests added
- `test_real_tagesschau_tokens_form_whole_words`: the real fixture groups into `["Guten","Tag,","hier","ist","die","Tagesschau","in","100","Sekunden."]` with exact global token spans.
- `test_group_times_span_their_first_and_last_token` / `test_punctuation_tokens_attach_to_the_preceding_word`: group start/end and punctuation attachment.
- `test_first_token_always_starts_a_group_even_without_a_leading_space`, `test_empty_input_has_no_groups`.
- `test_speaker_is_the_group_majority`, `test_speaker_tie_goes_to_the_first_token`, `test_unassigned_tokens_do_not_outvote_a_speaker`: majority + tie rule.
- `test_real_tagesschau_cues_and_lines_break_only_between_word_groups`: with `max_chars=20` every cue starts on a token with `space_before is True`, every rendered word is one whole group in order (no line split inside `Tagesschau`/`Sekunden.`), a two-line wrap is exercised.
- `test_a_word_split_into_tokens_and_longer_than_a_line_stays_whole`: a 60-char word made of two continuation tokens stays whole on one line.
- `test_token_split_transcript_scores_the_same_as_whole_words`: token-split vs whole-word transcript both score 1.0 against whole-word captions.
- `test_group_straddling_a_turn_boundary_gets_one_speaker_on_every_token`: midpoint-in-S2 group gives S2 to every token and no segment split inside the word.
- `test_youtube_shapes_canonicalise_to_one_watch_url`: 10 shapes incl. `m.`, `music.`, `&t=42s`, `&si=`, `&feature=share`, uppercase scheme/host, fragment.
- `test_non_youtube_urls_keep_their_path_and_clean_the_query`, `test_youtube_url_without_a_valid_id_is_only_cleaned`: tracking removal and invalid-id behaviour.
- `test_job_stores_the_canonical_youtube_url`: API job row and status echo store `https://www.youtube.com/watch?v=<id>`.
- `test_shorts_url_deduplicates_a_done_watch_job`: `shorts/<id>` after a done `watch?v=<id>` returns the same id with `deduplicated: true`; `depth: deep` still creates a second job.
- `test_dedup_falls_back_to_the_raw_url_of_a_legacy_row`: a pre-change raw `youtu.be/<id>` row is still found.
## Measured
- gate exit code: 0
- ```text
  ........................................................................ [ 33%]
  ........................................................................ [ 38%]
  ........................................................................ [ 44%]
  ........................................................................ [ 50%]
  ........................................................................ [ 55%]
  ........................................................................ [ 61%]
  ........................................................................ [ 66%]
  ........................................................................ [ 72%]
  ........................................................................ [ 77%]
  ........................................................................ [ 83%]
  ........................................................................ [ 88%]
  ........................................................................ [ 94%]
  .......................................................................  [100%]
  1295 passed in 12.80s
  GATE: PASS
  ```
## Deviations
- Adapted `tests/unit/test_segmenter.py::test_cjk_punctuation_breaks_without_inserting_spaces` → `test_cjk_punctuation_breaks_between_word_groups`: it encoded token-level behaviour by giving each CJK word `space_before=False`; under the mandated grouping rule those tokens are one word group. The test now uses `space_before=True` per CJK word (so each is a group) and expects spaces between them; it still proves `。` ends a cue.
- Adapted `tests/unit/test_captions.py::test_agreement_is_one_for_identical_words`: the constructed `Word`s used the default `space_before=False`, which under grouping collapses all 30 words into one group (below the 20-word floor). Added `space_before=True`; the assertion (agreement == 1.0) is unchanged.
- `WordGroup.speaker` counts only assigned speakers (unassigned `None` tokens do not vote); documented in the helper and covered by `test_unassigned_tokens_do_not_outvote_a_speaker`. The card says "majority of its tokens' speakers" without specifying `None`.
- Reworded the `asr/__init__.py` module docstring from "Every model" to "Every persisted model" because `WordGroup` is a derived view without `to_dict`.
## NOT proven
- No live job was run (instructions forbid live network). The judge's `scripts/caracal-youtube.sh` re-run on the Tagesschau clip is the end-to-end proof.
- The Postgres `find_recent_done` path for the legacy raw-URL fallback was exercised through the in-memory repo fake, not against `latitude` Postgres.
- Real-world CJK grouping: whisper gives CJK tokens `space_before=False`, so per the mandated rule an entire CJK run becomes one group (one long line/cue). Not changed, because requirement 1 defines grouping purely by `space_before`.
## Open
- Should CJK (or other space-less scripts) get a grouping exception so cue/line breaks remain possible? This would need a docs change to `asr.word_groups`.
- Should `music.youtube.com/shorts/<id>` (and other prefix/host combinations outside YouTube's real routing) map to a watch URL? The implementation applies every listed prefix on every listed host, as requirement 5 reads.
