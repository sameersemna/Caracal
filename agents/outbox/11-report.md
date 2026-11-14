STATUS: DONE

## What changed
- `src/caracal/asr/__init__.py`: frozen `Word`, `Segment`, `WhisperResult`, `Transcript`, `LangSpan` with `to_dict()`, stripped text, and the complete whisper code↔English-name table.
- `src/caracal/asr/whisper.py`: `WhisperClient.transcribe()` (explicit `language`, multipart, lock-scoped request, argmax detection with name fallback, retryable `dependency_down` on transport/5xx, `internal` on 4xx/bad bodies) and `health()`; never logs transcript text.
- `src/caracal/asr/merge.py`: overlap-midpoint word ownership, shifted timestamps, segment trimming/text rebuild, word-less midpoint retention, no_speech/repetition/empty filters with counts, id renumbering, language attribution and dominant-language selection.
- `src/caracal/asr/langid.py`: `language_spans`, `refine_spans` (30 s windows, `max_calls`) and `languages_summary` (`code/name/share/spans/confidence`).
- `src/caracal/asr/captions.py`: `parse_vtt` (tagged auto cues preferred, rolling duplicates ignored; manual evenly spread), `caption_agreement` (NFKC/punctuation/bracket normalisation, per-60 s-window edit distance, 20-word floor) and `normalise_word`.
- `tests/fakes/whisper.py`: `FakeWhisper` `httpx.MockTransport` serving the real fixtures, multipart decode/record, scripted errors and an events list for lock-order assertions.
- `tests/unit/test_whisper_client.py`, `tests/unit/test_merge.py`, `tests/unit/test_langid.py`, `tests/unit/test_captions.py`: unit coverage listed below.
- `tests/live/test_whisper_live.py`: real-server JFK transcription (`live`, not run here).

## Tests added
- `test_request_is_explicit_multipart_and_parses_real_fixture`: multipart fields (`language=auto`, `verbose_json`, temperature, translate, file bytes) and the real jfk parse (en, 0.9352, 3 segments, stripped text).
- `test_forced_language_uses_real_detection_not_the_echo`: forced `de` still yields `language_code="en"`, p=0.9971.
- `test_silence_yields_no_segments_and_low_confidence`: silence fixture → 0 segments, p≈0.321.
- `test_transport_failures_are_retryable_dependency_down`, `test_5xx_is_retryable_dependency_down`: connect/timeout/503 → `dependency_down`, retryable.
- `test_4xx_is_internal_and_not_retryable`, `test_malformed_bodies_are_internal`: 400, HTML, missing/!list segments → `internal`.
- `test_invalid_language_never_reaches_the_server`: bad hints → `invalid_input`, zero requests.
- `test_request_happens_inside_the_lock`: enter→request→exit order.
- `test_health_reports_the_root_status`: 200 true, 500/connect false.
- `test_language_falls_back_to_the_detected_name`: empty probabilities → full-name→ISO map.
- `test_real_overlap_pair_keeps_each_word_once`: exact word list for c0/c1 equals every c0 word with midpoint <60.5 plus every c1 word ≥60.5; one `'t`/`couldn` at the cut; monotonic starts; ids 0..n-1; zero dropped.
- `test_segments_are_trimmed_to_their_kept_words`: segment start/end/text shrink to kept words and the cut-owned word is kept once.
- `test_no_speech_hallucination_is_dropped`, `test_repeated_text_is_dropped_from_the_third_consecutive_time`, `test_empty_text_is_dropped`: each filter and its counter.
- `test_wordless_segments_are_kept_by_their_own_midpoint`: midpoint ownership without word timings.
- `test_language_code_is_the_one_with_most_speech_time`, `test_result_and_chunk_count_mismatch_is_internal`, `test_empty_inputs_produce_an_empty_transcript`, `test_models_serialise_to_json`.
- `test_consecutive_segments_with_the_same_language_form_one_span`, `test_span_below_the_confidence_floor_becomes_und`, `test_unknown_language_is_und`, `test_refine_relabels_windows_and_respects_max_calls`, `test_refine_skips_confident_and_short_chunks`, `test_languages_summary_shares_sum_to_one`, `test_empty_transcript_has_no_languages`, `test_lang_span_serialises_to_json`.
- `test_auto_captions_are_word_timed_and_not_duplicated_by_rolling_cues`: strictly increasing starts and the “a few years ago I broke into my own” phrase appears once.
- `test_manual_captions_are_spread_evenly_over_each_cue`: token order and even spread.
- `test_notes_styles_regions_and_cue_settings_are_ignored`, `test_tagged_cues_win_over_rolling_plain_cues`, `test_normalise_word_strips_punctuation_music_and_bracket_tags`.
- `test_agreement_with_real_auto_captions_is_at_least_eighty_percent`: real merge vs `talk_en_auto.vtt` = 0.8551 (≥0.80).
- `test_agreement_with_shuffled_captions_is_low` (<0.3), `test_agreement_is_none_when_either_side_has_too_few_words`, `test_agreement_is_one_for_identical_words`.

## Measured
- gate exit code: 0
```
83 files already formatted
==> mypy (strict)
Success: no issues found in 36 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 12%]
........................................................................ [ 25%]
........................................................................ [ 37%]
........................................................................ [ 50%]
........................................................................ [ 62%]
........................................................................ [ 75%]
........................................................................ [ 87%]
.......................................................................  [100%]
575 passed in 5.22s
GATE: PASS
```
- real caption agreement measured in a scratch script: `0.855072463768116`.

## Deviations
- `Segment` has one extra field, `lang_prob: float = 1.0`, after `speaker`. docs/01 §3 and the card list the Segment fields without a confidence carrier, but `language_spans(transcript)` must report “that chunk's `language_prob`” from the transcript alone; `merge_chunks` fills it from the chunk result. Positional construction of the documented fields is unchanged.
- `refine_spans` takes `chunk_results` as `(chunk_start_s, WhisperResult)` pairs. The card leaves the pairing untyped; absolute 30 s windows cannot be placed from chunk-local `WhisperResult`s without the chunk offset. The stage can build these pairs from its `chunks` list.
- `languages_summary` is required by the card but not listed in the docs/01 §3 layout note for `langid.py`; it lives in the allowed `asr/langid.py` file, so no new module was created.

## NOT proven
- `tests/live/test_whisper_live.py` was written but not run (worker rule: live tests are the judge's).
- No audio longer than 120 s was chunked/merged; the real pair covers the 1 s overlap and cut semantics but not the 600 s planner path.
- Code-switching refinement was exercised with a synthetic `recheck`, not real multilingual audio.
- Auto-caption parsing is proven on the recorded rolling fixture, not on other YouTube auto-caption layouts (e.g. multiple tagged lines per cue).

## Open
- Confirm the `(chunk_start_s, WhisperResult)` shape for `refine_spans.chunk_results` with the T2 transcribe/langid stage, or replace it with a `(Chunk, WhisperResult)` pair if that reads better there.
