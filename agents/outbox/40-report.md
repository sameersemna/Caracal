STATUS: DONE

## What changed
- `scripts/eval/__init__.py`: package marker for the eval harness.
- `scripts/eval/metrics.py`: pure WER, DER (optimal label mapping + reference-boundary collar), identity scoring and ROC/threshold calibration.
- `scripts/eval/cases.json`: six labelled cases (TED, All Ears English, JFK/TED splice, Caliendo, SoundCloud instrumental, German news) with options and expectations; URLs/paths only.
- `scripts/eval/run_eval.py`: REST-API runner (submit → poll → report/transcript/subtitles → metrics → Markdown), manual-caption download, on-the-fly ffmpeg splice, `/metrics` snapshot, `--strict`.
- `scripts/eval/README.md`: throwaway/production usage, case table, metric definitions, calibration caveat, known metric gap.
- `tests/unit/test_eval_metrics.py`: 21 tests over WER, DER, identity and ROC.
- `agents/outbox/40-report.md`: this report.

## Tests added
- `test_wer_identical_is_zero`, `test_wer_one_substitution`, `test_wer_insertion_counts_against_reference_length`, `test_wer_deletion_counts_against_reference_length`: edit-distance accounting against the reference length.
- `test_wer_uses_caption_agreement_normalisation`, `test_wer_empty_reference_reports_hypothesis_length`: shared `normalise_word` normalisation and the empty-reference edge.
- `test_der_perfect_is_zero`, `test_der_swapped_labels_are_remapped_to_zero`: exact DER and optimal label assignment.
- `test_der_miss`, `test_der_false_alarm`, `test_der_confusion_when_one_reference_speaker_is_split`: each error component in isolation.
- `test_der_collar_ignores_small_boundary_offsets`: a 0.2 s boundary offset scores 0 with a 0.25 s collar and > 0 without it.
- `test_der_accepts_plain_tuples_and_sums_to_the_error_rate`, `test_der_false_alarm_only_uses_hypothesis_time_as_denominator`: API shape, `der == miss + fa + confusion`, degenerate reference fallback.
- `test_identity_scores_every_outcome_including_abstention`, `test_identity_wrong_and_false_positive_against_unknown`, `test_identity_ignores_case_and_surrounding_punctuation`: correct/wrong/missing/abstained, null-as-correct for unknown, name normalisation.
- `test_roc_is_monotonic_and_spans_infinities`, `test_best_threshold_maximises_tpr_under_the_fpr_budget`, `test_best_threshold_prefers_a_higher_tpr_within_budget`, `test_best_threshold_abstains_when_no_threshold_fits`: ROC monotonicity, budget-respecting threshold, tie-break and no-fit fallback.

## Measured
- gate exit code: 0
- Last 15 lines of `bash scripts/gate.sh`:
```
........................................................................ [ 35%]
........................................................................ [ 40%]
........................................................................ [ 46%]
........................................................................ [ 52%]
........................................................................ [ 58%]
........................................................................ [ 64%]
........................................................................ [ 70%]
........................................................................ [ 75%]
........................................................................ [ 81%]
........................................................................ [ 87%]
........................................................................ [ 93%]
........................................................................ [ 99%]
..........                                                               [100%]
1234 passed in 10.67s
GATE: PASS
```

Live validation on a throwaway schema (`caracal_eval_dev_6dfacf`, port 11498, real whisper.cpp + pyannote + gemma4). The six cases were run in three invocations against the same schema; each reached `done`:
- `ted-levitin`: WER 411/2220 = **0.185**, language `en`, S1 **Daniel Levitin** 0.79, 3/3 expectations, 121.2 s.
- `all-ears-english`: S1 **Lindsay McMahon** 0.62, language `en`, 2/2, 33.3 s.
- `jfk-ted-splice`: **DER 0.083** (miss 0.082, false alarm 0.000, confusion 0.001), S1/S2 unnamed, 3/3, 29.6 s.
- `caliendo-impressions`: 7 labels, all unnamed, no name ≥ 0.6, 1/1, 123.8 s.
- `soundcloud-instrumental`: `music_detected: true`, languages `und`, 1/1, 39.4 s.
- `german-news`: language `de`, `en.srt` + `en.vtt` served, 2/2, 54.4 s.
All dev artifacts lived under `/tmp`; the throwaway schema, its Redis prefix (33 keys) and the data dir were removed.

## Deviations
- **Voiceprint calibration skipped** (allowed by the card): label embeddings are not exposed over the REST API. The report says so explicitly; `roc()`/`best_threshold()` are implemented and unit-tested for when an embedding export exists.
- **LLM call and whisper-seconds figures are absent from `/metrics`**: T4.1's `MetricsRegistry.incr_llm_call` and `add_whisper_seconds` exist but no pipeline client calls them, so `caracal_llm_calls_total` is not emitted and `caracal_whisper_audio_seconds_total` stays 0. The harness reads the series and reports "not recorded" rather than inventing numbers; search cost is real.
- `cases.json` carries an extra `options` object per case and a `source.splice` descriptor for the JFK case (the card's `{id, source, expect}` is the minimum shape). `run_eval.py` also accepts `--cases-file`, `--timeout`, `--poll`, `--request-timeout` and `--keep-work` beyond the listed flags.
- The JFK case encodes both `identity: {S1: null, S2: null}` and `speakers_unnamed: true` so an unexpected extra named label is caught too.
- The yt-dlp media/caption calls retry transient YouTube 403s three times with a 5 s delay; this was needed during validation (the first attempt was forbidden, the retry succeeded).

## NOT proven
- A single all-six-cases invocation on a brand-new schema: each case was exercised live, but the runs were split across three invocations on one throwaway schema, so a judge's fresh full run will produce slightly different aggregate timings.
- Calibration numbers (skipped, see above).
- Behaviour when YouTube blocks every download attempt: the splice case then ends as `error` (and `--strict` exits 1); it never silently passes.
- The live validation required `WHISPER_URL=http://127.0.0.1:8768` to be exported for the worker because this session's shell inherited a test dummy (`http://127.0.0.1:9`) that shadows `.env`. The harness itself does not set or depend on that variable; `scripts/live-suite.sh` runs in a normal shell.

## Open
- T4.1 metric gap: wire `incr_llm_call` / `add_whisper_seconds` into `LLMClient` and the whisper stage (or decide they are out of scope), so the eval's "LLM calls" and "Whisper audio" lines carry numbers. Until then the report labels them "not recorded".
- The AEE and TED name expectations depend on web verification (live: 0.62 and 0.79). They are expectations, not hard tests; a future web-provider outage will show as an unmet expectation, not a harness bug.
