STATUS: DONE

## What changed
- `src/caracal/analysis/review.py`: new LLM self-review layer — 30-segment even sampling (first+last), evidence-free report view, JSON-mode call, field-path validation, message cleaning (evidence ids stripped, ≤200 chars), dedupe and the 10-flag cap.
- `src/caracal/llm/prompts/review.md`: new review prompt (inconsistencies only, dotted field paths, no evidence ids, `{"flags": []}` when clean).
- `src/caracal/pipeline/stages.py`: `stage_review` builds the draft with the pure `assemble_report` (no files/report row written), reviews it and checkpoints `{flags, llm_meta}`; skips on quick depth, music (`non-speech audio`), a missing LLM and a failed LLM call (non-fatal); wired into `m2_registry`; `review` keeps its enum position before `assemble`.
- `src/caracal/report/assemble.py`: review-checkpoint flags are appended; all flags are de-duplicated by `(field, message)` across plan/content/identity/verify/review/deterministic; the chapter-coverage check is skipped for music jobs and transcripts below 20 segments; the contradicted-claim-repeated-in-summary case is explicitly left to the LLM.
- `tests/unit/test_review.py`: new unit coverage for sampling, field paths, message cleaning, JSON-mode prompt shape, validation/cap/dedupe.
- `tests/unit/test_assemble.py`: music and <20-segment coverage cases, the 20-segment boundary, review flag append and cross-source dedupe.
- `tests/unit/test_stages.py`: stage skips (quick/music/no LLM/invalid LLM output), draft built without report files, assemble appending the review checkpoint, registry wiring.
- `tests/live/test_review_live.py`: judge-run live checks (planted Jim Nantz/Daniel Levitin inconsistency, clean draft).

## Tests added
- `test_sample_segments_includes_first_and_last_evenly_spaced`: 100 segments → 30 unique, evenly spaced samples with ids 0 and 99.
- `test_sample_segments_keeps_a_short_transcript_whole`: short/empty transcripts are returned unchanged/empty.
- `test_valid_field_path`: top-level keys and dotted paths accepted; unknown sections, wrong case and malformed paths rejected.
- `test_clean_message_strips_evidence_ids_and_truncates`: `seg:`, `meta:`, `web:` ids removed, newlines collapsed, 200-char cap.
- `test_review_report_uses_json_mode_validates_dedupes_and_caps_flags`: JSON mode, no evidence payload in the report block, 30 sampled lines ending at the last segment, invalid/duplicate flags dropped, ≤10 kept.
- `test_review_report_keeps_an_empty_clean_review`: empty flag list yields no flags and correct metadata.
- `test_music_job_produces_no_chapter_coverage_flag`: music job with short chapters emits no `chapters` flag.
- `test_short_transcript_produces_no_chapter_coverage_flag` / `test_twenty_segments_still_apply_the_chapter_coverage_rule`: 19 segments skip, 20 apply.
- `test_review_checkpoint_flags_are_appended_to_the_report`: review checkpoint flags reach `quality.review_flags`.
- `test_review_flags_are_deduplicated_across_sources`: identical content/review flags appear once.
- `test_stage_review_skips_quick_music_and_missing_llm`: four skip paths, including invalid model output.
- `test_stage_review_builds_the_draft_without_writing_the_report`: draft in memory only (no report JSON/MD/row) while the prompt carries the assembled summary, speaker and first/last sampled lines.
- `test_stage_assemble_appends_the_review_checkpoint_flags`: end-to-end review checkpoint → assembled report.
- `test_m2_registry_wires_diarize_and_align_and_keeps_m1_stages` (extended): `Stage.REVIEW` is wired to `stage_review`.
- `tests/live/test_review_live.py`: planted speaker inconsistency yields ≥1 `summary`/`speakers` flag; clean draft yields ≤2.

## Measured
- gate exit code: 0
- ```
  ==> repository rules
  ==> pytest (unit + contract, live excluded)
  ........................................................................ [  6%]
  ........................................................................ [ 12%]
  ........................................................................ [ 18%]
  ........................................................................ [ 25%]
  ........................................................................ [ 31%]
  ........................................................................ [ 37%]
  ........................................................................ [ 43%]
  ........................................................................ [ 50%]
  ........................................................................ [ 56%]
  ........................................................................ [ 62%]
  ........................................................................ [ 68%]
  ........................................................................ [ 75%]
  ........................................................................ [ 81%]
  ........................................................................ [ 87%]
  ......................................................................   [100%]
  1150 passed in 10.62s
  GATE: PASS
  ```

## Deviations
- "LLM unavailable" is treated as both a missing client and a failed review call: `stage_review` catches `CaracalError`, logs `review.llm_failed` and returns `skipped / llm unavailable` instead of failing the job (docs/04 §11 and card requirement 3 call the review non-fatal). This is the only interpretation added beyond the literal skip list.

## NOT proven
- `tests/live/test_review_live.py` was not run (live tests are judge-run per the card); its two cases depend on real gemma4 behaviour.
- The unit fake proves JSON-mode plumbing, validation and the prompt contents, not how often gemma4 actually finds a contradiction.

## Open
- none
