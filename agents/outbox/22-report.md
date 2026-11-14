STATUS: DONE

## What changed
- `src/caracal/report/__init__.py`: package exports (`Report`, `ReportSummary`, `render_markdown`).
- `src/caracal/report/model.py`: every docs/03 §5 model (`Report`, `ReportSource`, `AudioInfo`, `Language`, `SpeakerIdentity`, `SpeakerCandidate`, `Summary`, `Context`, `Chapter`, `Topic`, `Entity`, `KeyQuote`, `Claim`, `Verification`, `VerificationSource`, `ReviewFlag`, `Translation`, `TranscriptRef`, `QualityInfo`, `Answer`, `Citation`, `KnownSpeaker`) plus docs/03 §3 `ReportSummary`/`SummarySpeaker` and `Report.summary_view()`; one `_ReportModel` base with `extra="forbid"`, `schema_version: Literal["1.0"]`, all confidences `ge=0/le=1`, empty-list defaults for optional sections.
- `src/caracal/report/evidence.py`: `EVIDENCE_ID_RE` (typed alternatives only), `normalise_ids`, `EvidenceItem` (kind/ref/excerpt, collapsed + 300-char cap), `EvidenceIndex` (`add`/`get`/`__contains__`/`ids`/`to_dict`), `build_evidence_index` for seg/meta/hint/vp/caption/web, and the generic `validate_citations`.
- `src/caracal/report/render_md.py`: `md_escape` and `render_markdown` — title, source/audio table, languages, speakers, summary, context, chapters with `mm:ss`/`h:mm:ss`, topics, entities grouped by type, blockquoted key quotes, claims with verdicts, translations, quality/review flags, evidence appendix with `mm:ss.s` ranges.
- `src/caracal/llm/schema.py`: evidence normalisation now uses `caracal.report.evidence.EVIDENCE_ID_RE` instead of its own permissive `[A-Za-z0-9_.-]+` pattern; docstrings updated.
- `tests/unit/test_llm_schema.py`: evidence table adapted to the typed ids (see Deviations).
- `tests/unit/test_report_model.py`, `tests/unit/test_evidence.py`, `tests/unit/test_render_md.py`: new.

## Tests added
- `test_full_report_round_trips_json_identically`: a full report built from the real TED fixture facts survives JSON → model → JSON byte-identically.
- `test_documented_top_level_fields_are_exactly_these`: pins the field sets of `Report`, `Summary`, `Context`, `SpeakerIdentity`, `Claim`, `Verification`, `VerificationSource`, `ReviewFlag`, `Answer`, `Citation`.
- `test_extra_field_is_rejected` / `test_extra_field_is_rejected_in_a_nested_model`: `extra="forbid"` works at both levels.
- `test_confidence_outside_zero_one_is_rejected` (5 models) and `test_schema_version_is_the_literal_one_point_zero`.
- `test_sections_a_stage_may_not_produce_default_to_empty_lists`: verification/translations/… default to `[]`.
- `test_summary_view_fields` / `..._tolerates_a_missing_title_and_name`: exact `ReportSummary` values.
- `test_index_contains_every_artifact_kind`: index built from the merged `talk_c0`+`talk_c1` transcript (49 segments), a yt-dlp-shaped `PlatformMetadata`, the manual VTT captions (36 groups), hints, voiceprints, probe tags and web hits has exactly the expected id set.
- `test_segment_item_carries_the_time_range_speaker_and_text`, `test_metadata_and_probe_tag_items`, `test_hint_voiceprint_caption_and_web_items`, `test_empty_arguments_build_an_empty_index`, `test_excerpts_are_whitespace_collapsed_and_capped`.
- `test_normalise_ids` (18 cases): `"seg:2." → seg:2`, `"seg:2 S1" → seg:2`, comma lists, `vp:S1`, `meta:title`, garbage → `[]`, `vp:SPEAKER_00` → `[]`, dedupe/order, non-str.
- `test_unknown_ids_are_dropped_but_the_item_stays`, `test_a_list_item_with_no_evidence_left_is_removed_and_counted`, `test_a_singular_uncited_object_is_kept_flagged_and_zeroed` (flag + confidence 0 for `context`), `test_unresolved_verification_sources_are_dropped_and_the_verdict_becomes_unverified`, `test_a_resolved_verification_source_keeps_the_verdict`, `test_evidence_index_is_exactly_the_cited_ids_in_citation_order`, `test_validation_is_idempotent` (second run drops 0, no duplicate flag, `dropped_uncited` unchanged), `test_walk_is_generic_and_handles_a_field_it_was_never_told_about` (test-only `Chapter` subclass with a nested `evidence` list).
- `test_headings_are_present_in_order`, `test_tables_languages_speakers_and_context`, `test_chapter_timestamps_switch_to_hours_past_one_hour`, `test_key_quotes_are_blockquotes_with_speaker_and_time`, `test_claims_show_their_verification_verdict`, `test_quality_and_review_flags_are_rendered`, `test_evidence_appendix_formats_segment_time_ranges`, `test_evidence_appendix_formats_hours`, `test_unidentified_speaker_and_empty_sections`.
- `test_md_escape_table` (9 cases) and `test_md_escape_inline_collapses_newlines`.
- `test_untrusted_content_renders_inert` / `test_content_cannot_break_a_table_row`: `<script>`, `[click](…)`, `![x](…)`, `| a | b |`, `# heading` in title, table cell, chapter, quote, entity, topic, speaker name, summary and both evidence kinds render inert; no `<script`, no `](http`, no `![x]`, no line beginning `# heading`, table row still exactly three cells.
- `test_coerce_normalises_evidence_ids[...]`: new trailing-dot cases (`seg:2.`, `caption:4.`, `meta:title.`) and the typed `vp:S1`/`vp:SPEAKER_00` split.

## Measured
- gate exit code: 0
- ```
........................................................................ [  8%]
........................................................................ [ 16%]
........................................................................ [ 24%]
........................................................................ [ 32%]
........................................................................ [ 40%]
........................................................................ [ 48%]
........................................................................ [ 56%]
........................................................................ [ 64%]
........................................................................ [ 72%]
........................................................................ [ 80%]
........................................................................ [ 88%]
........................................................................ [ 96%]
...............................                                          [100%]
895 passed in 8.48s
GATE: PASS
```
- Non-vacuity spot checks (mutated source, ran the affected tests, restored the file):
  - permissive evidence regex (`[A-Za-z0-9_.-]+`) → 10 failed;
  - citation filtering removed → 5 failed;
  - `md_escape` escaping removed → 8 failed;
  - duplicate `uncited` flags / `dropped_uncited =` instead of `+=` → idempotence test failed.

## Deviations
- `docs/06 S9` / card requirement 3 (escape `<` and `>`): they are rendered as `&lt;` / `&gt;` entities instead of backslash-escapes, and `(`/`)` are escaped too. Reason: the card's own accept criterion requires the rendered Markdown to contain no `<script` and no `](http`, which backslash-escaping `<` and not-escaping brackets' trailing `](` cannot satisfy. Entities are not raw HTML.
- `docs/03 §5` preamble says "every list item … has `evidence` and `confidence`", but the per-type listing (which the card calls binding and says to follow "exactly") omits `confidence` on `Claim`, `Chapter`, `Topic`, `Entity`, `KeyQuote`. I followed the listing, so those items carry `evidence` only and the "confidence 1.2 → error" accept case is tested on `Verification`, `SpeakerIdentity`, `Language` and `Context`.
- `tests/unit/test_llm_schema.py` adapted (declared as the card allows): the typed pattern makes `vp:SPEAKER_00` invalid (diarization labels are renamed `S1..Sn`, docs/04 §5) so that case became `vp:S1` + `vp:SPEAKER_00 → []`; trailing-dot cases were added, no assertion was weakened.
- `tests/unit/test_llm_schema.py` keeps one `# type: ignore`-free table but `test_evidence.py` needed no ignores.

## NOT proven
- No end-to-end pipeline call: `report/assemble` and the stages that will feed `build_evidence_index`/`validate_citations` do not exist yet (T2.5+); the builders were exercised only with the real fixture artifacts passed by keyword, including a `VoiceMatch`-shaped stand-in because `speakers/voiceprint.py` has no match type yet (only `enroll`), and web hits as mappings because `search/` has not landed.
- Markdown output is asserted as text, not by rendering it through a Markdown parser; "inert" is proven by the absent raw-HTML/link substrings the card names.
- Live search/LLM calls: none; `EVIDENCE_ID_RE` reuse in `llm/schema.py` is covered by unit coercion tests only.

## Open
- `build_evidence_index(voiceprint_matches=…)` currently takes `Mapping[label, Sequence[object]]` and reads `name`/`similarity` by attribute or mapping key; when the voiceprint stage adds a concrete `VoiceMatch` dataclass, type it there (no behaviour change expected).
- docs/03 §5 preamble vs the per-type field listing on item-level `confidence` (see Deviations) — if the preamble is meant literally, `Claim`/`Chapter`/`Topic`/`Entity`/`KeyQuote` need an optional `confidence` in a follow-up card.
