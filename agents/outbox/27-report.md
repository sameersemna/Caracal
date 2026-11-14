STATUS: DONE

## What changed
- src/caracal/analysis/identity.py: deterministic identity resolution — LLM mention extraction, per-segment verification, weighted candidate sources, first-name grouping, fusion, promotion/ambiguity flags, voiceprint links, role/context descriptions with a protected-attribute filter.
- src/caracal/llm/prompts/identity_mentions.md: JSON-mode mention/role/description prompt with typed evidence ids and explicit prohibition of protected-attribute inference.
- src/caracal/pipeline/stages.py: implemented `stage_identity` and checkpoint parsers for align stats, voiceprint matches and content entities; output `{speakers, flags}`.
- tests/unit/test_identity.py: mention verification, source scores, fusion math, promotion rules, sole-speaker metadata, voiceprint, hints, sanitisation, long-transcript windowing.
- tests/unit/test_stages.py: identity stage wiring, plan skip, LLM-unavailable skip.
- tests/live/test_identity_live.py: merged real TED transcript + real metadata, judge-run.

## Tests added
- test_self_intro_in_german_promotes_the_name: German self-intro yields a 0.85 `self_intro` candidate that is promoted.
- test_addressed_name_before_the_turn_is_a_weak_candidate: "thanks, Anna" from S1 before S2's turn gives S2 an `addressed_by_name` 0.6 candidate.
- test_hallucinated_mention_whose_name_is_not_in_the_segment_is_dropped: a name absent from the cited segment is dropped.
- test_referred_name_is_never_a_source: `referred` mentions never become candidates.
- test_self_intro_and_addressed_fuse_to_094: 0.85 + 0.6 fuses to exactly 0.94.
- test_fusion_is_capped_at_098: 0.85 + 0.6 + 0.7 fuses to the 0.98 cap.
- test_conflicting_voiceprint_and_self_intro_are_not_promoted: 0.96 voiceprint vs 0.85 self-intro → no name, `ambiguous identity`, both candidates, known speaker set.
- test_two_equal_candidates_fail_the_lead_margin: equal candidates → null name + flag.
- test_metadata_only_name_is_a_candidate_but_is_not_promoted: multi-speaker metadata-only stays a 0.5 candidate.
- test_sole_speaker_ted_case_promotes_the_metadata_name_at_075: sole speaker + metadata person entity → "Daniel Levitin" at 0.75.
- test_voiceprint_match_sets_the_known_speaker_and_similarity: `match` 0.962 → source `voiceprint`, `known_speaker_id`, similarity.
- test_weak_voiceprint_candidate_is_scored_similarity_times_06: `candidate` 0.55 → 0.33, no known speaker.
- test_hint_maps_to_the_label_with_the_matching_role: "host: Lex Fridman" maps to the host label at 0.7.
- test_bare_hint_applies_to_every_label_when_several_speak: multi-speaker bare hint → every label at 0.4.
- test_protected_attribute_in_a_description_is_replaced_by_the_role_fallback: "adult male" description → role-only fallback.
- test_prompt_forbids_protected_attributes: prompt text forbids gender/protected inference and names male/female.
- test_long_transcript_sends_first_40_segments_plus_the_run_up_to_turns: long prompt includes first 40 per speaker plus 3 before turns, capped at `LLM_CONTEXT_CHARS`.
- test_short_transcript_is_sent_whole: short transcript is not reduced.
- test_addressed_mention_too_far_before_the_turn_is_dropped: addressed mention outside the 3-segment window is dropped.
- test_self_intro_cited_from_another_speaker_is_dropped: self-intro cited on another label's segment is dropped.
- test_name_matching_ignores_case_and_diacritics_and_allows_a_first_name: "Jose Alvarez" matches "JOSÉ Alvarez".
- test_a_first_name_merges_into_the_full_name_it_starts: "Angela" merges into "Angela Merkel", fusing to 0.94.
- test_identity_stage_resolves_mentions_and_voiceprints_from_checkpoints: stage reads align/content/voiceprint checkpoints, JSON mode, stats and voiceprint fields in output.
- test_identity_stage_is_skipped_by_the_plan: plan skip passthrough.
- test_identity_stage_skips_without_an_llm: missing LLM → `skipped: llm unavailable`.
- test_live_identity_names_the_single_ted_speaker: merged TED transcript + live content entities + real metadata → S1 "Daniel Levitin" (judge-run).

## Measured
- gate exit code: 0
- Last 15 lines of `bash scripts/gate.sh`:
```
........................................................................ [ 14%]
........................................................................ [ 22%]
........................................................................ [ 29%]
........................................................................ [ 36%]
........................................................................ [ 44%]
........................................................................ [ 51%]
........................................................................ [ 58%]
........................................................................ [ 66%]
........................................................................ [ 73%]
........................................................................ [ 80%]
........................................................................ [ 88%]
........................................................................ [ 95%]
...........................................                              [100%]
979 passed in 8.92s
GATE: PASS
```

## Deviations
- docs/01 §3 lists `resolve_identities(...) -> list[SpeakerIdentity]`; it returns `IdentityResult` with `speakers` and `flags`, because card requirements 4-5 require the ambiguity flags returned alongside the identities. The stage output is exactly `{speakers: [...], flags: [...]}` as required.
- The metadata source is populated by intersecting the content stage's `person` entities with `meta:title/description/channel/uploader` (docs/04 §7 says "name appears in ..."). Deterministic free-text name extraction without the content entities is not specified; the sole-speaker rule already requires the content person entity.
- `SpeakerIdentity.evidence` includes the label's first segment in addition to candidate evidence so the item remains addressable under `validate_citations` (docs/04 §9). Candidates keep their own exact evidence lists.

## NOT proven
- The live identity test was not run here (no live network calls from the worker); the judge runs it. It depends on the real gemma4 content call returning a `person` entity for "Daniel Levitin" from the metadata.
- Real-gemma4 adherence to the role enum and mention recall is only exercised by the live test; unit tests use scripted JSON.
- The `web` candidate source remains absent in this stage; docs/04 §8 feeds it back from the verify stage, which is a later card.
- Very long real transcripts were only tested with a scripted LLM and synthetic segments, not with real gemma4 prompts.

## Open
- Confirm the docs/01 §3 signature should be updated to the `IdentityResult` return type (speakers + flags).
- Confirm that multi-speaker metadata names should continue to come from content `person` entities rather than a separate deterministic metadata parser.
