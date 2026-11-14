STATUS: DONE

## What changed
- `src/caracal/analysis/identity.py`: `in_character` is a mention relation and never an identity source; self-intro scores a full name 0.85 and a bare given name 0.55; promotion now also requires top confidence ≥ 0.6; metadata names link to a label only through the sole-speaker, own-speech/verified-addressed or channel/uploader+host rules; performance genres/tone halve self-intro/addressed scores and add the warn flag; the extraction prompt receives plan `likely_genre` and content `genre`/`tone`.
- `src/caracal/llm/prompts/identity_mentions.md`: documents `in_character` with neutral impression/parody/sketch/role-play/voice-over examples and adds the recording-type DATA block.
- `src/caracal/pipeline/stages.py`: parses `content_profile.likely_genre` from the plan checkpoint and `context.genre`/`context.tone` from the content checkpoint and passes them to `resolve_identities`.
- `tests/unit/test_identity.py`: per-fix unit tests; adapted existing tests whose expectations changed under the new scores/linking rules.
- `tests/unit/test_stages.py`: stage test proving the plan profile and content context reach the identity prompt and produce the performance flag.
- `tests/live/test_identity_live.py`: the TED live test now passes the live content genre/tone (no behavioural change for TED).

## Tests added
- `test_in_character_impression_line_is_never_an_identity_source`: an "It's me, John." line marked `in_character` yields no candidate.
- `test_single_given_name_self_intro_scores_055_and_is_not_promoted`: "I'm Lindsay" is a 0.55 candidate with `name: null`.
- `test_full_name_self_intro_still_promotes_at_085`: "I'm Lindsay McMahon" still promotes at 0.85 (Lindsay-style full name unchanged).
- `test_performance_content_halves_the_self_intro_and_flags_it`: plan genre `comedy` turns 0.85 into 0.425 and adds the `warn` "performance content: speaker identities uncertain" flag.
- `test_performance_content_halves_the_addressed_score`: content tone "sketch comedy" turns an addressed 0.6 into 0.3.
- `test_description_names_are_not_sprayed_onto_one_speaker`: two person names that only appear in a description produce no candidates for either label.
- `test_metadata_name_links_to_the_label_whose_speech_contains_it`: a description name is a candidate only for the label whose own segments contain it.
- `test_plan_and_content_genre_reach_the_identity_prompt`: `[plan:likely_genre]`, `[content:genre]` and `[content:tone]` appear in the rendered prompt.
- `test_identity_stage_passes_the_plan_genre_and_content_context`: the stage passes plan/content context and emits the performance flag in its checkpoint output.
- Adapted `test_metadata_only_name_is_a_candidate_but_is_not_promoted`: the name now links through the channel/uploader + `host` rule (0.5, not promoted); the guest label still gets nothing.
- Adapted `test_self_intro_and_addressed_fuse_to_094` and `test_fusion_is_capped_at_098`: use the full name "Anna Schmidt" because a bare given name is now 0.55 by rule.
- Adapted `test_weak_voiceprint_candidate_is_scored_similarity_times_06`: a 0.33 candidate is no longer promoted because of the ≥ 0.6 threshold.
- Adapted `test_prompt_forbids_protected_attributes`: supplies the new `{content_context}` placeholder and asserts `in_character` guidance.

## Measured
- gate exit code: 0
- Last 15 lines of `bash scripts/gate.sh`:
```
........................................................................ [ 14%]
........................................................................ [ 21%]
........................................................................ [ 29%]
........................................................................ [ 36%]
........................................................................ [ 43%]
........................................................................ [ 51%]
........................................................................ [ 58%]
........................................................................ [ 65%]
........................................................................ [ 72%]
........................................................................ [ 80%]
........................................................................ [ 87%]
........................................................................ [ 94%]
....................................................                     [100%]
988 passed in 9.02s
GATE: PASS
```
- Mutation checks (temporary local edits, reverted): forcing every metadata link made both linking tests fail; setting the performance factor to 1.0 failed both performance tests; restoring the single-name score to 0.85 failed the single-name test; dropping the 0.6 threshold failed the weak-voiceprint and single-name tests; removing `in_character` from the relation literal failed the in-character test.

## Deviations
- Adapted five existing tests where the required fixes change their expected values (fusion 0.94 now needs a full name; weak voiceprint 0.33 no longer promotes; metadata-only candidate now links via channel/uploader + host; prompt variable added). No test was weakened.
- No new Caliendo / All Ears English live tests were added: the repository has no transcript fixtures or URLs for those interviews, and the card says the judge re-runs them.

## NOT proven
- I did not run any live interview (no network/model/GPU in this session). Whether gemma4 labels the Caliendo impressions `in_character` and whether the plan/content genre resolves to a performance term is unverified here.
- The "Lindsay unchanged" live case is only proven for a full-name self-intro. If the episode's only evidence is the single given name "Lindsay", the required fix scores it 0.55 (< 0.6) and it is not promoted without a second source (e.g. an addressed mention) or a full name.

## Open
- The card's Accept says "Lindsay-style full-name self-intro behaviour unchanged" while Required fix 2 says a single given name scores 0.55 with a 0.6 promotion floor. The unit suite pins both: a bare "I'm Lindsay" stays a 0.55 candidate, and "I'm Lindsay McMahon" promotes at 0.85. The judge should confirm the live All Ears English episode contains a full name or a second source for Lindsay.
