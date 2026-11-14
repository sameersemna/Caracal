STATUS: DONE

## What changed
- `src/caracal/migrations/versions/0002_voiceprints.py`: new migration — `known_speakers` (unique `name_norm`), `voiceprints` (`vector(256)`, FK cascade to speakers, `job_id` FK `SET NULL`), HNSW `vector_cosine_ops` index; downgrade drops both.
- `src/caracal/db/tables.py`: `EMB_DIM`, `known_speakers`, `voiceprints`, HNSW index; no column named for a reserved word.
- `src/caracal/db/repo.py`: `normalise_name`/`normalise_text` (NFKC, casefold, punctuation→space, collapse), `validate_embedding` (256 finite floats), `KnownSpeakerRecord`, `VoiceMatch`, `SpeakerRepo` protocol, `PostgresSpeakerRepo` (upsert merge, samples counts, best-per-speaker cosine, delete-with-count in one transaction).
- `tests/fakes/repo.py`: `FakeSpeakerRepo` plus store fields; `FakeJobRepo.delete` nulls `voiceprints.job_id` (SET NULL) and keeps the embeddings.
- `tests/repo_contract.py`: `Repos` gains `speakers`; five new shared cases (normalisation merge incl. accents, nearest ordering/best-per-speaker/k, delete cascade + count, job delete survival, embedding validation).
- `src/caracal/speakers/voiceprint.py`: `classify()` against the calibrated thresholds; `enroll()` (job must be `done`, label must exist in the diarize checkpoint, stores the label embedding; only call site is the explicit API).
- `src/caracal/pipeline/stages.py`: `Repos.speakers`; `stage_voiceprint` (skip if diarize not done, `nearest(k=3)`, keep `match`/`candidate`, drop below candidate); wired into `m2_registry`.
- `src/caracal/api/schemas.py`: `KnownSpeaker`, `EnrollSpeakerIn`, `SpeakerDeleted`.
- `src/caracal/services.py`: `SpeakerService` (list/enroll/delete) and `JobService.list_speakers`/`enroll_speaker`/`delete_speaker`.
- `src/caracal/api/routes.py`: `GET /v1/speakers?q=`, `POST /v1/speakers/enroll` (201), `DELETE /v1/speakers/{id}` (204).
- `src/caracal/mcp/server.py`: `list_speakers`, `enroll_speaker`, `delete_speaker` tools; every description carries the biometric/consent note.
- `src/caracal/worker.py`, `src/caracal/cli.py`, `src/caracal/app.py`: wire `PostgresSpeakerRepo` into the repository bundles the worker and API already build (see Deviations).
- `tests/unit/test_diarize.py`: verdict-18 item 0 — `SPEAKER_01` first becomes `S1`, `SPEAKER_00` becomes `S2`, and turns/exclusive turns/embeddings follow the same mapping.
- `tests/unit/test_voiceprint.py`: classification boundaries, stage match/candidate/absent, k=3, skip cases, enrollment success/merge and every error path, never-auto-enroll, registry wiring.
- `tests/contract/test_api_speakers.py`: list/enroll/delete contract, q filter, notes omission, merge, 409/404/422/401 envelopes.
- `tests/contract/test_mcp_tools.py`: tool round-trip equal to REST, error codes, consent wording.
- `tests/contract/test_parity.py`: speakers row promoted to available; list/enroll/delete parity test.
- `tests/live/test_db.py`: live fixture uses `PostgresSpeakerRepo`; table-set assertion covers both new tables; new test asserts `vector(256)`, the HNSW cosine index definition and `job_id IS NULL` after a job delete.
- `tests/unit/test_db_tables.py`: table-set test now expects the seven documented tables (see Deviations).
- `docs/findings/errata.md`: one appended entry for the docs/01 §3 vs card 19 voiceprint-signature drift (the only docs file touched).

## Tests added
- `test_speaker_01_speaking_first_becomes_s1_and_the_mapping_follows_everywhere`: raw labels are unordered; a swapped `SPEAKER_01`/`SPEAKER_00` order renames, and embeddings, turns and exclusive turns all follow first appearance. Fails under the judge's `sorted(ordered)` mutation (verified: `(0.0,'S2') != (0.0,'S1')`).
- `case_speaker_name_normalisation_merges`: `"Daniel  Levitin"` == `"daniel levitin."` == `"DANIEL LEVITIN"` (one row), notes kept when omitted and updated when given, NFKC-composed vs decomposed `José` merge, punctuation-only names rejected.
- `case_speaker_nearest_best_per_speaker`: exactly one row per known speaker at its best sample, ordering by similarity, `k` honoured, name filter and sample counts.
- `case_speaker_delete_cascades_and_counts`: returns the number of removed embeddings, the speaker disappears, the other speaker's samples and matches survive, second delete is 404.
- `case_job_delete_keeps_voiceprints`: after the job row is deleted the voiceprint still matches (live test additionally proves `job_id IS NULL`).
- `case_speaker_embedding_validation`: wrong dimensional length, NaN, infinity and unknown speaker/job are rejected with the documented codes.
- `test_voiceprint_stage_keeps_matches_and_candidates_and_drops_the_rest`: 0.96 → `match`, 0.6 → `candidate`, 0.3 absent, and the pipeline stores no new voiceprint.
- `test_voiceprint_stage_asks_for_three_matches_per_label`: four candidates yield three rows — proves `nearest(k=3)`.
- `test_voiceprint_stage_is_skipped_when_diarize_was_skipped_or_failed`, `..._without_a_diarize_checkpoint`, `test_m2_registry_wires_the_voiceprint_stage`.
- `test_classify_uses_the_calibrated_thresholds` (0.72/0.55 inclusive boundaries) and `test_classify_follows_configured_thresholds`.
- `test_enroll_stores_the_labels_embedding_and_appends_to_the_same_name`: stored vector matches the checkpoint exactly (nearest = 1.0) and the same normalised name adds a second sample.
- `test_enroll_rejects_a_job_that_is_not_done`: queued/running/failed/cancelled all raise `job_not_ready`.
- `test_enroll_unknown_job_or_label_is_not_found`, `test_enroll_without_a_diarize_checkpoint_is_not_found`, `test_enroll_rejects_a_name_without_letters`.
- API: `test_list_is_empty_until_something_is_enrolled` (201 + samples/notes + q filter), `test_notes_are_omitted_from_the_wire_model_when_absent`, `test_enrolling_the_same_normalised_name_appends_a_sample`, `test_enroll_requires_a_finished_job` (409 `job_not_ready`), `test_enroll_unknown_job_or_label_is_404`, `test_enroll_body_validation_is_422`, `test_delete_removes_the_speaker_and_its_voiceprints` (204 + second delete 404), `test_speaker_routes_require_the_bearer_token`.
- MCP: `test_speaker_tools_round_trip_and_match_rest` (MCP list == REST list, delete reports `voiceprints_removed: 1`), `test_speaker_tool_errors_carry_the_error_code`, `test_speaker_tool_descriptions_state_the_biometric_consent_rule`, plus `test_speakers_return_equal_data` in the parity suite.
- Live: `test_voiceprints_are_vectors_with_an_hnsw_cosine_index_and_outlive_their_job`.

## Measured
- gate exit code: 0
```
==> pytest (unit + contract, live excluded)
........................................................................ [  8%]
........................................................................ [ 17%]
........................................................................ [ 26%]
........................................................................ [ 35%]
........................................................................ [ 44%]
........................................................................ [ 53%]
........................................................................ [ 62%]
........................................................................ [ 71%]
........................................................................ [ 80%]
........................................................................ [ 88%]
........................................................................ [ 97%]
..................                                                       [100%]
810 passed in 8.40s
GATE: PASS
```
- Mutation checks (each restored afterwards): `sorted(ordered)` in `_label_order` → item-0 test fails; `nearest(k=4)` → stage k-test fails; fake nearest per-sample rows → best-per-speaker case fails; fake overwrites notes with `None` → merge case fails; fake job delete drops voiceprints → survival case fails; removing the consent note from `list_speakers` → description test fails.
- Migration SQL rendered offline with `uv run alembic upgrade head --sql`: `embedding VECTOR(256)`, `CONSTRAINT fk_voiceprints_job_id_jobs FOREIGN KEY(job_id) REFERENCES jobs (id) ON DELETE SET NULL`, `CREATE INDEX ix_voiceprints_embedding_hnsw ON voiceprints USING hnsw (embedding vector_cosine_ops)`.

## Deviations
- `src/caracal/app.py`, `src/caracal/cli.py`, `src/caracal/worker.py` are outside the card's Files list. `SpeakerRepo` has to join the single repository bundle that `StageContext`, `AppDeps` and `Worker` pass around, so the two bundles that build real repositories gained `speakers=PostgresSpeakerRepo(engine)`. This is the same judge-omission wiring accepted for `pipeline/runner.py` in card 14 / verdict 18; no behaviour was changed beyond the field.
- `tests/unit/test_db_tables.py` is outside the Files list. It asserted "only the five tables exist" and `"voiceprints" not in metadata.tables`, which migration 0002 (required by this card) makes false by construction. The test now asserts the seven tables documented in docs/05 §1. It was extended, not weakened: the table-set equality is the strongest form and metadata.schema is still asserted.
- `normalise_name`, `normalise_text` and `validate_embedding` are public in `db/repo.py` so the fake and the Postgres repo share one implementation of the normalisation and dimension rules the contract compares; the card asks for the fake to agree exactly.
- Unspecified detail: `upsert_known` keeps the first-seen display `name` (a later variant only updates notes and `updated_at`). docs/04 §5 only requires the normalised merge, so this was chosen for stable display.

## NOT proven
- Live Postgres was not reachable from this session: the shell's `DATABASE_URL` points at `127.0.0.1:9` (connection refused), so `tests/live/test_db.py` was lint/mypy-checked and its DDL verified offline, but not executed. Still unverified at runtime: the ON CONFLICT upsert against the unique index, the grouped best-per-speaker query with float4 rounding, the pgvector asyncpg round-trip, and `ON DELETE SET NULL`.
- The HNSW index is created as docs/05 §1 requires but the exact best-per-speaker aggregate cannot consume its ordering; if the judge expects an index-backed plan, that is a deliberate correctness trade-off documented in `PostgresSpeakerRepo`'s docstring (see Open).
- Real pyannote embeddings flowing through the stage: only the fakes and the verdict-18 measurements (0.94–0.99 same / 0.29–0.31 different) are exercised; no live speaker run in this round.
- REST/MCP data equality is asserted for list/enroll/delete; other tools are unchanged this round.

## Open
- Should `nearest` switch to an approximate HNSW pre-filter (`ORDER BY embedding <=> q LIMIT n` then best-per-speaker) once the store grows, or stay exact? Voiceprints are small (docs/00 Decision table) and T4.2 is the recalibration card; a follow-up there can revisit it without changing the interface.
- docs/01 §3 still describes `speakers/voiceprint.py` as `match(embedding)` / `enroll(label, job_id, name)`; the card (requirement 3) mandates `classify` and `enroll(deps, job_id, label, name, notes)`. Recorded as an errata entry (`docs/findings/errata.md`, 2026-11-14 T2.3) for the docs owner; the code follows the card.
