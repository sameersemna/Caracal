# Errata

Append-only. Worker records doc problems here; the boss resolves them into the docs.

## 2026-11-14 T1.1 — upload directory name cannot match the `uploads.id` row key

**What:** docs/05 §3 places an upload at `uploads/<upload_id>/<filename>`, but `UploadRepo.create`
(`src/caracal/db/repo.py`) takes no id and generates its own `uuid.uuid4()` after the file must
already exist. T1.1's Files list excludes `db/repo.py`, so the API cannot make the directory name
equal the row id returned to the client.

**Where:** `src/caracal/ingest/upload.py` / `src/caracal/api/routes.py`; `caracal.db.repo.UploadRepo.create`.

**Evidence:** a `POST /v1/uploads` of 3 bytes returns `upload_id=41185387-…` while the row's
`path` is `data/uploads/32e6bbb0-…/x.mp3`; `record.path` (the authoritative artifact location) is
correct and the file is under `data/uploads/`, but the directory UUID is not the response id.

**Suggested resolution:** either add an optional `upload_id` parameter to `UploadRepo.create`
(then T1.1 can write directly into `uploads/<row id>/`), or change docs/05 §3 to state that the
upload directory is keyed by the storage id stored in `uploads.path` and that consumers must use
`path` rather than reconstructing it from `id`.

**Resolution (boss, verdict 08):** accepted for now; T1.6 adds `upload_id` to `UploadRepo.create` so the directory is `uploads/<upload_id>/`.

## 2026-11-14 T2.2 — diarization embeddings come from `DiarizeOutput.speaker_embeddings`, not a re-computed centroid

**What:** docs/04 §5 says the diarize stage stores "one embedding per label (centroid over that label's turns ≥ 1.5 s)". The measured pyannote community-1 output already returns one embedding per label (`DiarizeOutput.speaker_embeddings`, `(n, 256)`, rows in `speaker_diarization.labels()` order), and the round-18 card requires using it (judge-verified fact). Re-computing a centroid with a ≥ 1.5 s turn filter would require a second embedding pass over the waveform and could disagree with the pipeline's own labels.

**Where:** `src/caracal/speakers/diarize.py`; docs/04 §5.

**Evidence:** docs/00 "pyannote" row and the round-18 card: `speaker_embeddings` is `(n, 256)` whose rows follow `speaker_diarization.labels()` order; `Pipeline.from_pretrained(...)` + `.to(cuda)` returns that field directly.

**Suggested resolution:** update docs/04 §5 to say the stage stores the pipeline's `speaker_embeddings` (L2-normalised), and drop the "centroid over turns ≥ 1.5 s" wording (or record the ≥ 1.5 s filter as a future refinement that must be validated against this output).

## 2026-11-14 T2.3 — docs/01 §3's voiceprint signatures disagree with the card (and the code)

**What:** docs/01 §3 lists `speakers/voiceprint.py  match(embedding) -> list[VoiceMatch]; enroll(label, job_id, name)`. The round-19 card (requirement 3) instead specifies `classify(similarity, settings) -> "match" | "candidate" | None` and `enroll(service deps, job_id, speaker_label, name, notes)`, with the per-label cosine search living in the `voiceprint` stage (`nearest(embedding, k=3)` on `SpeakerRepo`) so the stage output carries the classes.

**Where:** `docs/01-architecture.md` §3 (layout table); implemented in `src/caracal/speakers/voiceprint.py` and `src/caracal/pipeline/stages.py::stage_voiceprint`.

**Evidence:** docs/01 §3 one-liner vs card 19 requirement 3 and requirement 4; the implemented module exposes `classify`, `enroll`, `EnrollmentDeps` and no `match()`.

**Suggested resolution:** update the docs/01 §3 row to `classify(similarity, settings); enroll(deps, job_id, label, name, notes)` and state that the stage performs the `nearest(k=3)` search, or reword it as prose (the module's public API is what the card binds).

## 2026-11-14 T3.5 — `GET /v1/search` is an object with `speaker`, not the bare list in docs/03 §2

**What:** docs/03 §2's REST table returns a bare list `[{"job_id","title","seg_id","start","text","rank"}]`, while docs/03 §3 requires every MCP tool to return a JSON object and round-38 card T3.5 §4 explicitly specifies `{"items": [{job_id, title, seg_id, start, speaker, text, rank}]}` for REST and MCP alike.

**Where:** docs/03 §2 (`GET /v1/search` row); `src/caracal/api/routes.py`, `src/caracal/api/schemas.py`.

**Evidence:** card 38 requirement 4 ("object, not list") and docs/03 §3 ("Every tool returns a JSON object ... structuredContent must be an object"); the implemented response is `SearchResults{items: [SearchItem]}`.

**Suggested resolution:** update the docs/03 §2 row to the object shape and add `speaker` to the item fields.
