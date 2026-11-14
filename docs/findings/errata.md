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
