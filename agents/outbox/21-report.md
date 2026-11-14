STATUS: DONE

## What changed
- `src/caracal/api/schemas.py`: added `SpeakerList` (`items: list[KnownSpeaker]`), the shared REST/MCP wrapper object for speaker lists.
- `src/caracal/services.py`: `JobService.list_speakers` now returns `SpeakerList`, so both front doors serialize the same model and cannot drift.
- `src/caracal/api/routes.py`: `GET /v1/speakers` returns `{"items": [...]}` via `SpeakerList`.
- `src/caracal/mcp/server.py`: dropped the local `RootModel[list[KnownSpeaker]]` (the defect: an array `structuredContent` is rejected by the MCP serialiser) and returns the shared `SpeakerList`; `SpeakerList` is re-exported from `api.schemas`.
- `tests/contract/test_mcp_tools.py`: adapted speaker list-shape assertions (card 21 declares this) and added the all-tools output-schema test.
- `tests/contract/test_api_speakers.py`, `tests/contract/test_parity.py`: adapted list-shape assertions to `{"items": [...]}` (card 21 declares this).
- `tests/contract/test_mcp_wire.py`: new ASGI JSON-RPC test over the mounted `/mcp` app for the six tools.

## Tests added
- `test_every_registered_tool_declares_an_object_output_schema`: loops `list_tools()` and asserts every one of the eight registered tools has a dict `output_schema` with `"type": "object"`; an exact expected-name set stops the loop going vacuous.
- `test_every_tool_returns_a_json_rpc_result_over_streamable_http`: initializes and calls `list_jobs`, `get_job`, `list_speakers`, `enroll_speaker`, `delete_speaker`, `cancel_job` through the mounted streamable-HTTP app over ASGI; each call must produce a JSON-RPC `result` (no `error`), `isError == false`, and an object `structuredContent` with the documented shape. Removing the wrapper (returning the old root list) makes both new tests fail with the exact SDK error path measured in verdict 20.

## Measured
- gate exit code: 0
```
==> pytest (unit + contract, live excluded)
........................................................................ [  8%]
........................................................................ [ 17%]
........................................................................ [ 26%]
........................................................................ [ 34%]
........................................................................ [ 43%]
........................................................................ [ 52%]
........................................................................ [ 61%]
........................................................................ [ 69%]
........................................................................ [ 78%]
........................................................................ [ 87%]
........................................................................ [ 96%]
...............................                                          [100%]
823 passed in 8.60s
GATE: PASS
```

## Deviations
- none

## NOT proven
- The judge's live demo (`tests/live/test_db.py` and the real serve + MCP `list_speakers` → `delete_speaker` against Postgres/Redis/GPU) was not run: the worker contract leaves the live layer to the judge. The transport defect is reproduced and covered over the real mounted ASGI app with fakes, and the REST/MCP payloads are asserted byte-equal in `test_speakers_return_equal_data`.

## Open
- none
