STATUS: DONE
## What changed
- `src/caracal/llm/client.py`: `call_tool(..., mode="tool"|"json")`; json mode sends `format:"json"` with the prepared schema in the system prompt and no tools, parses with `extract_json`, and repairs with assistant+user messages; an empty tool reply (no tool call, no text) retries the same attempt in json mode; `CallMeta.mode_used` records the mode that produced the answer.
- `src/caracal/analysis/content.py`: all content/window/reduce calls use `mode="json"`; completeness check for transcripts with ≥20 segments re-asks once for only the empty chapters/entities/claims/topics through a cached dynamic sub-model and merges the raw values before post-checks; `ContentLlmMeta.completeness` and `.reasked` record the outcome; LLM-facing chapters are `title, summary, first_seg, last_seg, evidence` and quotes are `seg_ids` only, with times derived from segments.
- `src/caracal/llm/prompts/content.md`, `content_window.md`, `content_reduce.md`: "Fill EVERY field; lists must not be empty when the transcript supports them", chapter `first_seg`/`last_seg` instructions, quote `seg_ids` instructions.
- `src/caracal/pipeline/runner.py`: `RunnerDeps.llm` passed into every `StageContext`.
- `src/caracal/worker.py`: one process-wide `LLMClient` sharing the worker's `httpx.AsyncClient`, injected (optionally) and handed to `RunnerDeps`.
- `tests/fakes/ollama.py`: `json_message` / `json_response` helpers for JSON-mode replies.
## Tests added
- `test_json_mode_sends_format_json_and_the_schema_in_the_system_prompt`: request body has `format:"json"`, no tools, the exact schema in the system prompt, and `mode_used=="json"`.
- `test_json_mode_repair_uses_assistant_and_user_messages`: invalid JSON-mode reply is repaired with an assistant echo + user validation error, still json mode.
- `test_empty_tool_reply_retries_the_attempt_in_json_mode`: first request is a tool call, second has `format:"json"` and no tools; `attempts==2`, `mode_used=="json"`.
- `test_chapters_are_derived_from_first_and_last_segment_ids`: numeric `start`/`end` from the model are ignored; bounds come from the segment ids; overlap removed and unresolvable chapter dropped.
- `test_chapter_with_reversed_segment_ids_uses_the_cited_span`: reversed ids fall back to the cited segments' span.
- `test_completeness_reasks_only_the_missing_sections_and_merges_them`: 20-segment transcript with empty chapters/entities triggers a second json call whose schema contains exactly `{chapters, entities}`, the note lists only them, results are merged, and `reasked`/`completeness`/`calls` are recorded.
- `test_completeness_is_skipped_for_short_transcripts` and `test_completeness_skips_the_reask_when_every_section_is_present`: no re-ask outside the card's conditions.
- `test_worker_wires_an_llm_client_into_the_content_stage`: a job run by `Worker` delivers a non-None `LLMClient` as `ctx.llm` to the content stage.
- Adapted `test_stage_content_writes_a_content_checkpoint_with_llm_meta` and the content tests to JSON-mode fakes (declared deviation) and asserted `format:"json"`/no tools there.
## Measured
- gate exit code: 0
- ```
........................................................................ [ 15%]
........................................................................ [ 22%]
........................................................................ [ 30%]
........................................................................ [ 38%]
........................................................................ [ 45%]
........................................................................ [ 53%]
........................................................................ [ 61%]
........................................................................ [ 68%]
........................................................................ [ 76%]
........................................................................ [ 84%]
........................................................................ [ 91%]
........................................................................ [ 99%]
....                                                                     [100%]
940 passed in 8.80s
GATE: PASS
```
## Deviations
- `tests/unit/test_content.py` and `test_stage_content_writes_a_content_checkpoint_with_llm_meta` now script JSON text replies (`json_response`) instead of tool calls, and the map-reduce request assertions check `format:"json"`/no tools instead of the tool name. Required by Required fix 2 (`mode="json"`), not by a chapter/quote model change alone; declared as the card's note asks.
- `tests/live/test_content_live.py` gained assertions for `llm_meta.completeness`, the entities count, and the invariant that an empty section is listed in `reasked`. Not run by me (live tests are judge-only).
## NOT proven
- No live gemma4 call was made from this session, so `tests/live/test_content_live.py` passing 3× and the judge's completeness probe ≥5/6 are unverified here; the json-mode request shape, parsing, repair and fallback are proven only against `FakeOllama`.
- The re-ask sub-schema's effect on the real model is assumed from the verdict's measurement (JSON mode complete 5/6); only its request shape and merge path are unit-tested.
## Open
- None.
