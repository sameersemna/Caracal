STATUS: DONE
## What changed
- src/caracal/analysis/qa.py: FTS retrieval (top 12, job-scoped), keyword expansion when <3 hits, ±1 neighbouring segments, report summary/context/chapters always in context, half-`LLM_CONTEXT_CHARS` cap, and deterministic citation enforcement.
- src/caracal/llm/prompts/qa.md: JSON-mode answer prompt (question-language answer, seg citations, Not-found rule, DATA blocks).
- src/caracal/llm/prompts/qa_keywords.md: JSON-mode keyword prompt in the transcript's dominant language.
- src/caracal/db/repo.py: `SearchHit.speaker`; `SegmentRepo.search(..., job_id=None, ...)` in the protocol and Postgres (`websearch_to_tsquery('simple')` + `ts_rank`, job filter, speaker column).
- tests/fakes/repo.py: fake search honours `job_id`, returns `speaker`, and approximates `OR`/conjunction semantics.
- tests/repo_contract.py: new `case_segments_search_job_filter_and_speaker` (fake + judge's live Postgres run).
- src/caracal/services.py: `ask_about_audio` (409 unless done, report/transcript context, lazy LLM client) and `search_transcripts` (422 blank query, 1–50 limit, done jobs only).
- src/caracal/api/schemas.py: `AskIn`, `SearchItem`, `SearchResults`.
- src/caracal/api/routes.py: `POST /v1/jobs/{id}/ask` and `GET /v1/search`.
- src/caracal/mcp/server.py: `ask_about_audio` and `search_transcripts` tools with LLM-facing descriptions.
- tests/unit/test_qa.py: retrieval, keyword step, neighbours, report context, cap, citation and injection tests.
- tests/contract/test_api_qa_search.py: REST/MCP ask+search, 404/409/422, done-only search, citation replacement.
- tests/contract/test_mcp_tools.py: MCP ask/search happy paths and error envelopes; registered-tool list updated.
- tests/contract/test_mcp_wire.py: ask/search over the real streamable-HTTP transport; required tool set updated.
- tests/contract/test_parity.py: ask + full-text search rows promoted from PENDING to AVAILABLE, with REST/MCP equality tests.
- tests/live/test_qa_live.py: judge-run TED temperature/Not-found/cortisol checks and the German-transcript English-question check.
- docs/findings/errata.md: recorded the docs/03 §2 `/v1/search` bare-list vs card §4 object/`speaker` discrepancy.

## Tests added
- test_keyword_expansion_runs_when_fewer_than_three_hits: 0 literal hits calls qa_keywords, then searches `Krakau or Polen or Gipfel`; also pins limit=12 and the job_id filter.
- test_two_hits_still_trigger_keyword_expansion: 2 hits still runs the keyword pass, together with the 3-hit no-keyword test pinning the <3 threshold.
- test_citation_text_and_start_come_from_the_real_segment: fabricated model start/text are replaced by the DB row values.
- test_citation_ids_tolerate_rendered_line_noise: "seg:1." and "seg:2 S1" resolve through the typed-evidence rule.
- test_unknown_citation_is_dropped_but_a_valid_one_is_kept: unknown ids are dropped without discarding valid citations.
- test_invented_segment_id_becomes_not_found: only-segment evidence gives exactly "Not found in the audio.", confidence 0.
- test_summary_only_citation_is_not_found: `meta:summary` is not a valid citation.
- test_injection_question_is_wrapped_as_data: question sits inside a DATA block next to the ignore-instructions guard and cannot bypass citation rules.
- test_answer_keeps_the_language_of_the_question: the answer prompt carries the language rule and the scripted non-English answer is preserved.
- test_neighbouring_segments_are_added_to_the_context: hits at 2/5/8 put segments 1–9 in the prompt; report summary/context/chapter title present.
- test_context_is_capped_at_half_the_llm_context_chars: report+transcript blocks stay within `LLM_CONTEXT_CHARS / 2`.
- test_question_validation: strips, rejects empty/whitespace and >1000 characters.
- test_qa_report_uses_the_dominant_language_context_and_chapters: `qa_report` picks German at 0.9 share and carries context/chapters.
- Contract: ask success with real citation, 409 for queued/running/failed, 404 unknown, 422 for empty/blank/too-long question, Not-found on invented citation, search done-only + speaker + limit, 422 blank query/limit 0/51, MCP ask/search parity and errors.
- Repo contract: job_id filter, speaker propagation, limit under the filter, unknown job id → [].
- Wire test: ask and search return JSON-RPC results with object structuredContent over `/mcp/`.

## Measured
- gate exit code: 0
```
........................................................................ [ 30%]
........................................................................ [ 36%]
........................................................................ [ 42%]
........................................................................ [ 48%]
........................................................................ [ 54%]
........................................................................ [ 61%]
........................................................................ [ 67%]
........................................................................ [ 73%]
........................................................................ [ 79%]
........................................................................ [ 85%]
........................................................................ [ 91%]
........................................................................ [ 97%]
............................                                             [100%]
1181 passed in 11.20s
GATE: PASS
```

## Deviations
- `GET /v1/search` returns `{"items": [...]}` with `speaker`; docs/03 §2 shows a bare list without it. The card (requirement 4) and docs/03 §3's object rule are binding for this round, so I followed the card and recorded the doc discrepancy in docs/findings/errata.md.
- None otherwise.

## NOT proven
- Live tests were written but not run (card reserves them for the judge): real gemma4 temperature answer/citation, 1998 World Cup Not-found, German→English keyword expansion, and `search_transcripts("cortisol")` against the TED job.
- The Postgres `job_id` filter was not executed locally; the shared fake contract case passes, and the judge's `tests/live/test_db.py` run exercises the Postgres implementation.
- The German live case needs `CARACAL_LIVE_GERMAN_URL` or an existing done German job; otherwise it skips (no German news URL is fixed by the card).
- No real LLM request was made from the unit/contract layers; a scripted client stands in.

## Open
- Should docs/03 §2's `GET /v1/search` row be updated to the object shape with `speaker` (errata entry)?
- Should the card fix a stable German news URL/fixture for the live keyword-expansion check?
