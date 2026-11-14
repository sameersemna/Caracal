STATUS: DONE
## What changed
- `src/caracal/report/assemble.py`: new pure `assemble_report()` — maps every checkpoint onto a `Report`, strips prose citations into evidence, tolerates missing/failed stages, builds the evidence index and runs `validate_citations`, and computes the deterministic docs/04 §11 review flags.
- `src/caracal/pipeline/stages.py`: real `stage_assemble` (writes `report.json` + `report.md` atomically, `reports.put(..., "1.0")`, section counts/uncited/flags output), wired into `m2_registry`; transcript/caption loaders for the assembler.
- `src/caracal/services.py`: `report()`, `report_markdown()`, `report_summary()` over `ReportRepo` with `not_found`/`job_not_ready` semantics.
- `src/caracal/api/routes.py`: `GET /v1/jobs/{id}/report?format=json|md` (md = `text/markdown; charset=utf-8`).
- `src/caracal/mcp/server.py`: `get_report` tool (section filter, unknown section → `invalid_input`, markdown `{format, content}`) and `report_summary` on the done `analyze_audio` result.
- Tests: new `tests/unit/test_assemble.py`, `tests/contract/test_api_report.py`, `tests/live/test_report_live.py`; extended `test_mcp_tools.py`, `test_mcp_wire.py`, `test_parity.py` (report row now available), `test_stages.py`.

## Tests added
- `test_full_checkpoints_assemble_a_valid_cited_report`: real TED fixture transcript + scripted content/identity → valid report, citations stripped/merged, quotes untouched, every cited id resolves, all stage statuses recorded.
- `test_missing_and_failed_stages_give_empty_sections_and_an_error_flag`: missing langid/content + failed identity → empty sections, unnamed align-derived speaker, failed-stage error flag.
- `test_music_job_uses_und_languages_and_has_no_quotes_or_claims`: plan music override, `und`, no quotes/claims, non-speech flag.
- `test_prose_citation_stripping_table`: bracketed/parenthesised id lists vs ordinary parenthetical text.
- `test_deterministic_review_flags`: each docs/04 §11 check (shares, chapter coverage, caption agreement, low-confidence share, failed stage).
- `test_stage_assemble_writes_report_json_markdown_and_store`, `test_stage_assemble_never_fails_because_a_section_is_missing`: stage files/DB/output and tolerant sections.
- `test_report_json_is_the_stored_report`, `test_report_markdown_is_served_as_utf8_markdown`, `test_report_before_assembly_is_409_job_not_ready`, `test_unknown_job_is_404`, `test_failed_job_without_a_report_is_404`, `test_invalid_report_format_is_422_invalid_input`.
- MCP: full report, `sections` filter, markdown, unknown section, `job_not_ready`, output-schema/tool-name set, `analyze_audio` `report_summary`; wire-level `get_report`; REST/MCP parity.
- Live (written, not run): `test_ted_report_validates_and_is_served_as_json_and_markdown`.

## Measured
- gate exit code: 0
- ```
........................................................................ [ 21%]
........................................................................ [ 28%]
........................................................................ [ 35%]
........................................................................ [ 42%]
........................................................................ [ 49%]
........................................................................ [ 56%]
........................................................................ [ 63%]
........................................................................ [ 71%]
........................................................................ [ 78%]
........................................................................ [ 85%]
........................................................................ [ 92%]
........................................................................ [ 99%]
.....                                                                    [100%]
1013 passed in 9.66s
GATE: PASS
```

## Deviations
- `docs/01 §3` already lists `report/assemble.py` at HEAD (commit e4e8292), so no doc edit was needed; no docs file was touched.
- `assemble_report` is documented as `(checkpoints, job, settings)`. It also takes optional keyword-only `transcript`, `caption_words` and `stage_statuses` because the transcript/caption artefacts are not stage checkpoints (the stage loads them from the job directory) and failed stages have no checkpoint to read their status from. The call form in the card is unchanged; the extras keep the function pure.
- `audio.music_detected` is `null` when no plan checkpoint recorded it (the schema allows `bool|null`); `false` only when the plan explicitly says so.
- Existing test files were extended only: the MCP tool-name set / wire `REQUIRED_TOOLS` gained `get_report`, the report parity row moved from pending to available, and the M2 registry test asserts `Stage.ASSEMBLE`.

## NOT proven
- The live test was not run (worker instructions): real TED job through `caracal serve` + worker, `speakers[0].name == "Daniel Levitin"`, ≥3 entities, ≥1 checkworthy claim, `quality.stages.assemble == "done"`, markdown H1.
- No real Postgres/Redis/GPU run locally; `ReportRepo` storage and the endpoint are covered with the fakes and the repository contract tests.
- Assembler tolerance is unit-tested for missing/failed checkpoints; malformed (non-JSON-shaped) checkpoint values are not treated as missing.

## Open
- `assemble` is non-critical per docs/01 §4: if it fails outside a missing section (e.g. a database error while storing the report), the job still finishes `done` and `GET /report` stays `409 job_not_ready`. The contract permits this, but the owner may prefer an `error` flag or a failed job in that case.
