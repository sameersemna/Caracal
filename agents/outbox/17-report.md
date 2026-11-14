STATUS: DONE

## What changed
- `src/caracal/llm/schema.py`: `prepare_schema` now rejects any property whose wire name (validation alias if set, else field name) is a JSON-Schema keyword (`RESERVED_FIELD_NAMES`, 13 names) and builds the schema with `by_alias=True`; `coerce` normalises evidence arrays (`evidence`, `*_evidence`, items marked `x-evidence`) into flattened, de-duplicated ids.
- `src/caracal/llm/client.py`: `call_tool` validates with `model_cls.model_validate(..., by_alias=True, by_name=True)` so aliased wire names and plain field names both validate.
- `src/caracal/llm/prompts/__init__.py`: added `render_transcript_lines` (`[seg:12] S2 03:14.2 | text`, `S?` when unknown, `h:mm:ss.s` past one hour), `render_metadata` (`[meta:title] ...`, DATA markers stripped), and `format_timestamp`; `wrap_data` now shares the marker sanitiser.
- `tests/unit/test_llm_schema.py`: test models now use the alias for the report `type` field and no reserved field names; new reserved-name, alias and evidence-normalisation coverage.
- `tests/unit/test_llm_client.py`: recorded fixtures are mapped to the `entity_type` wire name in-test; new aliased-model end-to-end tests.
- `tests/unit/test_llm_prompts.py`: new renderer tests.
- `tests/live/test_llm_live.py`: models use the `entity_type` alias; new realistic six-parallel-call content analysis over `talk_c0.json`.

## Tests added
- `test_prepare_schema_rejects_reserved_wire_names` (13 params): every keyword raises `ValueError("reserved field name: …")`.
- `test_prepare_schema_rejects_reserved_name_in_nested_model`: the check reaches models still under `$defs`.
- `test_prepare_schema_checks_the_alias_not_the_python_field_name`: a reserved `validation_alias` is rejected even though the Python field name is safe.
- `test_validation_alias_is_the_wire_name_and_keeps_the_field_name`: prepared schema exposes `entity_type`; `{"entity_type": "place"}` validates into `.type == "place"` and `model_dump()` keeps `type`.
- `test_aliased_nested_model_validates_wire_name_into_field_name`: the same holds inside a list of nested models.
- `test_coerce_normalises_evidence_ids` (9 params): `"seg:2 S1"`→`["seg:2"]`, `"seg:2, seg:5"`→both, `"see segment 2"`→dropped, `"meta:title"` kept, de-duplication, order kept, non-strings dropped.
- `test_coerce_normalises_evidence_suffix_and_x_evidence_marker`: `*_evidence` and `x-evidence` item markers both trigger the rule.
- `test_coerce_leaves_string_arrays_that_are_not_evidence_untouched`: the rule is name/marker based, not applied to every string array.
- `test_aliased_model_end_to_end_uses_wire_name_and_report_field_name`: request schema has `entity_type`, synonym `Location`→`place`, `"seg:1 S1"`→`["seg:1"]`, report field stays `type`.
- `test_aliased_model_accepts_the_plain_field_name_via_by_name`: a reply using `type` validates, proving `by_name=True`.
- `test_render_transcript_lines_*`, `test_format_timestamp_switches_to_hours_past_one_hour`, `test_render_metadata_*`: exact line format, unknown speaker, ordering, >1 h timestamps, DATA-marker neutralisation.
- `test_live_content_analysis_is_valid_and_cites_only_transcript_ids`: 6 parallel realistic calls, ≥5 valid, every cited id present in the rendered transcript (judge-run).

## Measured
- gate exit code: 0
- ```
  ==> repository rules
  ==> pytest (unit + contract, live excluded)
  ........................................................................ [  9%]
  ........................................................................ [ 19%]
  ........................................................................ [ 28%]
  ........................................................................ [ 38%]
  ........................................................................ [ 47%]
  ........................................................................ [ 57%]
  ........................................................................ [ 66%]
  ........................................................................ [ 76%]
  ........................................................................ [ 85%]
  ........................................................................ [ 95%]
  ....................................                                     [100%]
  756 passed in 7.10s
  GATE: PASS
  ```

## Deviations
- Existing unit tests had to change because the new contract makes their models illegal: `test_llm_schema.Outer.items` → `entries`, and `Entity.type` now carries `validation_alias="entity_type"` (docs/04 §1: LLM-facing models never use reserved names; aliases are the escape hatch). The recorded fixtures still use the pre-card wire name, so the tests map `type`→`entity_type` in-test (`_wire_arguments` / `_recorded_arguments`, and a quoted-key replace for the fenced-JSON fixture) instead of editing fixture files, which are outside the card's Files list. Recorded enum synonyms, inlining and coercion are still exercised; no assertion was removed or loosened.
- `format_timestamp` is an extra exported helper (public for direct unit testing) alongside the two renderers the card asks for; it is not a new module or dependency.
- `render_metadata` only strips `<<<`/`>>>` as specified; newlines inside a metadata value are left as-is (the card does not ask for line collapsing).

## NOT proven
- The live tests were written but not run (card instruction: the judge runs `tests/live/`). The ≥5/6 validity rate and the transcript-id citation check are therefore unverified in this session; module collection was verified.
- The `x-evidence` marker is honoured only where the card specifies it (on array items), not on the property schema itself.
- Evidence tokens with trailing punctuation follow the card's regex literally: `"seg:2."` normalises to `"seg:2."` because `.` is an allowed id character.

## Open
- A report model that needs both a literal `type` and a field actually named `entity_type` would collide on the alias; no convention for a second alias exists yet.
