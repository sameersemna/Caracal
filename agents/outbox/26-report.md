STATUS: DONE
## What changed
- `src/caracal/analysis/content.py`: map windows are merged deterministically (entities deduped by normalised text with majority type and evidence union; topics by normalised label with weight=max; chapters in time order with adjacent near-identical titles merged; quotes/claims deduped by normalised text with evidence union), followed by one small JSON-mode call for summary+context only; music mode adds a metadata-only summary/context call, empty quotes/claims and a "non-speech audio" ReviewFlag.
- `src/caracal/llm/prompts/content_reduce.md`: now a summary+context-only prompt with an explicit mode note for reduce vs music.
- `src/caracal/analysis/planner.py`: music uses the measured exclusive-turn speech share (< 0.15) when a diarize checkpoint exists and the lexical text rule otherwise; the music rule no longer skips diarize.
- `src/caracal/pipeline/stages.py`: DAG reordered to `langid → diarize → plan → align → voiceprint → content`; `stage_diarize` is no longer plan-gated; `stage_plan` derives speech share from exclusive turns; `stage_content` runs music mode from the plan checkpoint.
- `src/caracal/pipeline/runner.py`: logs `stage.done` (INFO, duration), `stage.skipped` (INFO, reason) and `stage.failed` (WARNING, code, error, retryable, attempts, duration) with no transcript text.
- `src/caracal/queue.py`: single `make_redis(settings, *, block_ms=5000)` factory with `socket_timeout=block_s+10`, `socket_connect_timeout=5`, `health_check_interval=30`, `retry_on_timeout=True`.
- `src/caracal/cli.py`, `src/caracal/app.py`: use `make_redis`; `caracal doctor` reports the JS runtime.
- `src/caracal/ingest/ytdlp.py`: `js_runtimes={"node": {"path": ...}}` when node is on PATH.
- `scripts/check_rules.py`: AST rule `redis-from-url` for `redis*.from_url(...)` outside `queue.py`.
- `pyproject.toml`, `uv.lock`: added `yt-dlp-ejs==0.8.0`.
## Tests added
- `test_deterministic_merge_dedupes_entities_topics_chapters_quotes_and_claims`: proves entity majority type/evidence union, topic weight=max, adjacent chapter merge, and quote/claim dedupe.
- `test_map_reduce_windows_overlap_by_one_segment_and_reduce_merges_deterministically`: proves the reduce call has only `summary`/`context` in its schema and deterministic window merge.
- `test_quote_spanning_two_segments_matches_the_joined_text`: keeps valid multi-segment quotes.
- `test_music_mode_analyses_metadata_only_and_flags_non_speech_audio`: proves no transcript in the prompt, empty quotes/claims, and the flag.
- `test_stage_content_music_mode_returns_no_quotes_or_claims`: proves the stage forwards plan music mode into the checkpoint.
- `test_diarization_speech_share_decides_music_without_an_llm_call`: 0.02 → music, 0.87 → not music, no LLM call for music.
- `test_music_text_fallback_is_seventy_percent_of_speech_time`: text fallback without a diarize checkpoint.
- `test_stage_plan_uses_the_diarization_speech_share`: 0.02/0.87 speech-share cases from a diarize checkpoint.
- `test_runner_logs_every_stage_outcome_without_transcript_text`: caplog proves done/skipped INFO and failed WARNING with code/message/retryable/attempts.
- `test_make_redis_sets_a_block_outliving_socket_timeout` and `...scales_the_socket_timeout...`: exact Redis kwargs.
- `test_redis_from_url_outside_queue_is_reported` and `...allowed_in_the_queue_factory`: check_rules coverage for import/alias forms.
- `test_js_runtimes_is_absent_when_node_is_not_on_path`: node-detection branch.
- Live: `test_live_map_reduce_with_a_small_context_budget_is_complete` (small `LLM_CONTEXT_CHARS`, all sections complete).
## Measured
- gate exit code: 0
- ```
........................................................................ [ 15%]
........................................................................ [ 22%]
........................................................................ [ 30%]
........................................................................ [ 37%]
........................................................................ [ 45%]
........................................................................ [ 52%]
........................................................................ [ 60%]
........................................................................ [ 67%]
........................................................................ [ 75%]
........................................................................ [ 83%]
........................................................................ [ 90%]
........................................................................ [ 98%]
..................                                                       [100%]
954 passed in 8.82s
GATE: PASS
```
## Deviations
- Declared test adaptations for the required behaviour changes: stage-order expectations (`langid → diarize → plan`), the non-critical failure call order, music-plan `run["diarize"]` assertions in `test_planner.py`, `test_stages.py` and `tests/live/test_content_live.py`, the plan-skip parametrisation no longer includes `Stage.DIARIZE`, the yt-dlp options table now asserts `js_runtimes`, and the CLI worker test monkeypatches `caracal.cli.make_redis`.
- `stage.failed` cannot use `extra={"message": ...}` because Python `logging` reserves `message`; the error text is in `extra["error"]` and is also rendered in the formatted log line (`stage.failed <code>: <message>`), while `code`, `retryable` and `attempts` stay as separate fields.
- `content_window.md` was not in the card's Files list, so windows still do not emit their own summary/context. The reduce call receives the partial findings plus metadata (the card's "window summaries/contexts (+ metadata)" intent) and writes only summary/context.
## NOT proven
- Live tests were not run (per the worker instructions): the real 12-min TED worker run with `LLM_CONTEXT_CHARS=6000`, the SoundCloud instrumental through pyannote, the worker log with no `queue read failed`, and YouTube with no JS warning are left to the judge.
- No real pyannote/Postgres/Redis worker pipeline was run locally; stage behaviour is covered with fakes and the checked-in unit/contract gate.
## Open
- The "non-speech audio" ReviewFlag currently lives on the content checkpoint (`ContentAnalysis.review_flags`) because `review`/`assemble` are still stubs; when those stages land it should be carried into `quality.review_flags`.
