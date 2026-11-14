STATUS: DONE
## What changed
- `src/caracal/analysis/verify.py`: `_identity_context` now prefers channel → uploader → title; `_mentions_context` checks channel/uploader (title only when neither exists); `_queries_for` prepends the deterministic `<candidate> <channel/uploader/title>` query for identity-confirm targets and still runs the model's differing query; `_corroboration_context` delegates to the same helper.
- `tests/unit/test_verify.py`: added the real-metadata test (`Lindsay All Ears English`); adapted the three existing identity query-string expectations from `Lindsay All Ears English Podcast` to `Lindsay All Ears English` (channel now wins over the title).

## Tests added
- `test_identity_query_uses_the_channel_not_the_episode_title`: with title "How to Learn English with a Podcast Thank You" and channel/uploader "All Ears English", the first search is exactly `Lindsay All Ears English`, the model's title-based query also runs with the channel appended, and S1 still promotes to `Lindsay McMahon`. Verified non-vacuous: reverting the context order, reverting the mention rule, and removing the deterministic prepend each make it fail.

## Measured
- gate exit code: 0
```
........................................................................ [ 26%]
........................................................................ [ 33%]
........................................................................ [ 39%]
........................................................................ [ 46%]
........................................................................ [ 53%]
........................................................................ [ 59%]
........................................................................ [ 66%]
........................................................................ [ 72%]
........................................................................ [ 79%]
........................................................................ [ 86%]
........................................................................ [ 92%]
........................................................................ [ 99%]
......                                                                   [100%]
1086 passed in 9.80s
GATE: PASS
```

## Deviations
- Existing test expectations changed: per the card, only query strings were adapted (`Lindsay All Ears English Podcast` → `Lindsay All Ears English`), because the identity context is now the channel rather than the title.
- The deterministic identity query is emitted only when some metadata context exists (channel, uploader, or title). With completely empty metadata it would be the bare candidate, which the model's queries already contain; emitting it as a second query would add no context. This keeps the existing verify-stage integration test green (`tests/unit/test_stages.py` supplies no metadata and asserts `queries_used == 1`); it is a no-op for the real job, which always has a channel.

## NOT proven
- The real All Ears English pipeline was not run here (no live provider calls from the worker); the judge's rerun is the acceptance check that S1 becomes "Lindsay McMahon".
- The real job's actual LLM-planned identity query is replicated from verdict 33's trace, not re-measured.

## Open
- none
