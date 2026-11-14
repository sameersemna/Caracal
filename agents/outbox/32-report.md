STATUS: DONE
## What changed
- `src/caracal/analysis/verify.py`: one corroboration follow-up search per identity/quote target whose proposed `full_name` / `attributed_to` appears on < 2 registrable domains; the new hits get fresh global `web:` ids, join the target's hits and every deterministic check re-runs; the rationale records `corroborated: true` when that follow-up is what satisfies the two-domain rule.
- `src/caracal/analysis/verify.py`: search accounting moved into `_SearchState` (`web:` numbering, `queries_used`, `cost_usd`) so planned queries, the claim retry and the corroboration query share one global sequence and the totals include the follow-up.
- `tests/unit/test_verify.py`: `ScriptedRouter` gained a `budget_after_calls` budget probe; added the judge's probe as embedded Brave data (`LINDSAY_CORROBORATION_BODY`) because `tests/fixtures/` is outside this card's file scope.
## Tests added
- `test_thin_name_match_triggers_one_corroboration_and_promotes`: first search is `brave_lindsay.json` truncated to the single apple.com/McMahon hit, the follow-up `"Lindsay McMahon" All Ears English` adds four real domains → speaker promoted to `Lindsay McMahon` (0.6175), evidence `web:2..web:5`, verdict `supported`, rationale ends `corroborated: true`, `queries_used == 2`, cost doubled. Fails if corroboration is removed (measured by mutation).
- `test_budget_exhausted_before_corroboration_keeps_the_confirmed_name`: budget probe spent after the first call → no second query, name stays `Lindsay` (confirmed-only, "web confirms Lindsay in this recording"). Fails if the budget guard is removed (the second call would appear in `router.calls`).
- `test_thin_quote_attribution_gets_one_corroboration_search`: attribution `John F. Kennedy` on one domain → follow-up `"<quote>" John F. Kennedy` over three domains → 0.5 `web` candidate and `mixed` verdict, rationale ends `corroborated: true`. Fails if corroboration is removed (measured by mutation).
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
........................................................................ [ 73%]
........................................................................ [ 79%]
........................................................................ [ 86%]
........................................................................ [ 93%]
........................................................................ [ 99%]
..                                                                       [100%]
1082 passed in 9.78s
GATE: PASS
```
- non-vacuity mutation: forcing `_corroborate` to return early makes both promotion tests fail (`assert router.calls` shows the missing follow-up); restoring it makes them pass.
## Deviations
- none. `tests/live/test_verify_live.py` was not modified: the existing assertions already match the required outcome and the card allows writing it only if needed.
- `corroborated: true` is recorded inside `Verification.rationale` (e.g. `...; corroborated: true`) because `Verification` has no boolean field and `report/model.py` is outside this card's file scope.
## NOT proven
- The live judge run (`tests/live/test_verify_live.py`) was not executed here (live tests are judge-run): the real Brave result set for the identity query may already contain two McMahon domains, in which case no follow-up fires and the assertion still passes; if it contains one, the follow-up `"Lindsay McMahon" All Ears English` is the query the judge measured at four domains.
- The corroboration fixture is embedded from the judge's probe rather than a fresh recording; it is parsed by the real `BraveProvider`, not a hand-built `SearchHit`.
- Cache/budget accounting inside `SearchRouter` itself is unchanged and assumed correct from T3.1; only the stage-side query count and cost include the follow-up.
## Open
- If the owner wants `corroborated` as a first-class report field, `Verification` in `report/model.py` and docs/03 §5 need to change (out of scope for T3.2).
