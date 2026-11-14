STATUS: DONE
## What changed
- `src/caracal/search/base.py`: `SearchHit` (frozen, dict/JSON round-trip), `SearchProvider` protocol, `SearchError(kind, status, retry_after)`, text/URL/query hygiene and HTTP status→error mapping.
- `src/caracal/search/brave.py`: `BraveProvider` with the measured GET request, `web.results[]` parsing, description+2 extra snippets, HTML/entity cleaning and auth/rate-limit/5xx mapping.
- `src/caracal/search/exa.py`: `ExaProvider` with the measured POST body (`contents.text.maxCharacters=2000`), Markdown-text clipping and cumulative `costDollars.total` counter.
- `src/caracal/search/router.py`: `SearchRouter`/`SearchResult` with configured order, key skipping, per-provider cache, per-job budget, auth disabling, one rate-limit retry (≤5 s), token buckets and summed cost.
- `src/caracal/search/__init__.py`: public exports for the search package.
- `src/caracal/queue.py`: added only `Keys.search_budget(job_id)` (`<prefix>search:budget:<job>`).
- `tests/fakes/search.py`: recorded-fixture response helpers, recording MockTransport handler, `FakeProvider`, `FakeClock`, `RecordingSleep`.
- `tests/unit/test_search_brave.py`, `test_search_exa.py`, `test_search_router.py`: unit coverage for requests, parsing, hygiene, errors, fallback, cache, budget, rate limits and cost.
- `tests/live/test_search_live.py`: judge-run live checks for both providers and the router cache (not run here).

## Tests added
- `test_brave_request_shape_and_parsed_fixture_hits`: exact URL/params/headers and real-fixture hit fields/ranks.
- `test_brave_cleans_strong_and_quot_entities_in_jfk_fixture`: `<strong>` removed and `&#x27;`/`&quot;` decoded.
- `test_brave_unauthorized_statuses_are_auth[401,403]`: 401/403 map to `auth`.
- `test_brave_recorded_422_subscription_token_invalid_is_auth`: real 422 body maps to `auth`.
- `test_brave_422_without_subscription_code_is_bad_response`: only the measured 422 code is auth.
- `test_brave_429_reports_retry_after_in_seconds`: `retry-after` parsed.
- `test_brave_429_falls_back_to_x_ratelimit_reset_header`: reset-header fallback parsed.
- `test_brave_5xx_is_unavailable`: 503 maps to `unavailable`.
- `test_brave_timeout_is_unavailable`: `httpx.ReadTimeout` maps to `unavailable`.
- `test_brave_unparsable_body_is_bad_response`: non-JSON maps to `bad_response`.
- `test_brave_drops_non_http_urls_and_keeps_rank_contiguous`: ftp/javascript/relative URLs dropped, ranks renumbered.
- `test_brave_never_logs_the_key_or_the_error_body`: planted key and recorded error body absent from caplog.
- `test_brave_caps_title_and_snippet_and_uses_two_extra_snippets`: 300/500 caps and exactly the first two extras.
- `test_brave_strips_control_characters`: C0/C1 controls removed, whitespace collapsed.
- `test_exa_request_shape_and_parsed_fixture_hits`: exact POST body (`maxCharacters=2000`), header, hit fields and $0.007 cost.
- `test_exa_cost_counter_accumulates_across_searches`: per-provider cost counter sums calls.
- `test_exa_recorded_invalid_key_is_auth`: real 401 body maps to `auth`.
- `test_exa_403_is_auth`: 403 maps to `auth`.
- `test_exa_429_reports_retry_after`: 429 delay parsed.
- `test_exa_503_is_unavailable` / `test_exa_connect_error_is_unavailable`: server/transport failures fall through as `unavailable`.
- `test_exa_unparsable_body_is_bad_response`: invalid JSON maps to `bad_response`.
- `test_exa_drops_non_http_urls`: non-http scheme dropped.
- `test_exa_never_logs_the_key_or_the_error_body`: planted key and real error text absent from caplog.
- `test_exa_caps_text_title_and_snippet`: 2000/300/500 caps enforced on oversized synthetic data.
- `test_router_uses_configured_order_and_normalises_the_query`: settings order wins over argument order and query is normalised.
- `test_router_falls_through_on_error_kinds[auth,unavailable,bad_response]`: each non-retryable kind falls through.
- `test_router_falls_through_on_empty_results`: empty hit list falls through.
- `test_router_retries_rate_limited_once_then_succeeds`: one retry after the indicated delay then success.
- `test_router_retries_rate_limited_once_then_falls_through`: delay capped at 5 s and second failure falls through.
- `test_router_auth_failure_disables_provider_and_logs_once`: provider skipped on later calls; `search.auth_failed` logged once.
- `test_router_cache_hit_skips_providers_and_budget`: second call is cached, no provider call, budget stays at 1; cache TTL is set.
- `test_router_budget_exhaustion_after_n_calls`: third call over a budget of 2 returns empty, logs `search.budget_exhausted`; budget TTL ≤ 1 day.
- `test_router_sums_provider_costs`: costs of the failed and successful providers are summed.
- `test_router_skips_provider_without_a_key`: Brave is skipped when its key is absent; Exa answers.
- `test_router_returns_empty_without_budget_when_no_provider_is_usable`: no providers means no budget key.
- `test_router_token_bucket_waits_for_brave_rate_limit`: 6th Brave call sleeps 0.2 s (5 req/s).
- `test_router_accepts_uuid_job_ids`: UUID job ids produce the budget key.
- `test_search_budget_key_format`: `search_budget` key shape.
- `test_router_caps_long_queries_at_400_chars`: over-long queries truncated without trailing space.
- `test_router_caches_the_provider_that_actually_answered`: fallback provider's hits are served from cache next time.
- `test_router_token_bucket_waits_for_exa_rate_limit`: 3rd Exa call sleeps 0.5 s (2 req/s).
- `test_router_empty_query_is_a_no_op`: blank query consumes no budget and calls no provider.
- `test_live_providers_return_lindsay_hits` (live): real Brave/Exa both return a McMahon hit.
- `test_live_router_prefers_brave_and_serves_the_second_call_from_cache` (live): Brave wins and the repeat call is cached.

## Measured
- gate exit code: 0
- last 15 lines of `bash scripts/gate.sh`:
  ```
........................................................................ [ 20%]
........................................................................ [ 27%]
........................................................................ [ 33%]
........................................................................ [ 40%]
........................................................................ [ 47%]
........................................................................ [ 54%]
........................................................................ [ 61%]
........................................................................ [ 67%]
........................................................................ [ 74%]
........................................................................ [ 81%]
........................................................................ [ 88%]
........................................................................ [ 95%]
...................................................                      [100%]
1059 passed in 9.80s
GATE: PASS
  ```
  (ruff, format, mypy strict and repository rules all pass earlier in the same run.)

## Deviations
- none

## NOT proven
- `tests/live/test_search_live.py` was written but not executed, per the card's instruction; live API keys and live latency/cost were therefore not exercised in this session.
- Auth disabling and the token buckets are per `SearchRouter` instance. The pipeline is expected to hold one router per process; behaviour across multiple routers in one process is not tested.
- Token-bucket behaviour is proven with an injected fake clock/sleep, not under real wall-clock concurrency.
- Cache-hash separator (`sha1("<query>\n<limit>")`) is an implementation choice; the card does not pin the exact concatenation, and the tests assert behaviour rather than a hard-coded digest.

## Open
- Should `SearchRouter` be wired into `StageContext`/services now, or is that explicitly left to T3.2/T3.3? This card's Files list excluded `services.py`, so no wiring was added.
