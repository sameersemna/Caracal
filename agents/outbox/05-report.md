STATUS: DONE

## What changed
- `src/caracal/queue.py`: `touch(consumer, entry_id) -> bool` and `ack(consumer, entry_id) -> bool` are single owner-checked Lua scripts (XPENDING owner test, then XCLAIM … JUSTID / XACK+XDEL); `rate_limit` is one Lua script (INCR, EXPIRE when TTL is −1, returns `{count, ttl}`) so a counter can never be left without a window.
- `src/caracal/pipeline/runner.py`: `RunnerDeps.still_owner: Callable[[], Awaitable[bool]]` (default always True); the runner checks it after the terminal-status check, before every stage (before the cancel check), again after a stage returns (before any checkpoint/stage-row/event write), in the timeout handler and before the final `set_status(done)`; returns `"lost"` and writes nothing after the loss.
- `src/caracal/worker.py`: injects `still_owner` as an owner-checked `touch` for the delivery's entry; a `"lost"` result logs `job.lost` and leaves the entry unacked; `_ack` reports (and skips) an ack refused because another consumer owns the entry; the periodic heartbeat task stops when it finds a new owner.
- `tests/unit/test_queue.py`: steal-back touch/ack tests, atomic rate-limit test, TTL-repair test; adapted `ack`/`touch` call sites.
- `tests/unit/test_runner.py`: `_deps` gained an `owner` flag sequence; three lost-ownership tests (before first stage, between stages, mid-stage).
- `tests/unit/test_worker.py`: end-to-end steal-back test: A loses, returns `"lost"`, logs `job.lost`, writes no checkpoint, does not ack, entry stays pending for B.
- `tests/live/test_queue_live.py`: real-Redis steal-back test (A touch False, owner stays B, late A ack False, B ack True) plus TTL-less rate-limit repair; adapted the existing ack/touch calls.

## Tests added
- `test_steal_back_touch_by_the_old_owner_is_refused_and_keeps_the_new_owner`: after B reclaims, A's `touch` returns False and XPENDING owner is still B (fails on the old plain `XCLAIM`).
- `test_steal_back_ack_by_the_old_owner_leaves_the_new_owners_entry_pending`: A's late `ack` returns False, the stream still has the entry, owner is B; B's ack then returns True and clears it (fails on the old plain `XACK`+`XDEL`).
- `test_rate_limit_increments_and_sets_the_window_in_one_command`: one `rate_limit` issues exactly one `EVAL` (no INCR/EXPIRE/TTL commands), so a crash cannot split the operation.
- `test_rate_limit_repairs_a_counter_left_without_a_ttl`: a key seeded with TTL −1 gets a window on the next call, and the count continues from the seeded value.
- `test_lost_ownership_before_the_first_stage_writes_nothing`: returns `"lost"`, no stage runs, no stage rows, no checkpoint, the only event is the initial `status`.
- `test_lost_ownership_after_a_stage_writes_nothing_for_the_next`: the finished stage's checkpoint is kept; nothing at all is written for the next stage and no `end` event is emitted.
- `test_stage_result_is_discarded_when_ownership_is_lost_while_it_runs`: a stage that loses ownership mid-run returns its result but neither its checkpoint nor its stage row is written.
- `test_ownership_loss_stops_the_job_without_ack_and_keeps_the_new_owners_entry` (worker): a reclaimed-in-flight job makes the runner return `"lost"`, the worker logs `job.lost`, writes no checkpoint, leaves the job `running`, does not ack, and the entry stays pending for worker-b.
- `test_steal_back_rejects_the_stale_owner` (live): the same scenario on real Redis, plus the rate-limit repair.

## Measured
- gate exit code: 0
- ```
==> ruff check
All checks passed!
==> ruff format --check
39 files already formatted
==> mypy (strict)
Success: no issues found in 17 source files
==> repository rules
==> pytest (unit + contract, live excluded)
........................................................................ [ 25%]
........................................................................ [ 50%]
........................................................................ [ 76%]
...................................................................      [100%]
283 passed in 2.81s
GATE: PASS
```
- Non-vacuity check (each fix temporarily reverted, then restored):
  - touch back to plain `XCLAIM` → `test_steal_back_touch_…` FAILED
  - ack back to plain `XACK`+`XDEL` → `test_steal_back_ack_…` FAILED
  - rate limit back to `INCR` + `EXPIRE NX` → `test_rate_limit_increments_and_sets_the_window_in_one_command` FAILED (commands were `INCRBY`, `EXPIRE`)
  - pre-stage ownership check removed → 2 runner tests FAILED
  - post-stage ownership check removed → 2 runner tests FAILED
  - worker no longer passing `still_owner` → worker test FAILED (job completed and ack was attempted)
  After every mutation the original files were restored and the gate rerun green.

## Deviations
- `tests/unit/test_queue.py`: adapted existing calls whose signature changed: `queue.ack(entry_id)` → `queue.ack("worker-a", entry_id)` in `test_enqueue_read_and_ack` and `test_every_key_and_channel_uses_the_prefix`; the `touch` return is now asserted `is True` in those two tests.
- `tests/live/test_queue_live.py`: adapted `queue.ack(entry_id)` → `queue.ack("live-worker", entry_id)` and asserted the `touch` result in the existing test; the new live test sleeps 20 ms before `min_idle_ms=0` reclaim so the idle time is measurably above zero on real Redis.
- The worker's `still_owner` implementation is an owner-checked `touch` (not a read-only XPENDING): it is atomic, judge-verified, and keeps the idle timer fresh at stage boundaries; the 60 s heartbeat task is unchanged.
- `still_owner` is also checked before the final `set_status(done)` and inside the timeout handler, so no write lands after ownership is lost on those paths either (stronger than the literal "before every stage").

## NOT proven
- The live test was written but not run in this session (worker instructions: live tests are the judge's to run); it uses only documented commands already verified in round 04 plus the judge-verified touch script.
- Real-Redis concurrency is not exercised by the gate; the unit coverage uses fakeredis 2.40.0 with `lupa` Lua, which mirrors real Redis for these scripts in a probe I ran, but timing under real load is the judge's attack.
- The `rate_limit` repair test alone would also pass with the old two-command code (`EXPIRE … NX` fixes a TTL-less key); the revert is caught by the single-command test instead.
- No test asserts the log level of `job.lost`, only that the message is emitted (WARNING).

## Open
- docs/05 §2 does not say whether `still_owner` should also re-touch at stage boundaries; I used an owner-checked touch (same Lua as the heartbeat) so the idle timer stays fresh. If the boss prefers a pure read, `JobQueue` would need a read-only owner query instead.
