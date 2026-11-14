# 09 — Harness: Claude (boss/judge) × Kilo (worker)

Adapted from the proven loop in `../text2image-mcp/docs/10-agent-harness.md`. A deterministic gate settles what an
exit code can settle; the judge settles the rest. **Finder and fixer are different actors.**

## 1. Roles

| Role | Who | May | May not |
|---|---|---|---|
| Worker | Kilo, agent `caracal-worker`, profile `ollama_deepseek.jsonc` (deepseek-v4.1-flash) | edit the card's Files, run the gate, commit on `task/<ID>-*` | touch `main`, push, edit `docs/`, edit tests not named, add undeclared deps, make live network calls |
| Boss / Judge | Claude Code | write docs + cards, launch the worker, review, run live checks, merge to `main` | write feature code (except fixtures, scripts/run-card.sh, docs) |
| Owner | Sameer | secrets, licenses, systemd, final approval at milestones | — |

## 2. Mailbox

```
agents/inbox/NN-<ID>.md       boss → worker   (card, refined; retries include "Required fixes")
agents/outbox/NN-report.md    worker → boss   (format §4)
agents/verdicts/NN.md         boss            (format §5)
agents/logs/NN.log            raw Kilo output (gitignored)
docs/findings/errata.md       worker records doc problems here (append-only); boss resolves
```
`NN` increments every round, including retries.

## 3. Round

1. Boss writes `agents/inbox/NN-<ID>.md`.
2. Boss runs `scripts/run-card.sh agents/inbox/NN-<ID>.md` (background). The worker reads the card, works, gates,
   commits on its branch, writes the outbox report, and exits.
3. Boss judges: `git diff main...task/<branch>`, gate, scope, contract, **non-vacuity** (neuter the code under the 2
   most important new tests, confirm they fail, restore), security, 3 attack inputs, live check if the card has one.
4. APPROVE → boss merges (`git merge --no-ff`), next card. REJECT → next round with "Required fixes".
   After 3 REJECTs on one card the boss splits the card or fixes the docs. NEEDS-OWNER → boss asks the owner.

## 4. Worker report (`agents/outbox/NN-report.md`)

```
STATUS: DONE | STOPPED
## What changed      - <file>: <one line>
## Tests added       - <test>: <what it proves>
## Measured          - gate exit code; tests passed/failed counts (paste the last lines of the gate output)
## Deviations        - <none | each with doc reference>
## NOT proven        - <what was not tested>
## Open              - <questions>
```

## 5. Verdict (`agents/verdicts/NN.md`)

```
# Verdict NN — <ID> — APPROVE | REJECT | NEEDS-OWNER
## Gate         ruff · format · mypy · rules · tests (n)
## Scope        ok | out-of-scope files
## Contract     deviations with doc refs
## Non-vacuity  <test>: neutered <what> → failed as expected | STILL PASSED
## Attacks      <input> → <result>
## Live         <what was run> → <result>
## Required fixes  1. <file:line — what, no code>
```
