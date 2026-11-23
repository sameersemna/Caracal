# Web UI Necessity Evaluation — Caracal

Decision: **No web user interface is required.** This document records the evaluation criteria, the
evidence, and the conditions that would justify revisiting the decision.

## 1. Evaluation criteria

| # | Criterion | Question applied |
|---|---|---|
| C1 | Target users & interaction model | Who consumes Caracal, and how do they interact with it? |
| C2 | Interaction complexity | Does the workflow need rich, stateful, interactive UI (drag-drop, live tuning, visual editing), or is it a one-shot submit → poll → retrieve pattern? |
| C3 | Data volume & visualization | Is there data that only makes sense explored visually? |
| C4 | Accessibility / non-technical users | Is any consumer excluded by a CLI/REST/MCP surface? |
| C5 | Deployment constraints | What does the runtime and security posture of the host permit or favor? |
| C6 | Maintenance overhead | Can the current team, process, and toolchain sustain a second stack? |
| C7 | Alignment with binding spec | Does the governing documentation admit a UI at all? |

## 2. Findings per criterion

- **C1 — Users are the owner and AI agents.** `docs/00` §Roles: Owner (Sameer, developer), Boss/Judge
  (Claude Code), Worker (Kilo). Consumers are scripts, `scripts/live-suite.sh`, and MCP clients
  (Claude Code/Desktop). No persona, in any doc or card, reaches Caracal through anything other than
  REST, MCP, or the CLI. The conversational UX layer already exists — it is whatever MCP client the
  user points at Caracal.
- **C2 — Interaction is batch and headless-friendly.** The API is one-shot: submit a source, get a
  queued→running→done job, stream progress (SSE or MCP progress notifications), fetch artifacts
  (report JSON/MD, transcript, SRT/VTT). Analysis is autonomous by design (`docs/00`: "It runs
  autonomously: one request in, one finished report out"). There is no stateful editing, no live
  audio scrubbing requirement, and no parameter-tuning loop that a UI would accelerate. The only
  interactive step is asking questions (`/v1/jobs/{id}/ask`), which is naturally a client-side
  conversation MCP already covers.
- **C3 — Artifacts are self-describing, not exploratory datasets.** Outputs are structured documents
  (Markdown/JSON report, text/SRT/VTT transcripts) consumed by downstream tooling. The volumes are
  modest (jobs list with cursor pagination; ≤ 50 search results). No dashboards, no drill-down grids,
  nothing that exceeds `curl` + an MCP client.
- **C4 — No non-technical consumers are in scope.** Every documented access path assumes a
  bearer-token-capable client. Nothing in `docs/`, `README.md`, `NOTES.md`, or git history mentions
  non-developer stakeholders or browsers.
- **C5 — Deployment favors headless.** Single LAN host (`promaxgb10-6116`, aarch64/GB10), systemd
  units in `deploy/`, `CARACAL_API_TOKEN` bearer auth, `CARACAL_ALLOWED_HOSTS` / localhost-default
  network posture, deliberately no MCP tool for biometric voiceprint export (`docs/03` §0: "REST only
  by design"). A web app would add a Node build toolchain to the host, a CORS/auth surface, and a
  token-storage pattern (a public SPA would embed the bearer token in the browser) — strictly worse
  for the stated posture.
- **C6 — Process cannot sustain a second stack cheaply.** Python-only repo: mypy `strict`, ruff, a
  test gate (`docs/07`), agent-driven cards where "the worker never edits docs" and the binding repo
  layout (`docs/01` §3) forbids inventing other top-level modules. A frontend adds a parallel stack
  (framework, bundler, UI tests, deployment artifact) that none of the existing tooling, harness, or
  cards cover. Maintenance cost is recurring; the benefit (C1–C4) is absent.
- **C7 — The binding spec excludes it.** `docs/00` §Decisions: "Interfaces — REST API and MCP server,
  equal citizens. Owner requirement. Parity is binding and tested." The implementation plan
  (`docs/08`, M0–M4) contains no UI milestone; the repo is at the M4/Operate stage (recent commits:
  T4.4–T4.5 translation/routing work). No card, verdict, or finding anywhere proposes a UI.

## 3. Conclusion

**A web UI is not required — now or on the current roadmap.** Every capability a hypothetical UI
would expose already has a richer consumer: REST for scripts and artifact plumbing, MCP for
conversational and agentic use, CLI (`caracal doctor|migrate|serve|worker`) for operation, and
Prometheus `/metrics` for observability. Operational visibility, the one place a "page" is often
wanted, is fully served by pointing any generic Grafana instance (zero Caracal code) at the existing
`/metrics` endpoint.

## 4. Revisit triggers (write these down so the decision is falsifiable)

Build (or reconsider) a UI only when at least one of these becomes true:

1. **Non-technical stakeholders** must review reports or approve speaker enrollment (consent for
   biometric data) without an agent or shell — e.g. Caracal is shared beyond the owner.
2. **Multi-tenant / public deployment** with self-service accounts — a real identity surface appears,
   which a UI would be the natural front for (and which would require the auth redesign of docs/06).
3. **Report curation workflows** emerge that need visual work: editing speaker assignments, calibrating
   voiceprint thresholds from `GET /v1/jobs/{id}/voiceprints` output, or reviewing review-flags at scale.

If a UI is ever built, the least-cost shape is a small read-mostly web app over the existing REST
service served by a separate process (no business-logic coupling; parity stays REST+MCP as `docs/03`
§0 requires) — re-evaluate the stack at that time, not now.

## 5. Assumptions to validate

- No undisclosed requirement for browser-based access by non-developers (none found in docs, cards,
  or git history; owner confirmation would settle it if in doubt).
- Caracal remains an owner-operated service on `promaxgb10-6116` rather than a public product.
- Observability needs stay at "counters and stage timings", which `/metrics` already provides.

## 6. Required action items from this decision

None that modify source. The decision closes the question; the only standing notes are the revisit
triggers in §4, which the team may fold into `docs/00` §Decisions if the Boss accepts it (that edit
is out of this session's scope — docs are owned by Claude per `docs/00` §Roles).
