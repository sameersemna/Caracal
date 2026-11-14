STATUS: DONE
## What changed
- `src/caracal/report/render_md.py`: `md_escape` now wraps URLs found in untrusted text (`(?:https?|ftp)://…` and `www.…`) in inline code spans (fence one backtick longer than the longest run inside the URL, padded when the URL starts/ends with a backtick so the fence cannot merge), and strips bidi embedding/override/isolate controls U+202A–U+202E and U+2066–U+2069 while keeping U+200E/U+200F.
- `tests/unit/test_render_md.py`: added a real markdown-it render test over a report whose every rendered free-text field carries the hostile payload, direct tests for URL masking/backtick fencing/bidi stripping, and adapted the three expectations that the mandated code-span behaviour changes.
- `pyproject.toml`, `uv.lock`: dev-only exact pins `markdown-it-py==3.0.0`, `linkify-it-py==2.0.3`.

## Tests added
- `test_gfm_render_of_hostile_report_has_no_content_tags_links_or_bidi`: renders the hostile report with `MarkdownIt("gfm-like", {"html": True, "linkify": True})` and asserts no `<a>`, `<img>`, `<script>`, `<svg>` or `<iframe>` tag, no `href`/`src` attribute from content, and no bidi control in the HTML, while the payloads themselves are present. Fails when URL masking or bidi stripping is reverted (verified both).
- `test_md_escape_table` (5 new cases): bare `http`/`https`/`ftp` and `www.` URLs become code spans, `<https://evil/a>` stays an escaped angle-bracket literal around a code span, and a URL containing backticks gets a two-backtick fence with padding.
- `test_md_escape_strips_bidi_controls_but_keeps_directional_marks`: all nine embedding/override/isolate controls removed and U+200E/U+200F preserved.

## Measured
- gate exit code: 0
- ```
  ........................................................................ [  7%]
  ........................................................................ [ 15%]
  ........................................................................ [ 23%]
  ........................................................................ [ 31%]
  ........................................................................ [ 39%]
  ........................................................................ [ 47%]
  ........................................................................ [ 55%]
  ........................................................................ [ 63%]
  ........................................................................ [ 71%]
  ........................................................................ [ 79%]
  ........................................................................ [ 87%]
  ........................................................................ [ 95%]
  ......................................                                   [100%]
  902 passed in 8.77s
  GATE: PASS
  ```
- Non-vacuity measured: masking disabled (`_URL_PATTERN` never matches) → 9 failed; bidi clause removed → 2 failed.

## Deviations
- Adapted existing tests (verdict 22 Required fix 1, card 23 Accept, docs/06 S9): `test_md_escape_table` cases `[click](http://evil)` / `![x](http://e/x.png)` now expect the URL (with its trailing `)`, as mandated by `\S+`) inside a code span, and the web evidence-appendix row now expects the web ref code-spanned. Both follow directly from "URLs … are rendered inside inline code spans"; no assertion was removed or weakened.
- `uv.lock` also records uv's re-resolution normalisation of unrelated platform markers (`cuda-*`, `nvidia-*`, `httpcore2`) and moves `markdown-it-py` 4.3.0 → 3.0.0; the downgrade is forced by the card's exact pin because `rich` (transitive) shares that package. Lock content stays consistent (`uv run` and the gate use it).

## NOT proven
- Linkify channels outside the card's URL grammar: `mailto:`, protocol-relative `//` and bare e-mail addresses are not code-spanned (card/verdict scope is `(?:https?|ftp)://\S+` and `www\.\S+`).
- A URL containing a literal `|` inside a table cell is still split by GFM's block-level table parser (pre-existing, integrity-only; it cannot produce tags/links).

## Open
- none
