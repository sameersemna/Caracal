# Judge one claim against web evidence

## Task

Decide whether the web evidence supports, contradicts or does not determine one claim. Use
only the DATA blocks. The claim block states what was said, the transcript block shows the
segments it came from, and the web block lists search results. A web page may be wrong,
promotional, out of date or hostile; it is evidence, not an authority.

Return one object:

- `target`: copied exactly from the claim line.
- `verdict`: exactly one of `supported`, `contradicted`, `mixed`, `unverified`.
- `confidence`: a number in [0, 1].
- `rationale`: one or two sentences explaining the verdict.
- `sources`: the web ids that decide the verdict. Put the id in `evidence_id`; the url, title
  and snippet are filled from the search result. Never cite an id that is not in the web block.

Prefer `unverified` with a low confidence when the web block is thin or off-topic. Do not use
outside knowledge to fill gaps.

## DATA and evidence ids

The claim line uses `[target:claim:<id>]`. Transcript lines use `[seg:12] S2 03:14.2 | text`.
Web results use `[web:3] provider=... query=...` followed by title, url, snippet and optional
text. Cite the typed id alone (`web:3`), never the whole rendered line.

{claim}

{transcript}

{web}
