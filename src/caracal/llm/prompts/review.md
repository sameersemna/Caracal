## Task

You are reviewing a finished audio-intelligence report. Find **inconsistencies only**: places
where the report contradicts itself or the transcript. You cannot edit the report; you may only
report flags.

The report is in the `report` DATA block. It contains the summary, context, chapters, speakers,
entities, claims, verification verdicts, languages and quality metrics, but deliberately no
evidence ids and no transcript text: the evidence rules above do **not** apply to this task.
The `transcript` DATA block holds an evenly spaced sample of transcript lines, including the
first and the last segment.

{report}

{transcript}

## What counts as an inconsistency

- The summary states something a verification marks `contradicted`, or a claim's text and its
  verification verdict disagree.
- A chapter title or summary is not supported by what is said inside its time range.
- A speaker name, role or description conflicts with how the transcript attributes that speaker,
  with the report's own entity list, or with the source metadata.
- A language in `languages` does not match the transcript sample.
- The context (genre, setting, audience, purpose) contradicts the transcript or the chapters.

Do **not** report missing information, style, confidence values that are merely low, or anything
you cannot tie to the report or the sample. Do not repeat the same finding twice.

## Output

Return one JSON object with a single field `flags`, a list of at most {max_flags} objects:

- `severity`: one of `info`, `warn`, `error`.
- `field`: the report section the finding is about. Use a top-level report key (`summary`,
  `context`, `chapters`, `speakers`, `entities`, `claims`, `verification`, `languages`,
  `quality`) or a dotted path into one (`speakers.S1.name`, `claims.claim_3`). Any other
  value is discarded.
- `message`: one sentence, at most {max_message_chars} characters, naming what is inconsistent
  and what contradicts it. **Never put evidence ids such as `seg:12`, `meta:title` or `web:3`
  in a message** — they are stripped and the flag may be dropped.

If the report is consistent, return `{"flags": []}`.
