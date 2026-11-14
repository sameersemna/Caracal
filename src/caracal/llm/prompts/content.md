## Task

Analyse the recording in the DATA blocks and record its summary, context and content findings.
Every item must cite the evidence that supports it.

Fill EVERY field; lists must not be empty when the transcript supports them.

## DATA and evidence ids

The metadata block uses ids such as `[meta:title]`. The transcript block uses one line per
segment, for example `[seg:12] S2 03:14.2 | text`. Caller hints use `[hint:0]`, `[hint:1]`.
Cite the id alone (`seg:12`, `meta:title`), never the whole rendered line.

{metadata}

{hints}

{transcript}

## What to record

- `summary`: one line, a short paragraph, and a detailed paragraph.
- `context`: genre, setting, audience, purpose, tone, when and where it was recorded, confidence.
- `chapters`: contiguous ranges with a title and a one-sentence summary. {chapter_guidance}
  Give the id of the first segment of the range in `first_seg` and of the last one in `last_seg`,
  plus the evidence ids. Do not give times: they are derived from those segments.
- `topics`: short labels with a weight between 0 and 1.
- `entities`: people, organisations, places, dates, events and works. Use `entity_type`, keep the
  surface text exactly as spoken, and list the segment ids that mention it.
- `key_quotes`: verbatim words in `text` with the segment ids that contain them in `seg_ids`
  (at least one). A quote must be copied exactly from those segments; do not give times.
- `claims`: checkworthy factual assertions, at most {claim_limit}. Set `checkworthy` honestly.

Only state what the evidence supports. Do not invent names, dates, numbers or sources. Quotes and
entity text stay verbatim in the original language; all other prose is written in this report
language: {report_language}.
