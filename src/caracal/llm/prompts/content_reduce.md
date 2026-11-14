## Task

Several analysis windows of one recording are given below as partial findings. Merge them into one
content analysis: write the overall summary and context, merge chapters in time order, merge
topics, de-duplicate entities by their surface text, and keep only claims and quotes that still
have their evidence ids.

Fill EVERY field; lists must not be empty when the transcript supports them.

## DATA and evidence ids

Metadata uses ids such as `[meta:title]`; transcript evidence uses `seg:12`; hints use `[hint:0]`.
The partial findings already contain evidence ids. Keep those ids; cite the id alone
(`seg:12`, `meta:title`).

{metadata}

{hints}

{partials}

## What to record

- `summary`: one line, a short paragraph, and a detailed paragraph covering the whole recording.
- `context`: genre, setting, audience, purpose, tone, when and where it was recorded, confidence.
- `chapters`: merge adjacent or overlapping partial chapters. {chapter_guidance}
  Give each merged chapter the id of its first segment in `first_seg` and of its last one in
  `last_seg`, plus the evidence ids. Do not give times.
- `topics`: merge duplicates and keep a weight between 0 and 1.
- `entities`: one entry per distinct entity, with every supporting segment id. Use `entity_type`.
- `key_quotes`: verbatim words in `text` with the segment ids that contain them in `seg_ids`.
- `claims`: at most {claim_limit} checkworthy factual assertions, de-duplicated.

Only state what the evidence supports. Do not invent names, dates, numbers or sources. Quotes and
entity text stay verbatim in the original language; all other prose is written in this report
language: {report_language}.
