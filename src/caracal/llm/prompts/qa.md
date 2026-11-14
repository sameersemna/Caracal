# Answering a question about one recording

## Task

Answer the question in the DATA block `question` using only what the `report` and
`transcript` blocks support. Write the answer in the language of the question, even when the
transcript and the report are in another language.

- Every factual statement needs at least one supporting transcript citation in `citations`.
  Put the segment id in `seg_id` as `seg:<n>`. Cite only segment ids that appear in the
  transcript block.
- `start` and `text` are optional and are replaced by Caracal with the real segment values;
  never rely on your own version of them.
- Never cite `meta:` ids and never answer from the report summary alone. If no transcript
  segment supports the answer, set `answer` to exactly "Not found in the audio." and
  `confidence` to 0 with an empty `citations` list.
- Do not use outside knowledge, do not guess, and keep the answer to at most four sentences.

## DATA blocks

{report}

{question}

{transcript}
