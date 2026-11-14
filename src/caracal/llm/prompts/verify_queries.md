# Verification queries

## Task

Write one or two short web-search queries for every target below. A later stage runs them and
judges the results, so the query must be able to find a page that settles the target.

- `claim`: search for the factual statement itself, including its distinctive numbers, names or
  dates. A query that can surface both confirming and contradicting pages is best.
- `identity_confirm`: combine the candidate name with the recording's channel, show or title so
  the search finds this person in this context. For example `All Ears English co-host Lindsay`.
- `quote_attribution`: put the sentence in double quotes so the search finds pages that
  attribute that exact sentence to someone.

Return a `queries` list with one entry per target. Copy `target` exactly from its target line.
Write each query in the language most likely to find the source (usually English).

## DATA

The target block uses ids such as `[target:claim:c1]` and `[target:speaker:S1]`. The metadata
block uses citable ids such as `[meta:title]` and `[meta:channel]`.

{context}

{targets}
