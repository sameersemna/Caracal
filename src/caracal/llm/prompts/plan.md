## Task

You are planning the analysis of one audio recording. Read the metadata and the first minutes of
the transcript, then record a content profile. The pipeline has already decided which stages run;
your answer can never change those decisions.

## DATA

The metadata block uses citable field ids such as `[meta:title]` and `[meta:description]`.
The transcript block uses one citable line per segment, for example `[seg:12] S2 03:14.2 | text`.

{metadata}

{transcript}

## What to record

- `likely_genre`: a short label such as interview, lecture, sermon, news, podcast, music, meeting,
  panel, audiobook or unknown.
- `expected_speakers`: the likely number of distinct voices, or null when you cannot tell.
- `needs_identity`: true when naming the speakers matters (an interview, a panel, a sermon).
- `checkworthy_domain`: true when the recording makes factual claims worth checking (news, politics,
  science, health, finance, history). False for fiction, music and casual conversation.

Use only the DATA blocks. Do not invent names, dates or numbers, and do not guess a genre the
evidence does not support.

Write every free-text value in this report language: {report_language}.
