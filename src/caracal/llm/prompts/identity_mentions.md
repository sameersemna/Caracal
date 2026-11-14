# Identity mentions

## Task

Read the speaker-tagged transcript and record the names that are actually spoken, plus one role and one
description for every speaker label. The names decide who is speaking; the role and description explain what
the speaker is doing in the recording.

Return two lists:

- `mentions`: every name that is spoken aloud, with the label it belongs to, the relation and the segment ids.
- `speakers`: one entry per speaker label, with its role and description.

The roles are: {roles}.

## DATA and evidence ids

The metadata block uses ids such as `[meta:title]`. The transcript block uses one line per segment, for example
`[seg:12] S2 03:14.2 | text`. Caller hints use `[hint:0]`, `[hint:1]`. The recording-type block uses ids such as
`[plan:likely_genre]`, `[content:genre]` and `[content:tone]`. Cite the id alone (`seg:12`, `meta:title`), never the
whole rendered line.

{metadata}

{hints}

{content_context}

{transcript}

## What to record

### mentions

Only record a name that you can see or hear in the transcript. Never infer a name from the topic, the channel
or your own knowledge. For every mention give `speaker_label`, `name`, `relation` and `seg_ids`.

`relation` must be exactly one of:

- `self_intro`: this speaker says their own name, for example "I'm Angela Merkel", "my name is ...",
  "ich bin ...", "je m'appelle ...", "me llamo ...". The cited segments must be spoken by this same label.
- `addressed`: another speaker says this name to or for this speaker, for example "thanks, Anna" or "welcome
  to the show, Anna" immediately before this speaker answers. The cited segments must be spoken by a
  *different* label and must be among the three segments directly before this speaker's next turn.
- `referred`: the name is merely talked about, for example a third person in a story. `referred` names are
  never used as an identity; record them only when the transcript makes the name explicit.
- `in_character`: the speaker is performing someone else when the name is said — an impression, a parody, a
  sketch, role-play, a prank call, a voice-over or an ad read, or the speaker is reading or quoting another
  person's words. For example an impressionist saying "It's me, John." as a famous coach, or an actor in a
  sketch introducing a character, is `in_character`, never `self_intro`. Use the recording type above as a
  signal: comedy, sketch, impression, parody, satire, skit and prank content is usually performed. `in_character`
  names are never used as an identity; record them so the extractor can set them aside.

A `self_intro` is a name the speaker gives for **themselves**, in their own voice, outside any performance. Cite
the segment or segments that actually contain the name. If you are unsure of the spelling, copy what the
transcript says.

### speakers

For every label in the transcript give:

- `role`: exactly one of the roles listed above. Use `other` when no role is clear. The role describes what
  the speaker does in the recording, not their personality.
- `description`: at most one short sentence about the speaker's role and the context, for example "hosts the
  recording" or "answers questions from the host". Write it in this report language: {report_language}.

A description must never infer or mention protected attributes. Never guess or state gender, age, ethnicity,
religion, health, disability or any other protected characteristic from the voice or the text. Do not use
words such as male, female, man or woman in a description. Describe only the role and what the speaker does
in the recording.

Only state what the evidence supports. Do not invent names, dates or sources.
