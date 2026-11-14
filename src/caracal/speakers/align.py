"""Word-to-speaker alignment, talk-time statistics and intro re-timing (docs/04 §5).

``assign_speakers`` gives every word the speaker of the exclusive (non-overlapping) turn
that contains its midpoint, falling back to the nearest turn within one second. Segments
are rebuilt split at speaker changes so a subtitle cue can never mix two speakers.

``retime_outside_speech`` repairs whisper's approximate word timings around long
non-speech intros (verdict 15): words that sit before the first turn, or in a gap longer
than two seconds, are moved into the following turn and spread evenly up to the first word
already inside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from caracal.asr import Segment, Transcript, Word, join_words
from caracal.speakers.diarize import Turn

_NEAREST_TURN_TOLERANCE_S = 1.0
_MIN_WORD_DURATION_S = 0.05
_GAP_S = 2.0


def _midpoint(word: Word) -> float:
    return (word.start + word.end) / 2.0


def _distance_to_turn(midpoint: float, turn: Turn) -> float:
    if midpoint < turn.start:
        return turn.start - midpoint
    if midpoint > turn.end:
        return midpoint - turn.end
    return 0.0


def _speaker_for(midpoint: float, turns: Sequence[Turn]) -> str | None:
    for turn in turns:
        if turn.start <= midpoint <= turn.end:
            return turn.label
    nearest: Turn | None = None
    nearest_distance = float("inf")
    for turn in turns:
        distance = _distance_to_turn(midpoint, turn)
        if distance < nearest_distance:
            nearest = turn
            nearest_distance = distance
    if nearest is not None and nearest_distance <= _NEAREST_TURN_TOLERANCE_S:
        return nearest.label
    return None


def assign_speakers(transcript: Transcript, exclusive_turns: Sequence[Turn]) -> Transcript:
    """Set ``word.speaker`` from the exclusive turns and split segments at speaker changes."""
    turns = sorted(exclusive_turns, key=lambda turn: (turn.start, turn.end))
    segments: list[Segment] = []
    next_id = 0
    for segment in transcript.segments:
        if not segment.words:
            segments.append(replace(segment, id=next_id, speaker=None))
            next_id += 1
            continue

        groups: list[list[Word]] = []
        for word in segment.words:
            label = _speaker_for(_midpoint(word), turns)
            labelled = replace(word, speaker=label)
            if groups and groups[-1][0].speaker == label:
                groups[-1].append(labelled)
            else:
                groups.append([labelled])

        for group in groups:
            label = group[0].speaker
            speaker = label if all(word.speaker == label for word in group) else None
            segments.append(
                Segment(
                    id=next_id,
                    start=group[0].start,
                    end=group[-1].end,
                    text=join_words(group),
                    words=tuple(group),
                    avg_logprob=segment.avg_logprob,
                    no_speech_prob=segment.no_speech_prob,
                    lang=segment.lang,
                    speaker=speaker,
                    lang_prob=segment.lang_prob,
                )
            )
            next_id += 1

    return Transcript(
        segments=tuple(segments),
        language_code=transcript.language_code,
        dropped=dict(transcript.dropped),
    )


def speaker_stats(transcript: Transcript) -> dict[str, dict[str, float | int]]:
    """Per-speaker talk time, share of labelled speech, first appearance and word count.

    ``share`` is the fraction of *labelled* speech, so the shares sum to 1 whenever at
    least one word was assigned a speaker.
    """
    talk_time: dict[str, float] = {}
    words: dict[str, int] = {}
    first_seen: dict[str, float] = {}
    for segment in transcript.segments:
        for word in segment.words:
            label = word.speaker
            if label is None:
                continue
            talk_time[label] = talk_time.get(label, 0.0) + max(0.0, word.end - word.start)
            words[label] = words.get(label, 0) + 1
            first_seen[label] = min(first_seen.get(label, word.start), word.start)

    total = sum(talk_time.values())
    return {
        label: {
            "talk_time_s": round(talk_time[label], 3),
            "share": round(talk_time[label] / total, 6) if total > 0.0 else 0.0,
            "first_seen_s": round(first_seen[label], 3),
            "words": words[label],
        }
        for label in sorted(talk_time)
    }


def _following_turn_index(midpoint: float, turns: Sequence[Turn]) -> int | None:
    """The turn a non-speech word should be moved into, or ``None`` when it is in speech."""
    if midpoint < turns[0].start:
        return 0
    for index in range(len(turns) - 1):
        current = turns[index]
        following = turns[index + 1]
        if following.start - current.end > _GAP_S and current.end <= midpoint < following.start:
            return index + 1
    return None


def _first_word_inside(turns: Sequence[Turn], words: Sequence[Word]) -> dict[int, float]:
    """For every turn, the start of the first word whose midpoint lies inside it."""
    anchors: dict[int, float] = {}
    for index, turn in enumerate(turns):
        for word in words:
            midpoint = _midpoint(word)
            if turn.start <= midpoint <= turn.end:
                anchors[index] = word.start
                break
        if index not in anchors:
            anchors[index] = turn.end
    return anchors


def retime_outside_speech(
    transcript: Transcript,
    turns: Sequence[Turn],
    *,
    max_shift_s: float = 30.0,
) -> Transcript:
    """Move words outside speech into the following turn (verdict 15, docs/04 §5).

    Words before the first turn or in a gap longer than two seconds are spread evenly from
    the following turn's start up to the first word already inside it, preserving order and
    giving every moved word at least a 0.05 s duration. A word is never shifted by more
    than ``max_shift_s``. Words already inside a turn are left untouched.
    """
    ordered = sorted(turns, key=lambda turn: (turn.start, turn.end))
    if not ordered:
        return transcript

    flat = [word for segment in transcript.segments for word in segment.words]
    anchors = _first_word_inside(ordered, flat)

    placements: dict[int, tuple[float, float]] = {}
    pending: dict[int, list[tuple[int, Word]]] = {}
    for position, word in enumerate(flat):
        index = _following_turn_index(_midpoint(word), ordered)
        if index is not None:
            pending.setdefault(index, []).append((position, word))

    for index, entries in pending.items():
        turn = ordered[index]
        anchor = anchors[index]
        span = max(0.0, anchor - turn.start)
        step = span / len(entries) if entries else 0.0
        for offset, (position, word) in enumerate(entries):
            new_start = turn.start + step * offset
            if abs(new_start - word.start) > max_shift_s:
                continue
            duration = max(word.end - word.start, _MIN_WORD_DURATION_S)
            placements[position] = (new_start, new_start + duration)

    if not placements:
        return transcript

    segments: list[Segment] = []
    position = 0
    for segment in transcript.segments:
        words: list[Word] = []
        changed = False
        for word in segment.words:
            placement = placements.get(position)
            position += 1
            if placement is None:
                words.append(word)
                continue
            changed = True
            words.append(replace(word, start=placement[0], end=placement[1]))
        if changed and words:
            segments.append(
                replace(
                    segment,
                    words=tuple(words),
                    start=min(word.start for word in words),
                    end=max(word.end for word in words),
                )
            )
        else:
            segments.append(segment)

    return Transcript(
        segments=tuple(segments),
        language_code=transcript.language_code,
        dropped=dict(transcript.dropped),
    )


__all__ = ["assign_speakers", "retime_outside_speech", "speaker_stats"]
