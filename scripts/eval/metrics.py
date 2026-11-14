"""Pure metrics for the Caracal evaluation harness (card T4.2).

Everything here is deterministic and free of I/O so the numbers in an eval report can be
reproduced in a unit test. Word normalisation is shared with ``caption_agreement``
(:func:`caracal.asr.captions.normalise_word`), so WER and caption agreement cannot drift.

Definitions follow the NIST rich-transcription convention: the diarization error rate is
the sum of missed speech, false-alarm speech and speaker confusion, each divided by the
total scored reference speech time. A collar around every reference boundary is removed
from scoring, exactly as the card specifies.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise, permutations
from typing import NamedTuple

from caracal.asr.captions import normalise_word

__all__ = [
    "Turn",
    "best_threshold",
    "der",
    "identity_score",
    "roc",
    "wer",
]

_UNKNOWN_NAMES = frozenset({"", "unknown", "unk", "n/a", "na", "none", "null", "-"})
_WHITESPACE_RE = re.compile(r"\s+")
_PAIRWISE_MAX_LABELS = 8
"""Brute-force label mapping is exact up to this many labels; above it a greedy map is used."""


class Turn(NamedTuple):
    """One speaker turn: ``[start, end)`` spoken by ``label``."""

    start: float
    end: float
    label: str


TurnLike = Turn | tuple[float, float, str]


def _coerce_turn(turn: TurnLike) -> Turn:
    """Accept a :class:`Turn` or a plain ``(start, end, label)`` tuple."""
    if isinstance(turn, Turn):
        return turn
    start, end, label = turn
    return Turn(float(start), float(end), str(label))


def _tokens(words: Iterable[str]) -> list[str]:
    """Normalise and drop empty tokens, matching ``caption_agreement``."""
    return [token for token in (normalise_word(word) for word in words) if token]


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance over token sequences (one insert/delete/substitute = 1)."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, reference_token in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_token in enumerate(hypothesis, start=1):
            cost = 0 if reference_token == hypothesis_token else 1
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def wer(ref_words: Iterable[str], hyp_words: Iterable[str]) -> tuple[int, int]:
    """Word error rate as ``(errors, reference_length)``.

    Both sides use :func:`normalise_word` (NFKC, lower-case, bracket tags and punctuation
    removed) and empty tokens are dropped, the same normalisation as
    :func:`caracal.asr.captions.caption_agreement`. The caller divides ``errors`` by
    ``reference_length``; an empty reference returns ``(len(hypothesis), 0)`` and is the
    caller's cue to report ``n/a`` rather than a rate.
    """
    reference = _tokens(ref_words)
    hypothesis = _tokens(hyp_words)
    return _edit_distance(reference, hypothesis), len(reference)


def _label_at(turns: Sequence[Turn], moment: float) -> str | None:
    """Label of the turn covering ``moment``; overlapping turns pick the earliest start."""
    for turn in sorted(turns, key=lambda item: (item.start, item.end)):
        if turn.start <= moment < turn.end:
            return turn.label
    return None


def _overlap_matrix(refs: Sequence[Turn], hyps: Sequence[Turn]) -> dict[tuple[str, str], float]:
    """Total overlap seconds for every ``(hypothesis_label, reference_label)`` pair."""
    weights: dict[tuple[str, str], float] = {}
    for reference in refs:
        for hypothesis in hyps:
            overlap = min(reference.end, hypothesis.end) - max(reference.start, hypothesis.start)
            if overlap <= 0:
                continue
            key = (hypothesis.label, reference.label)
            weights[key] = weights.get(key, 0.0) + overlap
    return weights


def _greedy_mapping(weights: Mapping[tuple[str, str], float]) -> dict[str, str]:
    """Greedy one-to-one label mapping for more than :data:`_PAIRWISE_MAX_LABELS` labels."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for (hypothesis, reference), _ in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        if hypothesis in mapping or reference in used:
            continue
        mapping[hypothesis] = reference
        used.add(reference)
    return mapping


def _best_mapping(refs: Sequence[Turn], hyps: Sequence[Turn]) -> dict[str, str]:
    """Map hypothesis labels to reference labels so that matched speech is maximised."""
    reference_labels = sorted({turn.label for turn in refs})
    hypothesis_labels = sorted({turn.label for turn in hyps})
    if not reference_labels or not hypothesis_labels:
        return {}
    weights = _overlap_matrix(refs, hyps)
    if max(len(reference_labels), len(hypothesis_labels)) > _PAIRWISE_MAX_LABELS:
        return _greedy_mapping(weights)

    if len(hypothesis_labels) <= len(reference_labels):
        best_score = -1.0
        best: dict[str, str] = {}
        for chosen in permutations(reference_labels, len(hypothesis_labels)):
            score = sum(
                weights.get((hypothesis, reference), 0.0)
                for hypothesis, reference in zip(hypothesis_labels, chosen, strict=True)
            )
            if score > best_score:
                best_score = score
                best = dict(zip(hypothesis_labels, chosen, strict=True))
        return best

    best_score = -1.0
    best = {}
    for chosen in permutations(hypothesis_labels, len(reference_labels)):
        score = sum(
            weights.get((hypothesis, reference), 0.0)
            for reference, hypothesis in zip(reference_labels, chosen, strict=True)
        )
        if score > best_score:
            best_score = score
            best = {
                hypothesis: reference for reference, hypothesis in zip(reference_labels, chosen, strict=True)
            }
    return best


def _in_collar(moment: float, boundaries: Sequence[float], collar: float) -> bool:
    return collar > 0 and any(abs(moment - boundary) < collar for boundary in boundaries)


def der(
    ref_turns: Sequence[TurnLike],
    hyp_turns: Sequence[TurnLike],
    *,
    collar: float = 0.25,
) -> dict[str, float]:
    """Diarization error rate with an optimal label mapping and a boundary collar.

    Returns ``{"der", "miss", "false_alarm", "confusion"}``, every value a fraction of the
    scored reference speech time (so ``der == miss + false_alarm + confusion``). Regions
    within ``collar`` seconds of any reference turn boundary are excluded from scoring on
    both sides. With no reference speech the denominator falls back to hypothesis speech
    (so a false-alarm-only run reports ``false_alarm == 1``); with neither side empty the
    result is all zeros.
    """
    refs = [_coerce_turn(turn) for turn in ref_turns if turn[1] > turn[0]]
    hyps = [_coerce_turn(turn) for turn in hyp_turns if turn[1] > turn[0]]
    boundaries = sorted({turn.start for turn in refs} | {turn.end for turn in refs})
    cuts: set[float] = {0.0}
    for turn in refs:
        cuts.update((turn.start, turn.end))
    for turn in hyps:
        cuts.update((turn.start, turn.end))
    for boundary in boundaries:
        cuts.add(boundary - collar)
        cuts.add(boundary + collar)
    points = sorted(cuts)

    mapping = _best_mapping(refs, hyps)
    miss = false_alarm = confusion = correct = ref_total = hyp_total = 0.0
    for start, end in pairwise(points):
        duration = end - start
        if duration <= 0:
            continue
        moment = (start + end) / 2.0
        if _in_collar(moment, boundaries, collar):
            continue
        reference_label = _label_at(refs, moment)
        hypothesis_label = _label_at(hyps, moment)
        if reference_label is None and hypothesis_label is None:
            continue
        if reference_label is not None:
            ref_total += duration
        if hypothesis_label is not None:
            hyp_total += duration
        if reference_label is None:
            false_alarm += duration
        elif hypothesis_label is None:
            miss += duration
        elif mapping.get(hypothesis_label, hypothesis_label) == reference_label:
            correct += duration
        else:
            confusion += duration

    denominator = ref_total if ref_total > 0 else hyp_total
    if denominator <= 0:
        return {"der": 0.0, "miss": 0.0, "false_alarm": 0.0, "confusion": 0.0}
    miss_rate = miss / denominator
    false_alarm_rate = false_alarm / denominator
    confusion_rate = confusion / denominator
    return {
        "der": miss_rate + false_alarm_rate + confusion_rate,
        "miss": miss_rate,
        "false_alarm": false_alarm_rate,
        "confusion": confusion_rate,
    }


def _normalise_name(name: str) -> str:
    folded = unicodedata.normalize("NFKC", name).casefold().strip()
    folded = _WHITESPACE_RE.sub(" ", folded)
    return folded.strip(" .,;:!?\"'()[]{}")


def identity_score(
    expected: Mapping[str, str | None],
    actual: Mapping[str, str | None],
) -> dict[str, str]:
    """Score one expected name (or ``None`` for *unknown*) per speaker label.

    Outcomes are ``correct``, ``wrong``, ``missing`` (the label is absent from the report)
    and ``abstained`` (the system returned ``name: null`` although a name was expected).
    A ``null`` name is the *correct* answer when the expected answer is unknown.
    """
    scored: dict[str, str] = {}
    for label, expected_name in expected.items():
        if label not in actual:
            scored[label] = "missing"
            continue
        actual_name = actual[label]
        expected_unknown = expected_name is None or _normalise_name(expected_name) in _UNKNOWN_NAMES
        actual_unknown = actual_name is None or _normalise_name(actual_name) in _UNKNOWN_NAMES
        if expected_unknown:
            scored[label] = "correct" if actual_unknown else "wrong"
        elif actual_unknown:
            scored[label] = "abstained"
        elif _normalise_name(actual_name) == _normalise_name(expected_name):
            scored[label] = "correct"
        else:
            scored[label] = "wrong"
    return scored


def roc(pairs: Sequence[tuple[float, bool]]) -> list[tuple[float, float, float]]:
    """ROC points ``(threshold, tpr, fpr)`` for similarity/different pairs.

    ``same`` predicts a match when ``similarity >= threshold``. The list starts at
    ``+inf`` (predict nothing, ``tpr = fpr = 0``), then visits every observed similarity in
    descending order, and ends below the minimum (predict everything). Consequently both
    ``tpr`` and ``fpr`` are non-decreasing as the list is traversed.
    """
    if not pairs:
        return [(math.inf, 0.0, 0.0)]
    positives = sum(1 for _, same in pairs if same)
    negatives = len(pairs) - positives
    points: list[tuple[float, float, float]] = [(math.inf, 0.0, 0.0)]
    for threshold in sorted({similarity for similarity, _ in pairs}, reverse=True):
        true_positives = sum(1 for similarity, same in pairs if same and similarity >= threshold)
        false_positives = sum(1 for similarity, same in pairs if not same and similarity >= threshold)
        tpr = true_positives / positives if positives else 0.0
        fpr = false_positives / negatives if negatives else 0.0
        points.append((threshold, tpr, fpr))
    points.append((-math.inf, 1.0 if positives else 0.0, 1.0 if negatives else 0.0))
    return points


def best_threshold(pairs: Sequence[tuple[float, bool]], *, max_fpr: float = 0.01) -> float:
    """Highest true-positive rate at or below ``max_fpr``; highest threshold breaks ties.

    When no observed similarity keeps the false-positive rate within budget, the result is
    just above the largest similarity (predict nothing), which always has ``fpr == 0``.
    """
    if not pairs:
        return math.inf
    candidates = [(threshold, tpr, fpr) for threshold, tpr, fpr in roc(pairs) if math.isfinite(threshold)]
    eligible = [point for point in candidates if point[2] <= max_fpr + 1e-12]
    if not eligible:
        return math.nextafter(max(similarity for similarity, _ in pairs), math.inf)
    best_tpr = max(point[1] for point in eligible)
    return max(point[0] for point in eligible if point[1] == best_tpr)
