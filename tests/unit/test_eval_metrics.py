"""Pure eval metrics: WER, DER with label mapping and collar, identity scoring, ROC (T4.2)."""

from __future__ import annotations

import math

import pytest
from scripts.eval.metrics import Turn, best_threshold, der, identity_score, roc, wer

# -- WER ---------------------------------------------------------------------


def test_wer_identical_is_zero() -> None:
    assert wer(["the", "cat", "sat"], ["the", "cat", "sat"]) == (0, 3)


def test_wer_one_substitution() -> None:
    assert wer(["a", "b", "c"], ["a", "x", "c"]) == (1, 3)


def test_wer_insertion_counts_against_reference_length() -> None:
    assert wer(["a", "b"], ["a", "x", "b"]) == (1, 2)


def test_wer_deletion_counts_against_reference_length() -> None:
    assert wer(["a", "x", "b"], ["a", "b"]) == (1, 3)


def test_wer_uses_caption_agreement_normalisation() -> None:
    assert wer(["Hello,", "WORLD!"], ["hello", "world"]) == (0, 2)
    assert wer(["[Music]", "Hello", "—"], ["hello"]) == (0, 1)


def test_wer_empty_reference_reports_hypothesis_length() -> None:
    assert wer([], ["a", "b"]) == (2, 0)


# -- DER ---------------------------------------------------------------------


REF = [Turn(0.0, 10.0, "S1"), Turn(10.0, 20.0, "S2")]


def test_der_perfect_is_zero() -> None:
    result = der(REF, REF)
    assert result == {"der": 0.0, "miss": 0.0, "false_alarm": 0.0, "confusion": 0.0}


def test_der_swapped_labels_are_remapped_to_zero() -> None:
    swapped = [Turn(0.0, 10.0, "S2"), Turn(10.0, 20.0, "S1")]
    result = der(REF, swapped)
    assert result["der"] == pytest.approx(0.0)
    assert result["confusion"] == pytest.approx(0.0)


def test_der_miss() -> None:
    result = der(REF, [Turn(0.0, 10.0, "S1")])
    assert result["miss"] == pytest.approx(0.5)
    assert result["false_alarm"] == 0.0
    assert result["confusion"] == 0.0
    assert result["der"] == pytest.approx(0.5)


def test_der_false_alarm() -> None:
    hyp = [Turn(0.0, 10.0, "S1"), Turn(10.0, 20.0, "S2"), Turn(20.0, 25.0, "S1")]
    result = der(REF, hyp)
    assert result["false_alarm"] == pytest.approx(0.25)
    assert result["miss"] == 0.0
    assert result["der"] == pytest.approx(0.25)


def test_der_confusion_when_one_reference_speaker_is_split() -> None:
    result = der([Turn(0.0, 10.0, "S1")], [Turn(0.0, 5.0, "S1"), Turn(5.0, 10.0, "S2")])
    assert result["confusion"] == pytest.approx(0.5)
    assert result["der"] == pytest.approx(0.5)


def test_der_collar_ignores_small_boundary_offsets() -> None:
    hyp = [Turn(0.0, 9.8, "S1"), Turn(9.8, 20.0, "S2")]
    assert der(REF, hyp, collar=0.25)["der"] == pytest.approx(0.0)
    assert der(REF, hyp, collar=0.0)["der"] > 0.0


def test_der_accepts_plain_tuples_and_sums_to_the_error_rate() -> None:
    result = der([(0.0, 10.0, "S1")], [(0.0, 4.0, "S1")], collar=0.0)
    assert result["der"] == pytest.approx(result["miss"] + result["false_alarm"] + result["confusion"])
    assert result["miss"] == pytest.approx(0.6)


def test_der_false_alarm_only_uses_hypothesis_time_as_denominator() -> None:
    result = der([], [Turn(0.0, 4.0, "S1")])
    assert result["false_alarm"] == pytest.approx(1.0)
    assert result["der"] == pytest.approx(1.0)
    assert der([], []) == {"der": 0.0, "miss": 0.0, "false_alarm": 0.0, "confusion": 0.0}


# -- identity ----------------------------------------------------------------


def test_identity_scores_every_outcome_including_abstention() -> None:
    expected = {
        "S1": "Daniel Levitin",
        "S2": "Lindsay McMahon",
        "S3": "unknown",
        "S4": None,
        "S5": "Nobody Here",
    }
    actual = {"S1": "daniel levitin", "S2": None, "S3": None, "S4": None}
    assert identity_score(expected, actual) == {
        "S1": "correct",
        "S2": "abstained",
        "S3": "correct",
        "S4": "correct",
        "S5": "missing",
    }


def test_identity_wrong_and_false_positive_against_unknown() -> None:
    assert identity_score({"S1": "Daniel Levitin"}, {"S1": "John F. Kennedy"}) == {"S1": "wrong"}
    assert identity_score({"S1": None}, {"S1": "John F. Kennedy"}) == {"S1": "wrong"}


def test_identity_ignores_case_and_surrounding_punctuation() -> None:
    assert identity_score({"S1": "Lindsay McMahon."}, {"S1": "  lindsay   mcmahon "}) == {"S1": "correct"}


# -- ROC ---------------------------------------------------------------------

PAIRS = [(0.95, True), (0.90, True), (0.30, False), (0.20, False)]


def test_roc_is_monotonic_and_spans_infinities() -> None:
    points = roc(PAIRS)
    assert points[0] == (math.inf, 0.0, 0.0)
    assert points[-1] == (-math.inf, 1.0, 1.0)
    tprs = [tpr for _, tpr, _ in points]
    fprs = [fpr for _, _, fpr in points]
    assert tprs == sorted(tprs)
    assert fprs == sorted(fprs)
    assert any(threshold == 0.90 and tpr == 1.0 and fpr == 0.0 for threshold, tpr, fpr in points)


def test_best_threshold_maximises_tpr_under_the_fpr_budget() -> None:
    threshold = best_threshold(PAIRS, max_fpr=0.01)
    assert threshold == pytest.approx(0.90)
    false_positives = sum(1 for similarity, same in PAIRS if not same and similarity >= threshold)
    assert false_positives == 0
    assert threshold <= min(similarity for similarity, same in PAIRS if same)


def test_best_threshold_prefers_a_higher_tpr_within_budget() -> None:
    pairs = [(0.8, True), (0.6, True), (0.7, False)]
    assert best_threshold(pairs, max_fpr=1.0) == pytest.approx(0.6)
    threshold = best_threshold(pairs, max_fpr=0.5)
    assert threshold == pytest.approx(0.8)
    assert sum(1 for similarity, same in pairs if not same and similarity >= threshold) == 0


def test_best_threshold_abstains_when_no_threshold_fits() -> None:
    pairs = [(0.9, False), (0.8, False)]
    threshold = best_threshold(pairs, max_fpr=0.0)
    assert threshold > max(similarity for similarity, _ in pairs)
    assert best_threshold([], max_fpr=0.01) == math.inf
    assert roc([]) == [(math.inf, 0.0, 0.0)]
