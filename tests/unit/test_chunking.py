"""Chunk planning: table cases and invariants over randomised silence sets (docs/04 §3)."""

from __future__ import annotations

import itertools
import random

import pytest

from caracal.audio.chunking import Chunk, plan_chunks

Overlap = tuple[float, float]


def _expected(*pairs: Overlap) -> list[Chunk]:
    return [Chunk(index=index, start_s=start, end_s=end) for index, (start, end) in enumerate(pairs)]


@pytest.mark.parametrize(
    ("duration_s", "silences", "expected"),
    [
        (300.0, [], [(0.0, 300.0)]),
        (900.0, [], [(0.0, 900.0)]),
        (901.0, [], [(0.0, 600.0), (599.0, 901.0)]),
        (2000.0, [], [(0.0, 600.0), (599.0, 1200.0), (1199.0, 2000.0)]),
        (1200.0, [(595.0, 605.0)], [(0.0, 600.0), (599.0, 1200.0)]),
        (
            1800.0,
            [(590.0, 600.0), (1190.0, 1200.0)],
            [(0.0, 595.0), (594.0, 1195.0), (1194.0, 1800.0)],
        ),
    ],
)
def test_plan_chunks_table(duration_s: float, silences: list[Overlap], expected: list[Overlap]) -> None:
    assert plan_chunks(duration_s, silences) == _expected(*expected)


def test_silence_nearest_the_target_wins() -> None:
    silences = [(400.0, 410.0), (585.0, 595.0), (850.0, 860.0)]

    chunks = plan_chunks(1400.0, silences)

    assert chunks[0].end_s == 590.0
    assert chunks[1].start_s == 589.0


def test_far_silence_is_ignored_and_target_is_hard_cut() -> None:
    silences = [(1000.0, 1010.0)]

    chunks = plan_chunks(1400.0, silences)

    assert chunks[0].end_s == 600.0
    assert chunks[1].start_s == 599.0


def test_tail_shorter_than_thirty_seconds_is_merged_into_one_chunk() -> None:
    chunks = plan_chunks(920.0, [(890.0, 900.0)])

    assert chunks == _expected((0.0, 920.0))


def test_tail_merge_keeps_a_previous_chunk() -> None:
    chunks = plan_chunks(1505.0, [(595.0, 605.0), (1480.0, 1490.0)])

    assert chunks == _expected((0.0, 600.0), (599.0, 1505.0))


def test_custom_target_and_overlap() -> None:
    chunks = plan_chunks(1000.0, [], target_s=300.0, max_s=400.0, overlap_s=0.5)

    assert chunks == _expected((0.0, 300.0), (299.5, 600.0), (599.5, 1000.0))


def test_twenty_five_minute_example_cuts_near_each_target() -> None:
    duration = 1500.0
    silences = [(36.5 + 37.0 * index, 37.5 + 37.0 * index) for index in range(40)]

    chunks = plan_chunks(duration, silences)

    assert len(chunks) == 3
    for index, target in enumerate((600.0, 1200.0), start=1):
        boundary = chunks[index].start_s + 1.0
        assert abs(boundary - target) <= 120.0


def test_single_chunk_boundary_is_inclusive() -> None:
    assert plan_chunks(900.0, [], max_s=900.0) == _expected((0.0, 900.0))


def _random_silences(rng: random.Random, duration: float) -> list[Overlap]:
    silences: list[Overlap] = []
    cursor = rng.uniform(0.0, 20.0)
    while cursor < duration:
        length = rng.uniform(0.2, 5.0)
        silences.append((cursor, cursor + length))
        cursor += length + rng.uniform(1.0, 120.0)
    return silences


def test_random_silence_sets_satisfy_chunk_invariants() -> None:
    rng = random.Random(20261114)  # noqa: S311 - deterministic test data, not crypto
    target_s = 600.0
    max_s = 900.0
    overlap_s = 1.0
    for _ in range(300):
        duration = rng.uniform(1.0, 5000.0)
        chunks = plan_chunks(duration, _random_silences(rng, duration), target_s=target_s, max_s=max_s)

        assert chunks
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
        assert chunks[0].start_s == 0.0
        assert chunks[-1].end_s == pytest.approx(duration)
        assert all(chunk.start_s < chunk.end_s for chunk in chunks)
        assert all(chunk.end_s - chunk.start_s <= max_s + overlap_s + 30.0 for chunk in chunks)
        for previous, following in itertools.pairwise(chunks):
            assert following.start_s > previous.start_s
            assert previous.end_s >= following.start_s
            assert previous.end_s - following.start_s == pytest.approx(overlap_s)
