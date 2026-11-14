"""Diarizer internals that must not need torch: label renaming, normalisation, device choice.

The pyannote call itself is exercised by ``tests/live/test_diarize_live.py``; here we test
the pure result-building contract and the availability/device logic with fakes, so the gate
stays green without the optional ``speakers`` extra (docs/07 §1).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.speakers.diarize import Diarizer, Turn, build_result

MakeSettings = Callable[..., Settings]


def _cuda(available: bool, count: int) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
        )
    )


def test_labels_are_renamed_by_first_appearance_across_turns_exclusive_and_embeddings() -> None:
    result = build_result(
        turns=[
            Turn(2.0, 3.0, "SPEAKER_01"),
            Turn(0.0, 1.0, "SPEAKER_00"),
            Turn(3.2, 4.0, "SPEAKER_01"),
        ],
        exclusive_turns=[
            Turn(0.0, 1.0, "SPEAKER_00"),
            Turn(2.0, 3.0, "SPEAKER_01"),
        ],
        embeddings={"SPEAKER_00": [0.0, 1.0], "SPEAKER_01": [3.0, 4.0]},
    )

    assert [(turn.start, turn.label) for turn in result.turns] == [
        (0.0, "S1"),
        (2.0, "S2"),
        (3.2, "S2"),
    ]
    assert [(turn.start, turn.label) for turn in result.exclusive_turns] == [(0.0, "S1"), (2.0, "S2")]
    assert result.num_speakers == 2
    assert set(result.embeddings) == {"S1", "S2"}
    assert [round(value, 6) for value in result.embeddings["S2"]] == [0.6, 0.8]


def test_turn_times_are_rounded_to_milliseconds() -> None:
    result = build_result(
        turns=[Turn(0.123456, 1.987654, "A")],
        exclusive_turns=[Turn(0.123456, 1.987654, "A")],
        embeddings={},
    )

    assert result.turns[0].start == 0.123
    assert result.turns[0].end == 1.988


def test_every_embedding_is_l2_normalised() -> None:
    result = build_result(
        turns=[Turn(0.0, 1.0, "A"), Turn(1.0, 2.0, "B")],
        exclusive_turns=[],
        embeddings={"A": [3.0, 4.0], "B": [0.0, -2.0, 0.0]},
    )

    for vector in result.embeddings.values():
        assert sum(value * value for value in vector) == pytest.approx(1.0)
    assert result.embeddings["S1"] == pytest.approx([0.6, 0.8])
    assert result.embeddings["S2"] == pytest.approx([0.0, -1.0, 0.0])


def test_zero_embedding_is_reported_as_an_internal_failure() -> None:
    with pytest.raises(CaracalError) as caught:
        build_result(
            turns=[Turn(0.0, 1.0, "A")],
            exclusive_turns=[],
            embeddings={"A": [0.0, 0.0]},
        )

    assert caught.value.code is ErrorCode.INTERNAL


def test_cpu_fallback_when_cuda_is_unavailable(make_settings: MakeSettings) -> None:
    diarizer = Diarizer(make_settings(DIARIZATION_DEVICE="cuda"))

    assert diarizer._select_device(_cuda(available=False, count=0)) == "cpu"
    assert diarizer._select_device(_cuda(available=True, count=0)) == "cpu"
    assert diarizer._select_device(_cuda(available=True, count=1)) == "cuda"


def test_explicit_cpu_device_is_not_probed(make_settings: MakeSettings) -> None:
    diarizer = Diarizer(make_settings(DIARIZATION_DEVICE="cpu"))

    assert diarizer._select_device(_cuda(available=True, count=1)) == "cpu"


def test_available_requires_both_modules_and_a_hf_token(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
    token = "hf_" + "test"

    assert Diarizer(make_settings()).available() is False
    assert Diarizer(make_settings(HF_TOKEN=token)).available() is True


def test_available_is_false_when_the_modules_are_missing(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    token = "hf_" + "test"

    assert Diarizer(make_settings(HF_TOKEN=token)).available() is False


class _Track:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _Annotation:
    def __init__(self, rows: list[tuple[_Track, str]]) -> None:
        self._rows = rows

    def itertracks(self, *, yield_label: bool = False) -> list[tuple[_Track, object, str]]:
        assert yield_label is True
        return [(track, object(), label) for track, label in self._rows]

    def labels(self) -> list[str]:
        return [label for _track, label in self._rows]


class _Output:
    def __init__(self, rows: list[tuple[_Track, str]], embeddings: list[list[float]]) -> None:
        self.speaker_diarization = _Annotation(rows)
        self.exclusive_speaker_diarization = _Annotation(rows)
        self.speaker_embeddings = embeddings


class _Pipeline:
    def __init__(self, output: _Output) -> None:
        self.output = output
        self.calls: list[tuple[dict[str, object], dict[str, int]]] = []

    def __call__(self, audio: dict[str, object], **kwargs: int) -> _Output:
        self.calls.append((audio, kwargs))
        return self.output


def _fake_audio_loader() -> Callable[[str], tuple[object, int]]:
    return lambda _path: (object(), 16000)


def test_infer_passes_num_speakers_only_when_given_and_renames_the_output(
    monkeypatch: pytest.MonkeyPatch, make_settings: MakeSettings
) -> None:
    output = _Output(
        rows=[(_Track(2.0, 3.0), "SPEAKER_01"), (_Track(0.0, 1.0), "SPEAKER_00")],
        embeddings=[[0.0, 2.0], [3.0, 4.0]],
    )
    pipeline = _Pipeline(output)
    monkeypatch.setattr(
        "caracal.speakers.diarize._import_torchaudio_load",
        _fake_audio_loader,
    )
    diarizer = Diarizer(make_settings(DIARIZATION_DEVICE="cpu"))

    first = diarizer._infer(pipeline, Path("audio.wav"), None)
    diarizer._infer(pipeline, Path("audio.wav"), 2)

    assert [turn.label for turn in first.turns] == ["S1", "S2"]
    assert first.turns[0].start == 0.0
    assert first.embeddings["S2"] == pytest.approx([0.0, 1.0])
    assert first.embeddings["S1"] == pytest.approx([0.6, 0.8])
    assert pipeline.calls[0][1] == {}
    assert pipeline.calls[1][1] == {"num_speakers": 2}
    assert pipeline.calls[0][0]["sample_rate"] == 16000
