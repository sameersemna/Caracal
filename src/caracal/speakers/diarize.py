"""pyannote speaker diarization behind a lazy, optional dependency (docs/04 §5).

The pipeline is loaded once per process on first use, on ``DIARIZATION_DEVICE`` with a
logged fallback to CPU when CUDA is unavailable, and every inference runs in a worker
thread so the event loop stays responsive. ``pyannote``/``torch`` are imported only inside
functions: the gate and deployments without the ``speakers`` extra never import them.

Labels are pyannote's ``SPEAKER_00…`` and are unordered, so every result is renamed
``S1..Sn`` by first chronological appearance. The rename is applied consistently to the
full turns, the non-overlapping exclusive turns and the per-label embeddings.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.log import get_logger

logger = get_logger("caracal.speakers.diarize")


@dataclass(frozen=True, slots=True)
class Turn:
    """One speaker turn with a stable ``S<n>`` label and millisecond-rounded bounds."""

    start: float
    end: float
    label: str

    def to_dict(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "label": self.label}


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    """Everything a diarize checkpoint keeps; embeddings are L2-normalised 256-d vectors."""

    turns: list[Turn]
    exclusive_turns: list[Turn]
    embeddings: dict[str, list[float]]
    num_speakers: int

    def to_dict(self) -> dict[str, object]:
        return {
            "num_speakers": self.num_speakers,
            "turns": [turn.to_dict() for turn in self.turns],
            "exclusive_turns": [turn.to_dict() for turn in self.exclusive_turns],
            "embeddings": {label: list(vector) for label, vector in self.embeddings.items()},
        }


def _module_available(name: str) -> bool:
    """True when ``name`` can be imported, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _import_pipeline_class() -> Any:
    module: Any = importlib.import_module("pyannote.audio")
    return module.Pipeline


def _import_torchaudio_load() -> Any:
    module: Any = importlib.import_module("torchaudio")
    return module.load


def _round_time(value: float) -> float:
    return round(float(value), 3)


def _label_order(*turn_lists: Sequence[Turn]) -> list[str]:
    """Raw labels in order of first chronological appearance across the given turn lists."""
    ordered: list[str] = []
    for turns in turn_lists:
        for turn in sorted(turns, key=lambda item: (item.start, item.end)):
            if turn.label not in ordered:
                ordered.append(turn.label)
    return ordered


def _normalise(vector: Sequence[float]) -> list[float]:
    """L2-normalise one embedding; an all-zero vector is a broken model output."""
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise CaracalError(ErrorCode.INTERNAL, "diarizer returned an empty speaker embedding")
    return [value / norm for value in values]


def build_result(
    *,
    turns: Sequence[Turn],
    exclusive_turns: Sequence[Turn],
    embeddings: Mapping[str, Sequence[float]],
    num_speakers: int | None = None,
) -> DiarizationResult:
    """Rename raw pyannote labels by first appearance and normalise the embeddings.

    Kept separate from the model call so the rename/normalise contract is testable without
    torch (docs/07 §5: only the model boundary may be faked).
    """
    ordered = _label_order(turns, exclusive_turns)
    for label in embeddings:
        if label not in ordered:
            ordered.append(label)
    mapping = {label: f"S{position}" for position, label in enumerate(ordered, start=1)}

    renamed_turns = [
        Turn(_round_time(turn.start), _round_time(turn.end), mapping[turn.label])
        for turn in sorted(turns, key=lambda item: (item.start, item.end))
    ]
    renamed_exclusive = [
        Turn(_round_time(turn.start), _round_time(turn.end), mapping[turn.label])
        for turn in sorted(exclusive_turns, key=lambda item: (item.start, item.end))
    ]
    normalised = {
        mapping[label]: _normalise(vector) for label, vector in embeddings.items() if label in mapping
    }
    return DiarizationResult(
        turns=renamed_turns,
        exclusive_turns=renamed_exclusive,
        embeddings=normalised,
        num_speakers=num_speakers if num_speakers is not None else len(ordered),
    )


class Diarizer:
    """Process-wide pyannote pipeline; the model is loaded lazily on first :meth:`run`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pipeline: Any | None = None
        self._lock = asyncio.Lock()
        self._fallback_logged = False

    def available(self) -> bool:
        """True when pyannote and torch are importable and a Hugging Face token is set."""
        if self._settings.HF_TOKEN is None:
            return False
        return _module_available("pyannote.audio") and _module_available("torch")

    def _select_device(self, torch_module: Any) -> str:
        """``DIARIZATION_DEVICE`` when usable, otherwise CPU (the fallback is logged once)."""
        requested = self._settings.DIARIZATION_DEVICE.strip() or "cpu"
        if requested == "cpu":
            return "cpu"
        try:
            usable = bool(torch_module.cuda.is_available()) and int(torch_module.cuda.device_count()) > 0
        except Exception:
            usable = False
        if usable:
            return requested
        if not self._fallback_logged:
            logger.warning(
                "diarization device unavailable; falling back to cpu",
                extra={"requested_device": requested},
            )
            self._fallback_logged = True
        return "cpu"

    def _load_pipeline(self) -> Any:
        token = self._settings.HF_TOKEN
        if token is None:
            raise CaracalError(ErrorCode.DEPENDENCY_DOWN, "speaker diarization needs HF_TOKEN")
        try:
            torch_module = _import_torch()
            pipeline_class = _import_pipeline_class()
            device = self._select_device(torch_module)
            pipeline = pipeline_class.from_pretrained(
                self._settings.DIARIZATION_MODEL,
                token=token.get_secret_value(),
            )
            pipeline.to(torch_module.device(device))
        except CaracalError:
            raise
        except Exception as exc:
            raise CaracalError(
                ErrorCode.DEPENDENCY_DOWN,
                "failed to load the diarization model",
                {"model": self._settings.DIARIZATION_MODEL, "error": type(exc).__name__},
            ) from exc
        logger.info(
            "diarization model loaded",
            extra={"model": self._settings.DIARIZATION_MODEL, "device": device},
        )
        return pipeline

    async def run(self, wav: Path, *, num_speakers: int | None = None) -> DiarizationResult:
        """Diarize one 16 kHz mono WAV, loading the pipeline on first use."""
        if not self.available():
            raise CaracalError(ErrorCode.DEPENDENCY_DOWN, "speaker diarization is unavailable")
        async with self._lock:
            if self._pipeline is None:
                self._pipeline = await asyncio.to_thread(self._load_pipeline)
            pipeline = self._pipeline
            return await asyncio.to_thread(self._infer, pipeline, wav, num_speakers)

    def _infer(self, pipeline: Any, wav: Path, num_speakers: int | None) -> DiarizationResult:
        try:
            load_audio = _import_torchaudio_load()
            waveform, sample_rate = load_audio(str(wav))
            audio_input: dict[str, Any] = {
                "waveform": waveform,
                "sample_rate": int(sample_rate),
            }
            if num_speakers is not None:
                output = pipeline(audio_input, num_speakers=num_speakers)
            else:
                output = pipeline(audio_input)
        except CaracalError:
            raise
        except Exception as exc:
            raise CaracalError(
                ErrorCode.DEPENDENCY_DOWN,
                "speaker diarization failed",
                {"error": type(exc).__name__},
            ) from exc

        annotation = output.speaker_diarization
        exclusive = output.exclusive_speaker_diarization
        turns = [
            Turn(float(turn.start), float(turn.end), str(label))
            for turn, _track, label in annotation.itertracks(yield_label=True)
        ]
        exclusive_turns = [
            Turn(float(turn.start), float(turn.end), str(label))
            for turn, _track, label in exclusive.itertracks(yield_label=True)
        ]
        embeddings: dict[str, Sequence[float]] = {}
        matrix = getattr(output, "speaker_embeddings", None)
        if matrix is not None:
            labels = list(annotation.labels())
            for index, label in enumerate(labels):
                if index < len(matrix):
                    embeddings[str(label)] = [float(value) for value in matrix[index]]
        return build_result(
            turns=turns,
            exclusive_turns=exclusive_turns,
            embeddings=embeddings,
            num_speakers=num_speakers,
        )


__all__ = ["DiarizationResult", "Diarizer", "Turn", "build_result"]
