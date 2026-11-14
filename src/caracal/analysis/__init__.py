"""Autonomy and analysis stages (docs/01 §3, docs/04 §2, §6).

The modules in this package are pure policy: they receive pipeline artifacts and an
``LLMClient`` and return validated, evidence-cited models. They never touch the filesystem,
the database or the network directly, which keeps them testable with the fakes in
``tests/fakes``.
"""

from __future__ import annotations

from typing import Literal, Protocol

#: Analysis depth accepted by ``JobOptions`` (docs/03 §2).
Depth = Literal["quick", "standard", "deep"]


class AnalysisOptions(Protocol):
    """The subset of ``JobOptions`` the analysis modules read.

    A structural protocol keeps ``analysis`` free of an import cycle with
    :mod:`caracal.pipeline.stages`, where the concrete pydantic model lives.
    """

    @property
    def depth(self) -> Depth: ...

    @property
    def diarize(self) -> Literal["auto", "on", "off"]: ...

    @property
    def verify(self) -> Literal["auto", "on", "off"]: ...

    @property
    def target_langs(self) -> list[str] | None: ...

    @property
    def speaker_hints(self) -> list[str]: ...

    @property
    def context_hint(self) -> str | None: ...


__all__ = ["AnalysisOptions", "Depth"]
