"""Rule-first stage planner (docs/04 §2).

The planner decides which later stages are worth running. Rules run first and are fully
deterministic, so the same artefacts always produce the same ``StagePlan``. The LLM fills
only the ``content_profile`` and can never change a ``run`` flag; its failure is a recorded
fallback, not a failed job.

The measured music case (verdict 14): an instrumental track makes whisper emit ``🎵🎵🎵``
segments and confidently label the language ``nn``. When most speech time is non-lexical the
plan marks the job as music, rewrites every language span to ``und`` and skips speaker
identity work, verification and translation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from caracal.analysis import AnalysisOptions
from caracal.asr import Segment, Transcript
from caracal.errors import CaracalError
from caracal.llm.client import LLMClient
from caracal.llm.prompts import load_prompt, render_metadata, render_transcript_lines, wrap_data

MUSIC_REASON: Final = "music detected"
MUSIC_CONTENT_REASON: Final = "music: describe the track, not speech"
AMBIENT_REASON: Final = "speech ratio below 0.05: music/ambient"
AMBIENT_CONTENT_REASON: Final = "music/ambient: short description only"
DIARIZE_OFF_REASON: Final = "diarization disabled by options"
DIARIZER_UNAVAILABLE_REASON: Final = "diarizer unavailable"
SHORT_CLIP_REASON: Final = "short clip: diarization is cheap"
SHORT_CLIP_IDENTITY_REASON: Final = "short clip: single-speaker identity logic"
QUICK_REASON: Final = "quick depth: summary, languages and subtitles only"
VERIFY_OFF_REASON: Final = "verification disabled by options"
VERIFY_ON_REASON: Final = "verification requested"
VERIFY_AUTO_REASON: Final = "auto verification: search configured"
VERIFY_AUTO_NO_SEARCH_REASON: Final = "auto verification: no search provider configured"
NO_TRANSLATION_REASON: Final = "no target language differs from the source"
LLM_PROFILE_REASON: Final = "content profile from the LLM"
LLM_UNAVAILABLE_REASON: Final = "llm unavailable"
NO_LLM_NEEDED_REASON: Final = "default profile: the LLM is not needed"

#: The stages the plan is allowed to gate (docs/04 §2, docs/01 §4).
PLANNED_STAGES: Final[tuple[str, ...]] = (
    "diarize",
    "align",
    "voiceprint",
    "content",
    "identity",
    "verify",
    "translate",
)

_MUSIC_THRESHOLD: Final = 0.70
_MUSIC_SPEECH_SHARE: Final = 0.15
_AMBIENT_THRESHOLD: Final = 0.05
_SHORT_CLIP_S: Final = 20.0
_FIRST_WINDOW_S: Final = 180.0
_TAG_RE: Final = re.compile(r"\[[^\]]*\]|\([^)]*\)")
_MUSIC_SYMBOLS: Final = frozenset("♪🎵🎶")
_STANDARD_SEARCH_BUDGET: Final = 12
_STANDARD_CLAIM_LIMIT: Final = 8
_DEEP_CLAIM_LIMIT: Final = 15


class ContentProfile(BaseModel):
    """What the LLM may contribute to a plan: a content profile and nothing else."""

    model_config = ConfigDict(extra="ignore")

    likely_genre: str = "unknown"
    expected_speakers: int | None = None
    needs_identity: bool = True
    checkworthy_domain: bool = False


class PlanBudgets(BaseModel):
    """How much searching and LLM work this job is allowed (docs/04 §2)."""

    model_config = ConfigDict(extra="forbid")

    search_queries: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    claims: int = Field(ge=0)


class StagePlan(BaseModel):
    """The checkpointed plan (docs/04 §2)."""

    model_config = ConfigDict(extra="forbid")

    run: dict[str, bool]
    reasons: dict[str, str]
    content_profile: ContentProfile
    budgets: PlanBudgets
    music_detected: bool = False
    languages: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class PlanInputs:
    """Everything the rules and the profile prompt need (docs/04 §2)."""

    duration: float
    speech_ratio: float
    options: AnalysisOptions
    speech_share: float | None = None
    transcript: Transcript | None = None
    languages: Sequence[Mapping[str, object]] = ()
    metadata: Mapping[str, str] = field(default_factory=dict)
    diarizer_available: bool = False
    search_configured: bool = False
    default_target_langs: Sequence[str] = ()
    search_max_queries: int = 24
    report_language: str = "en"


def _default_profile() -> ContentProfile:
    return ContentProfile()


def _has_lexical_text(text: str) -> bool:
    """True when ``text`` still contains a letter or digit after music/tag removal."""
    cleaned = _TAG_RE.sub(" ", text)
    return any(char.isalpha() or char.isdigit() for char in cleaned if char not in _MUSIC_SYMBOLS)


def _music_detected(transcript: Transcript | None, speech_share: float | None = None) -> bool:
    """True for non-speech audio.

    With a diarize checkpoint the measured discriminator is the exclusive-turn speech
    share (music 2 %, talk 87 % — verdict 25 D3). Without one, fall back to the text rule:
    at least 70 % of segment time is non-lexical (``🎵`` segments or hallucinated tags).
    """
    if speech_share is not None:
        return speech_share < _MUSIC_SPEECH_SHARE
    if transcript is None:
        return False
    total = 0.0
    non_lexical = 0.0
    for segment in transcript.segments:
        spoken = max(0.0, segment.end - segment.start)
        if spoken <= 0.0:
            continue
        total += spoken
        if not _has_lexical_text(segment.text):
            non_lexical += spoken
    return total > 0.0 and non_lexical / total >= _MUSIC_THRESHOLD


def _span_bounds(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Mapping):
        return None
    start = value.get("start")
    end = value.get("end")
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return float(start), float(end)


def _und_languages(
    transcript: Transcript | None,
    languages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Rewrite every language span to ``und`` while preserving the timeline."""
    spans: list[tuple[float, float]] = []
    for language in languages:
        raw_spans = language.get("spans")
        if not isinstance(raw_spans, (list, tuple)):
            continue
        for raw_span in raw_spans:
            bounds = _span_bounds(raw_span)
            if bounds is not None:
                spans.append(bounds)
    if not spans and transcript is not None:
        spans = [(segment.start, segment.end) for segment in transcript.segments]
    spans.sort()
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 0.01:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if not merged:
        return []
    return [
        {
            "code": "und",
            "name": "Unknown",
            "share": 1.0,
            "spans": [{"start": start, "end": end} for start, end in merged],
            "confidence": 1.0,
        }
    ]


def _first_segments(transcript: Transcript | None) -> tuple[Segment, ...]:
    if transcript is None:
        return ()
    return tuple(segment for segment in transcript.segments if segment.start < _FIRST_WINDOW_S)


def _resolve_targets(inputs: PlanInputs) -> list[str]:
    if inputs.options.target_langs is None:
        targets = list(inputs.default_target_langs)
    else:
        targets = list(inputs.options.target_langs)
    dominant: str | None = None
    best_share = -1.0
    for language in inputs.languages:
        code = language.get("code")
        share = language.get("share")
        if not isinstance(code, str) or code == "und":
            continue
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            share = 0.0
        if float(share) > best_share:
            dominant = code
            best_share = float(share)
    if dominant is None and inputs.transcript is not None:
        dominant = inputs.transcript.language_code
    return [target for target in targets if target and target != dominant]


def _set_skip(run: dict[str, bool], reasons: dict[str, str], stage: str, reason: str) -> None:
    """Skip ``stage`` unless an earlier, stronger rule already skipped it."""
    if not run[stage]:
        return
    run[stage] = False
    reasons[stage] = reason


def _set_reason(run: dict[str, bool], reasons: dict[str, str], stage: str, reason: str) -> None:
    if run[stage]:
        reasons[stage] = reason


def _skip_group(
    run: dict[str, bool],
    reasons: dict[str, str],
    stages: Sequence[str],
    reason: str,
) -> None:
    for stage in stages:
        _set_skip(run, reasons, stage, reason)


def _budgets(depth: str, search_max_queries: int) -> PlanBudgets:
    if depth == "quick":
        return PlanBudgets(search_queries=0, llm_calls=1, claims=0)
    if depth == "deep":
        return PlanBudgets(
            search_queries=max(0, search_max_queries),
            llm_calls=max(0, search_max_queries),
            claims=_DEEP_CLAIM_LIMIT,
        )
    return PlanBudgets(
        search_queries=_STANDARD_SEARCH_BUDGET,
        llm_calls=_STANDARD_SEARCH_BUDGET,
        claims=_STANDARD_CLAIM_LIMIT,
    )


def _profile_prompt(inputs: PlanInputs) -> str:
    metadata = render_metadata(dict(inputs.metadata))
    transcript = render_transcript_lines(_first_segments(inputs.transcript))
    return load_prompt(
        "plan",
        metadata=wrap_data("metadata", metadata),
        transcript=wrap_data("transcript", transcript),
        report_language=inputs.report_language,
    )


async def _profile(inputs: PlanInputs, llm: LLMClient | None) -> tuple[ContentProfile, str]:
    """Return the profile and a reason, never raising for an unavailable or failing LLM."""
    if _music_detected(inputs.transcript, inputs.speech_share):
        return _default_profile(), "music: no profile call"
    if inputs.options.depth == "quick":
        return _default_profile(), "quick depth: default profile"
    if llm is None:
        return _default_profile(), LLM_UNAVAILABLE_REASON
    try:
        result = await llm.call_tool(
            prompt=_profile_prompt(inputs),
            system=(
                "You are Caracal's planning stage. Read the DATA blocks and describe what kind "
                "of recording this is. You cannot change which pipeline stages run."
            ),
            model_cls=ContentProfile,
            tool_name="record_content_profile",
            tool_description="Record the likely genre and whether identity and verification matter.",
        )
    except CaracalError:
        return _default_profile(), LLM_UNAVAILABLE_REASON
    profile = result.value
    if not profile.likely_genre.strip():
        profile = profile.model_copy(update={"likely_genre": "unknown"})
    return profile, LLM_PROFILE_REASON


async def plan(inputs: PlanInputs, *, llm: LLMClient | None) -> StagePlan:
    """Build the deterministic stage plan, consulting the LLM only for the content profile."""
    run = {stage: True for stage in PLANNED_STAGES}
    reasons: dict[str, str] = {}
    music = _music_detected(inputs.transcript, inputs.speech_share)
    languages: list[dict[str, object]] | None = None

    if music:
        _skip_group(
            run,
            reasons,
            ("align", "voiceprint", "identity", "verify", "translate"),
            MUSIC_REASON,
        )
        _set_reason(run, reasons, "content", MUSIC_CONTENT_REASON)
        languages = _und_languages(inputs.transcript, inputs.languages)

    if inputs.speech_ratio < _AMBIENT_THRESHOLD:
        _skip_group(
            run,
            reasons,
            ("diarize", "align", "voiceprint", "identity", "verify", "translate"),
            AMBIENT_REASON,
        )
        _set_reason(run, reasons, "content", AMBIENT_CONTENT_REASON)

    if inputs.options.diarize == "off":
        _skip_group(
            run,
            reasons,
            ("diarize", "align", "voiceprint"),
            DIARIZE_OFF_REASON,
        )
    elif not inputs.diarizer_available:
        _skip_group(
            run,
            reasons,
            ("diarize", "align", "voiceprint"),
            DIARIZER_UNAVAILABLE_REASON,
        )

    if inputs.options.diarize == "auto" and inputs.duration < _SHORT_CLIP_S:
        _set_reason(run, reasons, "diarize", SHORT_CLIP_REASON)
        _set_reason(run, reasons, "identity", SHORT_CLIP_IDENTITY_REASON)

    if inputs.options.depth == "quick":
        _skip_group(
            run,
            reasons,
            ("diarize", "align", "voiceprint", "identity", "verify", "translate"),
            QUICK_REASON,
        )

    profile, profile_reason = await _profile(inputs, llm)
    reasons["profile"] = profile_reason

    if inputs.options.verify == "off":
        _set_skip(run, reasons, "verify", VERIFY_OFF_REASON)
    elif inputs.options.verify == "on":
        _set_reason(run, reasons, "verify", VERIFY_ON_REASON)
    elif inputs.search_configured:
        # The planner runs before identity, so it cannot know whether an unconfirmed name
        # exists; auto therefore means "run unless quick, music or no provider" (verdict 32)
        # and the verify stage itself reports "nothing to verify" when no target exists.
        _set_reason(run, reasons, "verify", VERIFY_AUTO_REASON)
    else:
        _set_skip(run, reasons, "verify", VERIFY_AUTO_NO_SEARCH_REASON)

    targets = _resolve_targets(inputs)
    if targets:
        target_text = ", ".join(targets)
        _set_reason(run, reasons, "translate", f"translate to {target_text}")
    else:
        _set_skip(run, reasons, "translate", NO_TRANSLATION_REASON)

    if run["content"] and "content" not in reasons:
        reasons["content"] = "content analysis"

    return StagePlan(
        run=run,
        reasons=reasons,
        content_profile=profile,
        budgets=_budgets(inputs.options.depth, inputs.search_max_queries),
        music_detected=music,
        languages=languages,
    )


__all__ = [
    "AMBIENT_REASON",
    "PLANNED_STAGES",
    "ContentProfile",
    "PlanBudgets",
    "PlanInputs",
    "StagePlan",
    "plan",
]
