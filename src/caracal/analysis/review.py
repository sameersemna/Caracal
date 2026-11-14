"""LLM self-review of the assembled report (docs/04 §11, card T3.4).

Deterministic review lives in :mod:`caracal.report.assemble`; this module is the second,
interpretive half. It receives a *draft* report (assembled but not yet persisted), builds a
review view that carries the report's claims and context but **no** transcript text, evidence
excerpts or evidence ids, and adds 30 evenly spaced transcript lines so the model can notice a
summary or a speaker name that the transcript contradicts.

The review can never edit the report. Its only output is a bounded list of
:class:`~caracal.report.model.ReviewFlag` items:

* at most :data:`MAX_FLAGS` survive;
* ``field`` must name a top-level ``Report`` key or a dotted path into one
  (``speakers.S1.name``, ``claims.claim_3``) — anything else is dropped;
* messages are stripped of evidence ids (the reviewer must not invent citations), collapsed
  to one line and truncated to :data:`MAX_MESSAGE_CHARS`;
* duplicates by ``(field, message)`` are removed.

The call uses JSON mode, because the measured gemma4 behaviour makes small tool calls
unreliable (docs/04 §1), and a failure is non-fatal for the caller: the stage skips the review
instead of failing the job.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from caracal.asr import Segment, Transcript
from caracal.llm.client import LLMClient
from caracal.llm.prompts import load_prompt, render_transcript_lines, wrap_data
from caracal.report.model import Report, ReviewFlag, Severity

#: Transcript lines the reviewer sees, evenly sampled across the whole recording.
SAMPLE_LINES: Final = 30
#: Hard cap on the flags one review may contribute.
MAX_FLAGS: Final = 10
#: Hard cap on a flag message, as documented in docs/03 §5.
MAX_MESSAGE_CHARS: Final = 200
_MAX_FIELD_CHARS: Final = 120

_REPORT_KEYS: Final[frozenset[str]] = frozenset(Report.model_fields)
_PATH_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_-]+")
_EVIDENCE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:seg|meta|web|vp|caption|hint):[A-Za-z0-9_.-]+")
_SPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"\s+([,.;:!?])")
_EMPTY_BRACKETS_RE: Final[re.Pattern[str]] = re.compile(r"[\[(]\s*[,;/&\s]*[\])]")


class ReviewLlmMeta(BaseModel):
    """How the review call went, checkpointed beside the flags (card T3.4)."""

    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    attempts: int = 0
    coerced: bool = False
    fallback_used: bool = False
    duration_s: float = 0.0
    sampled: int = 0
    raw_flags: int = 0
    kept_flags: int = 0
    dropped_flags: int = 0


@dataclass(frozen=True)
class ReviewResult:
    """The validated flags plus the call metadata the stage checkpoints."""

    flags: list[ReviewFlag]
    llm_meta: ReviewLlmMeta


class _LlmFlag(BaseModel):
    """One inconsistency as the model reports it; the field name is not a schema keyword."""

    model_config = ConfigDict(extra="ignore")

    severity: Severity = "warn"
    field: str
    message: str


class _LlmReview(BaseModel):
    """The review reply: a bounded list of flags and nothing else."""

    model_config = ConfigDict(extra="ignore")

    flags: list[_LlmFlag] = Field(default_factory=list)


def sample_segments(segments: Sequence[Segment], count: int = SAMPLE_LINES) -> list[Segment]:
    """``count`` evenly spaced segments, always including the first and the last.

    Short transcripts are returned whole. For longer ones the positions are computed
    deterministically (``round(i * (n - 1) / (count - 1))``), so a resumed review sees the
    same sample and the caller can assert that both ends of the recording are present.
    """
    items = list(segments)
    if not items or count <= 0:
        return []
    if len(items) <= count:
        return items
    if count == 1:
        return [items[0]]
    step = (len(items) - 1) / (count - 1)
    indexes = sorted({round(position * step) for position in range(count)})
    return [items[index] for index in indexes]


def _speaker_view(report: Report) -> list[dict[str, Any]]:
    return [
        {
            "label": speaker.label,
            "name": speaker.name,
            "confidence": speaker.confidence,
            "role": speaker.role,
            "description": speaker.description,
        }
        for speaker in report.speakers
    ]


def _report_view(report: Report) -> dict[str, Any]:
    """The reviewer's view: report sections only, with no transcript or evidence payload.

    Evidence ids, ``seg_ids``, verification source snippets and ``evidence_index`` are
    deliberately absent: the reviewer works from the report's assertions plus the sampled
    transcript lines, and is not allowed to add or repeat citations.
    """
    return {
        "summary": {
            "one_line": report.summary.one_line,
            "short": report.summary.short,
            "detailed": report.summary.detailed,
        },
        "context": {
            "genre": report.context.genre,
            "setting": report.context.setting,
            "audience": report.context.audience,
            "purpose": report.context.purpose,
            "tone": list(report.context.tone),
            "recorded_when": report.context.recorded_when,
            "recorded_where": report.context.recorded_where,
            "confidence": report.context.confidence,
        },
        "chapters": [
            {
                "start": round(chapter.start, 3),
                "end": round(chapter.end, 3),
                "title": chapter.title,
                "summary": chapter.summary,
            }
            for chapter in report.chapters
        ],
        "speakers": _speaker_view(report),
        "entities": [
            {"text": entity.text, "type": entity.type, "canonical": entity.canonical}
            for entity in report.entities
        ],
        "claims": [
            {
                "id": claim.id,
                "text": claim.text,
                "speaker": claim.speaker,
                "kind": claim.kind,
                "checkworthy": claim.checkworthy,
            }
            for claim in report.claims
        ],
        "verification": [
            {
                "target": item.target,
                "verdict": item.verdict,
                "confidence": item.confidence,
                "rationale": item.rationale,
            }
            for item in report.verification
        ],
        "languages": [
            {
                "code": language.code,
                "name": language.name,
                "share": language.share,
                "confidence": language.confidence,
            }
            for language in report.languages
        ],
        "quality": {
            "asr_avg_logprob": report.quality.asr_avg_logprob,
            "low_confidence_segments": report.quality.low_confidence_segments,
            "caption_agreement": report.quality.caption_agreement,
            "dropped_uncited": report.quality.dropped_uncited,
        },
    }


def valid_field_path(field: str) -> bool:
    """True when ``field`` is a top-level ``Report`` key or a dotted path into one.

    ``summary`` and ``quality.caption_agreement`` are valid; ``speakers.S1.name`` and
    ``claims.claim_3`` are valid dotted paths; ``transcript.text`` and ``Summary`` are not.
    """
    text = field.strip()
    if not text or len(text) > _MAX_FIELD_CHARS:
        return False
    parts = text.split(".")
    if parts[0] not in _REPORT_KEYS:
        return False
    return all(_PATH_TOKEN_RE.fullmatch(part) is not None for part in parts)


def clean_message(message: str) -> str:
    """One-line message without evidence ids, at most :data:`MAX_MESSAGE_CHARS` characters."""
    cleaned = _EVIDENCE_TOKEN_RE.sub(" ", message)
    cleaned = _EMPTY_BRACKETS_RE.sub(" ", cleaned)
    cleaned = _SPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned.strip(" \t,;:-")[:MAX_MESSAGE_CHARS].strip()


def _dedupe(flags: Sequence[ReviewFlag]) -> list[ReviewFlag]:
    """First flag per ``(field, message)`` pair, preserving order."""
    seen: set[tuple[str, str]] = set()
    kept: list[ReviewFlag] = []
    for flag in flags:
        key = (flag.field, flag.message)
        if key in seen:
            continue
        seen.add(key)
        kept.append(flag)
    return kept


def _validated_flags(raw: Sequence[_LlmFlag]) -> tuple[list[ReviewFlag], int]:
    """Validate, clean, de-duplicate and cap the model's flags; return (flags, dropped)."""
    flags: list[ReviewFlag] = []
    dropped = 0
    for item in raw:
        field = item.field.strip()
        message = clean_message(item.message)
        if not valid_field_path(field) or not message:
            dropped += 1
            continue
        flags.append(ReviewFlag(severity=item.severity, field=field, message=message))
    unique = _dedupe(flags)
    dropped += len(flags) - len(unique)
    if len(unique) > MAX_FLAGS:
        dropped += len(unique) - MAX_FLAGS
        unique = unique[:MAX_FLAGS]
    return unique, dropped


def _prompt(report: Report, transcript: Transcript | None) -> tuple[str, int]:
    """The rendered review prompt and the number of sampled transcript lines it carries."""
    sampled = sample_segments(transcript.segments if transcript is not None else ())
    prompt = load_prompt(
        "review",
        report=wrap_data("report", json.dumps(_report_view(report), ensure_ascii=False)),
        transcript=wrap_data("transcript", render_transcript_lines(sampled)),
        max_flags=str(MAX_FLAGS),
        max_message_chars=str(MAX_MESSAGE_CHARS),
    )
    return prompt, len(sampled)


async def review_report(
    report: Report,
    *,
    transcript: Transcript | None,
    llm: LLMClient,
) -> ReviewResult:
    """Run one LLM self-review over ``report`` and return bounded, validated flags.

    Raises :class:`~caracal.errors.CaracalError` only for a transport/protocol failure; the
    pipeline stage treats that as a non-fatal skip (docs/04 §11).
    """
    started = time.monotonic()
    prompt, sampled = _prompt(report, transcript)
    result = await llm.call_tool(
        prompt=prompt,
        system=(
            "You are Caracal's self-reviewer. Compare the report sections with the sampled "
            "transcript lines and flag only inconsistencies. You cannot edit the report."
        ),
        model_cls=_LlmReview,
        tool_name="flag_inconsistencies",
        tool_description="Record the inconsistencies between the report sections and the transcript.",
        mode="json",
        temperature=0.1,
        max_tokens=2048,
    )
    flags, dropped = _validated_flags(result.value.flags)
    meta = ReviewLlmMeta(
        calls=1,
        attempts=result.meta.attempts,
        coerced=result.meta.coerced,
        fallback_used=result.meta.fallback_used,
        duration_s=round(time.monotonic() - started, 3),
        sampled=sampled,
        raw_flags=len(result.value.flags),
        kept_flags=len(flags),
        dropped_flags=dropped,
    )
    return ReviewResult(flags=flags, llm_meta=meta)


__all__ = [
    "MAX_FLAGS",
    "MAX_MESSAGE_CHARS",
    "SAMPLE_LINES",
    "ReviewLlmMeta",
    "ReviewResult",
    "clean_message",
    "review_report",
    "sample_segments",
    "valid_field_path",
]
