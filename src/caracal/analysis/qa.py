"""Question answering over one recording and retrieval for cross-job search (docs/04 §12).

Retrieval is deterministic: the transcript index returns the highest-ranked segments for the
question, a keyword-expansion call widens recall when that first pass returns fewer than three
hits, and one neighbouring segment on each side of every hit restores the local context. The
report summary/context and chapter titles always go with the segments so the answer can rely on
what the recording is about without ever citing it.

Only the LLM call is non-deterministic. The citation rules are enforced here: a citation whose
segment id is not in the supplied context is dropped, the citation's start and text are copied
from the real segment, and an answer without one valid citation is replaced by the fixed
"Not found in the audio." answer with confidence 0 (docs/04 §12, docs/03 §5 ``Answer``).
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from caracal.asr import LANGUAGE_NAMES
from caracal.config import Route, Settings
from caracal.db.repo import SearchHit, SegmentRepo
from caracal.errors import CaracalError, ErrorCode
from caracal.llm.client import LLMClient
from caracal.llm.prompts import format_timestamp, load_prompt, wrap_data
from caracal.report.assemble import strip_prose_citations
from caracal.report.evidence import normalise_ids
from caracal.report.model import Answer, Citation, Report

_SEARCH_LIMIT: Final = 12
_MIN_HITS: Final = 3
_MAX_KEYWORDS: Final = 8
_MAX_QUESTION_CHARS: Final = 1000
_KEYWORD_SAMPLE_SEGMENTS: Final = 40
_NOT_FOUND: Final = "Not found in the audio."
_RESERVED_KEYWORDS: Final = frozenset({"or", "and", "not"})
_ANSWER_SYSTEM: Final = (
    "You are Caracal's question-answering engine for one recording. Use only the DATA "
    "blocks, cite only transcript segment ids that appear in them, and answer in the "
    "language of the question."
)
_KEYWORD_SYSTEM: Final = (
    "You extract search keywords for a transcript search. Use the report only as context "
    "and return only the requested JSON object."
)


class _LlmKeywords(BaseModel):
    """The keyword-expansion answer (JSON mode)."""

    model_config = ConfigDict(extra="ignore")

    keywords: list[str] = Field(default_factory=list)


class _LlmCitation(BaseModel):
    """One model-supplied citation; only ``seg_id`` is trusted."""

    model_config = ConfigDict(extra="ignore")

    seg_id: str | int | None = None
    start: float | None = None
    text: str | None = None


class _LlmAnswer(BaseModel):
    """The question-answering answer (JSON mode)."""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    confidence: float = 0.0
    citations: list[_LlmCitation] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class QaSegment:
    """One transcript segment as retrieval sees it."""

    seg_id: int
    start: float
    end: float
    speaker: str | None
    text: str


@dataclass(frozen=True)
class QaReport:
    """The report facts Q&A always supplies to the model (summary, context, chapters)."""

    summary: str
    context: str
    chapters: tuple[tuple[float, str], ...]
    language_code: str


def qa_report(report: Report) -> QaReport:
    """Reduce a report to the context Q&A adds to every question (docs/04 §12)."""
    summary = report.summary.short.strip() or report.summary.one_line.strip()
    context = report.context
    parts: list[str] = []
    for label, value in (
        ("genre", context.genre),
        ("setting", context.setting),
        ("audience", context.audience),
        ("purpose", context.purpose),
        ("recorded when", context.recorded_when),
        ("recorded where", context.recorded_where),
    ):
        if isinstance(value, str) and value.strip():
            parts.append(f"{label}: {value.strip()}")
    if context.tone:
        parts.append("tone: " + ", ".join(tone for tone in context.tone if tone.strip()))
    language = max(report.languages, key=lambda item: item.share, default=None)
    return QaReport(
        summary=summary,
        context="; ".join(parts),
        chapters=tuple((chapter.start, chapter.title) for chapter in report.chapters if chapter.title),
        language_code=language.code if language is not None else "en",
    )


def validate_question(question: str) -> str:
    """Strip and validate a question: non-empty, at most 1 000 characters (docs/04 §12)."""
    if not isinstance(question, str):
        raise CaracalError(ErrorCode.INVALID_INPUT, "question must be a string")
    stripped = question.strip()
    if not stripped:
        raise CaracalError(ErrorCode.INVALID_INPUT, "question must not be empty")
    if len(stripped) > _MAX_QUESTION_CHARS:
        raise CaracalError(
            ErrorCode.INVALID_INPUT,
            "question must be at most 1000 characters",
            {"max_chars": _MAX_QUESTION_CHARS},
        )
    return stripped


def _language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _report_text(report: QaReport) -> str:
    lines = [f"Summary: {report.summary}"]
    if report.context:
        lines.append(f"Context: {report.context}")
    if report.chapters:
        chapters = "; ".join(f"{format_timestamp(start)} {title}" for start, title in report.chapters)
        lines.append(f"Chapters: {chapters}")
    return "\n".join(lines)


def _segment_line(segment: QaSegment) -> str:
    speaker = segment.speaker.strip() if segment.speaker and segment.speaker.strip() else "S?"
    return f"[seg:{segment.seg_id}] {speaker} {format_timestamp(segment.start)} | {segment.text}"


def _hit_segment(hit: SearchHit) -> QaSegment:
    return QaSegment(
        seg_id=hit.seg_id,
        start=hit.start_s,
        end=hit.start_s,
        speaker=hit.speaker,
        text=hit.text,
    )


def _merge_hits(primary: Sequence[SearchHit], secondary: Sequence[SearchHit]) -> list[SearchHit]:
    """Merge two ranked passes, keeping the first occurrence of every (job, segment)."""
    seen: set[tuple[str, int]] = set()
    merged: list[SearchHit] = []
    for hit in (*primary, *secondary):
        key = (str(hit.job_id), hit.seg_id)
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)
    return merged


def _candidate_segments(
    hits: Sequence[SearchHit],
    transcript: Sequence[QaSegment],
) -> list[QaSegment]:
    """Each hit plus its immediate neighbours, in hit order, without duplicates."""
    positions = {segment.seg_id: index for index, segment in enumerate(transcript)}
    candidates: list[QaSegment] = []
    seen: set[int] = set()

    def add(segment: QaSegment) -> None:
        if segment.seg_id in seen:
            return
        seen.add(segment.seg_id)
        candidates.append(segment)

    for hit in hits:
        index = positions.get(hit.seg_id)
        if index is None:
            add(_hit_segment(hit))
            continue
        for position in (index - 1, index, index + 1):
            if 0 <= position < len(transcript):
                add(transcript[position])
    return candidates


def _context_blocks(
    report_text: str,
    segments: Sequence[QaSegment],
    limit: int,
) -> tuple[str, str, list[QaSegment]]:
    """Wrap the report and as many segment lines as fit in ``limit`` characters.

    The whole context (report block plus transcript block, including both DATA
    delimiters) stays within ``LLM_CONTEXT_CHARS / 2``; segments are added in ranked order
    and a line that would cross the budget stops the block, so every segment in the prompt
    remains citable.
    """
    budget = max(1, limit)
    empty_report = len(wrap_data("report", ""))
    empty_transcript = len(wrap_data("transcript", ""))
    text_budget = max(0, budget - empty_report - empty_transcript)
    if len(report_text) > text_budget:
        report_text = report_text[:text_budget]
    report_block = wrap_data("report", report_text)
    lines: list[str] = []
    chosen: list[QaSegment] = []
    for segment in segments:
        candidate = wrap_data("transcript", "\n".join([*lines, _segment_line(segment)]))
        if len(report_block) + len(candidate) > budget:
            break
        lines.append(_segment_line(segment))
        chosen.append(segment)
    return report_block, wrap_data("transcript", "\n".join(lines)), chosen


def _normalise_keywords(values: Sequence[str]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            continue
        keyword = " ".join(raw.split()).strip()
        if not keyword:
            continue
        key = keyword.casefold()
        if key in _RESERVED_KEYWORDS or key in seen:
            continue
        seen.add(key)
        keywords.append(keyword)
        if len(keywords) >= _MAX_KEYWORDS:
            break
    return keywords


async def _keywords(
    question: str,
    *,
    report: QaReport,
    transcript: Sequence[QaSegment],
    llm: LLMClient,
) -> list[str]:
    """Ask for transcript-language keywords when the literal search found too little."""
    sample = "\n".join(_segment_line(segment) for segment in transcript[:_KEYWORD_SAMPLE_SEGMENTS])
    prompt = load_prompt(
        "qa_keywords",
        language=_language_name(report.language_code),
        code=report.language_code,
        report=wrap_data("report", _report_text(report)),
        question=wrap_data("question", question),
        sample=wrap_data("transcript", sample),
    )
    result = await llm.call_tool(
        prompt=prompt,
        system=_KEYWORD_SYSTEM,
        model_cls=_LlmKeywords,
        tool_name="qa_keywords",
        tool_description="Return 3 to 8 search keywords in the transcript's dominant language.",
        mode="json",
        route=Route.QA,
    )
    return _normalise_keywords(result.value.keywords)


async def _retrieve(
    job_id: uuid.UUID,
    question: str,
    *,
    repos: SegmentRepo,
    report: QaReport,
    transcript: Sequence[QaSegment],
    llm: LLMClient,
) -> list[SearchHit]:
    """Top segments for the question, widened through keywords when recall is low."""
    hits = await repos.search(question, job_id=job_id, limit=_SEARCH_LIMIT)
    if len(hits) >= _MIN_HITS:
        return hits
    keywords = await _keywords(question, report=report, transcript=transcript, llm=llm)
    if not keywords:
        return hits
    expanded = await repos.search(" or ".join(keywords), job_id=job_id, limit=_SEARCH_LIMIT)
    return _merge_hits(hits, expanded)


def _citation_seg_id(raw: object) -> int | None:
    """Parse ``"seg:12"``, ``"12"`` or ``12``; rendered-line noise is tolerated.

    ``normalise_ids`` applies the typed-evidence rule, so ``"seg:2."``, ``"seg:2 S1"`` and
    ``"see seg:2"`` all become segment 2 (docs/04 §1, §9).
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw if raw >= 0 else None
    if not isinstance(raw, str):
        return None
    for found in normalise_ids(raw):
        if found.startswith("seg:"):
            try:
                return int(found[4:])
            except ValueError:
                continue
    value = raw.strip()
    if value.startswith("seg:"):
        value = value[4:]
    try:
        identifier = int(value)
    except ValueError:
        return None
    return identifier if identifier >= 0 else None


def _finalise(raw: _LlmAnswer, allowed: Mapping[int, QaSegment]) -> Answer:
    """Enforce the citation rules on one model answer (docs/04 §12)."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for item in raw.citations:
        seg_id = _citation_seg_id(item.seg_id)
        if seg_id is None or seg_id not in allowed or seg_id in seen:
            continue
        seen.add(seg_id)
        segment = allowed[seg_id]
        citations.append(Citation(seg_id=f"seg:{seg_id}", start=segment.start, text=segment.text))
    # The model sometimes repeats its evidence ids inside the prose (verdict 38). The ids
    # stay in ``citations``; only the prose is stripped, exactly like report prose.
    answer_text, _inline_ids = strip_prose_citations(raw.answer.strip())
    answer_text = answer_text.strip()
    if not citations or not answer_text:
        return Answer(answer=_NOT_FOUND, confidence=0.0, citations=[], evidence=[])
    confidence = float(raw.confidence)
    if not math.isfinite(confidence):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return Answer(
        answer=answer_text,
        confidence=confidence,
        citations=citations,
        evidence=[citation.seg_id for citation in citations],
    )


async def answer(
    job_id: uuid.UUID,
    question: str,
    *,
    repos: SegmentRepo,
    report: QaReport,
    transcript: Sequence[QaSegment],
    settings: Settings,
    llm: LLMClient,
) -> Answer:
    """Answer one question about a finished job (docs/01 §3, docs/04 §12)."""
    stripped = validate_question(question)
    hits = await _retrieve(
        job_id,
        stripped,
        repos=repos,
        report=report,
        transcript=transcript,
        llm=llm,
    )
    candidates = _candidate_segments(hits, transcript)
    budget = max(1, settings.LLM_CONTEXT_CHARS // 2)
    report_block, transcript_block, allowed_segments = _context_blocks(
        _report_text(report),
        candidates,
        budget,
    )
    prompt = load_prompt(
        "qa",
        report=report_block,
        question=wrap_data("question", stripped),
        transcript=transcript_block,
    )
    result = await llm.call_tool(
        prompt=prompt,
        system=_ANSWER_SYSTEM,
        model_cls=_LlmAnswer,
        tool_name="qa_answer",
        tool_description="Answer the question with citations to transcript segment ids.",
        mode="json",
        route=Route.QA,
    )
    allowed = {segment.seg_id: segment for segment in allowed_segments}
    return _finalise(result.value, allowed)


__all__ = [
    "QaReport",
    "QaSegment",
    "answer",
    "qa_report",
    "validate_question",
]
