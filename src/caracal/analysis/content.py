"""Evidence-cited content analysis with map-reduce over long transcripts (docs/04 §6).

The LLM sees the transcript as citable lines (``[seg:12] S2 03:14.2 | text``) and metadata as
``[meta:title] ...``, so every item it returns can be checked against an artifact. Everything
deterministic happens here: window splitting, entity de-duplication, chapter time validation
and quote fuzzy-matching. Items whose evidence does not resolve are dropped before the
report is ever assembled (the citation validator runs again there).
"""

from __future__ import annotations

import asyncio
import difflib
import json
import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from caracal.analysis import AnalysisOptions, Depth
from caracal.asr import Segment, Transcript
from caracal.config import Settings
from caracal.llm.client import CallMeta, LLMClient, ToolResult
from caracal.llm.prompts import load_prompt, render_metadata, render_transcript_lines, wrap_data
from caracal.report.evidence import normalise_ids
from caracal.report.model import (
    Chapter,
    Claim,
    ClaimKind,
    Context,
    Entity,
    EntityType,
    KeyQuote,
    Summary,
    Topic,
)

_QUOTE_MATCH_RATIO: Final = 0.85
_CHAPTER_TOLERANCE_S: Final = 2.0
_QUICK_CLAIMS: Final = 0
_STANDARD_CLAIMS: Final = 8
_DEEP_CLAIMS: Final = 15


class _LlmSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    one_line: str
    short: str = ""
    detailed: str = ""
    evidence: list[str] = Field(default_factory=list)


class _LlmContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    genre: str = "unknown"
    setting: str | None = None
    audience: str | None = None
    purpose: str | None = None
    tone: list[str] = Field(default_factory=list)
    recorded_when: str | None = None
    recorded_where: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class _LlmChapter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: float | None = None
    end: float | None = None
    title: str = ""
    summary: str = ""
    evidence: list[str] = Field(default_factory=list)


class _LlmTopic(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str = ""
    weight: float = 0.0
    evidence: list[str] = Field(default_factory=list)


class _LlmEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    type: EntityType = Field(validation_alias=AliasChoices("entity_type", "type"))
    mentions: list[str] = Field(default_factory=list)
    canonical: str | None = None
    url: str | None = None
    evidence: list[str] = Field(default_factory=list)


class _LlmKeyQuote(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    seg_id: str | None = None
    seg_ids: list[str] = Field(default_factory=list)
    speaker: str | None = None
    start: float | None = None
    why_notable: str = ""
    evidence: list[str] = Field(default_factory=list)


class _LlmClaim(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str | None = None
    text: str
    speaker: str | None = None
    seg_ids: list[str] = Field(default_factory=list)
    kind: ClaimKind = "fact"
    checkworthy: bool = False
    evidence: list[str] = Field(default_factory=list)


class _LlmWindow(BaseModel):
    """One map-window answer: partial findings with their evidence ids."""

    model_config = ConfigDict(extra="ignore")

    chapters: list[_LlmChapter] = Field(default_factory=list)
    topics: list[_LlmTopic] = Field(default_factory=list)
    entities: list[_LlmEntity] = Field(default_factory=list)
    key_quotes: list[_LlmKeyQuote] = Field(default_factory=list)
    claims: list[_LlmClaim] = Field(default_factory=list)


class _LlmContent(_LlmWindow):
    """The single-call answer: the partial findings plus summary and context."""

    summary: _LlmSummary
    context: _LlmContext


class _LlmQuick(BaseModel):
    """Quick depth: only the summary and context are requested."""

    model_config = ConfigDict(extra="ignore")

    summary: _LlmSummary
    context: _LlmContext


class ContentLlmMeta(BaseModel):
    """How the content analysis was produced, checkpointed beside it (card T2.5)."""

    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    attempts: int = 0
    coerced: bool = False
    fallback_used: bool = False
    duration_s: float = 0.0


class ContentAnalysis(BaseModel):
    """The validated content section of a report (docs/03 §5 types)."""

    model_config = ConfigDict(extra="forbid")

    summary: Summary
    context: Context
    chapters: list[Chapter] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    llm_meta: ContentLlmMeta = Field(default_factory=ContentLlmMeta)


@dataclass(frozen=True)
class _EvidenceScope:
    """The evidence ids a content analysis may legitimately cite."""

    allowed: frozenset[str]
    segments: Mapping[int, Segment]

    def valid(self, raw: str | Iterable[object] | None) -> list[str]:
        return [item for item in normalise_ids(raw) if item in self.allowed]

    def cited_segments(self, ids: Sequence[str]) -> list[Segment]:
        found: list[Segment] = []
        for item in ids:
            if not item.startswith("seg:"):
                continue
            try:
                segment = self.segments.get(int(item[4:]))
            except ValueError:
                continue
            if segment is not None:
                found.append(segment)
        return found


def _scope(
    transcript: Transcript,
    meta: Mapping[str, str],
    hints: Sequence[str],
) -> _EvidenceScope:
    allowed = {f"seg:{segment.id}" for segment in transcript.segments}
    allowed.update(f"meta:{key}" for key in meta)
    allowed.update(f"hint:{index}" for index in range(len(hints)))
    return _EvidenceScope(
        allowed=frozenset(allowed),
        segments={segment.id: segment for segment in transcript.segments},
    )


def _normalise_quote_text(text: str) -> str:
    normalised = unicodedata.normalize("NFKC", text).casefold()
    kept = "".join(char if char.isalnum() or char.isspace() else " " for char in normalised)
    return " ".join(kept.split())


def _entity_key(text: str) -> str:
    return _normalise_quote_text(text)


def _claim_limit(depth: Depth) -> int:
    if depth == "quick":
        return _QUICK_CLAIMS
    if depth == "deep":
        return _DEEP_CLAIMS
    return _STANDARD_CLAIMS


def _chapter_guidance(depth: Depth) -> str:
    if depth == "deep":
        return "Start a new chapter every 2-4 minutes."
    if depth == "quick":
        return "Do not produce chapters."
    return "Start a new chapter every 5-10 minutes."


def _summary(value: _LlmSummary, scope: _EvidenceScope) -> Summary:
    one_line = value.one_line.strip() or "No summary available."
    short = value.short.strip() or one_line
    detailed = value.detailed.strip() or short
    return Summary(
        one_line=one_line,
        short=short,
        detailed=detailed,
        evidence=scope.valid(value.evidence),
    )


def _context(value: _LlmContext, scope: _EvidenceScope) -> Context:
    genre = value.genre.strip() or "unknown"
    return Context(
        genre=genre,
        setting=value.setting.strip() if value.setting and value.setting.strip() else None,
        audience=value.audience.strip() if value.audience and value.audience.strip() else None,
        purpose=value.purpose.strip() if value.purpose and value.purpose.strip() else None,
        tone=[tone.strip() for tone in value.tone if tone.strip()],
        recorded_when=value.recorded_when,
        recorded_where=value.recorded_where,
        evidence=scope.valid(value.evidence),
        confidence=value.confidence,
    )


def _chapters(
    values: Sequence[_LlmChapter],
    scope: _EvidenceScope,
    duration: float,
) -> list[Chapter]:
    chapters: list[Chapter] = []
    for value in values:
        ids = scope.valid(value.evidence)
        cited = scope.cited_segments(ids)
        if not cited:
            continue
        derived_start = min(segment.start for segment in cited)
        derived_end = max(segment.end for segment in cited)
        start = value.start if isinstance(value.start, (int, float)) else derived_start
        end = value.end if isinstance(value.end, (int, float)) else derived_end
        if abs(float(start) - derived_start) > _CHAPTER_TOLERANCE_S:
            start = derived_start
        if abs(float(end) - derived_end) > _CHAPTER_TOLERANCE_S:
            end = derived_end
        start = min(max(float(start), 0.0), duration)
        end = min(max(float(end), 0.0), duration)
        if end <= start:
            end = min(duration, start + 1.0)
        if end <= start:
            continue
        chapters.append(
            Chapter(
                start=start,
                end=end,
                title=value.title.strip(),
                summary=value.summary.strip(),
                evidence=ids,
            )
        )

    chapters.sort(key=lambda chapter: (chapter.start, chapter.end))
    kept: list[Chapter] = []
    previous_end = 0.0
    for chapter in chapters:
        start = max(chapter.start, previous_end)
        if chapter.end <= start:
            continue
        kept.append(chapter.model_copy(update={"start": start}))
        previous_end = chapter.end
    return kept


def _topics(values: Sequence[_LlmTopic], scope: _EvidenceScope) -> list[Topic]:
    topics: list[Topic] = []
    for value in values:
        ids = scope.valid(value.evidence)
        label = value.label.strip()
        if not ids or not label:
            continue
        topics.append(Topic(label=label, weight=value.weight, evidence=ids))
    return topics


def _entities(values: Sequence[_LlmEntity], scope: _EvidenceScope) -> list[Entity]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for value in values:
        text = value.text.strip()
        ids = scope.valid(value.evidence)
        if not text or not ids:
            continue
        key = _entity_key(text) or text.casefold()
        mentions = [item for item in scope.valid(value.mentions) if item.startswith("seg:")]
        if key not in merged:
            order.append(key)
            merged[key] = {
                "text": text,
                "type": value.type,
                "mentions": mentions,
                "canonical": value.canonical,
                "url": value.url,
                "evidence": ids,
            }
            continue
        entry = merged[key]
        entry["mentions"] = list(dict.fromkeys([*entry["mentions"], *mentions]))
        entry["evidence"] = list(dict.fromkeys([*entry["evidence"], *ids]))
        if entry["canonical"] is None and value.canonical:
            entry["canonical"] = value.canonical
        if entry["url"] is None and value.url:
            entry["url"] = value.url
    return [
        Entity(
            text=merged[key]["text"],
            type=merged[key]["type"],
            mentions=merged[key]["mentions"],
            canonical=merged[key]["canonical"],
            url=merged[key]["url"],
            evidence=merged[key]["evidence"],
        )
        for key in order
    ]


def _quotes(values: Sequence[_LlmKeyQuote], scope: _EvidenceScope) -> list[KeyQuote]:
    quotes: list[KeyQuote] = []
    for value in values:
        cited_ids = scope.valid([value.seg_id or "", *value.seg_ids, *value.evidence])
        cited = scope.cited_segments(cited_ids)
        text = value.text.strip()
        if not cited or not text:
            continue
        reference = " ".join(segment.text for segment in cited)
        ratio = difflib.SequenceMatcher(
            None,
            _normalise_quote_text(text),
            _normalise_quote_text(reference),
        ).ratio()
        if ratio < _QUOTE_MATCH_RATIO:
            continue
        first = cited[0]
        quotes.append(
            KeyQuote(
                seg_id=f"seg:{first.id}",
                speaker=first.speaker or "S?",
                text=text,
                start=first.start,
                why_notable=value.why_notable.strip(),
            )
        )
    return quotes


def _claims(values: Sequence[_LlmClaim], scope: _EvidenceScope, limit: int) -> list[Claim]:
    claims: list[Claim] = []
    seen_ids: set[str] = set()
    for value in values:
        if len(claims) >= limit:
            break
        ids = scope.valid(value.evidence)
        seg_ids = [item for item in scope.valid(value.seg_ids) if item.startswith("seg:")]
        if not ids and not seg_ids:
            continue
        evidence = ids or seg_ids
        identifier = (value.id or "").strip()
        if not identifier or identifier in seen_ids:
            identifier = f"c{len(claims) + 1}"
            while identifier in seen_ids:
                identifier = f"c{len(seen_ids) + 1}"
        seen_ids.add(identifier)
        claims.append(
            Claim(
                id=identifier,
                text=value.text.strip(),
                speaker=(value.speaker or "").strip(),
                seg_ids=seg_ids,
                kind=value.kind,
                checkworthy=value.checkworthy,
                evidence=evidence,
            )
        )
    return claims


def _to_analysis(
    value: _LlmQuick | _LlmContent,
    scope: _EvidenceScope,
    duration: float,
    depth: Depth,
) -> ContentAnalysis:
    window = value if isinstance(value, _LlmWindow) else _LlmWindow()
    return ContentAnalysis(
        summary=_summary(value.summary, scope),
        context=_context(value.context, scope),
        chapters=_chapters(window.chapters, scope, duration),
        topics=_topics(window.topics, scope),
        entities=_entities(window.entities, scope),
        key_quotes=_quotes(window.key_quotes, scope),
        claims=_claims(window.claims, scope, _claim_limit(depth)),
    )


def _windows(segments: Sequence[Segment], limit: int) -> list[list[Segment]]:
    """Split at segment boundaries, at most ``limit`` characters, with one-segment overlap."""
    if limit <= 0:
        raise ValueError("LLM_CONTEXT_CHARS must be positive")
    windows: list[list[Segment]] = []
    current: list[Segment] = []
    for segment in segments:
        candidate = [*current, segment]
        if current and len(render_transcript_lines(candidate)) > limit:
            windows.append(current)
            overlap = [current[-1], segment]
            current = overlap if len(render_transcript_lines(overlap)) <= limit else [segment]
        else:
            current = candidate
    if current:
        windows.append(current)
    return windows


def _prompt_blocks(
    meta: Mapping[str, str],
    hints: Sequence[str],
) -> tuple[str, str]:
    metadata = wrap_data("metadata", render_metadata(dict(meta)))
    hint_lines = "\n".join(f"[hint:{index}] {hint}" for index, hint in enumerate(hints))
    return metadata, wrap_data("hints", hint_lines)


def _duration(transcript: Transcript) -> float:
    return max((segment.end for segment in transcript.segments), default=0.0)


async def _call[ModelT: BaseModel](
    llm: LLMClient,
    *,
    prompt: str,
    model_cls: type[ModelT],
    tool_name: str,
    tool_description: str,
) -> ToolResult[ModelT]:
    return await llm.call_tool(
        prompt=prompt,
        system=(
            "You are Caracal's content analyst. Use only the DATA blocks, cite an evidence id "
            "for every item, and write prose in the requested report language."
        ),
        model_cls=model_cls,
        tool_name=tool_name,
        tool_description=tool_description,
    )


async def _window_call(
    llm: LLMClient,
    *,
    segments: Sequence[Segment],
    index: int,
    total: int,
    metadata: str,
    hints: str,
    report_language: str,
    chapter_guidance: str,
    claim_limit: int,
) -> ToolResult[_LlmWindow]:
    prompt = load_prompt(
        "content_window",
        window_index=str(index),
        window_total=str(total),
        metadata=metadata,
        hints=hints,
        transcript=wrap_data("transcript", render_transcript_lines(segments)),
        report_language=report_language,
        chapter_guidance=chapter_guidance,
        claim_limit=str(claim_limit),
    )
    return await _call(
        llm,
        prompt=prompt,
        model_cls=_LlmWindow,
        tool_name="record_content_window",
        tool_description="Record the chapters, topics, entities, quotes and claims in this window.",
    )


def _partials_json(partials: Sequence[_LlmWindow]) -> str:
    payload = [partial.model_dump(mode="json") for partial in partials]
    return json.dumps(payload, ensure_ascii=False)


async def analyse_content(
    transcript: Transcript,
    meta: Mapping[str, str],
    hints: Sequence[str],
    options: AnalysisOptions,
    settings: Settings,
    *,
    llm: LLMClient,
) -> ContentAnalysis:
    """Analyse a transcript into evidence-cited report sections (docs/04 §6)."""
    started = time.monotonic()
    depth = options.depth
    scope = _scope(transcript, meta, hints)
    duration = _duration(transcript)
    metadata, hints_block = _prompt_blocks(meta, hints)
    report_language = settings.CARACAL_REPORT_LANGUAGE
    chapter_guidance = _chapter_guidance(depth)
    claim_limit = _claim_limit(depth)
    rendered = render_transcript_lines(transcript.segments)
    metals: list[CallMeta] = []

    if len(rendered) <= settings.LLM_CONTEXT_CHARS:
        prompt = load_prompt(
            "content",
            metadata=metadata,
            hints=hints_block,
            transcript=wrap_data("transcript", rendered),
            report_language=report_language,
            chapter_guidance=chapter_guidance,
            claim_limit=str(claim_limit),
        )
        if depth == "quick":
            quick_result = await _call(
                llm,
                prompt=prompt,
                model_cls=_LlmQuick,
                tool_name="record_content",
                tool_description="Record the summary and context of the recording.",
            )
            metals.append(quick_result.meta)
            analysis = _to_analysis(quick_result.value, scope, duration, depth)
        else:
            full_result = await _call(
                llm,
                prompt=prompt,
                model_cls=_LlmContent,
                tool_name="record_content",
                tool_description="Record the summary, context and cited content findings.",
            )
            metals.append(full_result.meta)
            analysis = _to_analysis(full_result.value, scope, duration, depth)
    else:
        window_segments = _windows(transcript.segments, settings.LLM_CONTEXT_CHARS)
        window_results = await asyncio.gather(
            *(
                _window_call(
                    llm,
                    segments=segments,
                    index=index,
                    total=len(window_segments),
                    metadata=metadata,
                    hints=hints_block,
                    report_language=report_language,
                    chapter_guidance=chapter_guidance,
                    claim_limit=claim_limit,
                )
                for index, segments in enumerate(window_segments, start=1)
            )
        )
        metals.extend(result.meta for result in window_results)
        partials = [result.value for result in window_results]
        reduce_prompt = load_prompt(
            "content_reduce",
            metadata=metadata,
            hints=hints_block,
            partials=wrap_data("partials", _partials_json(partials)),
            report_language=report_language,
            chapter_guidance=chapter_guidance,
            claim_limit=str(claim_limit),
        )
        if depth == "quick":
            quick_reduced = await _call(
                llm,
                prompt=reduce_prompt,
                model_cls=_LlmQuick,
                tool_name="record_content",
                tool_description="Merge the window summaries into a summary and context.",
            )
            metals.append(quick_reduced.meta)
            analysis = _to_analysis(quick_reduced.value, scope, duration, depth)
        else:
            full_reduced = await _call(
                llm,
                prompt=reduce_prompt,
                model_cls=_LlmContent,
                tool_name="record_content",
                tool_description="Merge the window findings into one cited content analysis.",
            )
            metals.append(full_reduced.meta)
            analysis = _to_analysis(full_reduced.value, scope, duration, depth)

    analysis.llm_meta = ContentLlmMeta(
        calls=len(metals),
        attempts=sum(meta.attempts for meta in metals),
        coerced=any(meta.coerced for meta in metals),
        fallback_used=any(meta.fallback_used for meta in metals),
        duration_s=round(time.monotonic() - started, 3),
    )
    return analysis


__all__ = [
    "ContentAnalysis",
    "ContentLlmMeta",
    "analyse_content",
]
