"""Assemble the report from the pipeline's stage checkpoints (docs/01 §4, T2.7).

The assembler is a **pure** function over data the pipeline already persisted: it never calls
a network, a database or the filesystem. Missing or skipped stages become empty sections and
a recorded stage status instead of an exception, so a partial job still produces a valid
``Report`` (docs/03 §5).

Two jobs beyond field mapping live here:

* **Citations in prose** (verdict 26): the model sometimes embeds its evidence list in the
  text (``"TED Talk [meta:title, meta:channel]"``). Prose fields are stripped of those
  bracketed/parenthesised id lists and the ids are merged into the item's ``evidence``; quote
  text is never touched.
* **Deterministic review** (docs/04 §11, deterministic half only — the LLM review is T3.4):
  speaker shares, chapter coverage, caption agreement, low-confidence segments and failed
  stages each become a :class:`ReviewFlag`.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import ValidationError

from caracal import __version__
from caracal.report.evidence import (
    EvidenceIndex,
    EvidenceItem,
    build_evidence_index,
    normalise_ids,
    validate_citations,
)
from caracal.report.model import (
    SCHEMA_VERSION,
    AudioInfo,
    Chapter,
    Claim,
    Context,
    Entity,
    KeyQuote,
    Language,
    QualityInfo,
    Report,
    ReportSource,
    ReviewFlag,
    SpeakerIdentity,
    Summary,
    Topic,
    TranscriptRef,
    Verification,
)

if TYPE_CHECKING:
    from caracal.asr import Transcript
    from caracal.asr.captions import CaptionWord
    from caracal.config import Settings
    from caracal.db.repo import JobRecord

#: A bracketed or parenthesised run that carries no nested brackets.
_BRACKET_RE: Final[re.Pattern[str]] = re.compile(r"[\[(]([^\[\]()]*)[\])]")

#: What may remain inside a bracket once its evidence ids are removed.
_SEPARATOR_RE: Final[re.Pattern[str]] = re.compile(r"[\s,;/&+·-]*")

_SPACE_BEFORE_PUNCT_RE: Final[re.Pattern[str]] = re.compile(r"[ \t]+([,.;:!?])")
_MULTISPACE_RE: Final[re.Pattern[str]] = re.compile(r"[ \t]{2,}")

_CHAPTER_COVERAGE_MIN: Final = 0.8
_SPEAKER_SHARE_TOLERANCE: Final = 0.05
_CAPTION_AGREEMENT_MIN: Final = 0.6
_LOW_CONFIDENCE_LOGPROB: Final = -1.0
_LOW_CONFIDENCE_SHARE_MAX: Final = 0.10
_DESCRIPTION_EXCERPT_CHARS: Final = 500

_SOURCE_KINDS: Final = frozenset({"upload", "direct", "platform"})
_AUDIO_QUALITIES: Final = frozenset({"good", "fair", "poor"})
_STAGE_STATUSES: Final = frozenset({"pending", "running", "done", "skipped", "failed"})


def _mapping(value: object) -> dict[str, Any] | None:
    """``value`` as a plain mapping, or ``None``."""
    if isinstance(value, Mapping):
        return cast("dict[str, Any]", value)
    return None


def _text(value: object) -> str | None:
    """A stripped non-empty string, or ``None``."""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _string_list(value: object) -> list[str]:
    """Every string entry of a list, stripped and non-empty."""
    if not isinstance(value, (list, tuple)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _number(value: object, default: float = 0.0) -> float:
    """A finite float, or ``default`` (bools are never numbers)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _integer(value: object, default: int) -> int:
    """An int, or ``default`` (bools are never numbers)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _output(checkpoints: Mapping[str, Mapping[str, Any] | None], stage: str) -> dict[str, Any] | None:
    """The ``output`` object of a stage checkpoint, or ``None`` when it is absent."""
    checkpoint = checkpoints.get(stage)
    if not isinstance(checkpoint, Mapping):
        return None
    return _mapping(checkpoint.get("output"))


def _merge_ids(existing: Sequence[str], extra: Sequence[str]) -> list[str]:
    """``existing`` followed by the unseen ids in ``extra``."""
    merged = list(existing)
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged


def strip_prose_citations(text: str) -> tuple[str, list[str]]:
    """Remove bracketed/parenthesised evidence-id lists from prose (verdict 26).

    ``"TED Talk [meta:title, meta:channel]"`` -> ``("TED Talk", ["meta:title", "meta:channel"])``.
    A bracket that also carries ordinary words (``"(see seg:2 for the story)"``) is left
    intact: only a pure evidence-id list is treated as a citation. The returned text is only
    whitespace-normalised, never otherwise rewritten.
    """
    found: list[str] = []

    def _replace(match: re.Match[str]) -> str:
        content = match.group(1)
        ids = normalise_ids(content)
        if not ids:
            return match.group(0)
        remainder = content
        for evidence_id in ids:
            remainder = remainder.replace(evidence_id, "")
        if _SEPARATOR_RE.fullmatch(remainder) is None:
            return match.group(0)
        found.extend(evidence_id for evidence_id in ids if evidence_id not in found)
        return ""

    stripped = _BRACKET_RE.sub(_replace, text)
    if not found:
        return text, []
    stripped = _MULTISPACE_RE.sub(" ", stripped)
    stripped = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", stripped)
    return stripped.strip(" \t"), found


def _strip(field: str) -> tuple[str, list[str]]:
    """Strip citations from one optional prose string (``""`` when absent)."""
    return strip_prose_citations(field)


def _source(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    job: JobRecord,
) -> ReportSource:
    """Map ingest + job + normalize onto ``ReportSource``."""
    ingest = _output(checkpoints, "ingest") or {}
    normalize = _output(checkpoints, "normalize") or {}
    probe = _mapping(normalize.get("probe")) or {}
    stats = _mapping(normalize.get("stats")) or {}
    platform = _mapping(ingest.get("platform"))
    duration = _number(stats.get("duration_s"), _number(probe.get("duration_s"), 0.0))

    kind_value = _text(ingest.get("source_kind")) or job.source_kind
    kind = cast("Any", kind_value if kind_value in _SOURCE_KINDS else job.source_kind)
    description = _text(platform.get("description")) if platform is not None else None
    if description is not None:
        description = description[:_DESCRIPTION_EXCERPT_CHARS]

    return ReportSource(
        kind=kind,
        url=_text(ingest.get("url")) or _text(job.source_url),
        title=(_text(platform.get("title")) if platform is not None else None) or _text(job.title),
        uploader=_text(platform.get("uploader")) if platform is not None else None,
        channel=_text(platform.get("channel")) if platform is not None else None,
        upload_date=_text(platform.get("upload_date")) if platform is not None else None,
        platform=_text(platform.get("extractor")) if platform is not None else None,
        duration_s=duration,
        description_excerpt=description,
        original_format=_text(probe.get("format_name")) or "unknown",
        sha256=_text(ingest.get("sha256")) or _text(job.sha256) or "",
    )


def _music_flag(checkpoints: Mapping[str, Mapping[str, Any] | None]) -> bool | None:
    """The plan's ``music_detected`` flag, or ``None`` when no plan recorded it."""
    plan = _output(checkpoints, "plan")
    if plan is None or not isinstance(plan.get("music_detected"), bool):
        return None
    return cast("bool", plan.get("music_detected"))


def _music_detected(checkpoints: Mapping[str, Mapping[str, Any] | None]) -> bool:
    """True only when the plan positively detected music."""
    return _music_flag(checkpoints) is True


def _audio(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    duration_s: float,
) -> AudioInfo:
    """Map normalize's probe/stats onto ``AudioInfo`` (music comes from the plan)."""
    normalize = _output(checkpoints, "normalize") or {}
    probe = _mapping(normalize.get("probe")) or {}
    stats = _mapping(normalize.get("stats")) or {}
    quality_value = _text(stats.get("quality"))
    quality = cast(
        "Any",
        quality_value if quality_value in _AUDIO_QUALITIES else "poor",
    )
    loudness = stats.get("loudness_lufs")
    return AudioInfo(
        duration_s=duration_s,
        sample_rate=_integer(probe.get("sample_rate"), 16000),
        channels=_integer(probe.get("channels"), 1),
        loudness_lufs=_number(loudness) if isinstance(loudness, (int, float)) else None,
        speech_ratio=_number(stats.get("speech_ratio"), 0.0),
        music_detected=_music_flag(checkpoints),
        quality=quality,
        notes=[],
    )


def _languages(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    music: bool,
) -> list[Language]:
    """Languages from langid, replaced by the plan's ``und`` timeline for music."""
    raw: object = None
    if music:
        plan = _output(checkpoints, "plan")
        if plan is not None:
            raw = plan.get("languages")
    if not isinstance(raw, list):
        langid = _output(checkpoints, "langid") or {}
        raw = langid.get("languages")
    languages: list[Language] = []
    for item in raw if isinstance(raw, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            languages.append(Language.model_validate(payload))
        except ValidationError:
            continue
    return languages


def _segment_ids(transcript: Transcript | None, label: str) -> list[str]:
    """Every segment the label speaks in, as ``seg:<id>`` evidence."""
    if transcript is None:
        return []
    ids: list[str] = []
    for segment in transcript.segments:
        spoken = segment.speaker == label or any(word.speaker == label for word in segment.words)
        if spoken:
            ids.append(f"seg:{segment.id}")
    return ids


def _unnamed_speaker(
    label: str,
    stats: Mapping[str, Any],
    transcript: Transcript | None,
) -> SpeakerIdentity:
    """One align-derived speaker with no name hypothesis (identity skipped/failed)."""
    return SpeakerIdentity(
        label=label,
        talk_time_s=_number(stats.get("talk_time_s"), 0.0),
        share=_number(stats.get("share"), 0.0),
        first_seen_s=_number(stats.get("first_seen_s"), 0.0),
        name=None,
        confidence=0.0,
        candidates=[],
        description=None,
        evidence=_segment_ids(transcript, label),
    )


def _speakers(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    transcript: Transcript | None,
) -> list[SpeakerIdentity]:
    """Verify's refined speakers when present, else identity's, else the align statistics.

    The verify stage re-runs the identity fusion with the web sources and checkpoints the
    refined speakers itself; a stage never rewrites another stage's checkpoint (docs/04 §8
    step 5). Falling back to identity keeps jobs whose verify stage was skipped unchanged.
    """
    verify = _output(checkpoints, "verify")
    raw = verify.get("speakers") if verify is not None else None
    if not isinstance(raw, list) or not raw:
        identity = _output(checkpoints, "identity")
        raw = identity.get("speakers") if identity is not None else None
    speakers: list[SpeakerIdentity] = []
    for item in raw if isinstance(raw, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            speakers.append(SpeakerIdentity.model_validate(payload))
        except ValidationError:
            continue
    if speakers:
        for speaker in speakers:
            if speaker.description:
                stripped, ids = _strip(speaker.description)
                speaker.description = stripped or None
                speaker.evidence = _merge_ids(speaker.evidence, ids)
        return speakers

    align = _output(checkpoints, "align")
    stats = _mapping(align.get("speakers")) if align is not None else None
    if stats is None:
        return []
    rebuilt: list[SpeakerIdentity] = []
    for label, value in stats.items():
        entry = _mapping(value)
        if not isinstance(label, str) or entry is None:
            continue
        rebuilt.append(_unnamed_speaker(label, entry, transcript))
    return rebuilt


def _summary(value: object) -> Summary:
    """Map content's summary, stripping any citations embedded in its prose."""
    payload = _mapping(value) or {}
    evidence = normalise_ids(payload.get("evidence"))
    one_line, ids_a = _strip(str(payload.get("one_line") or ""))
    short, ids_b = _strip(str(payload.get("short") or ""))
    detailed, ids_c = _strip(str(payload.get("detailed") or ""))
    return Summary(
        one_line=one_line,
        short=short,
        detailed=detailed,
        evidence=_merge_ids(_merge_ids(_merge_ids(evidence, ids_a), ids_b), ids_c),
    )


def _context(value: object) -> Context:
    """Map content's context, stripping citations from every prose field and the tone."""
    payload = _mapping(value) or {}
    evidence = normalise_ids(payload.get("evidence"))
    collected: list[str] = []
    fields: dict[str, str | None] = {}
    for name in ("genre", "setting", "audience", "purpose", "recorded_when", "recorded_where"):
        stripped, ids = _strip(str(payload.get(name) or ""))
        fields[name] = stripped or None
        collected = _merge_ids(collected, ids)
    tone: list[str] = []
    raw_tone = payload.get("tone")
    for entry in raw_tone if isinstance(raw_tone, list) else []:
        if not isinstance(entry, str):
            continue
        stripped, ids = _strip(entry)
        collected = _merge_ids(collected, ids)
        if stripped:
            tone.append(stripped)
    confidence = payload.get("confidence")
    return Context(
        genre=fields["genre"] or "unknown",
        setting=fields["setting"],
        audience=fields["audience"],
        purpose=fields["purpose"],
        tone=tone,
        recorded_when=fields["recorded_when"],
        recorded_where=fields["recorded_where"],
        evidence=_merge_ids(evidence, collected),
        confidence=_number(confidence, 0.0),
    )


def _chapters(value: object) -> list[Chapter]:
    """Map content's chapters, stripping citations from title and summary."""
    chapters: list[Chapter] = []
    for item in value if isinstance(value, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            chapter = Chapter.model_validate(payload)
        except ValidationError:
            continue
        title, ids_a = _strip(chapter.title)
        summary, ids_b = _strip(chapter.summary)
        chapter.title = title
        chapter.summary = summary
        chapter.evidence = _merge_ids(_merge_ids(chapter.evidence, ids_a), ids_b)
        chapters.append(chapter)
    return chapters


def _topics(value: object) -> list[Topic]:
    """Map content's topics, stripping citations from the label."""
    topics: list[Topic] = []
    for item in value if isinstance(value, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            topic = Topic.model_validate(payload)
        except ValidationError:
            continue
        label, ids = _strip(topic.label)
        topic.label = label
        topic.evidence = _merge_ids(topic.evidence, ids)
        topics.append(topic)
    return topics


def _entities(value: object) -> list[Entity]:
    """Map content's named entities unchanged (their text is not prose)."""
    entities: list[Entity] = []
    for item in value if isinstance(value, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            entities.append(Entity.model_validate(payload))
        except ValidationError:
            continue
    return entities


def _key_quotes(value: object) -> list[KeyQuote]:
    """Map content's quotes verbatim — quote text is never altered (verdict 26)."""
    quotes: list[KeyQuote] = []
    for item in value if isinstance(value, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            quotes.append(KeyQuote.model_validate(payload))
        except ValidationError:
            continue
    return quotes


def _claims(value: object) -> list[Claim]:
    """Map content's claims, stripping citations from the claim text."""
    claims: list[Claim] = []
    for item in value if isinstance(value, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            claim = Claim.model_validate(payload)
        except ValidationError:
            continue
        text, ids = _strip(claim.text)
        claim.text = text
        claim.evidence = _merge_ids(claim.evidence, ids)
        claims.append(claim)
    return claims


def _flag_list(value: object) -> list[ReviewFlag]:
    """Validated review flags from one checkpoint list."""
    flags: list[ReviewFlag] = []
    for item in value if isinstance(value, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            flags.append(ReviewFlag.model_validate(payload))
        except ValidationError:
            continue
    return flags


def _review_flags(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    content: Mapping[str, Any] | None,
    identity: Mapping[str, Any] | None,
) -> list[ReviewFlag]:
    """Flags the plan/content stages and the speakers' final owner produced.

    The verify checkpoint already carries the identity flags forward (plus the recomputed
    ambiguity and the web attribution notes), so when it exists it replaces the identity
    checkpoint's flags instead of duplicating them.
    """
    plan = _output(checkpoints, "plan")
    verify = _output(checkpoints, "verify")
    # A skipped checkpoint persists an empty output; only a verify run that actually wrote
    # its refined flags replaces the identity flags.
    verify_flags = verify if verify is not None and "flags" in verify else None
    flags: list[ReviewFlag] = []
    for source in (plan, content, verify_flags if verify_flags is not None else identity):
        if source is None:
            continue
        flags.extend(_flag_list(source.get("review_flags")))
        flags.extend(_flag_list(source.get("flags")))
    return flags


def _verifications(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
) -> list[Verification]:
    """The verify checkpoint's verdicts, validated for the report (docs/03 §5)."""
    verify = _output(checkpoints, "verify")
    raw = verify.get("verifications") if verify is not None else None
    verifications: list[Verification] = []
    for item in raw if isinstance(raw, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        try:
            verifications.append(Verification.model_validate(payload))
        except ValidationError:
            continue
    return verifications


def _stage_statuses(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    """``{stage: status}`` for every stage, in canonical order (docs/03 §5)."""
    from caracal.pipeline.stages import Stage

    statuses: dict[str, str] = {}
    for stage in Stage:
        name = stage.value
        override = overrides.get(name) if overrides is not None else None
        if isinstance(override, str) and override in _STAGE_STATUSES:
            statuses[name] = override
            continue
        checkpoint = checkpoints.get(name)
        raw = checkpoint.get("status") if isinstance(checkpoint, Mapping) else None
        statuses[name] = raw if isinstance(raw, str) and raw in _STAGE_STATUSES else "pending"
    statuses[Stage.ASSEMBLE.value] = "done"
    return statuses


def _chapter_coverage(chapters: Sequence[Chapter], duration_s: float) -> float:
    """Fraction of the audio covered by at least one chapter."""
    if duration_s <= 0.0 or not chapters:
        return 0.0
    spans = sorted(
        (max(0.0, chapter.start), min(duration_s, chapter.end))
        for chapter in chapters
        if chapter.end > chapter.start
    )
    covered = 0.0
    current_start = -1.0
    current_end = -1.0
    for start, end in spans:
        if end <= current_end:
            continue
        if start > current_end:
            covered += max(0.0, current_end - current_start)
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    covered += max(0.0, current_end - current_start)
    return min(1.0, covered / duration_s)


def _deterministic_flags(
    *,
    speakers: Sequence[SpeakerIdentity],
    chapters: Sequence[Chapter],
    duration_s: float,
    caption_agreement: float | None,
    low_confidence_segments: int,
    segment_count: int,
    stages: Mapping[str, str],
) -> list[ReviewFlag]:
    """The deterministic half of docs/04 §11."""
    flags: list[ReviewFlag] = []

    if speakers:
        share_sum = sum(speaker.share for speaker in speakers)
        if abs(share_sum - 1.0) > _SPEAKER_SHARE_TOLERANCE:
            flags.append(
                ReviewFlag(
                    severity="warn",
                    field="speakers",
                    message=f"speaker shares sum to {share_sum:.3f} (expected 1.0 ± 0.05)",
                )
            )

    coverage = _chapter_coverage(chapters, duration_s)
    if duration_s > 0.0 and coverage < _CHAPTER_COVERAGE_MIN:
        flags.append(
            ReviewFlag(
                severity="warn",
                field="chapters",
                message=f"chapters cover {coverage:.1%} of the audio (expected at least 80%)",
            )
        )

    if caption_agreement is not None and caption_agreement < _CAPTION_AGREEMENT_MIN:
        flags.append(
            ReviewFlag(
                severity="warn",
                field="quality.caption_agreement",
                message=f"caption agreement {caption_agreement:.2f} is below 0.6",
            )
        )

    if segment_count > 0:
        share = low_confidence_segments / segment_count
        if share > _LOW_CONFIDENCE_SHARE_MAX:
            flags.append(
                ReviewFlag(
                    severity="warn",
                    field="quality.low_confidence_segments",
                    message=(
                        f"{low_confidence_segments} of {segment_count} segments "
                        f"({share:.1%}) have avg_logprob < -1.0"
                    ),
                )
            )

    for name, status in stages.items():
        if status == "failed":
            flags.append(
                ReviewFlag(
                    severity="error",
                    field=f"stages.{name}",
                    message=f"stage {name} failed",
                )
            )

    return flags


def _quality(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    *,
    transcript: Transcript | None,
    content: Mapping[str, Any] | None,
    identity: Mapping[str, Any] | None,
    speakers: Sequence[SpeakerIdentity],
    chapters: Sequence[Chapter],
    duration_s: float,
    stage_statuses: Mapping[str, str],
) -> QualityInfo:
    """ASR signals, every stage status and the deterministic review flags."""
    segments = transcript.segments if transcript is not None else ()
    segment_count = len(segments)
    asr_avg_logprob = (
        sum(segment.avg_logprob for segment in segments) / segment_count if segment_count else 0.0
    )
    low_confidence = sum(1 for segment in segments if segment.avg_logprob < _LOW_CONFIDENCE_LOGPROB)
    langid = _output(checkpoints, "langid") or {}
    raw_agreement = langid.get("caption_agreement")
    caption_agreement = _number(raw_agreement) if isinstance(raw_agreement, (int, float)) else None

    flags = _review_flags(checkpoints, content, identity)
    flags.extend(
        _deterministic_flags(
            speakers=speakers,
            chapters=chapters,
            duration_s=duration_s,
            caption_agreement=caption_agreement,
            low_confidence_segments=low_confidence,
            segment_count=segment_count,
            stages=stage_statuses,
        )
    )
    return QualityInfo(
        asr_avg_logprob=asr_avg_logprob,
        low_confidence_segments=low_confidence,
        caption_agreement=caption_agreement,
        dropped_uncited=0,
        stages=dict(stage_statuses),
        review_flags=flags,
    )


def _transcript_ref(
    settings: Settings,
    job_id: uuid.UUID,
    transcript: Transcript | None,
) -> TranscriptRef:
    """Segment/word counts and the REST links to every transcript artefact."""
    segments = transcript.segments if transcript is not None else ()
    base = settings.CARACAL_PUBLIC_BASE_URL.rstrip("/")
    prefix = f"{base}/v1/jobs/{job_id}"
    return TranscriptRef(
        segments=len(segments),
        words=sum(len(segment.words) for segment in segments),
        links={
            "json": f"{prefix}/transcript?format=json",
            "txt": f"{prefix}/transcript?format=txt",
            "srt": f"{prefix}/subtitles/source.srt",
            "vtt": f"{prefix}/subtitles/source.vtt",
        },
    )


def _platform_metadata(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    job: JobRecord,
) -> Any:
    """A ``PlatformMetadata``-shaped evidence source (``meta:<field>`` ids)."""
    from caracal.ingest.ytdlp import PlatformMetadata

    ingest = _output(checkpoints, "ingest") or {}
    platform = _mapping(ingest.get("platform")) or {}
    title = _text(platform.get("title")) or _text(job.title)
    if not platform and title is None:
        return None
    return PlatformMetadata(
        id=_text(platform.get("id")),
        title=title,
        uploader=_text(platform.get("uploader")),
        channel=_text(platform.get("channel")),
        channel_url=_text(platform.get("channel_url")),
        upload_date=_text(platform.get("upload_date")),
        duration=_number(platform.get("duration")) if platform.get("duration") is not None else None,
        description=_text(platform.get("description")),
        tags=_string_list(platform.get("tags")),
        categories=_string_list(platform.get("categories")),
        chapters=[],
        language=_text(platform.get("language")),
        view_count=_integer(platform.get("view_count"), 0)
        if platform.get("view_count") is not None
        else None,
        webpage_url=_text(platform.get("webpage_url")),
        extractor=_text(platform.get("extractor")),
    )


def _hints(job: JobRecord) -> list[str]:
    """The free-text hints the analysis stages saw, in their order (``hint:<n>``)."""
    options = job.options if isinstance(job.options, Mapping) else {}
    hints: list[str] = []
    context_hint = _text(options.get("context_hint"))
    if context_hint is not None:
        hints.append(context_hint)
    hints.extend(_string_list(options.get("speaker_hints")))
    return hints


def _verify_web_hits(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
) -> list[dict[str, str]]:
    """The verify checkpoint's numbered hits, restricted to valid ``web:<n>`` ids.

    The verify stage numbers hits globally from 1 in search-call order (card T3.2), so the
    ids are taken from the checkpoint rather than re-derived from the list position.
    """
    verify = _output(checkpoints, "verify")
    raw = verify.get("hits") if verify is not None else None
    hits: list[dict[str, str]] = []
    for item in raw if isinstance(raw, list) else []:
        payload = _mapping(item)
        if payload is None:
            continue
        hit_id = _text(payload.get("id"))
        if hit_id is None or re.fullmatch(r"web:\d+", hit_id) is None:
            continue
        hits.append(
            {
                "id": hit_id,
                "url": _text(payload.get("url")) or "",
                "title": _text(payload.get("title")) or "",
                "snippet": _text(payload.get("snippet")) or "",
            }
        )
    return hits


def _one_based_web_ids(index: EvidenceIndex) -> EvidenceIndex:
    """Shift ``build_evidence_index``'s zero-based web ids to the documented ``web:1``.

    The verify stage numbers hits globally from 1 in search-call order (card T3.2), while
    ``build_evidence_index`` numbers the list it receives from 0. Re-keying keeps a
    verification source's ``web:<n>`` pointing at the hit it was judged against.
    """
    items = index.to_dict()
    shifted: dict[str, EvidenceItem] = {}
    for key, item in items.items():
        if not key.startswith("web:"):
            shifted[key] = item
            continue
        try:
            position = int(key.split(":", 1)[1])
        except ValueError:
            continue
        shifted[f"web:{position + 1}"] = item
    return EvidenceIndex(shifted)


def _evidence_index(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    *,
    job: JobRecord,
    transcript: Transcript | None,
    caption_words: Sequence[CaptionWord] | None,
) -> EvidenceIndex:
    """The evidence a report may cite, built from the persisted artifacts (docs/04 §9)."""
    normalize = _output(checkpoints, "normalize") or {}
    probe = _mapping(normalize.get("probe")) or {}
    raw_tags = _mapping(probe.get("tags")) or {}
    tags = {str(key): str(value) for key, value in raw_tags.items()}
    voiceprint = _output(checkpoints, "voiceprint") or {}
    matches = _mapping(voiceprint.get("matches")) or {}
    index = build_evidence_index(
        transcript=transcript,
        platform_meta=_platform_metadata(checkpoints, job),
        probe_tags=tags,
        hints=_hints(job),
        voiceprint_matches=cast("Mapping[str, Sequence[object]]", matches),
        caption_words=caption_words,
        web_hits=_verify_web_hits(checkpoints),
    )
    return _one_based_web_ids(index)


def assemble_report(
    checkpoints: Mapping[str, Mapping[str, Any] | None],
    *,
    job: JobRecord,
    settings: Settings,
    transcript: Transcript | None = None,
    caption_words: Sequence[CaptionWord] | None = None,
    stage_statuses: Mapping[str, str] | None = None,
) -> Report:
    """Build a complete, citation-validated ``Report`` from the stage checkpoints.

    ``checkpoints`` maps every stage name to its persisted checkpoint (or ``None``). Missing
    or failed stages yield empty sections and a stage status, never an exception. ``transcript``
    and ``caption_words`` are the two artifacts that are not stage checkpoints; the assemble
    stage loads them from the job directory and passes them in, keeping this function pure.
    """
    content = _output(checkpoints, "content")
    identity = _output(checkpoints, "identity")
    source = _source(checkpoints, job)
    duration_s = source.duration_s
    music = _music_detected(checkpoints)
    speakers = _speakers(checkpoints, transcript)
    chapters = _chapters(content.get("chapters") if content is not None else None)
    statuses = _stage_statuses(checkpoints, stage_statuses)

    report = Report(
        schema_version=SCHEMA_VERSION,
        job_id=job.id,
        created_at=datetime.now(UTC),
        pipeline_version=__version__,
        source=source,
        audio=_audio(checkpoints, duration_s),
        languages=_languages(checkpoints, music),
        speakers=speakers,
        summary=_summary(content.get("summary") if content is not None else None),
        context=_context(content.get("context") if content is not None else None),
        chapters=chapters,
        topics=_topics(content.get("topics") if content is not None else None),
        entities=_entities(content.get("entities") if content is not None else None),
        key_quotes=_key_quotes(content.get("key_quotes") if content is not None else None),
        claims=_claims(content.get("claims") if content is not None else None),
        verification=_verifications(checkpoints),
        translations=[],
        transcript_ref=_transcript_ref(settings, job.id, transcript),
        quality=_quality(
            checkpoints,
            transcript=transcript,
            content=content,
            identity=identity,
            speakers=speakers,
            chapters=chapters,
            duration_s=duration_s,
            stage_statuses=statuses,
        ),
    )

    index = _evidence_index(
        checkpoints,
        job=job,
        transcript=transcript,
        caption_words=caption_words,
    )
    validated, _dropped = validate_citations(report, index)
    return validated


__all__ = ["assemble_report", "strip_prose_citations"]
