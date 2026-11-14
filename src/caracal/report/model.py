"""The report schema (docs/03 §5) — the one artefact every interface returns.

Field names and enum members are exactly those of docs/03 §5; every model forbids unknown
fields, every confidence is constrained to ``[0, 1]`` and every timestamp/duration is a
float in seconds. Sections a stage may not have produced (verification when ``verify`` is
skipped, translations, chapters, ...) default to empty lists, so a partial report is still a
valid report.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from caracal.report.evidence import EvidenceItem

#: The only schema version this model tree describes (docs/03 §5).
SCHEMA_VERSION = "1.0"

EntityType = Literal["person", "org", "place", "date", "event", "work", "other"]
ClaimKind = Literal["fact", "opinion", "prediction", "quote"]
Verdict = Literal["supported", "contradicted", "unverified", "mixed"]
Severity = Literal["info", "warn", "error"]
TranslationStatus = Literal["done", "failed"]
CandidateSource = Literal["self_intro", "addressed_by_name", "metadata", "voiceprint", "web", "hint"]
AudioQuality = Literal["good", "fair", "poor"]
SourceKind = Literal["upload", "direct", "platform"]


class _ReportModel(BaseModel):
    """Every report model rejects unknown fields (docs/03 §5)."""

    model_config = ConfigDict(extra="forbid")


class ReportSource(_ReportModel):
    """Where the media came from (docs/03 §5 ``source``)."""

    kind: SourceKind
    url: str | None = None
    title: str | None = None
    uploader: str | None = None
    channel: str | None = None
    upload_date: str | None = None
    platform: str | None = None
    duration_s: float
    description_excerpt: str | None = None
    original_format: str
    sha256: str


class AudioInfo(_ReportModel):
    """What the audio itself measures (docs/03 §5 ``audio``)."""

    duration_s: float
    sample_rate: int
    channels: int
    loudness_lufs: float | None = None
    speech_ratio: float
    music_detected: bool | None = None
    quality: AudioQuality
    notes: list[str] = Field(default_factory=list)


class LanguageSpan(_ReportModel):
    """One contiguous run of a language on the timeline."""

    start: float
    end: float


class Language(_ReportModel):
    """One detected language with its share of the speech."""

    code: str
    name: str
    share: float
    spans: list[LanguageSpan] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class SpeakerCandidate(_ReportModel):
    """One name hypothesis for a diarized label, with where it came from."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[CandidateSource] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class SpeakerIdentity(_ReportModel):
    """One diarized speaker, its name hypotheses and its voiceprint link."""

    label: str
    talk_time_s: float
    share: float
    first_seen_s: float
    name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    role: str | None = None
    candidates: list[SpeakerCandidate] = Field(default_factory=list)
    known_speaker_id: uuid.UUID | None = None
    voiceprint_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    description: str | None = None
    evidence: list[str] = Field(default_factory=list)


class Summary(_ReportModel):
    """The layered summary (docs/03 §5)."""

    one_line: str
    short: str
    detailed: str
    evidence: list[str] = Field(default_factory=list)


class Context(_ReportModel):
    """Genre, setting and tone inferred from the evidence (docs/03 §5)."""

    genre: str
    setting: str | None = None
    audience: str | None = None
    purpose: str | None = None
    tone: list[str] = Field(default_factory=list)
    recorded_when: str | None = None
    recorded_where: str | None = None
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class Chapter(_ReportModel):
    """One chapter of the recording."""

    start: float
    end: float
    title: str
    summary: str
    evidence: list[str] = Field(default_factory=list)


class Topic(_ReportModel):
    """One topic with its relative weight."""

    label: str
    weight: float
    evidence: list[str] = Field(default_factory=list)


class Entity(_ReportModel):
    """A named entity and the segments that mention it."""

    text: str
    type: EntityType
    mentions: list[str] = Field(default_factory=list)
    canonical: str | None = None
    url: str | None = None
    evidence: list[str] = Field(default_factory=list)


class KeyQuote(_ReportModel):
    """A notable verbatim quote, addressed by its segment."""

    seg_id: str
    speaker: str
    text: str
    start: float
    why_notable: str


class Claim(_ReportModel):
    """A checkworthy (or explicitly not checkworthy) assertion."""

    id: str
    text: str
    speaker: str
    seg_ids: list[str] = Field(default_factory=list)
    kind: ClaimKind
    checkworthy: bool
    evidence: list[str] = Field(default_factory=list)


class VerificationSource(_ReportModel):
    """One search hit that was used to judge a verification target."""

    url: str
    title: str = ""
    snippet: str = ""
    evidence_id: str


class Verification(_ReportModel):
    """The web verdict for one claim or speaker hypothesis (docs/04 §8)."""

    target: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    sources: list[VerificationSource] = Field(default_factory=list)


class ReviewFlag(_ReportModel):
    """One self-review finding; it never edits the report (docs/04 §11)."""

    severity: Severity
    field: str
    message: str


class Translation(_ReportModel):
    """One requested subtitle translation."""

    lang: str
    status: TranslationStatus
    links: dict[str, str] = Field(default_factory=dict)


class TranscriptRef(_ReportModel):
    """Where the transcript artefacts are and how big they are."""

    segments: int
    words: int
    links: dict[str, str] = Field(default_factory=dict)


class QualityInfo(_ReportModel):
    """ASR quality signals, stage statuses and review flags."""

    asr_avg_logprob: float
    low_confidence_segments: int
    caption_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    dropped_uncited: int = 0
    stages: dict[str, str] = Field(default_factory=dict)
    review_flags: list[ReviewFlag] = Field(default_factory=list)


class Citation(_ReportModel):
    """One transcript segment cited by an answer."""

    seg_id: str
    start: float
    text: str


class Answer(_ReportModel):
    """The answer to one question about a finished job (docs/03 §2 ``ask``)."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class KnownSpeaker(_ReportModel):
    """An enrolled voice (docs/03 §3 ``list_speakers`` / ``enroll_speaker``)."""

    id: uuid.UUID
    name: str
    notes: str | None = None
    samples: int
    created_at: datetime
    updated_at: datetime


class SummarySpeaker(_ReportModel):
    """The speaker stub inside :class:`ReportSummary` (docs/03 §3)."""

    label: str
    name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class ReportSummary(_ReportModel):
    """The small report view MCP ``analyze_audio`` returns inline (docs/03 §3)."""

    title: str
    duration_s: float
    languages: list[str] = Field(default_factory=list)
    speakers: list[SummarySpeaker] = Field(default_factory=list)
    summary: str
    genre: str


class Report(_ReportModel):
    """The complete, evidence-backed audio intelligence report (docs/03 §5)."""

    schema_version: Literal["1.0"] = "1.0"
    job_id: uuid.UUID
    created_at: datetime
    pipeline_version: str
    source: ReportSource
    audio: AudioInfo
    languages: list[Language] = Field(default_factory=list)
    speakers: list[SpeakerIdentity] = Field(default_factory=list)
    summary: Summary
    context: Context
    chapters: list[Chapter] = Field(default_factory=list)
    topics: list[Topic] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    verification: list[Verification] = Field(default_factory=list)
    translations: list[Translation] = Field(default_factory=list)
    transcript_ref: TranscriptRef
    quality: QualityInfo
    evidence_index: dict[str, EvidenceItem] = Field(default_factory=dict)

    def summary_view(self) -> ReportSummary:
        """The compact view of this report (docs/03 §3 ``ReportSummary``)."""
        return ReportSummary(
            title=self.source.title or "",
            duration_s=self.audio.duration_s,
            languages=[language.code for language in self.languages],
            speakers=[
                SummarySpeaker(label=speaker.label, name=speaker.name, confidence=speaker.confidence)
                for speaker in self.speakers
            ],
            summary=self.summary.one_line,
            genre=self.context.genre,
        )


__all__ = [
    "SCHEMA_VERSION",
    "Answer",
    "AudioInfo",
    "AudioQuality",
    "CandidateSource",
    "Chapter",
    "Citation",
    "Claim",
    "ClaimKind",
    "Context",
    "Entity",
    "EntityType",
    "KeyQuote",
    "KnownSpeaker",
    "Language",
    "LanguageSpan",
    "QualityInfo",
    "Report",
    "ReportSource",
    "ReportSummary",
    "ReviewFlag",
    "Severity",
    "SourceKind",
    "SpeakerCandidate",
    "SpeakerIdentity",
    "Summary",
    "SummarySpeaker",
    "Topic",
    "TranscriptRef",
    "Translation",
    "TranslationStatus",
    "Verdict",
    "Verification",
    "VerificationSource",
]
