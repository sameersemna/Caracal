"""Speaker identity resolution: mention extraction, deterministic verification and fusion.

The LLM is used for one thing only — reading the speaker-tagged transcript and answering
"which names are said, by whom, and in what relation" plus a role and a role/context-only
description per label (``identity_mentions.md``). Everything that decides an identity is
deterministic here (docs/04 §7):

* a ``self_intro`` or ``addressed`` mention is kept only when the name really occurs in the
  cited segments and those segments belong to the right speaker;
* ``referred`` and ``in_character`` names are never an identity source — an impressionist's
  "It's me, John." names the character, not the performer;
* candidates come from named, weighted sources and are fused with
  ``confidence = 1 - product(1 - score_i)`` capped at 0.98;
* a name is promoted only when it leads the runner-up by ≥ 0.2 and its confidence is ≥ 0.6,
  and metadata alone can only promote in the sole-speaker case;
* metadata names are linked to a label by evidence, never sprayed from the description;
* performance content (comedy, sketch, impressions, parody …) halves self-intro/addressed
  scores and flags the identities as uncertain;
* descriptions never infer protected attributes — a description that does is replaced by
  the role-only fallback.

The module is pure policy: it receives artifacts and an ``LLMClient`` and returns validated
report models. It never touches the filesystem, the database or the network.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from caracal.asr import Segment, Transcript
from caracal.config import Route, Settings
from caracal.db.repo import normalise_name
from caracal.errors import CaracalError
from caracal.llm.client import LLMClient
from caracal.llm.prompts import (
    load_prompt,
    render_metadata,
    render_transcript_lines,
    wrap_data,
)
from caracal.log import get_logger
from caracal.report.evidence import normalise_ids
from caracal.report.model import (
    Entity,
    ReviewFlag,
    SpeakerCandidate,
    SpeakerIdentity,
)
from caracal.speakers.voiceprint import MatchClass

logger = get_logger("caracal.analysis.identity")

IdentityRole = Literal[
    "host",
    "guest",
    "interviewer",
    "interviewee",
    "caller",
    "narrator",
    "presenter",
    "preacher",
    "moderator",
    "panelist",
    "other",
]
MentionRelation = Literal["self_intro", "addressed", "referred", "in_character"]
CandidateSourceName = Literal["self_intro", "addressed_by_name", "metadata", "voiceprint", "web", "hint"]

#: Every role the prompt may return, in the order the prompt lists them.
IDENTITY_ROLES: Final[tuple[IdentityRole, ...]] = (
    "host",
    "guest",
    "interviewer",
    "interviewee",
    "caller",
    "narrator",
    "presenter",
    "preacher",
    "moderator",
    "panelist",
    "other",
)

#: Base score per source (docs/04 §7).
SELF_INTRO_SCORE: Final = 0.85
SINGLE_NAME_SELF_INTRO_SCORE: Final = 0.55
ADDRESSED_SCORE: Final = 0.60
METADATA_SCORE: Final = 0.50
SOLE_SPEAKER_METADATA_SCORE: Final = 0.75
HINT_SCORE: Final = 0.70
BARE_HINT_SCORE: Final = 0.40
VOICEPRINT_CANDIDATE_FACTOR: Final = 0.60
#: The cross-verification stage's contributions (docs/04 §8 steps 5-6).
WEB_CONFIRM_SCORE: Final = 0.15
WEB_ATTRIBUTION_SCORE: Final = 0.50
MAX_CONFIDENCE: Final = 0.98
PROMOTION_MARGIN: Final = 0.20
PROMOTION_MIN_CONFIDENCE: Final = 0.60
SOLE_SPEAKER_SHARE: Final = 0.90
PERFORMANCE_SCORE_FACTOR: Final = 0.50
PERFORMANCE_FLAG: Final = "performance content: speaker identities uncertain"

#: Long transcripts send only each speaker's first segments plus the run-up to their turns.
MENTION_FIRST_SEGMENTS: Final = 40
MENTION_ADDRESS_LOOKBACK: Final = 3

_SOURCE_ORDER: Final[tuple[CandidateSourceName, ...]] = (
    "self_intro",
    "addressed_by_name",
    "metadata",
    "voiceprint",
    "web",
    "hint",
)

#: Protected attributes that must never reach a description (docs/04 §7).
_PROTECTED_WORDS: Final = (
    "male",
    "female",
    "man",
    "men",
    "woman",
    "women",
    "boy",
    "girl",
    "transgender",
    "nonbinary",
    "non-binary",
    "gay",
    "lesbian",
    "bisexual",
    "queer",
    "heterosexual",
    "young",
    "old",
    "elderly",
    "teenager",
    "child",
    "kid",
    "black",
    "white",
    "asian",
    "african",
    "caucasian",
    "latino",
    "latina",
    "hispanic",
    "arab",
    "indian",
    "muslim",
    "christian",
    "jewish",
    "hindu",
    "sikh",
    "atheist",
    "disabled",
    "blind",
    "deaf",
    "pregnant",
    "healthy",
)
_PROTECTED_RE: Final = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(word) for word in _PROTECTED_WORDS) + r")(?![A-Za-z])",
    re.IGNORECASE,
)

_ROLE_DESCRIPTION: Final[dict[IdentityRole, str]] = {
    "host": "hosts the recording",
    "guest": "appears as a guest",
    "interviewer": "interviews the other speakers",
    "interviewee": "is interviewed",
    "caller": "calls in to the recording",
    "narrator": "narrates the recording",
    "presenter": "presents the recording",
    "preacher": "preaches in the recording",
    "moderator": "moderates the discussion",
    "panelist": "speaks as a panelist",
    "other": "speaks in the recording",
}

_META_NAME_FIELDS: Final[tuple[str, ...]] = ("title", "description", "channel", "uploader")
_SOLE_NAME_FIELDS: Final[frozenset[str]] = frozenset({"title", "uploader", "channel"})
_CHANNEL_NAME_FIELDS: Final[frozenset[str]] = frozenset({"channel", "uploader"})
_HOST_ROLES: Final[frozenset[IdentityRole]] = frozenset({"host", "presenter"})

#: Genres/topics whose voices are performances, so a spoken name is not the speaker's own (docs/04 §7).
_PERFORMANCE_TERMS: Final[tuple[str, ...]] = (
    "comedy",
    "sketch",
    "impression",
    "parody",
    "satire",
    "skit",
    "prank",
)


class _LlmMention(BaseModel):
    """One name the model heard, with the speaker it belongs to and the cited segments."""

    model_config = ConfigDict(extra="ignore")

    speaker_label: str
    name: str
    relation: MentionRelation
    seg_ids: list[str] = Field(default_factory=list)


class _LlmSpeakerInfo(BaseModel):
    """The role and role/context-only description the model wrote for one label."""

    model_config = ConfigDict(extra="ignore")

    speaker_label: str
    role: IdentityRole = "other"
    description: str | None = None


class _LlmIdentity(BaseModel):
    """The identity extraction answer (JSON mode)."""

    model_config = ConfigDict(extra="ignore")

    mentions: list[_LlmMention] = Field(default_factory=list)
    speakers: list[_LlmSpeakerInfo] = Field(default_factory=list)


class IdentityResult(BaseModel):
    """The identity stage output: one ``SpeakerIdentity`` per label plus review flags."""

    model_config = ConfigDict(extra="forbid")

    speakers: list[SpeakerIdentity] = Field(default_factory=list)
    flags: list[ReviewFlag] = Field(default_factory=list)


@dataclass(frozen=True)
class VoiceprintMatch:
    """One voiceprint search hit for a diarized label (docs/04 §5)."""

    known_speaker_id: UUID
    name: str
    similarity: float
    match_class: MatchClass


@dataclass(frozen=True)
class WebSource:
    """One ``web`` source the verify stage feeds back into the fusion (docs/04 §8).

    ``replace`` names the existing candidate a confirmation expands (``Lindsay`` →
    ``Lindsay McMahon``). When it is ``None`` the source either strengthens a candidate
    with the same name or adds a new one at ``score`` (a quote attribution is 0.5 and can
    never be promoted on its own). ``flag`` becomes an informational review flag.
    """

    label: str
    name: str
    score: float
    evidence: Sequence[str]
    replace: str | None = None
    flag: str | None = None


@dataclass
class _Group:
    """All sources that point at one normalised name."""

    display: str
    norm: str
    scores: dict[CandidateSourceName, float] = field(default_factory=dict)
    evidence: dict[CandidateSourceName, list[str]] = field(default_factory=dict)
    known_speaker_id: UUID | None = None
    voiceprint_similarity: float | None = None
    match_similarity: float = -1.0

    def add(
        self,
        source: CandidateSourceName,
        score: float,
        evidence: Sequence[str],
        *,
        known_speaker_id: UUID | None = None,
        voiceprint_similarity: float | None = None,
        match_similarity: float | None = None,
    ) -> None:
        """Record a source; repeated sightings of one source keep its strongest score."""
        clamped = min(max(float(score), 0.0), 1.0)
        self.scores[source] = max(self.scores.get(source, 0.0), clamped)
        bucket = self.evidence.setdefault(source, [])
        for item in evidence:
            if item not in bucket:
                bucket.append(item)
        if source == "voiceprint" and known_speaker_id is not None:
            similarity = float(match_similarity if match_similarity is not None else clamped)
            if similarity > self.match_similarity:
                self.match_similarity = similarity
                self.known_speaker_id = known_speaker_id
                self.voiceprint_similarity = float(
                    voiceprint_similarity if voiceprint_similarity is not None else similarity
                )

    def merged_with(self, other: _Group) -> None:
        """Fold a first-name group into the full name it belongs to."""
        for source, score in other.scores.items():
            self.scores[source] = max(self.scores.get(source, 0.0), score)
            bucket = self.evidence.setdefault(source, [])
            for item in other.evidence.get(source, []):
                if item not in bucket:
                    bucket.append(item)
        if other.match_similarity > self.match_similarity:
            self.match_similarity = other.match_similarity
            self.known_speaker_id = other.known_speaker_id
            self.voiceprint_similarity = other.voiceprint_similarity


def _fold(text: str) -> str:
    """Case- and diacritic-insensitive comparison form; punctuation becomes a space."""
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    spaced = "".join(
        " " if not (char.isalnum() or char.isspace()) else char for char in without_marks.casefold()
    )
    return " ".join(spaced.split())


def _name_occurs(name: str, text: str) -> bool:
    """Whether ``name`` occurs in ``text``, allowing a first-name-only match (docs/04 §7)."""
    folded = _fold(name)
    if not folded:
        return False
    haystack = f" {_fold(text)} "
    if f" {folded} " in haystack:
        return True
    tokens = folded.split()
    return len(tokens) > 1 and len(tokens[0]) >= 2 and f" {tokens[0]} " in haystack


def _self_intro_score(name: str) -> float:
    """A spoken full name is strong evidence; a bare given name is weak (docs/04 §7)."""
    return SELF_INTRO_SCORE if len(_fold(name).split()) >= 2 else SINGLE_NAME_SELF_INTRO_SCORE


def _performance_content(*labels: str | Sequence[str]) -> bool:
    """Whether any genre or tone label names performance content (docs/04 §7)."""
    for label in labels:
        values = (label,) if isinstance(label, str) else label
        for value in values:
            text = value.casefold()
            if any(term in text for term in _PERFORMANCE_TERMS):
                return True
    return False


def _group_key(name: str) -> str:
    """The merge key for a name, or ``""`` when the model returned no usable name."""
    try:
        return normalise_name(name)
    except CaracalError:
        return ""


def _mentioned_segments(segments: Sequence[Segment], limit: int) -> list[Segment]:
    """The transcript subset the mention prompt may see (docs/04 §7, card T2.6).

    Short transcripts are sent whole. For long ones every speaker contributes its first
    ``MENTION_FIRST_SEGMENTS`` segments plus the ``MENTION_ADDRESS_LOOKBACK`` segments before
    each of its turns (where a name can be addressed to it), capped at ``limit`` characters
    by dropping the latest selected segments.
    """
    if limit <= 0:
        raise ValueError("LLM_CONTEXT_CHARS must be positive")
    ordered = list(segments)
    if len(render_transcript_lines(ordered)) <= limit:
        return ordered

    selected: set[int] = set()
    seen: dict[str, int] = {}
    for position, segment in enumerate(ordered):
        label = segment.speaker
        if label is None:
            continue
        count = seen.get(label, 0)
        if count < MENTION_FIRST_SEGMENTS:
            selected.add(position)
        seen[label] = count + 1

    previous: str | None = None
    for position, segment in enumerate(ordered):
        if segment.speaker is not None and segment.speaker != previous:
            selected.update(range(max(0, position - MENTION_ADDRESS_LOOKBACK), position))
        previous = segment.speaker

    chosen = [ordered[position] for position in sorted(selected)]
    while len(chosen) > 1 and len(render_transcript_lines(chosen)) > limit:
        chosen.pop()
    return chosen


def _cited_segment_ids(raw: Sequence[str]) -> list[int]:
    """The transcript ids a mention cites; anything that is not ``seg:<n>`` is ignored."""
    ids: list[int] = []
    for item in normalise_ids(list(raw)):
        if not item.startswith("seg:"):
            continue
        try:
            identifier = int(item[4:])
        except ValueError:
            continue
        if identifier not in ids:
            ids.append(identifier)
    return ids


def _turn_starts(segments: Sequence[Segment]) -> dict[str, list[int]]:
    """For every label, the positions at which one of its turns starts."""
    starts: dict[str, list[int]] = {}
    previous: str | None = None
    for position, segment in enumerate(segments):
        label = segment.speaker
        if label is not None and label != previous:
            starts.setdefault(label, []).append(position)
        previous = label
    return starts


def _addressed_positions(starts: Sequence[int]) -> set[int]:
    """The positions in the three segments before each of a label's turns."""
    allowed: set[int] = set()
    for start in starts:
        allowed.update(range(max(0, start - MENTION_ADDRESS_LOOKBACK), start))
    return allowed


def _verified_mention(
    mention: _LlmMention,
    *,
    label: str,
    segments: Sequence[Segment],
    positions: Mapping[int, int],
    starts: Mapping[str, list[int]],
) -> list[str] | None:
    """The cited ``seg:`` ids when the mention survives verification, else ``None``.

    ``self_intro``: every cited segment is spoken by the same label.
    ``addressed``: every cited segment belongs to a *different* label and sits in the three
    segments before one of the target's turns.
    In both cases the name must actually occur in the cited text (docs/04 §7).
    """
    ids = _cited_segment_ids(mention.seg_ids)
    if not ids:
        return None
    cited: list[Segment] = []
    cited_positions: list[int] = []
    for identifier in ids:
        position = positions.get(identifier)
        if position is None:
            return None
        cited.append(segments[position])
        cited_positions.append(position)

    if not _name_occurs(mention.name, " ".join(segment.text for segment in cited)):
        return None

    if mention.relation == "self_intro":
        if any(segment.speaker != label for segment in cited):
            return None
    elif mention.relation == "addressed":
        if any(segment.speaker is None or segment.speaker == label for segment in cited):
            return None
        window = _addressed_positions(starts.get(label, []))
        if any(position not in window for position in cited_positions):
            return None
    else:
        return None
    return [f"seg:{identifier}" for identifier in ids]


def _add_candidate(
    groups: dict[str, _Group],
    name: str,
    source: CandidateSourceName,
    score: float,
    evidence: Sequence[str],
    *,
    known_speaker_id: UUID | None = None,
    voiceprint_similarity: float | None = None,
    match_similarity: float | None = None,
) -> None:
    key = _group_key(name)
    display = " ".join(name.split())
    if not key or not display:
        return
    group = groups.get(key)
    if group is None:
        group = _Group(display=display, norm=key)
        groups[key] = group
    group.add(
        source,
        score,
        evidence,
        known_speaker_id=known_speaker_id,
        voiceprint_similarity=voiceprint_similarity,
        match_similarity=match_similarity,
    )


def _merge_first_names(groups: dict[str, _Group]) -> None:
    """Merge a bare first name into the one full name that starts with it (docs/04 §7)."""
    full = [group for group in groups.values() if len(group.norm.split()) > 1]
    for key in list(groups):
        single = groups.get(key)
        if single is None or len(single.norm.split()) != 1:
            continue
        matches = [group for group in full if group.norm.split()[0] == single.norm]
        if len(matches) != 1:
            continue
        matches[0].merged_with(single)
        del groups[key]


def _fuse(scores: Mapping[CandidateSourceName, float]) -> float:
    """``1 - product(1 - score_i)`` over independent sources, capped at 0.98 (docs/04 §7)."""
    remaining = 1.0
    for score in scores.values():
        remaining *= 1.0 - min(max(float(score), 0.0), 1.0)
    return min(round(1.0 - remaining, 6), MAX_CONFIDENCE)


def _candidate(group: _Group) -> SpeakerCandidate:
    sources = [source for source in _SOURCE_ORDER if source in group.scores]
    evidence: list[str] = []
    for source in sources:
        for item in group.evidence.get(source, []):
            if item not in evidence:
                evidence.append(item)
    return SpeakerCandidate(
        name=group.display,
        confidence=_fuse(group.scores),
        sources=sources,
        evidence=evidence,
    )


def _promote_speaker(speaker: SpeakerIdentity, *, sole: bool, flags: list[ReviewFlag]) -> None:
    """Apply the documented promotion rule to a speaker's already-fused candidates.

    This is the one place that decides whether a candidate becomes the speaker's ``name``
    (docs/04 §7): the top candidate must lead the runner-up by ``PROMOTION_MARGIN`` and be
    at least ``PROMOTION_MIN_CONFIDENCE``; a candidate backed only by metadata can promote
    in the sole-speaker case at ``SOLE_SPEAKER_METADATA_SCORE``. Two competing candidates
    that do not resolve add the "ambiguous identity" flag.
    """
    candidates = sorted(
        speaker.candidates,
        key=lambda candidate: (-candidate.confidence, candidate.name.casefold()),
    )
    speaker.candidates = candidates
    promoted: SpeakerCandidate | None = None
    if candidates:
        top = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        metadata_only = set(top.sources) <= {"metadata"}
        metadata_promotable = (
            sole and top.confidence >= SOLE_SPEAKER_METADATA_SCORE and "metadata" in top.sources
        )
        leads = second is None or top.confidence - second.confidence >= PROMOTION_MARGIN
        confident = top.confidence >= PROMOTION_MIN_CONFIDENCE
        if leads and confident and (not metadata_only or metadata_promotable):
            promoted = top
        elif len(candidates) >= 2:
            flags.append(
                ReviewFlag(
                    severity="warn",
                    field=f"speakers.{speaker.label}.name",
                    message="ambiguous identity",
                )
            )
    speaker.name = promoted.name if promoted is not None else None
    speaker.confidence = promoted.confidence if promoted is not None else 0.0


def _noisy_or(left: float, right: float) -> float:
    """Combine two independent source scores, capped at ``MAX_CONFIDENCE``."""
    combined = 1.0 - (1.0 - min(max(left, 0.0), 1.0)) * (1.0 - min(max(right, 0.0), 1.0))
    return min(round(combined, 6), MAX_CONFIDENCE)


def _merge_unique(existing: Sequence[str], extra: Sequence[str]) -> list[str]:
    """``existing`` followed by the unseen entries of ``extra``."""
    merged = list(existing)
    for item in extra:
        if item not in merged:
            merged.append(item)
    return merged


def _apply_web_source(speaker: SpeakerIdentity, source: WebSource) -> None:
    """Fold one web source into a speaker's candidate groups (docs/04 §8 step 5-6)."""
    display = " ".join(source.name.split())
    key = _group_key(display)
    if not key:
        return
    target: SpeakerCandidate | None = None
    if source.replace is not None:
        replace_key = _group_key(source.replace)
        target = next(
            (candidate for candidate in speaker.candidates if _group_key(candidate.name) == replace_key),
            None,
        )
        if target is None:
            return
    else:
        target = next(
            (candidate for candidate in speaker.candidates if _group_key(candidate.name) == key),
            None,
        )
    if target is None:
        target = SpeakerCandidate(name=display, confidence=0.0, sources=[], evidence=[])
        speaker.candidates.append(target)
    target.name = display
    target.confidence = _noisy_or(target.confidence, source.score)
    if "web" not in target.sources:
        target.sources.append("web")
    target.evidence = _merge_unique(target.evidence, source.evidence)
    speaker.evidence = _merge_unique(speaker.evidence, source.evidence)


def apply_web_sources(
    speakers: Sequence[SpeakerIdentity],
    flags: Sequence[ReviewFlag],
    sources: Sequence[WebSource],
) -> IdentityResult:
    """Re-run the deterministic identity fusion with the verify stage's web sources.

    The verify stage never rewrites the identity checkpoint (docs/04 §8 step 5): it hands the
    already-fused speakers back to this function, which adds each web source with the noisy-or
    rule (confirm +0.15, attribution 0.5), expands a given name to the full name found on the
    web, and re-applies the same promotion rule. ``ambiguous identity`` flags are recomputed
    from the refined candidates so confirming a name can resolve a previous ambiguity, while
    every other flag (performance content, ...) is carried through unchanged.
    """
    refined = [speaker.model_copy(deep=True) for speaker in speakers]
    by_label = {speaker.label: speaker for speaker in refined}
    for source in sources:
        speaker = by_label.get(source.label)
        if speaker is not None:
            _apply_web_source(speaker, source)

    kept = [flag for flag in flags if flag.message != "ambiguous identity"]
    stats: dict[str, Mapping[str, float | int]] = {
        speaker.label: {"share": speaker.share} for speaker in refined
    }
    sole = _sole_labels(stats)
    for speaker in refined:
        _promote_speaker(speaker, sole=speaker.label in sole, flags=kept)
    for source in sources:
        if source.flag:
            kept.append(
                ReviewFlag(
                    severity="info",
                    field=f"speakers.{source.label}.name",
                    message=source.flag,
                )
            )
    return IdentityResult(speakers=refined, flags=kept)


def _safe_description(info: _LlmSpeakerInfo | None, role: IdentityRole) -> str:
    """The model's role/context description, or the role-only fallback when it is unsafe."""
    fallback = _ROLE_DESCRIPTION[role]
    if info is None or info.description is None:
        return fallback
    text = " ".join(info.description.split())
    if not text or _PROTECTED_RE.search(text):
        return fallback
    return text


def _sole_labels(stats: Mapping[str, Mapping[str, float | int]]) -> set[str]:
    """The labels that own ≥ 90 % of talk time (the sole-speaker rule)."""
    if len(stats) <= 1:
        return set(stats)
    return {label for label, values in stats.items() if float(values.get("share", 0.0)) >= SOLE_SPEAKER_SHARE}


def _metadata_candidates(
    entities: Sequence[Entity],
    metadata: Mapping[str, str],
    *,
    stats: Mapping[str, Mapping[str, float | int]],
    sole: set[str],
    roles: Mapping[str, IdentityRole],
    label_text: Mapping[str, str],
    addressed_names: Mapping[str, Sequence[str]],
) -> list[tuple[str, str, float, list[str]]]:
    """``(label, name, score, evidence)`` for person names actually linked to a label.

    A metadata name is a candidate only when the label is the sole speaker, when the name
    occurs in that label's own speech or in a verified ``addressed`` mention of it, or when
    the name is the channel/uploader and the label is the host/presenter (docs/04 §7). A
    description full of guests therefore never sprays every name onto one label.
    """
    found: list[tuple[str, str, float, list[str]]] = []
    for entity in entities:
        if entity.type != "person":
            continue
        matched = [
            field
            for field in _META_NAME_FIELDS
            if field in metadata and _name_occurs(entity.text, metadata[field])
        ]
        if not matched:
            continue
        evidence = [f"meta:{field}" for field in matched]
        sole_name = any(field in _SOLE_NAME_FIELDS for field in matched)
        channel_name = any(field in _CHANNEL_NAME_FIELDS for field in matched)
        for label in stats:
            linked = (
                label in sole
                or _name_occurs(entity.text, label_text.get(label, ""))
                or any(_name_occurs(entity.text, name) for name in addressed_names.get(label, ()))
                or (channel_name and roles.get(label) in _HOST_ROLES)
            )
            if not linked:
                continue
            score = SOLE_SPEAKER_METADATA_SCORE if (label in sole and sole_name) else METADATA_SCORE
            found.append((label, entity.text, score, evidence))
    return _dedupe_metadata_names(found)


def _dedupe_metadata_names(
    entries: Sequence[tuple[str, str, float, list[str]]],
) -> list[tuple[str, str, float, list[str]]]:
    """Keep only the fuller name when one metadata name is a token subset of another."""
    token_sets = [(label, set(_fold(name).split())) for label, name, _score, _evidence in entries]
    kept: list[tuple[str, str, float, list[str]]] = []
    for position, (label, name, score, evidence) in enumerate(entries):
        tokens = token_sets[position][1]
        if any(other_label == label and tokens < other_tokens for other_label, other_tokens in token_sets):
            continue
        kept.append((label, name, score, evidence))
    return kept


def _hint_candidates(
    hints: Sequence[str],
    stats: Mapping[str, Mapping[str, float | int]],
    roles: Mapping[str, IdentityRole],
) -> list[tuple[str, str, float, list[str]]]:
    """``(label, name, score, evidence)`` for the caller's ``speaker_hints``."""
    found: list[tuple[str, str, float, list[str]]] = []
    labels = list(stats)
    for index, raw in enumerate(hints):
        text = raw.strip()
        if not text:
            continue
        evidence = [f"hint:{index}"]
        if ":" in text:
            role_text, name = text.split(":", 1)
            role = role_text.strip().casefold()
            name = name.strip()
            if not name or role not in IDENTITY_ROLES:
                continue
            for label in labels:
                if roles.get(label) == role:
                    found.append((label, name, HINT_SCORE, evidence))
            continue
        if len(labels) == 1:
            found.append((labels[0], text, HINT_SCORE, evidence))
        else:
            for label in labels:
                found.append((label, text, BARE_HINT_SCORE, evidence))
    return found


async def _extract(
    llm: LLMClient,
    *,
    transcript: Transcript,
    metadata: Mapping[str, str],
    hints: Sequence[str],
    report_language: str,
    context_chars: int,
    profile_genre: str | None,
    content_genre: str | None,
    content_tone: Sequence[str],
) -> _LlmIdentity:
    """One JSON-mode call: mentions plus the role/description for every label."""
    segments = _mentioned_segments(transcript.segments, context_chars)
    hint_lines = "\n".join(f"[hint:{index}] {hint}" for index, hint in enumerate(hints))
    context_lines: list[str] = []
    if profile_genre:
        context_lines.append(f"[plan:likely_genre] {profile_genre}")
    if content_genre:
        context_lines.append(f"[content:genre] {content_genre}")
    if content_tone:
        context_lines.append(f"[content:tone] {', '.join(content_tone)}")
    prompt = load_prompt(
        "identity_mentions",
        metadata=wrap_data("metadata", render_metadata(dict(metadata))),
        hints=wrap_data("hints", hint_lines),
        transcript=wrap_data("transcript", render_transcript_lines(segments)),
        content_context=wrap_data("content_context", "\n".join(context_lines)),
        roles=", ".join(IDENTITY_ROLES),
        report_language=report_language,
    )
    result = await llm.call_tool(
        prompt=prompt,
        system=(
            "You are Caracal's identity analyst. Use only the DATA blocks, cite the segment ids "
            "for every name you report, and never infer protected attributes."
        ),
        model_cls=_LlmIdentity,
        tool_name="record_identity_mentions",
        tool_description="Record every name mention and a role/description for each speaker label.",
        mode="json",
        route=Route.IDENTITY,
    )
    return result.value


async def resolve_identities(
    transcript: Transcript,
    metadata: Mapping[str, str],
    hints: Sequence[str],
    entities: Sequence[Entity],
    voiceprint_matches: Mapping[str, Sequence[VoiceprintMatch]],
    stats: Mapping[str, Mapping[str, float | int]],
    settings: Settings,
    *,
    llm: LLMClient,
    profile_genre: str | None = None,
    content_genre: str | None = None,
    content_tone: Sequence[str] = (),
) -> IdentityResult:
    """Resolve every diarized label against its candidates (docs/04 §7).

    ``stats`` is :func:`caracal.speakers.align.speaker_stats` output; the talk time, share and
    first appearance in each ``SpeakerIdentity`` come from it. ``profile_genre`` and the content
    ``genre``/``tone`` only gate the performance-content rule. The returned ``flags`` are the
    "ambiguous identity" and performance warnings the report's review section carries.
    """
    if not stats:
        return IdentityResult()

    performance = _performance_content(profile_genre or "", content_genre or "", content_tone)
    extracted = await _extract(
        llm,
        transcript=transcript,
        metadata=metadata,
        hints=hints,
        report_language=settings.CARACAL_REPORT_LANGUAGE,
        context_chars=settings.LLM_CONTEXT_CHARS,
        profile_genre=profile_genre,
        content_genre=content_genre,
        content_tone=content_tone,
    )
    infos = {info.speaker_label: info for info in extracted.speakers if info.speaker_label in stats}
    roles: dict[str, IdentityRole] = {
        label: infos[label].role if label in infos else "other" for label in stats
    }

    positions = {segment.id: position for position, segment in enumerate(transcript.segments)}
    starts = _turn_starts(transcript.segments)
    sole = _sole_labels(stats)

    groups_by_label: dict[str, dict[str, _Group]] = {label: {} for label in stats}
    addressed_names: dict[str, list[str]] = {label: [] for label in stats}

    for mention in extracted.mentions:
        label = mention.speaker_label
        if label not in groups_by_label:
            continue
        verified = _verified_mention(
            mention,
            label=label,
            segments=transcript.segments,
            positions=positions,
            starts=starts,
        )
        if verified is None:
            continue
        if mention.relation == "self_intro":
            score = (
                _self_intro_score(mention.name) * PERFORMANCE_SCORE_FACTOR
                if performance
                else _self_intro_score(mention.name)
            )
            _add_candidate(groups_by_label[label], mention.name, "self_intro", score, verified)
        elif mention.relation == "addressed":
            score = ADDRESSED_SCORE * PERFORMANCE_SCORE_FACTOR if performance else ADDRESSED_SCORE
            _add_candidate(
                groups_by_label[label],
                mention.name,
                "addressed_by_name",
                score,
                verified,
            )
            addressed_names[label].append(mention.name)

    for label in stats:
        for match in voiceprint_matches.get(label, ()):
            if match.match_class == "match":
                _add_candidate(
                    groups_by_label[label],
                    match.name,
                    "voiceprint",
                    match.similarity,
                    [f"vp:{label}"],
                    known_speaker_id=match.known_speaker_id,
                    voiceprint_similarity=match.similarity,
                    match_similarity=match.similarity,
                )
            else:
                _add_candidate(
                    groups_by_label[label],
                    match.name,
                    "voiceprint",
                    match.similarity * VOICEPRINT_CANDIDATE_FACTOR,
                    [f"vp:{label}"],
                )

    label_text = {
        label: "\n".join(segment.text for segment in transcript.segments if segment.speaker == label)
        for label in stats
    }
    for label, name, score, meta_evidence in _metadata_candidates(
        entities,
        metadata,
        stats=stats,
        sole=sole,
        roles=roles,
        label_text=label_text,
        addressed_names=addressed_names,
    ):
        if label in groups_by_label:
            _add_candidate(groups_by_label[label], name, "metadata", score, meta_evidence)

    for label, name, score, hint_evidence in _hint_candidates(hints, stats, roles):
        if label in groups_by_label:
            _add_candidate(groups_by_label[label], name, "hint", score, hint_evidence)

    identities: list[SpeakerIdentity] = []
    flags: list[ReviewFlag] = []
    if performance:
        flags.append(ReviewFlag(severity="warn", field="speakers", message=PERFORMANCE_FLAG))
    for label, values in stats.items():
        groups = groups_by_label[label]
        _merge_first_names(groups)
        candidates = sorted(
            (_candidate(group) for group in groups.values()),
            key=lambda candidate: (-candidate.confidence, candidate.name.casefold()),
        )

        all_evidence: list[str] = []
        for candidate in candidates:
            for item in candidate.evidence:
                if item not in all_evidence:
                    all_evidence.append(item)
        # A speaker identity always cites the label's first segment, so it stays addressable
        # even when every name hypothesis was dropped (docs/04 §9).
        first_segment = next(
            (f"seg:{segment.id}" for segment in transcript.segments if segment.speaker == label),
            None,
        )
        if first_segment is not None and first_segment not in all_evidence:
            all_evidence.append(first_segment)

        matched_group = max(
            (group for group in groups.values() if group.known_speaker_id is not None),
            key=lambda group: group.match_similarity,
            default=None,
        )

        identity = SpeakerIdentity(
            label=label,
            talk_time_s=float(values.get("talk_time_s", 0.0)),
            share=float(values.get("share", 0.0)),
            first_seen_s=float(values.get("first_seen_s", 0.0)),
            name=None,
            confidence=0.0,
            role=roles.get(label),
            candidates=candidates,
            known_speaker_id=matched_group.known_speaker_id if matched_group is not None else None,
            voiceprint_similarity=(
                matched_group.voiceprint_similarity if matched_group is not None else None
            ),
            description=_safe_description(infos.get(label), roles[label]),
            evidence=all_evidence,
        )
        _promote_speaker(identity, sole=label in sole, flags=flags)
        identities.append(identity)

    logger.info(
        "identity resolved",
        extra={
            "speakers": len(identities),
            "named": sum(1 for identity in identities if identity.name is not None),
            "flags": len(flags),
        },
    )
    return IdentityResult(speakers=identities, flags=flags)


__all__ = [
    "ADDRESSED_SCORE",
    "IDENTITY_ROLES",
    "MAX_CONFIDENCE",
    "MENTION_ADDRESS_LOOKBACK",
    "MENTION_FIRST_SEGMENTS",
    "METADATA_SCORE",
    "PERFORMANCE_FLAG",
    "PROMOTION_MARGIN",
    "PROMOTION_MIN_CONFIDENCE",
    "SELF_INTRO_SCORE",
    "SINGLE_NAME_SELF_INTRO_SCORE",
    "SOLE_SPEAKER_METADATA_SCORE",
    "WEB_ATTRIBUTION_SCORE",
    "WEB_CONFIRM_SCORE",
    "IdentityResult",
    "IdentityRole",
    "VoiceprintMatch",
    "WebSource",
    "apply_web_sources",
    "resolve_identities",
]
