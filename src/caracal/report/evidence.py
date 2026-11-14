"""Evidence index, evidence-id normalisation and the citation validator (docs/04 §9).

Evidence ids are **typed** — ``seg:12``, ``meta:title``, ``web:3``, ``hint:0``, ``vp:S1``,
``caption:4`` — so sentence punctuation can never become part of an id. The measured gemma4
behaviour (docs/04 §1) copies whole rendered transcript prefixes (``"seg:2 S1"``) and
``"seg:2."``; only the typed token itself is an id (verdict 17 carry-forward).

The index is built deterministically from the pipeline artifacts a job already has, and
:func:`validate_citations` walks the report generically through pydantic introspection, so a
new evidence-bearing field is handled without teaching the validator about it. The module is
deliberately light at import time (no whisper/yt-dlp/torch imports) because
``caracal.llm.schema`` reuses :data:`EVIDENCE_ID_RE`; the artifact types are imported only for
type checking and read through attributes at runtime.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from caracal.asr import Transcript
    from caracal.asr.captions import CaptionWord
    from caracal.ingest.ytdlp import PlatformMetadata
    from caracal.report.model import Report

#: One typed evidence id embedded in arbitrary LLM or transcript text.
EVIDENCE_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(?:seg:\d+|caption:\d+|web:\d+|hint:\d+|vp:S\d+|meta:[a-z][a-z0-9_]*)"
)

#: The evidence kinds an index entry can have (docs/04 §9).
EvidenceKind = Literal["seg", "meta", "web", "hint", "vp", "caption"]

#: Hard cap on an evidence excerpt, applied after whitespace collapsing (docs/03 §5).
EXCERPT_MAX_CHARS: Final = 300

#: Platform metadata fields that become ``meta:<field>`` evidence (docs/04 §9).
META_FIELDS: Final = (
    "title",
    "description",
    "channel",
    "uploader",
    "upload_date",
    "tags",
    "categories",
)

#: Caption words are grouped into cue-sized evidence items (docs/04 §9).
CAPTION_GROUP_WORDS: Final = 10

_TAG_KEY_RE: Final = re.compile(r"[^a-z0-9_]+")


def normalise_ids(values: str | Iterable[object] | None) -> list[str]:
    """Flatten ``values`` into typed evidence ids, de-duplicated, keeping first-seen order.

    Anything that is not a typed id is discarded: ``"seg:2."`` becomes ``"seg:2"``,
    ``"seg:2 S1"`` becomes ``"seg:2"``, ``"garbage"`` becomes ``[]`` (docs/04 §1, §9).
    """
    if values is None:
        return []
    candidates: list[object]
    if isinstance(values, str):
        candidates = [values]
    elif isinstance(values, Iterable):
        candidates = list(values)
    else:
        candidates = [values]

    ids: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        for found in EVIDENCE_ID_RE.findall(candidate):
            if found in seen:
                continue
            seen.add(found)
            ids.append(found)
    return ids


class EvidenceItem(BaseModel):
    """One addressable fact: its kind, a human-readable locator and a short excerpt."""

    model_config = ConfigDict(extra="forbid")

    kind: EvidenceKind
    ref: str
    excerpt: str = ""

    @field_validator("ref", "excerpt", mode="before")
    @classmethod
    def _collapse_whitespace(cls, value: object) -> str:
        """Collapse every whitespace run (including newlines) to one space."""
        if value is None:
            return ""
        return " ".join(str(value).split())

    @field_validator("excerpt")
    @classmethod
    def _truncate(cls, value: str) -> str:
        """Keep the excerpt at the documented 300-character cap."""
        return value[:EXCERPT_MAX_CHARS]


class EvidenceIndex:
    """The evidence a job can cite, keyed by typed id (docs/04 §9)."""

    def __init__(self, items: Mapping[str, EvidenceItem] | None = None) -> None:
        self._items: dict[str, EvidenceItem] = dict(items or {})

    def add(self, item_id: str, *, kind: EvidenceKind, ref: str, excerpt: str = "") -> None:
        """Record one evidence item under ``item_id`` (later additions replace earlier ones)."""
        self._items[item_id] = EvidenceItem(kind=kind, ref=ref, excerpt=excerpt)

    def get(self, item_id: str) -> EvidenceItem | None:
        """The item stored under ``item_id``, or ``None``."""
        return self._items.get(item_id)

    def __contains__(self, item_id: object) -> bool:
        return isinstance(item_id, str) and item_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    def ids(self) -> list[str]:
        """Every indexed id, in insertion order."""
        return list(self._items)

    def to_dict(self) -> dict[str, EvidenceItem]:
        """A shallow copy of the id → item mapping (report ``evidence_index``)."""
        return dict(self._items)


def _field(obj: object, name: str) -> object:
    """Read ``name`` from a mapping or an object, returning ``None`` when absent."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ", ".join(_as_text(item) for item in cast("Sequence[object]", value))
    return str(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _tag_slug(key: object) -> str | None:
    """Turn an ffprobe tag key into the ``meta:tag_<key>`` suffix, or ``None``."""
    slug = _TAG_KEY_RE.sub("_", str(key).strip().lower()).strip("_")
    if not slug or not slug[0].isalpha():
        return None
    return slug


def _segment_ref(start: float, end: float, speaker: str | None) -> str:
    ref = f"{start:.1f}-{end:.1f}s"
    return f"{ref} {speaker}" if speaker else ref


def build_evidence_index(
    *,
    transcript: Transcript | None = None,
    platform_meta: PlatformMetadata | None = None,
    probe_tags: Mapping[str, str] | None = None,
    hints: Sequence[str] | None = None,
    voiceprint_matches: Mapping[str, Sequence[object]] | None = None,
    caption_words: Sequence[CaptionWord] | None = None,
    web_hits: Sequence[object] | None = None,
) -> EvidenceIndex:
    """Build the index a report may cite (docs/04 §9), tolerating any missing artifact.

    ``voiceprint_matches`` maps a speaker label to that label's matches (objects or mappings
    with ``name`` and ``similarity``); the best match by similarity becomes ``vp:<label>``.
    ``web_hits`` are objects or mappings with ``url``, ``title`` and ``snippet``.
    """
    index = EvidenceIndex()

    if transcript is not None:
        for segment in transcript.segments:
            index.add(
                f"seg:{segment.id}",
                kind="seg",
                ref=_segment_ref(segment.start, segment.end, segment.speaker),
                excerpt=segment.text,
            )

    if platform_meta is not None:
        for field in META_FIELDS:
            value = getattr(platform_meta, field, None)
            text = _as_text(value)
            if text:
                index.add(f"meta:{field}", kind="meta", ref=field, excerpt=text)

    for key, value in (probe_tags or {}).items():
        slug = _tag_slug(key)
        if slug is None:
            continue
        text = _as_text(value)
        if text:
            index.add(f"meta:tag_{slug}", kind="meta", ref=f"tag {key}", excerpt=text)

    for position, hint in enumerate(hints or []):
        text = _as_text(hint)
        if text:
            index.add(f"hint:{position}", kind="hint", ref=f"hint {position}", excerpt=text)

    for label, matches in (voiceprint_matches or {}).items():
        candidates = list(matches)
        if not candidates:
            continue
        best = max(candidates, key=lambda match: _as_float(_field(match, "similarity")))
        name = _as_text(_field(best, "name"))
        similarity = _as_float(_field(best, "similarity"))
        index.add(
            f"vp:{label}",
            kind="vp",
            ref=f"{name} ({similarity:.3f})" if name else f"similarity {similarity:.3f}",
            excerpt=f"voiceprint match for {label} at cosine similarity {similarity:.3f}",
        )

    words = list(caption_words or [])
    for group_number, start in enumerate(range(0, len(words), CAPTION_GROUP_WORDS)):
        group = words[start : start + CAPTION_GROUP_WORDS]
        index.add(
            f"caption:{group_number}",
            kind="caption",
            ref=f"{group[0].start:.1f}-{group[-1].end:.1f}s",
            excerpt=" ".join(word.text for word in group),
        )

    for position, hit in enumerate(web_hits or []):
        url = _as_text(_field(hit, "url"))
        title = _as_text(_field(hit, "title"))
        snippet = _as_text(_field(hit, "snippet"))
        excerpt = f"{title} — {snippet}" if title and snippet else title or snippet
        index.add(f"web:{position}", kind="web", ref=url, excerpt=excerpt)

    return index


class _Stats:
    """Mutable counters for one validation run."""

    def __init__(self) -> None:
        self.dropped = 0
        self.cited: list[str] = []


def validate_citations(report: Report, index: EvidenceIndex) -> tuple[Report, int]:
    """Drop every citation that does not resolve, mutating ``report`` in place (docs/04 §9).

    The walk is generic: any ``evidence: list[str]`` field is normalised and filtered, any
    list item left without evidence is removed and counted, and any singular evidence-bearing
    object that is left uncited keeps its place but loses its confidence and gains an
    ``uncited`` review flag. A list item whose scalar ``evidence_id`` does not resolve (a
    verification source) is removed; a verdict left without sources becomes ``unverified``
    (docs/04 §8). ``report.evidence_index`` is then exactly the ids the report cites, and
    ``quality.dropped_uncited`` grows by the number of dropped items.

    Running it twice changes nothing further: every remaining id resolves, no item is dropped
    again, the ``uncited`` flag is not duplicated and ``dropped_uncited`` only grows.
    """
    from caracal.report.model import ReviewFlag

    stats = _Stats()
    _walk(report, "", index, stats, report, ReviewFlag)
    resolved: dict[str, EvidenceItem] = {}
    for evidence_id in stats.cited:
        item = index.get(evidence_id)
        if item is not None:
            resolved[evidence_id] = item
    report.evidence_index = resolved
    report.quality.dropped_uncited += stats.dropped
    return report, stats.dropped


def _read_field(model: BaseModel, name: str) -> object:
    """Read a pydantic field by name (the walk is generic, mypy only sees ``BaseModel``)."""
    return getattr(model, name, None)


def _set_field(model: BaseModel, name: str, value: object) -> None:
    """Write a pydantic field by name (same reason as :func:`_read_field`)."""
    setattr(model, name, value)


def _walk(
    model: BaseModel,
    path: str,
    index: EvidenceIndex,
    stats: _Stats,
    root: Report,
    review_flag_cls: type[Any],
) -> None:
    """Normalise every evidence field reachable from ``model`` and enforce the verdict rule."""
    for name in type(model).model_fields:
        if name == "evidence_index":
            continue
        value = getattr(model, name)
        if name == "evidence" and isinstance(value, list):
            kept = [evidence_id for evidence_id in normalise_ids(value) if evidence_id in index]
            value[:] = kept
            stats.cited.extend(kept)
            continue
        if name == "evidence_id" and isinstance(value, str):
            resolved = next((evidence_id for evidence_id in normalise_ids(value) if evidence_id in index), "")
            if resolved != value:
                setattr(model, name, resolved)
            if resolved:
                stats.cited.append(resolved)
            continue
        _walk_field(value, f"{path}.{name}" if path else name, index, stats, root, review_flag_cls)

    verdict = _read_field(model, "verdict")
    sources = _read_field(model, "sources")
    if isinstance(verdict, str) and isinstance(sources, list) and verdict != "unverified" and not sources:
        _set_field(model, "verdict", "unverified")


def _walk_field(
    value: object,
    path: str,
    index: EvidenceIndex,
    stats: _Stats,
    root: Report,
    review_flag_cls: type[Any],
) -> None:
    if isinstance(value, BaseModel):
        _walk(value, path, index, stats, root, review_flag_cls)
        if "evidence" in type(value).model_fields and not _read_field(value, "evidence"):
            _flag_uncited(root, value, path, review_flag_cls)
    elif isinstance(value, list):
        _walk_list(value, path, index, stats, root, review_flag_cls)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, BaseModel):
                _walk_field(item, f"{path}.{key}", index, stats, root, review_flag_cls)


def _walk_list(
    items: list[Any],
    path: str,
    index: EvidenceIndex,
    stats: _Stats,
    root: Report,
    review_flag_cls: type[Any],
) -> None:
    kept: list[Any] = []
    for position, item in enumerate(items):
        if isinstance(item, BaseModel):
            _walk(item, f"{path}[{position}]", index, stats, root, review_flag_cls)
            if _should_drop(item):
                stats.dropped += 1
                continue
        kept.append(item)
    items[:] = kept


def _should_drop(item: BaseModel) -> bool:
    """True when a list item has lost all of its evidence (docs/04 §9)."""
    fields = type(item).model_fields
    if "evidence" in fields:
        return not _read_field(item, "evidence")
    if "evidence_id" in fields:
        return not _read_field(item, "evidence_id")
    return False


def _flag_uncited(root: Report, model: BaseModel, path: str, review_flag_cls: type[Any]) -> None:
    """Zero the confidence of an uncited singular object and flag it once."""
    if "confidence" in type(model).model_fields:
        _set_field(model, "confidence", 0.0)
    flags = root.quality.review_flags
    for flag in flags:
        if flag.severity == "warn" and flag.field == path and flag.message == "uncited":
            return
    flags.append(review_flag_cls(severity="warn", field=path, message="uncited"))


__all__ = [
    "CAPTION_GROUP_WORDS",
    "EVIDENCE_ID_RE",
    "EXCERPT_MAX_CHARS",
    "META_FIELDS",
    "EvidenceIndex",
    "EvidenceItem",
    "EvidenceKind",
    "build_evidence_index",
    "normalise_ids",
    "validate_citations",
]
