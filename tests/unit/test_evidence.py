"""Evidence index, id normalisation and the generic citation validator (docs/04 §9).

The fixtures are the real recordings the ASR tests use: the two overlapping TED chunks
(merged with :func:`merge_chunks`, exactly as the pipeline does), the manual English
captions and a ``PlatformMetadata`` shaped like the yt-dlp ingest checkpoint.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from tests.fakes.whisper import load_fixture

from caracal.asr import Transcript
from caracal.asr.captions import CaptionWord, parse_vtt
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.ingest.ytdlp import PlatformMetadata
from caracal.report.evidence import (
    EvidenceIndex,
    EvidenceItem,
    build_evidence_index,
    normalise_ids,
    validate_citations,
)
from caracal.report.model import (
    AudioInfo,
    Chapter,
    Claim,
    Context,
    QualityInfo,
    Report,
    ReportSource,
    ReviewFlag,
    Summary,
    TranscriptRef,
    Verification,
    VerificationSource,
)

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
JOB_ID = uuid.UUID("6f5c5f4e-0d5b-4a3e-9d1e-2f6b0f3f9a10")
CREATED_AT = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class _VoiceMatch:
    """Stand-in for the voiceprint stage's match record (name + cosine similarity)."""

    name: str
    similarity: float


def _transcript() -> Transcript:
    """The real overlapping TED chunks merged exactly as the pipeline merges them."""
    chunks = (Chunk(index=0, start_s=0.0, end_s=61.0), Chunk(index=1, start_s=60.0, end_s=120.0))
    merged = merge_chunks(
        [parse_whisper_response(load_fixture("talk_c0")), parse_whisper_response(load_fixture("talk_c1"))],
        chunks,
    )
    segments = tuple(
        replace(segment, speaker="S1") if segment.id == 0 else segment for segment in merged.segments
    )
    return replace(merged, segments=segments)


def _platform_meta(description: str | None = None) -> PlatformMetadata:
    return PlatformMetadata(
        id="8jPQjjsBbIc",
        title="How to stay calm when you know you'll be stressed",
        uploader="TED",
        channel="TED",
        channel_url="https://www.youtube.com/@TED",
        upload_date="20151124",
        duration=120.0,
        description=description or "Neuroscientist Daniel Levitin offers strategies for stress.",
        tags=["psychology", "stress"],
        categories=["Education"],
        chapters=[],
        language="en",
        view_count=1000,
        webpage_url="https://www.youtube.com/watch?v=8jPQjjsBbIc",
        extractor="youtube",
    )


def _caption_words() -> list[CaptionWord]:
    return parse_vtt((FIXTURES / "captions" / "talk_en_manual.vtt").read_text(encoding="utf-8"))


def _report() -> Report:
    return Report(
        job_id=JOB_ID,
        created_at=CREATED_AT,
        pipeline_version="0.1.0",
        source=ReportSource(kind="upload", duration_s=10.0, original_format="wav", sha256="a" * 64),
        audio=AudioInfo(duration_s=10.0, sample_rate=16000, channels=1, speech_ratio=0.9, quality="good"),
        summary=Summary(one_line="x", short="y", detailed="z", evidence=["seg:1"]),
        context=Context(genre="talk", confidence=0.9, evidence=["seg:1"]),
        transcript_ref=TranscriptRef(segments=3, words=9),
        quality=QualityInfo(asr_avg_logprob=-0.1, low_confidence_segments=0),
    )


def _index() -> EvidenceIndex:
    index = EvidenceIndex()
    for segment_id in (1, 2, 3):
        index.add(
            f"seg:{segment_id}",
            kind="seg",
            ref=f"{segment_id}.0-{segment_id + 1}.0s S1",
            excerpt=f"segment {segment_id}",
        )
    index.add("web:0", kind="web", ref="https://example.org/hit", excerpt="a search hit")
    return index


# --- the index itself -------------------------------------------------------------------


def test_index_contains_every_artifact_kind() -> None:
    transcript = _transcript()
    captions = _caption_words()
    index = build_evidence_index(
        transcript=transcript,
        platform_meta=_platform_meta(),
        probe_tags={"artist": "Daniel Levitin", "TAG Name!": "x"},
        hints=["Levitin is the speaker"],
        voiceprint_matches={"S1": [_VoiceMatch("Someone Else", 0.41), _VoiceMatch("Daniel Levitin", 0.93)]},
        caption_words=captions,
        web_hits=[{"url": "https://example.org/stress", "title": "Stress", "snippet": "A snippet."}],
    )

    expected = {f"seg:{segment.id}" for segment in transcript.segments}
    expected |= {
        "meta:title",
        "meta:description",
        "meta:channel",
        "meta:uploader",
        "meta:upload_date",
        "meta:tags",
        "meta:categories",
        "meta:tag_artist",
        "meta:tag_tag_name",
        "hint:0",
        "vp:S1",
        "web:0",
    }
    expected |= {f"caption:{number}" for number in range(math.ceil(len(captions) / 10))}

    assert set(index.ids()) == expected
    assert len(index) == len(expected)


def test_segment_item_carries_the_time_range_speaker_and_text() -> None:
    index = build_evidence_index(transcript=_transcript())

    first = index.get("seg:0")
    assert first is not None
    assert first.kind == "seg"
    assert first.ref == "0.2-15.4s S1"
    assert first.excerpt == "A few years ago, I broke into my own house."
    assert "seg:0" in index
    assert index.get("seg:999") is None


def test_metadata_and_probe_tag_items() -> None:
    index = build_evidence_index(platform_meta=_platform_meta(), probe_tags={"artist": "Daniel Levitin"})

    title = index.get("meta:title")
    assert title is not None
    assert (title.kind, title.ref, title.excerpt) == (
        "meta",
        "title",
        "How to stay calm when you know you'll be stressed",
    )
    tags = index.get("meta:tags")
    assert tags is not None
    assert tags.excerpt == "psychology, stress"
    tag = index.get("meta:tag_artist")
    assert tag is not None
    assert (tag.ref, tag.excerpt) == ("tag artist", "Daniel Levitin")


def test_hint_voiceprint_caption_and_web_items() -> None:
    captions = _caption_words()
    index = build_evidence_index(
        hints=["The speaker introduces himself"],
        voiceprint_matches={"S1": [_VoiceMatch("Someone Else", 0.41), _VoiceMatch("Daniel Levitin", 0.93)]},
        caption_words=captions,
        web_hits=[{"url": "https://example.org/stress", "title": "Stress", "snippet": "A snippet."}],
    )

    hint = index.get("hint:0")
    assert hint is not None
    assert (hint.kind, hint.excerpt) == ("hint", "The speaker introduces himself")

    voiceprint = index.get("vp:S1")
    assert voiceprint is not None
    assert voiceprint.kind == "vp"
    assert voiceprint.ref == "Daniel Levitin (0.930)"
    assert voiceprint.excerpt == "voiceprint match for S1 at cosine similarity 0.930"

    caption = index.get("caption:0")
    assert caption is not None
    assert caption.kind == "caption"
    assert caption.excerpt == " ".join(word.text for word in captions[:10])
    assert caption.ref == f"{captions[0].start:.1f}-{captions[9].end:.1f}s"
    assert index.get(f"caption:{math.ceil(len(captions) / 10) - 1}") is not None
    assert index.get(f"caption:{math.ceil(len(captions) / 10)}") is None

    web = index.get("web:0")
    assert web is not None
    assert (web.kind, web.ref, web.excerpt) == ("web", "https://example.org/stress", "Stress — A snippet.")


def test_empty_arguments_build_an_empty_index() -> None:
    index = build_evidence_index()
    assert len(index) == 0
    assert index.ids() == []
    assert build_evidence_index(transcript=None, platform_meta=None, probe_tags={}, hints=[]).ids() == []


def test_excerpts_are_whitespace_collapsed_and_capped() -> None:
    description = "first line\nsecond   line " + "x" * 400
    index = build_evidence_index(platform_meta=_platform_meta(description=description))

    item = index.get("meta:description")
    assert item is not None
    assert item.excerpt.startswith("first line second line x")
    assert "\n" not in item.excerpt
    assert len(item.excerpt) == 300

    direct = EvidenceItem(kind="seg", ref="0.0-1.0s", excerpt="  a\n\n b  ")
    assert direct.excerpt == "a b"


# --- id normalisation -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("seg:2.", ["seg:2"]),
        ("seg:2 S1", ["seg:2"]),
        ("seg:2, seg:5", ["seg:2", "seg:5"]),
        ("vp:S1", ["vp:S1"]),
        ("meta:title", ["meta:title"]),
        ("garbage", []),
        ("see segment two", []),
        ("[seg:7] S3 00:05.0 | hi", ["seg:7"]),
        ("caption:4", ["caption:4"]),
        ("hint:0", ["hint:0"]),
        ("web:3", ["web:3"]),
        ("vp:SPEAKER_00", []),
        ("meta:Title", []),
        (["seg:2", "seg:2", "web:3"], ["seg:2", "web:3"]),
        (["web:3", "seg:1"], ["web:3", "seg:1"]),
        (["seg:2."], ["seg:2"]),
        ([42, "seg:1"], ["seg:1"]),
        (None, []),
    ],
)
def test_normalise_ids(value: object, expected: list[str]) -> None:
    assert normalise_ids(value) == expected


# --- citation validation ----------------------------------------------------------------


def test_unknown_ids_are_dropped_but_the_item_stays() -> None:
    report = _report()
    report.claims = [
        Claim(
            id="c1",
            text="Minus 40 is where the two scales meet.",
            speaker="S1",
            seg_ids=["seg:7"],
            kind="fact",
            checkworthy=True,
            evidence=["seg:1", "seg:99", "web:0."],
        )
    ]

    validated, dropped = validate_citations(report, _index())

    assert validated.claims[0].evidence == ["seg:1", "web:0"]
    assert dropped == 0


def test_a_list_item_with_no_evidence_left_is_removed_and_counted() -> None:
    report = _report()
    report.chapters = [
        Chapter(start=0.0, end=1.0, title="kept", summary="s", evidence=["seg:2"]),
        Chapter(start=1.0, end=2.0, title="dropped", summary="s", evidence=["seg:99"]),
    ]

    validated, dropped = validate_citations(report, _index())

    assert [chapter.title for chapter in validated.chapters] == ["kept"]
    assert dropped == 1
    assert validated.quality.dropped_uncited == 1


def test_a_singular_uncited_object_is_kept_flagged_and_zeroed() -> None:
    report = _report()
    report.summary = Summary(one_line="x", short="y", detailed="z", evidence=["seg:99"])
    report.context = Context(genre="talk", confidence=0.9, evidence=["seg:99"])

    validated, dropped = validate_citations(report, _index())

    assert validated.summary.evidence == []
    assert validated.context.evidence == []
    assert validated.context.confidence == 0.0
    assert dropped == 0
    assert ReviewFlag(severity="warn", field="summary", message="uncited") in validated.quality.review_flags
    assert ReviewFlag(severity="warn", field="context", message="uncited") in validated.quality.review_flags


def test_unresolved_verification_sources_are_dropped_and_the_verdict_becomes_unverified() -> None:
    report = _report()
    report.verification = [
        Verification(
            target="claim:c1",
            verdict="supported",
            confidence=0.9,
            rationale="a hit confirmed it",
            sources=[
                VerificationSource(url="https://example.org", title="t", snippet="s", evidence_id="web:9")
            ],
        )
    ]

    validated, dropped = validate_citations(report, _index())

    assert validated.verification[0].sources == []
    assert validated.verification[0].verdict == "unverified"
    assert dropped == 1


def test_a_resolved_verification_source_keeps_the_verdict() -> None:
    report = _report()
    report.verification = [
        Verification(
            target="claim:c1",
            verdict="supported",
            confidence=0.9,
            rationale="a hit confirmed it",
            sources=[
                VerificationSource(url="https://example.org", title="t", snippet="s", evidence_id="web:0")
            ],
        )
    ]

    validated, _ = validate_citations(report, _index())

    assert validated.verification[0].verdict == "supported"
    assert validated.verification[0].sources[0].evidence_id == "web:0"


def test_evidence_index_is_exactly_the_cited_ids_in_citation_order() -> None:
    report = _report()
    report.chapters = [Chapter(start=0.0, end=1.0, title="t", summary="s", evidence=["seg:2"])]
    report.claims = [
        Claim(
            id="c1",
            text="t",
            speaker="S1",
            seg_ids=["seg:1"],
            kind="fact",
            checkworthy=True,
            evidence=["seg:1", "seg:99"],
        )
    ]
    report.verification = [
        Verification(
            target="claim:c1",
            verdict="supported",
            confidence=0.9,
            rationale="r",
            sources=[
                VerificationSource(url="https://example.org", title="t", snippet="s", evidence_id="web:0")
            ],
        )
    ]

    validated, _ = validate_citations(report, _index())

    assert list(validated.evidence_index) == ["seg:1", "seg:2", "web:0"]
    assert validated.evidence_index["seg:1"] == EvidenceItem(
        kind="seg", ref="1.0-2.0s S1", excerpt="segment 1"
    )


def test_validation_is_idempotent() -> None:
    report = _report()
    report.summary = Summary(one_line="x", short="y", detailed="z", evidence=["seg:99"])
    report.chapters = [Chapter(start=0.0, end=1.0, title="gone", summary="s", evidence=["seg:99"])]
    report.verification = [
        Verification(
            target="claim:c1",
            verdict="supported",
            confidence=0.9,
            rationale="r",
            sources=[
                VerificationSource(url="https://example.org", title="t", snippet="s", evidence_id="web:9")
            ],
        )
    ]

    first, first_dropped = validate_citations(report, _index())
    snapshot = first.model_dump()
    second, second_dropped = validate_citations(first, _index())

    assert first_dropped == 2
    assert second_dropped == 0
    assert second.model_dump() == snapshot
    assert second.quality.dropped_uncited == 2
    warnings = [flag for flag in second.quality.review_flags if flag.field == "summary"]
    assert len(warnings) == 1


class _NestedEvidence(BaseModel):
    """A field the validator has no special knowledge of."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[str] = []


class _ChapterWithNote(Chapter):
    """Test-only extension of a report model with one extra nested evidence field."""

    model_config = ConfigDict(extra="forbid")

    note: _NestedEvidence = _NestedEvidence()


def test_walk_is_generic_and_handles_a_field_it_was_never_told_about() -> None:
    report = _report()
    chapter = _ChapterWithNote(
        start=0.0,
        end=1.0,
        title="t",
        summary="s",
        evidence=["seg:1"],
        note=_NestedEvidence(evidence=["seg:99", "seg:2."]),
    )
    report.chapters = [chapter]

    validated, _ = validate_citations(report, _index())

    note = validated.chapters[0].note
    assert isinstance(note, _NestedEvidence)
    assert note.evidence == ["seg:2"]
