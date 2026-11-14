"""Report schema (docs/03 §5): exactly the documented fields, JSON round trip, summary view.

The example report is built from real facts of the TED fixture used by the ASR tests
(``tests/fixtures/whisper/talk_c0.json`` + ``talk_c1.json``): Daniel Levitin, Montreal,
minus 40 where the two temperature scales meet.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel, ValidationError

from caracal.report.evidence import EvidenceItem
from caracal.report.model import (
    SCHEMA_VERSION,
    Answer,
    AudioInfo,
    Chapter,
    Citation,
    Claim,
    Context,
    Entity,
    KeyQuote,
    KnownSpeaker,
    Language,
    LanguageSpan,
    QualityInfo,
    Report,
    ReportSource,
    ReportSummary,
    ReviewFlag,
    SpeakerCandidate,
    SpeakerIdentity,
    Summary,
    Topic,
    TranscriptRef,
    Translation,
    Verification,
    VerificationSource,
)

JOB_ID = uuid.UUID("6f5c5f4e-0d5b-4a3e-9d1e-2f6b0f3f9a10")
CREATED_AT = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
SHA256 = "a3f1" + "0" * 60


def _full_report() -> Report:
    """A report that populates every section docs/03 §5 defines."""
    return Report(
        schema_version="1.0",
        job_id=JOB_ID,
        created_at=CREATED_AT,
        pipeline_version="0.1.0",
        source=ReportSource(
            kind="platform",
            url="https://www.youtube.com/watch?v=8jPQjjsBbIc",
            title="How to stay calm when you know you'll be stressed",
            uploader="TED",
            channel="TED",
            upload_date="20151124",
            platform="youtube",
            duration_s=120.0,
            description_excerpt="Neuroscientist Daniel Levitin offers strategies for stress.",
            original_format="webm",
            sha256=SHA256,
        ),
        audio=AudioInfo(
            duration_s=120.0,
            sample_rate=16000,
            channels=1,
            loudness_lufs=-18.4,
            speech_ratio=0.91,
            music_detected=False,
            quality="good",
            notes=["trimmed to the first 120 s of the talk"],
        ),
        languages=[
            Language(
                code="en",
                name="English",
                share=1.0,
                spans=[LanguageSpan(start=0.0, end=120.0)],
                confidence=0.997,
            )
        ],
        speakers=[
            SpeakerIdentity(
                label="S1",
                talk_time_s=112.5,
                share=1.0,
                first_seen_s=0.18,
                name="Daniel Levitin",
                confidence=0.86,
                role="narrator",
                candidates=[
                    SpeakerCandidate(
                        name="Daniel Levitin",
                        confidence=0.86,
                        sources=["self_intro", "metadata"],
                        evidence=["seg:1", "meta:title"],
                    )
                ],
                known_speaker_id=None,
                voiceprint_similarity=0.93,
                description="adult male, calm, appears to be the speaker of the talk",
                evidence=["seg:0", "seg:1"],
            )
        ],
        summary=Summary(
            one_line="A neuroscientist explains how to prepare for stressful moments.",
            short="Levitin tells the story of breaking into his own house and draws lessons about stress.",
            detailed="Daniel Levitin describes a cold Montreal night, a broken door and the police, "
            "then turns the anecdote into practical advice about anticipating stress.",
            evidence=["seg:0", "seg:7"],
        ),
        context=Context(
            genre="talk",
            setting="stage",
            audience="general public",
            purpose="explain a practical technique",
            tone=["calm", "humorous"],
            recorded_when="2015",
            recorded_where="TED stage",
            evidence=["seg:0", "meta:title"],
            confidence=0.78,
        ),
        chapters=[
            Chapter(
                start=0.18,
                end=61.0,
                title="Breaking into my own house",
                summary="A cold night in Montreal and a door that would not open.",
                evidence=["seg:0", "seg:6"],
            ),
            Chapter(
                start=61.0,
                end=120.0,
                title="Pre-mortems and the two temperature scales",
                summary="Why anticipating failure reduces stress.",
                evidence=["seg:7", "seg:48"],
            ),
        ],
        topics=[
            Topic(label="stress", weight=0.82, evidence=["seg:7"]),
            Topic(label="memory", weight=0.41, evidence=["seg:12"]),
        ],
        entities=[
            Entity(
                text="Daniel Levitin",
                type="person",
                mentions=["seg:1"],
                canonical="Daniel Levitin",
                url="https://en.wikipedia.org/wiki/Daniel_Levitin",
                evidence=["seg:1"],
            ),
            Entity(text="Montreal", type="place", mentions=["seg:2"], evidence=["seg:2"]),
        ],
        key_quotes=[
            KeyQuote(
                seg_id="seg:0",
                speaker="S1",
                text="A few years ago, I broke into my own house.",
                start=0.18,
                why_notable="Opening anecdote that frames the talk.",
            )
        ],
        claims=[
            Claim(
                id="c1",
                text="Minus 40 is where the two temperature scales meet.",
                speaker="S1",
                seg_ids=["seg:7"],
                kind="fact",
                checkworthy=True,
                evidence=["seg:7"],
            )
        ],
        verification=[
            Verification(
                target="claim:c1",
                verdict="supported",
                confidence=0.88,
                rationale="Reference tables confirm the scales meet at minus 40.",
                sources=[
                    VerificationSource(
                        url="https://example.org/temperature-scales",
                        title="Celsius and Fahrenheit",
                        snippet="The scales cross at -40 degrees.",
                        evidence_id="web:0",
                    )
                ],
            )
        ],
        translations=[
            Translation(lang="de", status="done", links={"srt": "out/de.srt", "vtt": "out/de.vtt"})
        ],
        transcript_ref=TranscriptRef(
            segments=49,
            words=812,
            links={"json": "out/transcript.json", "txt": "out/transcript.txt"},
        ),
        quality=QualityInfo(
            asr_avg_logprob=-0.42,
            low_confidence_segments=3,
            caption_agreement=0.91,
            dropped_uncited=2,
            stages={"transcribe": "done", "verify": "done", "translate": "done"},
            review_flags=[ReviewFlag(severity="warn", field="summary", message="uncited")],
        ),
        evidence_index={
            "seg:0": EvidenceItem(
                kind="seg", ref="0.2-15.4s S1", excerpt="A few years ago, I broke into my own house."
            ),
            "seg:7": EvidenceItem(
                kind="seg", ref="30.1-34.2s S1", excerpt="Minus 40 is where the two scales meet."
            ),
            "web:0": EvidenceItem(
                kind="web", ref="https://example.org/temperature-scales", excerpt="Celsius and Fahrenheit"
            ),
        },
    )


def test_full_report_round_trips_json_identically() -> None:
    report = _full_report()
    payload = report.model_dump_json()
    reparsed = Report.model_validate_json(payload)

    assert reparsed.model_dump_json() == payload
    assert reparsed == report


def test_documented_top_level_fields_are_exactly_these() -> None:
    assert set(Report.model_fields) == {
        "schema_version",
        "job_id",
        "created_at",
        "pipeline_version",
        "source",
        "audio",
        "languages",
        "speakers",
        "summary",
        "context",
        "chapters",
        "topics",
        "entities",
        "key_quotes",
        "claims",
        "verification",
        "translations",
        "transcript_ref",
        "quality",
        "evidence_index",
    }
    assert set(Summary.model_fields) == {"one_line", "short", "detailed", "evidence"}
    assert set(Context.model_fields) == {
        "genre",
        "setting",
        "audience",
        "purpose",
        "tone",
        "recorded_when",
        "recorded_where",
        "evidence",
        "confidence",
    }
    assert set(SpeakerIdentity.model_fields) == {
        "label",
        "talk_time_s",
        "share",
        "first_seen_s",
        "name",
        "confidence",
        "role",
        "candidates",
        "known_speaker_id",
        "voiceprint_similarity",
        "description",
        "evidence",
    }
    assert set(Claim.model_fields) == {"id", "text", "speaker", "seg_ids", "kind", "checkworthy", "evidence"}
    assert set(Verification.model_fields) == {"target", "verdict", "confidence", "rationale", "sources"}
    assert set(VerificationSource.model_fields) == {"url", "title", "snippet", "evidence_id"}
    assert set(ReviewFlag.model_fields) == {"severity", "field", "message"}
    assert set(Answer.model_fields) == {"answer", "confidence", "citations", "evidence"}
    assert set(Citation.model_fields) == {"seg_id", "start", "text"}


def test_extra_field_is_rejected() -> None:
    payload = _full_report().model_dump()
    payload["unexpected"] = 1

    with pytest.raises(ValidationError) as caught:
        Report.model_validate(payload)

    assert caught.value.errors()[0]["type"] == "extra_forbidden"


def test_extra_field_is_rejected_in_a_nested_model() -> None:
    with pytest.raises(ValidationError) as caught:
        Summary(one_line="x", short="y", detailed="z", extra="nope")

    assert caught.value.errors()[0]["loc"] == ("extra",)


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (Verification, {"target": "claim:c1", "verdict": "supported", "confidence": 1.2, "rationale": "x"}),
        (Verification, {"target": "claim:c1", "verdict": "supported", "confidence": -0.1, "rationale": "x"}),
        (
            SpeakerIdentity,
            {"label": "S1", "talk_time_s": 1.0, "share": 1.0, "first_seen_s": 0.0, "confidence": 1.2},
        ),
        (Language, {"code": "en", "name": "English", "share": 1.0, "confidence": 1.01}),
        (Context, {"genre": "talk", "confidence": 2.0}),
    ],
)
def test_confidence_outside_zero_one_is_rejected(model: type[BaseModel], payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_schema_version_is_the_literal_one_point_zero() -> None:
    assert SCHEMA_VERSION == "1.0"
    payload = _full_report().model_dump()
    payload["schema_version"] = "2.0"

    with pytest.raises(ValidationError):
        Report.model_validate(payload)


def test_sections_a_stage_may_not_produce_default_to_empty_lists() -> None:
    report = Report(
        job_id=JOB_ID,
        created_at=CREATED_AT,
        pipeline_version="0.1.0",
        source=ReportSource(kind="upload", duration_s=11.0, original_format="wav", sha256=SHA256),
        audio=AudioInfo(duration_s=11.0, sample_rate=16000, channels=1, speech_ratio=0.9, quality="good"),
        summary=Summary(one_line="x", short="y", detailed="z"),
        context=Context(genre="speech", confidence=0.5),
        transcript_ref=TranscriptRef(segments=1, words=2),
        quality=QualityInfo(asr_avg_logprob=-0.1, low_confidence_segments=0),
    )

    dumped = report.model_dump()
    for section in ("languages", "speakers", "chapters", "topics", "entities", "key_quotes"):
        assert dumped[section] == []
    assert dumped["claims"] == []
    assert dumped["verification"] == []
    assert dumped["translations"] == []
    assert dumped["evidence_index"] == {}
    assert dumped["quality"]["review_flags"] == []


def test_summary_view_fields() -> None:
    view = _full_report().summary_view()

    assert isinstance(view, ReportSummary)
    assert view.model_dump() == {
        "title": "How to stay calm when you know you'll be stressed",
        "duration_s": 120.0,
        "languages": ["en"],
        "speakers": [{"label": "S1", "name": "Daniel Levitin", "confidence": 0.86}],
        "summary": "A neuroscientist explains how to prepare for stressful moments.",
        "genre": "talk",
    }


def test_summary_view_tolerates_a_missing_title_and_name() -> None:
    report = _full_report()
    report.source.title = None
    report.speakers[0].name = None

    view = report.summary_view()

    assert view.title == ""
    assert view.speakers[0].name is None


def test_evidence_index_entries_are_the_documented_shape() -> None:
    entry = _full_report().evidence_index["seg:0"]
    assert entry.model_dump() == {
        "kind": "seg",
        "ref": "0.2-15.4s S1",
        "excerpt": "A few years ago, I broke into my own house.",
    }
    with pytest.raises(ValidationError):
        EvidenceItem(kind="video", ref="x")


def test_known_speaker_and_answer_models_are_documented() -> None:
    speaker = KnownSpeaker(
        id=JOB_ID,
        name="Daniel Levitin",
        notes=None,
        samples=2,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )
    assert speaker.model_dump()["samples"] == 2

    answer = Answer(
        answer="He broke into his own house.",
        confidence=0.72,
        citations=[Citation(seg_id="seg:0", start=0.18, text="A few years ago, I broke into my own house.")],
        evidence=["seg:0"],
    )
    assert answer.model_dump()["citations"][0]["seg_id"] == "seg:0"
