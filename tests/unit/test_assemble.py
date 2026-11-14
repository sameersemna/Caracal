"""Report assembly from stage checkpoints (T2.7, docs/03 §5, docs/04 §9 + §11).

The realistic checkpoint set is built from the recorded TED fixture
(``tests/fixtures/whisper/talk_c0.json``) plus scripted content/identity outputs, so the
evidence ids the assembler must resolve are the real segment ids of the fixture.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from tests.fakes.repo import make_fake_repos
from tests.fakes.whisper import load_fixture
from tests.repo_contract import Repos

from caracal.asr import Segment, Transcript, Word
from caracal.config import Settings
from caracal.db.repo import JobRecord
from caracal.ingest.ytdlp import PlatformMetadata
from caracal.pipeline.stages import Stage
from caracal.report.assemble import assemble_report, strip_prose_citations
from caracal.report.evidence import build_evidence_index
from caracal.report.model import Report

URL = "https://www.youtube.com/watch?v=8jPQjjsBbIc"
SHA256 = "a3f1" + "0" * 60


def _transcript(speaker: str | None = None) -> Transcript:
    """The real TED fixture as a merged transcript (real segment ids and timings)."""
    payload = load_fixture("talk_c0")
    segments: list[Segment] = []
    for index, raw in enumerate(payload["segments"]):
        words = tuple(
            Word(
                text=item["word"],
                start=float(item["start"]),
                end=float(item["end"]),
                prob=float(item.get("probability", 1.0)),
                space_before=True,
            )
            for item in raw.get("words", [])
        )
        segments.append(
            Segment(
                id=index,
                start=float(raw["start"]),
                end=float(raw["end"]),
                text=raw["text"],
                words=words,
                avg_logprob=float(raw.get("avg_logprob", 0.0)),
                no_speech_prob=float(raw.get("no_speech_prob", 0.0)),
                lang="en",
                speaker=speaker,
            )
        )
    return Transcript(segments=tuple(segments), language_code="en", dropped={})


def _checkpoint(status: str, output: dict[str, Any] | None) -> dict[str, Any]:
    return {"status": status, "reason": None, "output": output or {}}


def _metadata() -> dict[str, Any]:
    return {
        "id": "8jPQjjsBbIc",
        "title": "How to stay calm when you know you'll be stressed",
        "uploader": "TED",
        "channel": "TED",
        "channel_url": "https://www.youtube.com/@TED",
        "upload_date": "20151124",
        "duration": 61.0,
        "description": "Neuroscientist Daniel Levitin offers strategies for stress.",
        "tags": ["TED", "stress"],
        "categories": ["Education"],
        "chapters": [],
        "language": "en",
        "view_count": 1_000_000,
        "webpage_url": URL,
        "extractor": "youtube",
    }


def _full_checkpoints(duration: float = 61.0) -> dict[str, dict[str, Any] | None]:
    """A complete, realistic set: every stage that M2 runs has a checkpoint."""
    checkpoints: dict[str, dict[str, Any] | None] = {
        Stage.INGEST.value: _checkpoint(
            "done",
            {
                "source_kind": "platform",
                "url": URL,
                "final_url": URL,
                "filename": "audio.webm",
                "size_bytes": 12345,
                "sha256": SHA256,
                "platform": _metadata(),
                "captions": [],
            },
        ),
        Stage.NORMALIZE.value: _checkpoint(
            "done",
            {
                "probe": {
                    "format_name": "webm",
                    "duration_s": duration,
                    "bit_rate": 128000,
                    "audio_streams": 1,
                    "video_streams": 0,
                    "codec": "opus",
                    "sample_rate": 48000,
                    "channels": 2,
                    "tags": {"artist": "TED"},
                },
                "stats": {
                    "duration_s": duration,
                    "speech_ratio": 0.91,
                    "loudness_lufs": -18.4,
                    "quality": "good",
                    "silences": [],
                },
            },
        ),
        Stage.CHUNK.value: _checkpoint(
            "done",
            {"chunks": [{"index": 0, "start_s": 0.0, "end_s": duration, "path": "chunks/000.wav"}]},
        ),
        Stage.TRANSCRIBE.value: _checkpoint(
            "done",
            {
                "segments": 23,
                "words": 200,
                "language_code": "en",
                "dropped": {},
                "chunk_langs": [{"start_s": 0.0, "code": "en", "prob": 0.99}],
            },
        ),
        Stage.LANGID.value: _checkpoint(
            "done",
            {
                "languages": [
                    {
                        "code": "en",
                        "name": "English",
                        "share": 1.0,
                        "spans": [{"start": 0.0, "end": duration}],
                        "confidence": 0.99,
                    }
                ],
                "caption_agreement": 0.92,
                "caption_lang": "en",
            },
        ),
        Stage.DIARIZE.value: _checkpoint(
            "done",
            {
                "num_speakers": 1,
                "turns": [{"start": 0.0, "end": duration, "label": "S1"}],
                "exclusive_turns": [{"start": 0.0, "end": duration, "label": "S1"}],
                "embeddings": {"S1": [1.0, *([0.0] * 255)]},
            },
        ),
        Stage.PLAN.value: _checkpoint(
            "done",
            {
                "run": {"content": True, "identity": True, "verify": False, "translate": False},
                "reasons": {"content": "content analysis"},
                "content_profile": {
                    "likely_genre": "talk",
                    "expected_speakers": 1,
                    "needs_identity": True,
                    "checkworthy_domain": True,
                },
                "budgets": {"search_queries": 12, "llm_calls": 4, "claims": 8},
                "music_detected": False,
                "languages": None,
            },
        ),
        Stage.ALIGN.value: _checkpoint(
            "done",
            {
                "speakers": {
                    "S1": {
                        "talk_time_s": round(duration - 1.5, 3),
                        "share": 1.0,
                        "first_seen_s": 0.18,
                        "words": 200,
                    }
                },
                "retimed_words": 3,
            },
        ),
        Stage.VOICEPRINT.value: _checkpoint(
            "done",
            {
                "matches": {
                    "S1": [
                        {
                            "known_speaker_id": str(uuid.uuid4()),
                            "name": "Daniel Levitin",
                            "similarity": 0.93,
                            "class": "match",
                        }
                    ]
                }
            },
        ),
        Stage.CONTENT.value: _checkpoint(
            "done",
            {
                "summary": {
                    "one_line": (
                        "Neuroscientist Daniel Levitin explains how stress clouds "
                        "thinking [meta:title, meta:channel]"
                    ),
                    "short": "Levitin tells a story about breaking into his own house.",
                    "detailed": "Levitin turns a cold Montreal night into advice about stress.",
                    "evidence": ["seg:0"],
                },
                "context": {
                    "genre": "TED Talk (meta:title)",
                    "setting": "stage",
                    "audience": "general public",
                    "purpose": "explain a practical technique",
                    "tone": ["calm", "humorous"],
                    "recorded_when": "2015",
                    "recorded_where": "TED stage",
                    "evidence": ["seg:1"],
                    "confidence": 0.78,
                },
                "chapters": [
                    {
                        "start": 0.0,
                        "end": duration / 2,
                        "title": "Breaking into my own house [seg:0, seg:1]",
                        "summary": "A cold night in Montreal and a door that would not open.",
                        "evidence": ["seg:0"],
                    },
                    {
                        "start": duration / 2,
                        "end": duration,
                        "title": "The pre-mortem",
                        "summary": "A technique for anticipating stress.",
                        "evidence": ["seg:12"],
                    },
                ],
                "topics": [
                    {"label": "stress (seg:2)", "weight": 0.8, "evidence": []},
                    {"label": "memory", "weight": 0.4, "evidence": ["seg:5"]},
                ],
                "entities": [
                    {
                        "text": "Daniel Levitin",
                        "type": "person",
                        "mentions": ["seg:0"],
                        "evidence": ["seg:0"],
                    },
                    {
                        "text": "Montreal",
                        "type": "place",
                        "mentions": ["seg:2"],
                        "evidence": ["seg:2"],
                    },
                    {
                        "text": "TED",
                        "type": "org",
                        "mentions": ["seg:1"],
                        "evidence": ["seg:1"],
                    },
                ],
                "key_quotes": [
                    {
                        "seg_id": "seg:4",
                        "speaker": "S1",
                        "text": "I broke into my own house [seg:4]",
                        "start": 12.0,
                        "why_notable": "The story that anchors the talk.",
                    }
                ],
                "claims": [
                    {
                        "id": "c1",
                        "text": "Minus 40 is where the Celsius and Fahrenheit scales meet [seg:3]",
                        "speaker": "S1",
                        "seg_ids": ["seg:3"],
                        "kind": "fact",
                        "checkworthy": True,
                        "evidence": [],
                    }
                ],
                "review_flags": [],
                "llm_meta": {"calls": 2},
            },
        ),
        Stage.IDENTITY.value: _checkpoint(
            "done",
            {
                "speakers": [
                    {
                        "label": "S1",
                        "talk_time_s": round(duration - 1.5, 3),
                        "share": 1.0,
                        "first_seen_s": 0.18,
                        "name": "Daniel Levitin",
                        "confidence": 0.75,
                        "role": "presenter",
                        "candidates": [],
                        "description": "adult male, calm (seg:0)",
                        "evidence": ["seg:0", "vp:S1"],
                    }
                ],
                "flags": [],
            },
        ),
        Stage.VERIFY.value: _checkpoint("skipped", None),
        Stage.TRANSLATE.value: _checkpoint("skipped", None),
        Stage.SUBTITLES.value: _checkpoint("done", {"cues": 23, "files": {}}),
        Stage.REVIEW.value: _checkpoint("skipped", None),
        Stage.ASSEMBLE.value: None,
    }
    return checkpoints


async def _job(repos: Repos, *, title: str | None = "How to stay calm") -> JobRecord:
    return await repos.jobs.create(
        source_kind="platform",
        source_url=URL,
        title=title,
        options={"speaker_hints": [], "context_hint": None},
    )


def _expected_index(transcript: Transcript) -> set[str]:
    meta = PlatformMetadata(
        id="8jPQjjsBbIc",
        title="How to stay calm when you know you'll be stressed",
        uploader="TED",
        channel="TED",
        channel_url="https://www.youtube.com/@TED",
        upload_date="20151124",
        duration=61.0,
        description="Neuroscientist Daniel Levitin offers strategies for stress.",
        tags=["TED", "stress"],
        categories=["Education"],
        chapters=[],
        language="en",
        view_count=1_000_000,
        webpage_url=URL,
        extractor="youtube",
    )
    return set(
        build_evidence_index(
            transcript=transcript,
            platform_meta=meta,
            probe_tags={"artist": "TED"},
            voiceprint_matches={"S1": [{"name": "Daniel Levitin", "similarity": 0.93}]},
        ).ids()
    )


def _collect_evidence(value: object, found: set[str] | None = None) -> set[str]:
    """Every id cited anywhere in a report (generic pydantic walk)."""
    found = found if found is not None else set()
    if isinstance(value, BaseModel):
        for name in type(value).model_fields:
            if name == "evidence_index":
                continue
            item = getattr(value, name)
            if name == "evidence" and isinstance(item, list):
                found.update(entry for entry in item if isinstance(entry, str))
            else:
                _collect_evidence(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_evidence(item, found)
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_evidence(item, found)
    return found


async def test_full_checkpoints_assemble_a_valid_cited_report(
    make_settings: Any,
    tmp_path: Path,
) -> None:
    settings: Settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _job(repos)
    transcript = _transcript()

    report = assemble_report(
        _full_checkpoints(),
        job=job,
        settings=settings,
        transcript=transcript,
    )

    assert isinstance(report, Report)
    assert report.job_id == job.id
    assert report.source.title == "How to stay calm when you know you'll be stressed"
    assert report.source.platform == "youtube"
    assert report.source.sha256 == SHA256
    assert report.audio.duration_s == pytest.approx(61.0)
    assert report.audio.sample_rate == 48000
    assert report.audio.music_detected is False
    assert [language.code for language in report.languages] == ["en"]
    assert [speaker.name for speaker in report.speakers] == ["Daniel Levitin"]

    # Verdict 26: citations leave the prose and join the item's evidence.
    assert report.summary.one_line == ("Neuroscientist Daniel Levitin explains how stress clouds thinking")
    assert {"meta:title", "meta:channel"} <= set(report.summary.evidence)
    assert report.context.genre == "TED Talk"
    assert "meta:title" in report.context.evidence
    assert report.chapters[0].title == "Breaking into my own house"
    assert {"seg:0", "seg:1"} <= set(report.chapters[0].evidence)
    assert report.topics[0].label == "stress"
    assert "seg:2" in report.topics[0].evidence
    assert report.claims[0].text == "Minus 40 is where the Celsius and Fahrenheit scales meet"
    assert "seg:3" in report.claims[0].evidence
    assert report.speakers[0].description == "adult male, calm"
    assert "seg:0" in report.speakers[0].evidence

    # Quotes are verbatim: the bracketed text stays exactly as the speaker said it.
    assert report.key_quotes[0].text == "I broke into my own house [seg:4]"

    # Every cited id resolves against an independently built index.
    expected = _expected_index(transcript)
    cited = _collect_evidence(report)
    assert cited
    assert cited <= expected
    assert set(report.evidence_index) == cited

    # Quality and stage bookkeeping.
    assert set(report.quality.stages) == {stage.value for stage in Stage}
    assert report.quality.stages[Stage.ASSEMBLE.value] == "done"
    assert report.quality.stages[Stage.REVIEW.value] == "skipped"
    assert report.quality.caption_agreement == pytest.approx(0.92)
    assert report.quality.low_confidence_segments == 0
    assert report.quality.asr_avg_logprob == pytest.approx(
        sum(segment.avg_logprob for segment in transcript.segments) / len(transcript.segments)
    )
    assert not [flag for flag in report.quality.review_flags if flag.severity == "error"]
    assert report.transcript_ref.segments == 23
    assert report.transcript_ref.words == sum(len(segment.words) for segment in transcript.segments)
    assert report.transcript_ref.links["json"].endswith(f"/v1/jobs/{job.id}/transcript?format=json")


async def test_missing_and_failed_stages_give_empty_sections_and_an_error_flag(
    make_settings: Any,
    tmp_path: Path,
) -> None:
    settings: Settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _job(repos)
    checkpoints = _full_checkpoints()
    checkpoints[Stage.CONTENT.value] = _checkpoint("failed", None)
    checkpoints[Stage.IDENTITY.value] = None
    checkpoints[Stage.LANGID.value] = None

    report = assemble_report(
        checkpoints,
        job=job,
        settings=settings,
        transcript=_transcript(speaker="S1"),
        stage_statuses={
            **{stage.value: "done" for stage in Stage},
            Stage.CONTENT.value: "failed",
            Stage.IDENTITY.value: "skipped",
            Stage.LANGID.value: "failed",
        },
    )

    assert report.languages == []
    assert report.chapters == []
    assert report.topics == []
    assert report.entities == []
    assert report.key_quotes == []
    assert report.claims == []
    assert report.summary.one_line == ""
    assert report.context.genre == "unknown"
    # Identity failed but align ran: one unnamed speaker per label from its statistics.
    assert [speaker.label for speaker in report.speakers] == ["S1"]
    assert report.speakers[0].name is None
    assert report.speakers[0].talk_time_s == pytest.approx(59.5)
    assert report.speakers[0].share == pytest.approx(1.0)
    assert report.speakers[0].first_seen_s == pytest.approx(0.18)
    assert report.quality.stages[Stage.CONTENT.value] == "failed"
    assert report.quality.stages[Stage.LANGID.value] == "failed"
    errors = [
        flag
        for flag in report.quality.review_flags
        if flag.severity == "error" and flag.field == f"stages.{Stage.CONTENT.value}"
    ]
    assert errors and errors[0].message == "stage content failed"


async def test_music_job_uses_und_languages_and_has_no_quotes_or_claims(
    make_settings: Any,
    tmp_path: Path,
) -> None:
    settings: Settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _job(repos, title="Instrumental")
    checkpoints = _full_checkpoints(duration=180.0)
    checkpoints[Stage.PLAN.value] = _checkpoint(
        "done",
        {
            "run": {"content": True, "identity": False, "verify": False, "translate": False},
            "reasons": {"content": "music: describe the track, not speech"},
            "content_profile": {
                "likely_genre": "unknown",
                "expected_speakers": None,
                "needs_identity": False,
                "checkworthy_domain": False,
            },
            "budgets": {"search_queries": 0, "llm_calls": 1, "claims": 8},
            "music_detected": True,
            "languages": [
                {
                    "code": "und",
                    "name": "Unknown",
                    "share": 1.0,
                    "spans": [{"start": 0.0, "end": 180.0}],
                    "confidence": 1.0,
                }
            ],
        },
    )
    checkpoints[Stage.CONTENT.value] = _checkpoint(
        "done",
        {
            "summary": {
                "one_line": "An instrumental track.",
                "short": "An instrumental track.",
                "detailed": "An instrumental track.",
                "evidence": ["meta:title"],
            },
            "context": {
                "genre": "music",
                "tone": ["instrumental"],
                "evidence": ["meta:title"],
                "confidence": 0.6,
            },
            "chapters": [],
            "topics": [],
            "entities": [],
            "key_quotes": [],
            "claims": [],
            "review_flags": [{"severity": "warn", "field": "content", "message": "non-speech audio"}],
            "llm_meta": {"calls": 1},
        },
    )
    checkpoints[Stage.IDENTITY.value] = _checkpoint("skipped", None)
    checkpoints[Stage.ALIGN.value] = _checkpoint("skipped", None)

    report = assemble_report(
        checkpoints,
        job=job,
        settings=settings,
        transcript=None,
        stage_statuses={**{stage.value: "done" for stage in Stage}, Stage.IDENTITY.value: "skipped"},
    )

    assert report.audio.music_detected is True
    assert [language.code for language in report.languages] == ["und"]
    assert report.languages[0].name == "Unknown"
    assert report.key_quotes == []
    assert report.claims == []
    assert report.speakers == []
    assert any(flag.message == "non-speech audio" for flag in report.quality.review_flags)


@pytest.mark.parametrize(
    ("text", "expected_text", "expected_ids"),
    [
        (
            "TED Talk [meta:title, meta:channel]",
            "TED Talk",
            ["meta:title", "meta:channel"],
        ),
        ("A talk (seg:1; seg:2)", "A talk", ["seg:1", "seg:2"]),
        ("Quote [seg:3]", "Quote", ["seg:3"]),
        ("Keep (parenthetical aside) text", "Keep (parenthetical aside) text", []),
        ("No ids at all", "No ids at all", []),
        ("Already clean", "Already clean", []),
        ("Two [seg:1] lists (meta:title)", "Two lists", ["seg:1", "meta:title"]),
    ],
)
def test_prose_citation_stripping_table(
    text: str,
    expected_text: str,
    expected_ids: list[str],
) -> None:
    stripped, ids = strip_prose_citations(text)
    assert stripped == expected_text
    assert ids == expected_ids


async def test_deterministic_review_flags(
    make_settings: Any,
    tmp_path: Path,
) -> None:
    settings: Settings = make_settings(CARACAL_DATA_DIR=tmp_path)
    repos = make_fake_repos()
    job = await _job(repos)
    transcript = _transcript()

    def _flags(checkpoints: Mapping[str, Any], *, stage_statuses: Mapping[str, str] | None = None) -> list:
        report = assemble_report(
            checkpoints,
            job=job,
            settings=settings,
            transcript=transcript,
            stage_statuses=stage_statuses,
        )
        return report.quality.review_flags

    # Speaker shares do not sum to 1 (±0.05).
    checkpoints = _full_checkpoints()
    checkpoints[Stage.IDENTITY.value] = _checkpoint(
        "done",
        {
            "speakers": [
                {
                    "label": "S1",
                    "talk_time_s": 30.0,
                    "share": 0.4,
                    "first_seen_s": 0.0,
                    "name": None,
                    "confidence": 0.0,
                    "evidence": ["seg:0"],
                },
                {
                    "label": "S2",
                    "talk_time_s": 30.0,
                    "share": 0.4,
                    "first_seen_s": 1.0,
                    "name": None,
                    "confidence": 0.0,
                    "evidence": ["seg:1"],
                },
            ]
        },
    )
    flags = _flags(checkpoints)
    assert any(flag.field == "speakers" and "0.800" in flag.message for flag in flags)

    # Chapters cover less than 80 % of the duration.
    checkpoints = _full_checkpoints()
    checkpoints[Stage.CONTENT.value] = _checkpoint(
        "done",
        {
            **(_full_checkpoints()[Stage.CONTENT.value] or {}),
            "chapters": [
                {
                    "start": 0.0,
                    "end": 6.0,
                    "title": "Intro",
                    "summary": "Short.",
                    "evidence": ["seg:0"],
                }
            ],
        },
    )
    flags = _flags(checkpoints)
    assert any(flag.field == "chapters" and "9.8%" in flag.message for flag in flags)

    # Caption agreement below 0.6.
    checkpoints = _full_checkpoints()
    assert checkpoints[Stage.LANGID.value] is not None
    langid = checkpoints[Stage.LANGID.value]
    assert langid is not None
    output = langid["output"]
    assert isinstance(output, dict)
    output["caption_agreement"] = 0.4
    flags = _flags(checkpoints)
    assert any(flag.field == "quality.caption_agreement" for flag in flags)

    # More than 10 % of segments have avg_logprob < -1.0.
    noisy = Transcript(
        segments=tuple(
            Segment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=segment.words,
                avg_logprob=-1.5 if segment.id < 5 else -0.1,
                no_speech_prob=segment.no_speech_prob,
                lang=segment.lang,
            )
            for segment in transcript.segments
        ),
        language_code="en",
        dropped={},
    )
    report = assemble_report(
        _full_checkpoints(),
        job=job,
        settings=settings,
        transcript=noisy,
    )
    assert any(flag.field == "quality.low_confidence_segments" for flag in report.quality.review_flags)

    # A failed stage becomes an error flag naming it.
    flags = _flags(
        _full_checkpoints(),
        stage_statuses={**{stage.value: "done" for stage in Stage}, Stage.VERIFY.value: "failed"},
    )
    errors = [flag for flag in flags if flag.severity == "error"]
    assert [flag.field for flag in errors] == [f"stages.{Stage.VERIFY.value}"]
    assert errors[0].message == "stage verify failed"


# --- T3.2: verify checkpoint -----------------------------------------------------------


async def test_verify_speakers_win_and_web_hits_become_evidence(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_PUBLIC_BASE_URL="http://caracal.test")
    repos = make_fake_repos()
    job = await _job(repos)
    checkpoints = _full_checkpoints()
    checkpoints[Stage.VERIFY.value] = _checkpoint(
        "done",
        {
            "verifications": [
                {
                    "target": "claim:c1",
                    "verdict": "supported",
                    "confidence": 0.7,
                    "rationale": "two independent sources agree",
                    "sources": [
                        {
                            "url": "https://www.example.org/fact",
                            "title": "Fact check",
                            "snippet": "The fact is correct.",
                            "evidence_id": "web:1",
                        }
                    ],
                }
            ],
            "speakers": [
                {
                    "label": "S1",
                    "talk_time_s": 59.5,
                    "share": 1.0,
                    "first_seen_s": 0.18,
                    "name": "Lindsay McMahon",
                    "confidence": 0.62,
                    "role": "host",
                    "candidates": [
                        {
                            "name": "Lindsay McMahon",
                            "confidence": 0.62,
                            "sources": ["self_intro", "web"],
                            "evidence": ["seg:0", "web:1"],
                        }
                    ],
                    "description": "hosts the show",
                    "evidence": ["seg:0", "web:1"],
                }
            ],
            "flags": [
                {
                    "severity": "info",
                    "field": "speakers.S1.name",
                    "message": "quote attributed to John F. Kennedy on the web",
                }
            ],
            "hits": [
                {
                    "id": "web:1",
                    "url": "https://www.example.org/fact",
                    "title": "Fact check",
                    "snippet": "The fact is correct.",
                    "text": None,
                    "provider": "brave",
                    "query": "the fact",
                }
            ],
            "queries_used": 1,
            "cost_usd": 0.0,
        },
    )

    report = assemble_report(
        checkpoints,
        job=job,
        settings=settings,
        transcript=_transcript(speaker="S1"),
    )

    assert report.speakers[0].name == "Lindsay McMahon"
    assert report.speakers[0].confidence == pytest.approx(0.62)
    assert report.verification[0].verdict == "supported"
    assert report.verification[0].sources[0].evidence_id == "web:1"
    assert report.evidence_index["web:1"].kind == "web"
    assert report.evidence_index["web:1"].ref == "https://www.example.org/fact"
    assert report.evidence_index["web:1"].excerpt == "Fact check — The fact is correct."
    assert "web:0" not in report.evidence_index
    flag_messages = [flag.message for flag in report.quality.review_flags]
    assert flag_messages.count("quote attributed to John F. Kennedy on the web") == 1


async def test_unknown_web_source_is_dropped_and_the_verdict_forced_unverified(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_PUBLIC_BASE_URL="http://caracal.test")
    repos = make_fake_repos()
    job = await _job(repos)
    checkpoints = _full_checkpoints()
    checkpoints[Stage.VERIFY.value] = _checkpoint(
        "done",
        {
            "verifications": [
                {
                    "target": "claim:c1",
                    "verdict": "supported",
                    "confidence": 0.9,
                    "rationale": "invented source",
                    "sources": [
                        {
                            "url": "https://www.example.org/missing",
                            "title": "Missing",
                            "snippet": "not indexed",
                            "evidence_id": "web:99",
                        }
                    ],
                }
            ],
            "speakers": [],
            "flags": [],
            "hits": [
                {
                    "id": "web:1",
                    "url": "https://www.example.org/fact",
                    "title": "Fact check",
                    "snippet": "The fact is correct.",
                    "text": None,
                    "provider": "brave",
                    "query": "the fact",
                }
            ],
            "queries_used": 1,
            "cost_usd": 0.0,
        },
    )

    report = assemble_report(
        checkpoints,
        job=job,
        settings=settings,
        transcript=_transcript(speaker="S1"),
    )

    assert report.verification[0].verdict == "unverified"
    assert report.verification[0].sources == []
    assert report.quality.dropped_uncited == 1
    assert "web:99" not in report.evidence_index


async def test_skipped_verify_keeps_the_identity_flags(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    settings = make_settings(CARACAL_PUBLIC_BASE_URL="http://caracal.test")
    repos = make_fake_repos()
    job = await _job(repos)
    checkpoints = _full_checkpoints()
    identity = checkpoints[Stage.IDENTITY.value]
    assert identity is not None
    output = dict(identity["output"])
    output["flags"] = [
        {
            "severity": "warn",
            "field": "speakers",
            "message": "performance content: speaker identities uncertain",
        }
    ]
    checkpoints[Stage.IDENTITY.value] = _checkpoint("done", output)
    checkpoints[Stage.VERIFY.value] = _checkpoint("skipped", None)

    report = assemble_report(
        checkpoints,
        job=job,
        settings=settings,
        transcript=_transcript(speaker="S1"),
    )

    assert [flag.message for flag in report.quality.review_flags] == [
        "performance content: speaker identities uncertain"
    ]
