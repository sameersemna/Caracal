"""Markdown rendering: section order, untrusted-content escaping, timestamps (docs/06 S9)."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

import pytest
from markdown_it import MarkdownIt

from caracal.report.evidence import EvidenceItem
from caracal.report.model import (
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
    SpeakerCandidate,
    SpeakerIdentity,
    Summary,
    Topic,
    TranscriptRef,
    Translation,
    Verification,
    VerificationSource,
)
from caracal.report.render_md import md_escape, render_markdown

JOB_ID = uuid.UUID("6f5c5f4e-0d5b-4a3e-9d1e-2f6b0f3f9a10")
CREATED_AT = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)

INJECTION = "<script>alert(1)</script> [click](http://evil) ![x](http://e/x.png) | a | b | # heading"

HOSTILE_PAYLOADS = (
    "http://evil.example",
    "www.evil.example/x",
    "<https://evil/a>",
    "[x](javascript:alert(1))",
    "<img src=x onerror=1>",
    "http://evil.example/`tick`",
    "safe\u202egnp.exe",
)
HOSTILE = " ".join(HOSTILE_PAYLOADS)

HEADINGS = [
    "# How to stay calm when you know you'll be stressed",
    "## Source and audio",
    "## Languages",
    "## Speakers",
    "## Summary",
    "## Context",
    "## Chapters",
    "## Topics",
    "## Entities",
    "## Key quotes",
    "## Claims",
    "## Translations",
    "## Quality",
    "## Evidence index",
]


def _report() -> Report:
    return Report(
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
            original_format="webm",
            sha256="a" * 64,
        ),
        audio=AudioInfo(
            duration_s=120.0,
            sample_rate=16000,
            channels=1,
            loudness_lufs=-18.4,
            speech_ratio=0.91,
            music_detected=False,
            quality="good",
            notes=["trimmed to the first 120 s"],
        ),
        languages=[Language(code="en", name="English", share=1.0, confidence=0.997)],
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
                        evidence=["seg:1"],
                    )
                ],
                description="adult male, calm",
                evidence=["seg:0"],
            )
        ],
        summary=Summary(
            one_line="A neuroscientist explains how to prepare for stressful moments.",
            short="Levitin breaks into his own house.",
            detailed="The anecdote turns into advice about anticipating stress.",
            evidence=["seg:0"],
        ),
        context=Context(
            genre="talk",
            setting="stage",
            audience="general public",
            purpose="explain a technique",
            tone=["calm", "humorous"],
            recorded_when="2015",
            recorded_where="TED stage",
            evidence=["seg:0"],
            confidence=0.78,
        ),
        chapters=[
            Chapter(start=194.2, end=198.0, title="A cold night", summary="s", evidence=["seg:7"]),
            Chapter(start=3725.0, end=3785.0, title="Past the hour", summary="s", evidence=["seg:7"]),
        ],
        topics=[Topic(label="stress", weight=0.82, evidence=["seg:7"])],
        entities=[
            Entity(text="Daniel Levitin", type="person", mentions=["seg:1"], evidence=["seg:1"]),
            Entity(text="Montreal", type="place", mentions=["seg:2"], evidence=["seg:2"]),
        ],
        key_quotes=[
            KeyQuote(
                seg_id="seg:0",
                speaker="S1",
                text="A few years ago, I broke into my own house.",
                start=0.18,
                why_notable="Opening anecdote",
            )
        ],
        claims=[
            Claim(
                id="c1",
                text="Minus 40 is where the two scales meet.",
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
                rationale="Reference tables confirm it.",
                sources=[
                    VerificationSource(
                        url="https://example.org/scales",
                        title="Celsius and Fahrenheit",
                        snippet="They cross at -40.",
                        evidence_id="web:0",
                    )
                ],
            )
        ],
        translations=[Translation(lang="de", status="done", links={"srt": "out/de.srt"})],
        transcript_ref=TranscriptRef(segments=49, words=812),
        quality=QualityInfo(
            asr_avg_logprob=-0.42,
            low_confidence_segments=3,
            caption_agreement=0.91,
            dropped_uncited=2,
            stages={"transcribe": "done", "verify": "skipped"},
            review_flags=[ReviewFlag(severity="warn", field="summary", message="uncited")],
        ),
        evidence_index={
            "seg:12": EvidenceItem(kind="seg", ref="194.2-198.0s S2", excerpt="Minus 40 is where they meet."),
            "web:0": EvidenceItem(
                kind="web", ref="https://example.org/scales", excerpt="Celsius and Fahrenheit"
            ),
        },
    )


def _hostile_report() -> Report:
    """``_report()`` with every free-text field the renderer prints set to hostile content."""
    report = _report()
    report.source.url = HOSTILE
    report.source.title = HOSTILE
    report.source.uploader = HOSTILE
    report.source.channel = HOSTILE
    report.source.upload_date = HOSTILE
    report.source.platform = HOSTILE
    report.source.original_format = HOSTILE
    report.source.sha256 = HOSTILE
    report.source.description_excerpt = HOSTILE
    report.audio.notes = [HOSTILE]

    report.languages[0].code = HOSTILE
    report.languages[0].name = HOSTILE

    speaker = report.speakers[0]
    speaker.label = HOSTILE
    speaker.name = HOSTILE
    speaker.role = HOSTILE
    speaker.description = HOSTILE
    speaker.candidates[0].name = HOSTILE
    speaker.candidates[0].sources = [HOSTILE]

    report.summary.one_line = HOSTILE
    report.summary.short = HOSTILE
    report.summary.detailed = HOSTILE

    context = report.context
    context.genre = HOSTILE
    context.setting = HOSTILE
    context.audience = HOSTILE
    context.purpose = HOSTILE
    context.tone = [HOSTILE]
    context.recorded_when = HOSTILE
    context.recorded_where = HOSTILE

    report.chapters[0].title = HOSTILE
    report.chapters[0].summary = HOSTILE
    report.topics[0].label = HOSTILE

    entity = report.entities[0]
    entity.text = HOSTILE
    entity.mentions = [HOSTILE]
    entity.canonical = HOSTILE
    entity.url = HOSTILE

    report.key_quotes[0].text = HOSTILE
    report.key_quotes[0].speaker = HOSTILE

    claim = report.claims[0]
    claim.id = HOSTILE
    claim.text = HOSTILE
    claim.speaker = HOSTILE
    verification = report.verification[0]
    verification.target = f"claim:{HOSTILE}"
    verification.rationale = HOSTILE
    verification.sources[0].url = HOSTILE
    verification.sources[0].title = HOSTILE
    verification.sources[0].snippet = HOSTILE
    verification.sources[0].evidence_id = HOSTILE

    report.translations[0].lang = HOSTILE
    report.translations[0].links = {"srt": HOSTILE}

    report.quality.stages = {HOSTILE: HOSTILE}
    report.quality.review_flags[0].field = HOSTILE
    report.quality.review_flags[0].message = HOSTILE

    report.evidence_index = {HOSTILE: EvidenceItem(kind="web", ref=HOSTILE, excerpt=HOSTILE)}
    return report


def test_headings_are_present_in_order() -> None:
    rendered = render_markdown(_report())

    positions = [rendered.index(heading) for heading in HEADINGS]
    assert positions == sorted(positions)
    assert rendered.startswith("# How to stay calm")


def test_tables_languages_speakers_and_context() -> None:
    rendered = render_markdown(_report())

    assert "| Source kind | platform |" in rendered
    assert "| Speech ratio | 91.0% |" in rendered
    assert "- en — English (100.0% of speech, confidence 1.00)" in rendered
    assert "- **S1** — Daniel Levitin (confidence 0.86, role: narrator)" in rendered
    assert "  - candidate Daniel Levitin (confidence 0.86; sources: self\\_intro, metadata)" in rendered
    assert "- Genre: talk" in rendered
    assert "- Tone: calm, humorous" in rendered
    assert "- Confidence: 0.78" in rendered


def test_chapter_timestamps_switch_to_hours_past_one_hour() -> None:
    rendered = render_markdown(_report())

    assert "- 03:14\u201303:18 \u2014 **A cold night** \u2014 s" in rendered
    assert "- 1:02:05\u20131:03:05 \u2014 **Past the hour** \u2014 s" in rendered


def test_key_quotes_are_blockquotes_with_speaker_and_time() -> None:
    rendered = render_markdown(_report())

    assert "> A few years ago, I broke into my own house.\n>\n> — S1, 00:00" in rendered


def test_claims_show_their_verification_verdict() -> None:
    rendered = render_markdown(_report())

    assert "**c1** — Minus 40 is where the two scales meet. (S1; fact; checkworthy)" in rendered
    assert "  - verdict: supported (confidence 0.88)" in rendered
    assert "  - source [web:0] Celsius and Fahrenheit" in rendered


def test_quality_and_review_flags_are_rendered() -> None:
    rendered = render_markdown(_report())

    assert "- ASR average logprob: -0.420" in rendered
    assert "- Caption agreement: 0.91" in rendered
    assert "- Dropped uncited: 2" in rendered
    assert "- transcribe: done" in rendered
    assert "- [warn] summary — uncited" in rendered


def test_evidence_appendix_formats_segment_time_ranges() -> None:
    rendered = render_markdown(_report())

    assert "- [seg:12] 03:14.2\u201303:18.0 S2 \u2014 Minus 40 is where they meet." in rendered
    assert "- [web:0] `https://example.org/scales` — Celsius and Fahrenheit" in rendered


def test_evidence_appendix_formats_hours() -> None:
    report = _report()
    report.evidence_index = {"seg:40": EvidenceItem(kind="seg", ref="3700.0-3701.5s S1", excerpt="later")}

    assert "- [seg:40] 1:01:40.0\u20131:01:41.5 S1 \u2014 later" in render_markdown(report)


def test_unidentified_speaker_and_empty_sections() -> None:
    report = _report()
    report.speakers[0].name = None
    report.key_quotes = []
    report.claims = []
    report.verification = []
    report.translations = []
    report.topics = []
    report.entities = []

    rendered = render_markdown(report)

    assert "- **S1** — unidentified (confidence 0.86, role: narrator)" in rendered
    assert rendered.count("- none") == 5


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<script>", "&lt;script&gt;"),
        ("[click](http://evil)", "\\[click\\]\\(`http://evil)`"),
        ("![x](http://e/x.png)", "\\!\\[x\\]\\(`http://e/x.png)`"),
        ("# heading", "\\# heading"),
        ("| a | b |", "\\| a \\| b \\|"),
        ("a_b*c`,", "a\\_b\\*c\\`,"),
        ("back\\slash", "back\\\\slash"),
        ("a\nb", "a\nb"),
        ("a\x00b\x7f", "ab"),
        ("http://evil.example", "`http://evil.example`"),
        ("see www.evil.example/x now", "see `www.evil.example/x` now"),
        ("ftp://evil.example/f", "`ftp://evil.example/f`"),
        ("<https://evil/a>", "&lt;`https://evil/a`&gt;"),
        ("http://evil.example/`x`", "`` http://evil.example/`x` ``"),
    ],
)
def test_md_escape_table(value: str, expected: str) -> None:
    assert md_escape(value) == expected


def test_md_escape_inline_collapses_newlines() -> None:
    assert md_escape("a\n  b", inline=True) == "a b"


def test_md_escape_strips_bidi_controls_but_keeps_directional_marks() -> None:
    value = "a\u202ab\u202bc\u202cd\u202de\u202ef\u2066g\u2067h\u2068i\u2069j\u200ek\u200fl"

    assert md_escape(value) == "abcdefghij\u200ek\u200fl"


def test_untrusted_content_renders_inert() -> None:
    report = _report()
    report.source.title = INJECTION
    report.audio.notes = [INJECTION]
    report.chapters[0].title = INJECTION
    report.chapters[0].summary = INJECTION
    report.key_quotes[0].text = INJECTION
    report.entities[0].text = INJECTION
    report.topics[0].label = INJECTION
    report.speakers[0].name = INJECTION
    report.summary.short = INJECTION
    report.evidence_index = {
        "seg:12": EvidenceItem(kind="seg", ref="194.2-198.0s S2", excerpt=INJECTION),
        "web:0": EvidenceItem(kind="web", ref="https://example.org/scales", excerpt=INJECTION),
    }

    rendered = render_markdown(report)

    assert "<script" not in rendered
    assert "</script" not in rendered
    assert "](http" not in rendered
    assert "![x]" not in rendered
    assert "\n# heading" not in rendered
    assert "\\# heading" in rendered
    assert "| a | b |" not in rendered
    assert "\\| a \\| b \\|" in rendered


def test_gfm_render_of_hostile_report_has_no_content_tags_links_or_bidi() -> None:
    rendered = render_markdown(_hostile_report())
    html = MarkdownIt("gfm-like", {"html": True, "linkify": True}).render(rendered)

    # Every payload reached the document, so the renderer was actually exercised.
    assert "http://evil.example" in html
    assert "www.evil.example/x" in html
    assert "javascript:alert(1)" in html
    assert "safegnp.exe" in html
    assert "<code>" in html

    # Content never becomes markup: no dangerous tag and no href/src from content.
    for tag in ("a", "img", "script", "svg", "iframe"):
        assert re.search(rf"<{tag}\b", html) is None, tag
    assert re.search(r"<[a-zA-Z][^>]*\b(?:href|src)=", html) is None

    # Bidi embedding/override/isolate controls are gone.
    for codepoint in (*range(0x202A, 0x202F), *range(0x2066, 0x206A)):
        assert chr(codepoint) not in html


def test_content_cannot_break_a_table_row() -> None:
    report = _report()
    report.audio.notes = [INJECTION]

    rendered = render_markdown(report)
    notes_row = next(line for line in rendered.splitlines() if line.startswith("| Audio notes |"))

    assert notes_row == f"| Audio notes | {md_escape(INJECTION, inline=True)} |"
