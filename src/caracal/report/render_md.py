"""Markdown rendering of a report (docs/03 §3 ``get_report`` with ``format=markdown``).

Transcripts, platform metadata, web snippets and LLM output are **data**, never markup
(docs/06 S9). Everything that reaches the document from the report passes through
:func:`md_escape`, so the rendered Markdown contains no raw HTML and no link or image syntax
that originated from content. Only the literal scaffolding written here is unescaped.
"""

from __future__ import annotations

from caracal.report.evidence import EvidenceItem
from caracal.report.model import Entity, Report

#: Characters that can turn content into markup, links or images (docs/06 S9).
_ESCAPE_CHARS = "`*_[]()#|!"

_DEFAULT_TITLE = "Audio intelligence report"


def md_escape(text: str, *, inline: bool = False) -> str:
    """Escape ``text`` so it renders as literal Markdown, never as markup (docs/06 S9).

    ``&`` and angle brackets become HTML entities (so a ``<script>`` in a transcript can never
    be raw HTML), backslashes are doubled, then every Markdown-significant character is
    escaped. ASCII control characters are removed except ``\\n``; ``inline`` collapses all
    whitespace (including newlines) to single spaces, which is what table cells and
    blockquotes need.
    """
    cleaned = "".join(
        character
        for character in str(text)
        if character == "\n" or (ord(character) >= 0x20 and ord(character) != 0x7F)
    )
    escaped = cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace("\\", "\\\\")
    for character in _ESCAPE_CHARS:
        escaped = escaped.replace(character, f"\\{character}")
    if inline:
        escaped = " ".join(escaped.split())
    return escaped


def _cell(value: object) -> str:
    """One table cell: em dash for an absent value, otherwise escaped single-line text."""
    if value is None or value == "":
        return "\u2014"
    return md_escape(str(value), inline=True)


def _clock(seconds: float) -> str:
    """``mm:ss`` below one hour, ``h:mm:ss`` from one hour on."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _appendix_time(seconds: float) -> str:
    """``mm:ss.s`` below one hour, ``h:mm:ss.s`` from one hour on."""
    value = max(0.0, seconds)
    hours, rest = divmod(value, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes):02d}:{secs:04.1f}"


def _appendix_ref(item: EvidenceItem) -> str:
    """Render a segment ref as clock times; every other ref is already human-readable."""
    if item.kind != "seg":
        return item.ref
    span, _, speaker = item.ref.partition(" ")
    start_text, _, end_text = span.rstrip("s").partition("-")
    try:
        start = float(start_text)
        end = float(end_text)
    except ValueError:
        return item.ref
    rendered = f"{_appendix_time(start)}\u2013{_appendix_time(end)}"
    return f"{rendered} {speaker}" if speaker else rendered


def _source_section(report: Report) -> list[str]:
    source = report.source
    audio = report.audio
    rows: list[tuple[str, str]] = [
        ("Source kind", _cell(source.kind)),
        ("URL", _cell(source.url)),
        ("Title", _cell(source.title)),
        ("Uploader", _cell(source.uploader)),
        ("Channel", _cell(source.channel)),
        ("Upload date", _cell(source.upload_date)),
        ("Platform", _cell(source.platform)),
        ("Source duration", _cell(f"{source.duration_s:.1f} s")),
        ("Original format", _cell(source.original_format)),
        ("SHA-256", _cell(source.sha256)),
        ("Audio duration", _cell(f"{audio.duration_s:.1f} s")),
        ("Sample rate", _cell(f"{audio.sample_rate} Hz")),
        ("Channels", _cell(str(audio.channels))),
        ("Loudness", _cell(f"{audio.loudness_lufs:.1f} LUFS" if audio.loudness_lufs is not None else None)),
        ("Speech ratio", _cell(f"{audio.speech_ratio:.1%}")),
        ("Music detected", _cell("unknown" if audio.music_detected is None else str(audio.music_detected))),
        ("Audio quality", _cell(audio.quality)),
        ("Audio notes", _cell("; ".join(audio.notes) if audio.notes else None)),
    ]
    lines = ["## Source and audio", "", "| Field | Value |", "|---|---|"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return lines


def _languages_section(report: Report) -> list[str]:
    lines = ["## Languages", ""]
    if not report.languages:
        lines.append("- none detected")
        return lines
    for language in report.languages:
        lines.append(
            f"- {md_escape(language.code)} \u2014 {md_escape(language.name)} "
            f"({language.share:.1%} of speech, confidence {language.confidence:.2f})"
        )
    return lines


def _speakers_section(report: Report) -> list[str]:
    lines = ["## Speakers", ""]
    if not report.speakers:
        lines.append("- none")
        return lines
    for speaker in report.speakers:
        name = md_escape(speaker.name) if speaker.name else "unidentified"
        details = [f"confidence {speaker.confidence:.2f}"]
        if speaker.role:
            details.append(f"role: {md_escape(speaker.role)}")
        lines.append(f"- **{md_escape(speaker.label)}** \u2014 {name} ({', '.join(details)})")
        if speaker.description:
            lines.append(f"  - {md_escape(speaker.description)}")
        for candidate in speaker.candidates:
            sources = ", ".join(md_escape(source) for source in candidate.sources) or "no source"
            lines.append(
                f"  - candidate {md_escape(candidate.name)} "
                f"(confidence {candidate.confidence:.2f}; sources: {sources})"
            )
    return lines


def _summary_section(report: Report) -> list[str]:
    lines = ["## Summary", "", f"**{md_escape(report.summary.one_line)}**"]
    for paragraph in (report.summary.short, report.summary.detailed):
        if paragraph:
            lines.extend(["", md_escape(paragraph)])
    return lines


def _context_section(report: Report) -> list[str]:
    context = report.context
    lines = ["## Context", "", f"- Genre: {md_escape(context.genre)}"]
    for label, value in (
        ("Setting", context.setting),
        ("Audience", context.audience),
        ("Purpose", context.purpose),
        ("Recorded when", context.recorded_when),
        ("Recorded where", context.recorded_where),
    ):
        if value:
            lines.append(f"- {label}: {md_escape(value)}")
    if context.tone:
        lines.append(f"- Tone: {', '.join(md_escape(tone) for tone in context.tone)}")
    lines.append(f"- Confidence: {context.confidence:.2f}")
    return lines


def _chapters_section(report: Report) -> list[str]:
    lines = ["## Chapters", ""]
    if not report.chapters:
        lines.append("- none")
        return lines
    separator = " \u2014 "
    for chapter in report.chapters:
        parts = [f"{_clock(chapter.start)}\u2013{_clock(chapter.end)}", f"**{md_escape(chapter.title)}**"]
        if chapter.summary:
            parts.append(md_escape(chapter.summary))
        lines.append(f"- {separator.join(parts)}")
    return lines


def _topics_section(report: Report) -> list[str]:
    lines = ["## Topics", ""]
    if not report.topics:
        lines.append("- none")
        return lines
    lines.extend(f"- {md_escape(topic.label)} (weight {topic.weight:.2f})" for topic in report.topics)
    return lines


def _entities_section(report: Report) -> list[str]:
    lines = ["## Entities", ""]
    if not report.entities:
        lines.append("- none")
        return lines
    groups: dict[str, list[Entity]] = {}
    for entity in report.entities:
        groups.setdefault(entity.type, []).append(entity)
    for kind, entities in groups.items():
        lines.append(f"### {md_escape(kind)}")
        for entity in entities:
            extras: list[str] = []
            if entity.mentions:
                extras.append("mentions: " + ", ".join(md_escape(mention) for mention in entity.mentions))
            if entity.canonical:
                extras.append("canonical: " + md_escape(entity.canonical))
            if entity.url:
                extras.append(md_escape(entity.url))
            suffix = f" ({'; '.join(extras)})" if extras else ""
            lines.append(f"- {md_escape(entity.text)}{suffix}")
    return lines


def _quotes_section(report: Report) -> list[str]:
    lines = ["## Key quotes", ""]
    if not report.key_quotes:
        lines.append("- none")
        return lines
    for quote in report.key_quotes:
        lines.extend(
            [
                f"> {md_escape(quote.text, inline=True)}",
                ">",
                f"> \u2014 {md_escape(quote.speaker)}, {_clock(quote.start)}",
                "",
            ]
        )
    return lines


def _claims_section(report: Report) -> list[str]:
    lines = ["## Claims", ""]
    if not report.claims:
        lines.append("- none")
        return lines
    verdicts = {verification.target: verification for verification in report.verification}
    for claim in report.claims:
        label = "checkworthy" if claim.checkworthy else "not checkworthy"
        lines.append(
            f"- **{md_escape(claim.id)}** \u2014 {md_escape(claim.text)} "
            f"({md_escape(claim.speaker)}; {md_escape(claim.kind)}; {label})"
        )
        verdict = verdicts.get(f"claim:{claim.id}")
        if verdict is None:
            lines.append("  - verdict: not verified")
            continue
        lines.append(f"  - verdict: {md_escape(verdict.verdict)} (confidence {verdict.confidence:.2f})")
        if verdict.rationale:
            lines.append(f"  - rationale: {md_escape(verdict.rationale)}")
        for source in verdict.sources:
            lines.append(
                f"  - source [{md_escape(source.evidence_id)}] {md_escape(source.title or source.url)}"
            )
    return lines


def _translations_section(report: Report) -> list[str]:
    lines = ["## Translations", ""]
    if not report.translations:
        lines.append("- none")
        return lines
    for translation in report.translations:
        links = ", ".join(f"{md_escape(key)}: {md_escape(value)}" for key, value in translation.links.items())
        suffix = f" ({links})" if links else ""
        lines.append(f"- {md_escape(translation.lang)}: {md_escape(translation.status)}{suffix}")
    return lines


def _quality_section(report: Report) -> list[str]:
    quality = report.quality
    lines = [
        "## Quality",
        "",
        f"- ASR average logprob: {quality.asr_avg_logprob:.3f}",
        f"- Low-confidence segments: {quality.low_confidence_segments}",
    ]
    if quality.caption_agreement is not None:
        lines.append(f"- Caption agreement: {quality.caption_agreement:.2f}")
    lines.append(f"- Dropped uncited: {quality.dropped_uncited}")
    if quality.stages:
        lines.extend(["", "### Stages", ""])
        lines.extend(f"- {md_escape(name)}: {md_escape(status)}" for name, status in quality.stages.items())
    if quality.review_flags:
        lines.extend(["", "### Review flags", ""])
        lines.extend(
            f"- [{md_escape(flag.severity)}] {md_escape(flag.field)} \u2014 {md_escape(flag.message)}"
            for flag in quality.review_flags
        )
    return lines


def _evidence_section(report: Report) -> list[str]:
    lines = ["## Evidence index", ""]
    if not report.evidence_index:
        lines.append("- none")
        return lines
    for evidence_id, item in report.evidence_index.items():
        lines.append(
            f"- [{md_escape(evidence_id)}] {md_escape(_appendix_ref(item))} \u2014 {md_escape(item.excerpt)}"
        )
    return lines


def render_markdown(report: Report) -> str:
    """Render ``report`` as Markdown with every content string escaped (docs/06 S9)."""
    title = report.source.title or _DEFAULT_TITLE
    sections: list[str] = [f"# {md_escape(title)}", ""]
    sections.extend(_source_section(report))
    sections.append("")
    sections.extend(_languages_section(report))
    sections.append("")
    sections.extend(_speakers_section(report))
    sections.append("")
    sections.extend(_summary_section(report))
    sections.append("")
    sections.extend(_context_section(report))
    sections.append("")
    sections.extend(_chapters_section(report))
    sections.append("")
    sections.extend(_topics_section(report))
    sections.append("")
    sections.extend(_entities_section(report))
    sections.append("")
    sections.extend(_quotes_section(report))
    sections.append("")
    sections.extend(_claims_section(report))
    sections.append("")
    sections.extend(_translations_section(report))
    sections.append("")
    sections.extend(_quality_section(report))
    sections.append("")
    sections.extend(_evidence_section(report))
    return "\n".join(sections) + "\n"


__all__ = ["md_escape", "render_markdown"]
