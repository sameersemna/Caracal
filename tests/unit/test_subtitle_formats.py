"""SRT/WebVTT rendering: exact bytes, escaping, long timestamps, round-trip and ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tests.fakes.whisper import load_fixture

from caracal.asr.captions import parse_vtt
from caracal.asr.merge import merge_chunks
from caracal.asr.whisper import parse_whisper_response
from caracal.audio.chunking import Chunk
from caracal.subtitles.formats import to_srt, to_vtt
from caracal.subtitles.segmenter import Cue, cues_from_transcript


def _cue(
    index: int,
    start: float,
    end: float,
    lines: list[str],
    *,
    speaker: str | None = None,
) -> Cue:
    return Cue(index=index, start=start, end=end, lines=lines, speaker=speaker, word_span=(0, 0))


def _ffmpeg_accepts(path: Path) -> None:
    """Let ffmpeg demux and stream-copy the subtitle file; any parse error exits non-zero.

    ``-c:s copy`` is required on ffmpeg 6.1: the null muxer has no subtitle encoder, so the
    card's literal ``-f null -`` command fails with "Output file does not contain any
    stream" even for a hand-written valid SRT (measured on this host).
    """
    binary = shutil.which("ffmpeg")
    assert binary is not None
    command = [
        binary,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-c:s",
        "copy",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(  # noqa: S603 - argv list, local ffmpeg only
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr.strip() == ""


def test_srt_exact_text_for_two_cues() -> None:
    cues = [
        _cue(1, 1.0, 3.5, ["Hello there"]),
        _cue(2, 3.6, 5.04, ["Second line one", "Second line two"], speaker="S1"),
    ]
    assert to_srt(cues) == (
        "1\n"
        "00:00:01,000 --> 00:00:03,500\n"
        "Hello there\n"
        "\n"
        "2\n"
        "00:00:03,600 --> 00:00:05,040\n"
        "Second line one\n"
        "Second line two\n"
    )


def test_vtt_exact_text_for_two_cues() -> None:
    cues = [
        _cue(1, 1.0, 3.5, ["Hello there"]),
        _cue(2, 3.6, 5.04, ["Second line one", "Second line two"], speaker="S1"),
    ]
    assert to_vtt(cues) == (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:03.500\n"
        "Hello there\n"
        "\n"
        "00:00:03.600 --> 00:00:05.040\n"
        "<v S1>Second line one\n"
        "Second line two\n"
    )


def test_vtt_voice_tags_can_be_disabled() -> None:
    cues = [_cue(1, 0.0, 1.0, ["Hi"], speaker="S2")]
    assert "<v S2>" in to_vtt(cues)
    assert "<v S2>" not in to_vtt(cues, voice_tags=False)


def test_vtt_escapes_markup_and_ampersands_while_srt_keeps_the_text() -> None:
    cues = [_cue(1, 0.0, 1.0, ["<script>&"])]
    assert "&lt;script&gt;&amp;" in to_vtt(cues)
    assert "<script>&" in to_srt(cues)


def test_hours_may_exceed_two_digits() -> None:
    cues = [_cue(1, 360_000.0, 360_001.5, ["late"])]
    assert "100:00:00,000 --> 100:00:01,500" in to_srt(cues)
    assert "100:00:00.000 --> 100:00:01.500" in to_vtt(cues)


def test_both_formats_end_with_exactly_one_newline() -> None:
    cues = [_cue(1, 0.0, 1.0, ["a"]), _cue(2, 1.1, 2.0, ["b"])]
    for rendered in (to_srt(cues), to_vtt(cues)):
        assert rendered.endswith("\n")
        assert not rendered.endswith("\n\n")


def test_vtt_round_trips_through_the_platform_caption_parser() -> None:
    chunks = [Chunk(index=0, start_s=0.0, end_s=61.0), Chunk(index=1, start_s=60.0, end_s=120.0)]
    results = [
        parse_whisper_response(load_fixture("talk_c0")),
        parse_whisper_response(load_fixture("talk_c1")),
    ]
    transcript = merge_chunks(results, chunks)
    cues = cues_from_transcript(transcript)
    assert len(cues) > 20

    expected = [token for cue in cues for line in cue.lines for token in line.split()]
    parsed = parse_vtt(to_vtt(cues))

    assert [word.text for word in parsed] == expected


def test_ffmpeg_accepts_the_rendered_files(tmp_path: Path) -> None:
    chunks = [Chunk(index=0, start_s=0.0, end_s=61.0), Chunk(index=1, start_s=60.0, end_s=120.0)]
    results = [
        parse_whisper_response(load_fixture("talk_c0")),
        parse_whisper_response(load_fixture("talk_c1")),
    ]
    cues = cues_from_transcript(merge_chunks(results, chunks))

    srt = tmp_path / "out.srt"
    vtt = tmp_path / "out.vtt"
    srt.write_text(to_srt(cues), encoding="utf-8")
    vtt.write_text(to_vtt(cues), encoding="utf-8")

    _ffmpeg_accepts(srt)
    _ffmpeg_accepts(vtt)
