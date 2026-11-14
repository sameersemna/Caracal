"""Real-local ffmpeg/ffprobe behaviour: probe, normalize, analyse, cut and hardening.

Media is generated in ``tmp_path`` with lavfi (see ``tests/fixtures/media/README.md``); no
binary fixtures are committed and no network is used.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from caracal.audio import ffmpeg
from caracal.audio.ffmpeg import analyse, cut, normalize, probe
from caracal.errors import CaracalError, ErrorCode


def _ffmpeg(*args: str, cwd: Path | None = None) -> None:
    """Run the local ffmpeg for fixture generation (argv list, never a shell)."""
    binary = shutil.which("ffmpeg")
    assert binary is not None
    completed = subprocess.run(  # noqa: S603 - argv list, local test fixture only
        [binary, "-hide_banner", "-nostdin", "-v", "error", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(f"fixture ffmpeg failed: {completed.stderr}")


def _tone(
    path: Path,
    *,
    duration_s: float = 1.2,
    rate: int = 16000,
    channels: int = 1,
    metadata: dict[str, str] | None = None,
) -> Path:
    args = [
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration_s}",
        "-ac",
        str(channels),
        "-ar",
        str(rate),
    ]
    for key, value in (metadata or {}).items():
        args.extend(["-metadata", f"{key}={value}"])
    args.append(str(path))
    _ffmpeg(*args)
    return path


def _tone_silence_tone(path: Path) -> Path:
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono:d=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-filter_complex",
        "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
        "-map",
        "[out]",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(path),
    )
    return path


def _silence(path: Path, *, duration_s: float = 2.0) -> Path:
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=16000:cl=mono:d={duration_s}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(path),
    )
    return path


def _video_only(path: Path) -> Path:
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=1:size=160x120:rate=10",
        "-c:v",
        "mpeg4",
        "-an",
        str(path),
    )
    return path


@pytest.fixture
def recorded_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record every argv while still executing it."""
    calls: list[list[str]] = []
    real = asyncio.create_subprocess_exec

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        calls.append([str(arg) for arg in args])
        return await real(*args, **kwargs)

    monkeypatch.setattr(ffmpeg.asyncio, "create_subprocess_exec", wrapper)
    return calls


def _assert_whitelists(calls: list[list[str]]) -> None:
    assert calls, "expected every ffmpeg/ffprobe invocation to be recorded"
    for argv in calls:
        assert argv[0].endswith(("ffmpeg", "ffprobe"))
        protocol_index = argv.index("-protocol_whitelist")
        assert argv[protocol_index + 1] == "file"
        format_index = argv.index("-format_whitelist")
        assert argv[format_index + 1] == ffmpeg._ALLOWED_FORMATS
        if "-i" in argv:
            input_index = argv.index("-i")
            assert protocol_index < input_index
            assert format_index < input_index


def _assert_clean_failure(dst: Path) -> None:
    assert not dst.exists()
    assert not dst.with_name(f"{dst.name}.part").exists()


async def test_probe_reads_first_audio_stream(tmp_path: Path) -> None:
    source = _tone(tmp_path / "stereo.m4a", rate=44100, channels=2)
    info = await probe(source)

    assert info.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
    assert info.audio_streams == 1
    assert info.video_streams == 0
    assert info.codec == "aac"
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert info.duration_s == pytest.approx(1.2, abs=0.05)
    assert info.bit_rate is not None and info.bit_rate > 0


async def test_probe_keeps_only_allowed_lowercased_tags(tmp_path: Path) -> None:
    source = _tone(
        tmp_path / "tagged.mp3",
        metadata={
            "title": "Fixture Title",
            "artist": "Fixture Artist",
            "publisher": "Fixture Publisher",
            "comment": "Fixture comment",
            "copyright": "must be dropped",
            "encoder": "Lavfi",
        },
    )
    info = await probe(source)

    assert info.tags["title"] == "Fixture Title"
    assert info.tags["artist"] == "Fixture Artist"
    assert info.tags["publisher"] == "Fixture Publisher"
    assert info.tags["comment"] == "Fixture comment"
    assert "copyright" not in info.tags
    assert all(key == key.lower() for key in info.tags)
    assert all(len(value) <= 500 for value in info.tags.values())


async def test_normalize_produces_16k_mono_pcm_wav(tmp_path: Path, recorded_calls: list[list[str]]) -> None:
    source = _tone(tmp_path / "stereo.mp3", duration_s=1.2, rate=44100, channels=2)
    source_info = await probe(source)
    dst = tmp_path / "audio" / "audio.wav"

    returned = await normalize(source, dst, max_duration_s=60)
    output = await probe(dst)

    assert returned == source_info
    assert output.format_name == "wav"
    assert output.codec == "pcm_s16le"
    assert output.sample_rate == 16000
    assert output.channels == 1
    assert output.audio_streams == 1
    assert output.duration_s == pytest.approx(source_info.duration_s or 0.0, abs=0.05)
    assert not dst.with_name("audio.wav.part").exists()
    _assert_whitelists(recorded_calls)


async def test_normalize_rejects_over_duration_before_writing(tmp_path: Path) -> None:
    source = _tone(tmp_path / "three.wav", duration_s=3.0)
    dst = tmp_path / "out" / "audio.wav"

    with pytest.raises(CaracalError) as excinfo:
        await normalize(source, dst, max_duration_s=2)

    assert excinfo.value.code is ErrorCode.TOO_LARGE
    _assert_clean_failure(dst)


def _hostile_inputs(directory: Path) -> dict[str, Path]:
    playlist = directory / "evil.m3u8"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:1\n"
        "#EXTINF:1.0,\nhttp://127.0.0.1:9/x.ts\n#EXT-X-ENDLIST\n",
        encoding="utf-8",
    )
    concat = directory / "evil.wav"
    concat.write_text("ffconcat version 1.0\nfile '/etc/hostname'\n", encoding="utf-8")
    noise = directory / "noise.mp3"
    noise.write_bytes(bytes(range(256)) * 64)
    empty = directory / "empty.mp3"
    empty.write_bytes(b"")
    return {"hls": playlist, "concat": concat, "noise": noise, "empty": empty}


@pytest.mark.parametrize("kind", ["hls", "concat", "noise", "empty"])
async def test_hostile_inputs_are_unsupported_and_leave_nothing_behind(
    tmp_path: Path, kind: str, recorded_calls: list[list[str]]
) -> None:
    source = _hostile_inputs(tmp_path)[kind]
    dst = tmp_path / "out" / "audio.wav"

    with pytest.raises(CaracalError) as probe_error:
        await probe(source)
    assert probe_error.value.code is ErrorCode.UNSUPPORTED_MEDIA

    with pytest.raises(CaracalError) as normalize_error:
        await normalize(source, dst, max_duration_s=60)
    assert normalize_error.value.code is ErrorCode.UNSUPPORTED_MEDIA
    _assert_clean_failure(dst)
    _assert_whitelists(recorded_calls)


async def test_video_without_audio_is_unsupported(tmp_path: Path) -> None:
    source = _video_only(tmp_path / "silent.mp4")

    with pytest.raises(CaracalError) as excinfo:
        await probe(source)

    assert excinfo.value.code is ErrorCode.UNSUPPORTED_MEDIA
    assert excinfo.value.message == "no audio stream"


async def test_analyse_finds_one_silence_and_speech_ratio(
    tmp_path: Path, recorded_calls: list[list[str]]
) -> None:
    source = _tone_silence_tone(tmp_path / "tone-silence-tone.wav")

    stats = await analyse(source)

    assert stats.duration_s == pytest.approx(6.0, abs=0.1)
    assert len(stats.silences) == 1
    start, end = stats.silences[0]
    assert start == pytest.approx(2.0, abs=0.15)
    assert end == pytest.approx(4.0, abs=0.15)
    assert stats.speech_ratio == pytest.approx(0.67, abs=0.05)
    assert stats.quality in {"good", "fair"}
    _assert_whitelists(recorded_calls)


async def test_analyse_pure_silence_is_poor(tmp_path: Path) -> None:
    source = _silence(tmp_path / "silence.wav", duration_s=2.0)

    stats = await analyse(source)

    assert stats.duration_s == pytest.approx(2.0, abs=0.1)
    assert stats.speech_ratio < 0.05
    assert stats.quality == "poor"


async def test_cut_extracts_the_requested_range(tmp_path: Path) -> None:
    source = _tone(tmp_path / "source.wav", duration_s=4.0)
    dst = tmp_path / "chunks" / "part.wav"

    await cut(source, dst, 1.0, 2.5)
    info = await probe(dst)

    assert info.format_name == "wav"
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.duration_s == pytest.approx(1.5, abs=0.05)
    assert not dst.with_name("part.wav.part").exists()


@pytest.mark.parametrize(("start_s", "end_s"), [(-0.1, 1.0), (1.0, 1.0), (2.0, 1.0)])
async def test_cut_rejects_invalid_ranges(tmp_path: Path, start_s: float, end_s: float) -> None:
    source = _tone(tmp_path / "source.wav")
    dst = tmp_path / "part.wav"

    with pytest.raises(CaracalError) as excinfo:
        await cut(source, dst, start_s, end_s)

    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    _assert_clean_failure(dst)


async def test_run_kills_the_process_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    processes: list[asyncio.subprocess.Process] = []
    real = asyncio.create_subprocess_exec

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        process = await real(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(ffmpeg.asyncio, "create_subprocess_exec", wrapper)
    argv = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=60",
        "-f",
        "null",
        "-",
    ]

    with pytest.raises(CaracalError) as excinfo:
        await ffmpeg._run(argv, timeout_s=0.01)

    assert excinfo.value.code is ErrorCode.INTERNAL
    assert excinfo.value.message == "ffmpeg timed out"
    assert excinfo.value.retryable is False
    assert len(processes) == 1
    assert processes[0].returncode is not None


async def test_missing_binary_is_dependency_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _name: None)

    with pytest.raises(CaracalError) as excinfo:
        await probe(tmp_path / "anything.wav")

    assert excinfo.value.code is ErrorCode.DEPENDENCY_DOWN
    assert excinfo.value.details["binary"] == "ffprobe"


async def test_process_runner_uses_argv_exec_and_devnull_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    real = asyncio.create_subprocess_exec

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        recorded.append((args, kwargs))
        return await real(*args, **kwargs)

    monkeypatch.setattr(ffmpeg.asyncio, "create_subprocess_exec", wrapper)

    await probe(_tone(tmp_path / "clip.wav"))

    assert recorded
    for args, kwargs in recorded:
        assert isinstance(args[0], str)
        assert kwargs["stdin"] is asyncio.subprocess.DEVNULL
        assert kwargs["stdout"] is asyncio.subprocess.PIPE
        assert kwargs["stderr"] is asyncio.subprocess.PIPE
        assert "shell" not in kwargs


async def test_normalize_removes_the_partial_file_when_conversion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tone(tmp_path / "source.wav", duration_s=1.0)
    dst = tmp_path / "out" / "audio.wav"
    real_run = ffmpeg._run

    async def failing_run(
        argv: list[str], *, timeout_s: float, stderr_limit: int | None = 4096
    ) -> tuple[bytes, bytes]:
        if argv[0] == "ffprobe":
            return await real_run(argv, timeout_s=timeout_s, stderr_limit=stderr_limit)
        await asyncio.to_thread(Path(argv[-1]).write_bytes, b"partial output")
        raise ffmpeg._ProcessFailed(1, b"", b"decode failed")

    monkeypatch.setattr(ffmpeg, "_run", failing_run)

    with pytest.raises(CaracalError) as excinfo:
        await normalize(source, dst, max_duration_s=60)

    assert excinfo.value.code is ErrorCode.UNSUPPORTED_MEDIA
    _assert_clean_failure(dst)
