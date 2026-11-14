"""yt-dlp ingest: options, SSRF pre-check, playlist/live/size guards and error mapping.

yt-dlp is replaced at the ``yt_dlp.YoutubeDL`` class boundary by :class:`FakeYoutubeDL`,
which records the options it was constructed with, returns scripted info dicts, creates
the files a real download would create and invokes the progress hooks. The autouse
``_stdout_guard`` fails any test that lets a byte reach stdout: stdout is the MCP stdio
channel, so the fake writes an ``ERROR:`` line there whenever the options lack a logger.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yt_dlp
from tests.fakes.resolver import PRIVATE_IP, PUBLIC_IP, FakeResolver

from caracal.config import Settings
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.ytdlp import (
    CaptionTrack,
    DownloadedAudio,
    PlatformMetadata,
    _Cancelled,
    _YtdlpLogger,
    build_options,
    download_audio,
    fetch_metadata,
    fetch_platform_captions,
)
from caracal.storage import JobFiles

URL = "https://youtube.com/watch?v=abc"
_YTDLP_UTILS = yt_dlp.utils


@pytest.fixture(autouse=True)
def _stdout_guard(capsys: pytest.CaptureFixture[str]) -> Iterator[None]:
    """stdout is the MCP stdio channel: every test must leave it empty."""
    yield
    assert capsys.readouterr().out == ""


@dataclass
class FakeCall:
    url: str
    download: bool
    options: dict[str, Any]


class FakeYoutubeDL:
    def __init__(self, factory: FakeYoutubeDLFactory, options: dict[str, Any], index: int) -> None:
        self._factory = factory
        self.options = options
        self._index = index

    def __enter__(self) -> FakeYoutubeDL:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def extract_info(self, url: str, download: bool = True) -> dict[str, Any]:
        self._factory.calls.append(FakeCall(url=url, download=download, options=self.options))
        if not self.options.get("logger"):
            # Simulate yt-dlp's default terminal output; the adapter must prevent it.
            print("ERROR: yt-dlp wrote to stdout")
        result = self._factory.result(self._index)
        if isinstance(result, BaseException):
            raise result
        match_filter = self.options.get("match_filter")
        if callable(match_filter):
            match_filter(result)
        if download:
            home = Path(self.options["paths"]["home"])
            for name in self._factory.create_files:
                path = home / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(self._factory.file_bytes)
            if self._factory.set_cancel is not None:
                self._factory.set_cancel.set()
            for hook in self.options.get("progress_hooks", ()):
                hook({"status": "downloading"})
        return result


class FakeYoutubeDLFactory:
    """A callable recording every ``YoutubeDL(options)`` construction and its script."""

    def __init__(self) -> None:
        self.script: list[Any] = []
        self.calls: list[FakeCall] = []
        self.constructed = 0
        self.create_files: list[str] = []
        self.file_bytes = b"media"
        self.set_cancel: threading.Event | None = None

    def __call__(self, options: dict[str, Any]) -> FakeYoutubeDL:
        index = self.constructed
        self.constructed += 1
        return FakeYoutubeDL(self, options, index)

    def result(self, index: int) -> Any:
        if index >= len(self.script):
            raise AssertionError(f"no scripted result for YoutubeDL construction {index}")
        return self.script[index]


def _install(monkeypatch: pytest.MonkeyPatch) -> FakeYoutubeDLFactory:
    factory = FakeYoutubeDLFactory()
    monkeypatch.setattr(yt_dlp, "YoutubeDL", factory)
    return factory


def _resolver() -> FakeResolver:
    resolver = FakeResolver()
    resolver.add("youtube.com", PUBLIC_IP)
    resolver.add("latitude", PRIVATE_IP)
    return resolver


def _info(**overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "_type": "video",
        "id": "abc",
        "title": "A talk",
        "duration": 123.0,
        "webpage_url": URL,
    }
    info.update(overrides)
    return info


def _meta(language: str | None = "en") -> PlatformMetadata:
    return PlatformMetadata(
        id="abc",
        title="A talk",
        uploader=None,
        channel=None,
        channel_url=None,
        upload_date=None,
        duration=123.0,
        description=None,
        tags=[],
        categories=[],
        chapters=[],
        language=language,
        view_count=None,
        webpage_url=URL,
        extractor="youtube",
    )


def _files(tmp_path: Path) -> JobFiles:
    files = JobFiles(data_dir=tmp_path, job_id=uuid4())
    files.ensure()
    return files


# --- options builder (docs/06 S4) -----------------------------------------------------


def test_metadata_options_are_exactly_the_hardened_set(make_settings: Any, tmp_path: Path) -> None:
    settings: Settings = make_settings(
        CARACAL_MAX_UPLOAD_MB=10,
        CARACAL_MAX_DURATION_S=60,
        YTDLP_COOKIES_FILE=tmp_path / "cookies.txt",
    )
    options = build_options(settings, dest_dir=tmp_path, cancel=threading.Event(), purpose="metadata")

    assert options["quiet"] is True
    assert options["no_warnings"] is True
    assert options["noprogress"] is True
    assert isinstance(options["logger"], _YtdlpLogger)
    assert options["noplaylist"] is True
    assert options["allowed_extractors"] == ["default", "-generic"]
    assert options["enable_file_urls"] is False
    assert options["socket_timeout"] == 30
    assert options["retries"] == 3
    assert options["fragment_retries"] == 3
    assert options["restrictfilenames"] is True
    assert options["paths"] == {"home": str(tmp_path)}
    assert options["outtmpl"] == {"default": "source.%(ext)s"}
    assert options["max_filesize"] == 10 * 1024 * 1024
    assert options["format"] == "bestaudio/best"
    assert options["postprocessors"] == []
    assert options["skip_download"] is True
    assert options["cookiefile"] == str(tmp_path / "cookies.txt")
    assert len(options["progress_hooks"]) == 1
    assert callable(options["match_filter"])

    for forbidden in ("exec", "postprocessor_args", "external_downloader"):
        assert forbidden not in options
    for template in options["outtmpl"].values():
        assert "%(title" not in template
        assert "%(uploader" not in template


def test_audio_options_do_not_skip_or_write_subtitles(make_settings: Any, tmp_path: Path) -> None:
    settings: Settings = make_settings()
    options = build_options(settings, dest_dir=tmp_path, cancel=threading.Event(), purpose="audio")
    assert "skip_download" not in options
    assert "writesubtitles" not in options
    assert "subtitleslangs" not in options
    assert "cookiefile" not in options


def test_captions_options_request_base_and_original_track(make_settings: Any, tmp_path: Path) -> None:
    settings: Settings = make_settings()
    options = build_options(
        settings,
        dest_dir=tmp_path,
        cancel=threading.Event(),
        purpose="captions",
        lang="en",
    )
    assert options["skip_download"] is True
    assert options["writesubtitles"] is True
    assert options["writeautomaticsub"] is True
    assert options["subtitlesformat"] == "vtt"
    assert options["subtitleslangs"] == ["en", "en-orig"]


def test_match_filter_returns_reasons_for_live_and_long(make_settings: Any, tmp_path: Path) -> None:
    settings: Settings = make_settings(CARACAL_MAX_DURATION_S=60)
    options = build_options(settings, dest_dir=tmp_path, cancel=threading.Event(), purpose="audio")
    match_filter = options["match_filter"]

    assert match_filter({"is_live": True}) == "live streams are not supported"
    assert match_filter({"live_status": "is_upcoming"}) == "live streams are not supported"
    assert match_filter({"duration": 61}) == "source exceeds the maximum duration"
    assert match_filter({"duration": 60}) is None
    assert match_filter({"duration": None}) is None


def test_logger_maps_levels_and_redacts(make_settings: Any, tmp_path: Path, caplog: Any) -> None:
    settings: Settings = make_settings()
    options = build_options(settings, dest_dir=tmp_path, cancel=threading.Event(), purpose="audio")
    logger = options["logger"]

    with caplog.at_level(logging.DEBUG, logger="caracal.ingest.ytdlp"):
        logger.debug("debug hf_abcdefghijklmnopqrstuvwxyz")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")

    records = [record for record in caplog.records if record.name == "caracal.ingest.ytdlp"]
    assert [record.levelno for record in records] == [
        logging.DEBUG,
        logging.DEBUG,
        logging.WARNING,
        logging.WARNING,
    ]
    assert all("hf_" not in record.getMessage() for record in records)


def test_progress_hook_raises_when_the_event_is_set(make_settings: Any, tmp_path: Path) -> None:
    cancel = threading.Event()
    cancel.set()
    options = build_options(settings=make_settings(), dest_dir=tmp_path, cancel=cancel, purpose="audio")
    hook = options["progress_hooks"][0]
    with pytest.raises(_Cancelled):
        hook({"status": "downloading"})


# --- SSRF pre-check (docs/06 S3/S4) ---------------------------------------------------


@pytest.mark.parametrize("url", ["http://127.0.0.1/watch", "http://latitude:6379/"])
async def test_metadata_runs_ssrf_check_before_constructing_ytdlp(
    url: str, make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _install(monkeypatch)
    with pytest.raises(CaracalError) as excinfo:
        await fetch_metadata(url, make_settings(), resolver=_resolver())
    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED
    assert factory.constructed == 0


@pytest.mark.parametrize("url", ["http://127.0.0.1/watch", "http://latitude:6379/"])
async def test_audio_runs_ssrf_check_before_constructing_ytdlp(
    url: str, make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    with pytest.raises(CaracalError) as excinfo:
        await download_audio(
            url,
            _files(tmp_path),
            make_settings(),
            resolver=_resolver(),
            cancel=threading.Event(),
        )
    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED
    assert factory.constructed == 0


@pytest.mark.parametrize("url", ["http://127.0.0.1/watch", "http://latitude:6379/"])
async def test_captions_run_ssrf_check_before_constructing_ytdlp(
    url: str, make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    with pytest.raises(CaracalError) as excinfo:
        await fetch_platform_captions(url, _meta(), _files(tmp_path), make_settings(), resolver=_resolver())
    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED
    assert factory.constructed == 0


# --- playlists, live streams, duration -------------------------------------------------


async def test_playlist_url_is_rejected_without_contacting_ytdlp(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _install(monkeypatch)
    with pytest.raises(CaracalError) as excinfo:
        await fetch_metadata("https://youtube.com/playlist?list=PL123", make_settings(), resolver=_resolver())
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.message == "one item per job"
    assert factory.constructed == 0


@pytest.mark.parametrize("result_type", ["playlist", "multi_video"])
async def test_playlist_info_is_rejected(
    result_type: str, make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _install(monkeypatch)
    factory.script.append(_info(_type=result_type))
    with pytest.raises(CaracalError) as excinfo:
        await fetch_metadata(URL, make_settings(), resolver=_resolver())
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.message == "one item per job"
    assert factory.constructed == 1


async def test_live_stream_is_invalid_input(make_settings: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    factory.script.append(_info(is_live=True))
    with pytest.raises(CaracalError) as excinfo:
        await fetch_metadata(URL, make_settings(), resolver=_resolver())
    assert excinfo.value.code is ErrorCode.INVALID_INPUT
    assert excinfo.value.message == "live streams are not supported"


async def test_duration_over_the_limit_is_too_large(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _install(monkeypatch)
    factory.script.append(_info(duration=101))
    with pytest.raises(CaracalError) as excinfo:
        await fetch_metadata(URL, make_settings(CARACAL_MAX_DURATION_S=100), resolver=_resolver())
    assert excinfo.value.code is ErrorCode.TOO_LARGE
    assert excinfo.value.message == "source exceeds the maximum duration"


# --- metadata mapping -----------------------------------------------------------------


async def test_fetch_metadata_maps_every_documented_field(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    description = "x" * 6_000
    factory.script.append(
        _info(
            uploader="Uploader",
            channel="Channel",
            channel_url="https://youtube.com/@channel",
            upload_date="20260101",
            description=description,
            tags=["tag-a", "tag-b"],
            categories=["Education"],
            chapters=[{"start_time": 0, "title": "Intro"}],
            language="en",
            view_count=42,
            extractor="youtube",
        )
    )

    meta = await fetch_metadata(URL, make_settings(), resolver=_resolver())

    assert meta.id == "abc"
    assert meta.title == "A talk"
    assert meta.uploader == "Uploader"
    assert meta.channel == "Channel"
    assert meta.channel_url == "https://youtube.com/@channel"
    assert meta.upload_date == "20260101"
    assert meta.duration == 123.0
    assert meta.description is not None and len(meta.description) == 5_000
    assert meta.tags == ["tag-a", "tag-b"]
    assert meta.categories == ["Education"]
    assert meta.chapters == [{"start_time": 0, "title": "Intro"}]
    assert meta.language == "en"
    assert meta.view_count == 42
    assert meta.webpage_url == URL
    assert meta.extractor == "youtube"
    assert factory.calls[0].download is False
    assert factory.calls[0].options["skip_download"] is True


async def test_fetch_metadata_missing_fields_become_none_and_empty(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = _install(monkeypatch)
    factory.script.append({"_type": "video"})
    meta = await fetch_metadata(URL, make_settings(), resolver=_resolver())
    assert meta.id is None
    assert meta.title is None
    assert meta.duration is None
    assert meta.description is None
    assert meta.tags == []
    assert meta.categories == []
    assert meta.chapters == []
    assert meta.view_count is None


# --- audio download -------------------------------------------------------------------


async def test_download_audio_keeps_one_source_file_and_hashes_it(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    factory.file_bytes = b"audio-bytes"
    factory.create_files = ["source.webm"]
    factory.script.append(_info())
    files = _files(tmp_path)

    result = await download_audio(URL, files, make_settings(), resolver=_resolver(), cancel=threading.Event())

    assert result == DownloadedAudio(
        path=files.source_dir / "source.webm",
        ext="webm",
        size_bytes=len(b"audio-bytes"),
        sha256=hashlib.sha256(b"audio-bytes").hexdigest(),
    )
    assert files.source_dir.joinpath("source.webm").read_bytes() == b"audio-bytes"
    assert factory.calls[0].download is True
    assert "skip_download" not in factory.calls[0].options


async def test_download_audio_removes_stale_source_files(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    factory.create_files = ["source.m4a"]
    factory.script.append(_info())
    files = _files(tmp_path)
    stale = files.source_dir / "source.old"
    stale.write_bytes(b"old")

    await download_audio(URL, files, make_settings(), resolver=_resolver(), cancel=threading.Event())

    assert not stale.exists()
    assert sorted(path.name for path in files.source_dir.iterdir()) == ["source.m4a"]


async def test_cancel_aborts_the_download_and_removes_the_partial_file(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    cancel = threading.Event()
    factory.set_cancel = cancel
    factory.create_files = ["source.webm.part"]
    factory.script.append(_info())
    files = _files(tmp_path)

    with pytest.raises(CaracalError) as excinfo:
        await download_audio(URL, files, make_settings(), resolver=_resolver(), cancel=cancel)

    assert excinfo.value.code is ErrorCode.SOURCE_UNAVAILABLE
    assert excinfo.value.message == "download cancelled"
    assert list(files.source_dir.iterdir()) == []


# --- captions -------------------------------------------------------------------------


async def test_manual_captions_beat_automatic(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    files = _files(tmp_path)
    captions_file = files.captions_dir / "source.en.vtt"
    factory.create_files = ["source.en.vtt", "source.en-orig.vtt"]
    factory.script.append(_info(subtitles={"en": [{"ext": "vtt"}]}, automatic_captions={"en-orig": []}))
    factory.script.append(_info(requested_subtitles={"en": {"filepath": str(captions_file)}}))

    tracks = await fetch_platform_captions(URL, _meta("en"), files, make_settings(), resolver=_resolver())

    assert tracks == [CaptionTrack(lang="en", kind="manual", path=files.captions("en"))]
    assert files.captions("en").read_bytes() == b"media"
    assert sorted(path.name for path in files.captions_dir.iterdir()) == ["en.vtt"]
    assert factory.calls[0].download is False
    assert factory.calls[1].download is True
    assert factory.calls[1].options["subtitleslangs"] == ["en", "en-orig"]


async def test_original_automatic_caption_beats_translated_one(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    files = _files(tmp_path)
    captions_file = files.captions_dir / "source.en-orig.vtt"
    factory.create_files = ["source.en-orig.vtt"]
    factory.script.append(
        _info(
            subtitles={},
            automatic_captions={"en-de": [{"ext": "vtt"}], "en-orig": [{"ext": "vtt"}]},
        )
    )
    factory.script.append(_info(requested_subtitles={"en-orig": {"filepath": str(captions_file)}}))

    tracks = await fetch_platform_captions(URL, _meta("en"), files, make_settings(), resolver=_resolver())

    assert tracks == [CaptionTrack(lang="en", kind="auto", path=files.captions("en"))]
    assert files.captions("en").read_bytes() == b"media"


async def test_first_manual_language_is_used_when_metadata_has_none(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    files = _files(tmp_path)
    captions_file = files.captions_dir / "source.de.vtt"
    factory.create_files = ["source.de.vtt"]
    factory.script.append(_info(subtitles={"de": [{"ext": "vtt"}]}, automatic_captions={}))
    factory.script.append(_info(requested_subtitles={"de": {"filepath": str(captions_file)}}))

    tracks = await fetch_platform_captions(URL, _meta(None), files, make_settings(), resolver=_resolver())

    assert tracks == [CaptionTrack(lang="de", kind="manual", path=files.captions("de"))]


async def test_no_captions_is_empty_and_does_not_download(
    make_settings: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = _install(monkeypatch)
    factory.script.append(_info(subtitles={}, automatic_captions={}))
    tracks = await fetch_platform_captions(
        URL, _meta("en"), _files(tmp_path), make_settings(), resolver=_resolver()
    )
    assert tracks == []
    assert factory.constructed == 1


# --- error mapping --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected_code", "expected_message", "retryable"),
    [
        (
            "ERROR: [youtube] abc: Private video. https://youtube.com/watch?v=abc",
            ErrorCode.SOURCE_UNAVAILABLE,
            "video is private",
            False,
        ),
        (
            "ERROR: [youtube] abc: Video unavailable https://youtube.com/watch?v=abc",
            ErrorCode.SOURCE_UNAVAILABLE,
            "video is unavailable",
            False,
        ),
        (
            "ERROR: This video has been removed by the uploader",
            ErrorCode.SOURCE_UNAVAILABLE,
            "video is unavailable",
            False,
        ),
        (
            "ERROR: The uploader has not made this video available in your country",
            ErrorCode.SOURCE_UNAVAILABLE,
            "video is geo-restricted",
            False,
        ),
        (
            "ERROR: This video is age-restricted",
            ErrorCode.SOURCE_UNAVAILABLE,
            "video is age-restricted",
            False,
        ),
        (
            "ERROR: Sign in to confirm you're not a bot",
            ErrorCode.SOURCE_UNAVAILABLE,
            "authentication required",
            False,
        ),
        (
            "ERROR: No suitable extractor found for URL http://latitude:6379/",
            ErrorCode.SOURCE_UNAVAILABLE,
            "unsupported site",
            False,
        ),
        (
            "ERROR: File is larger than max-filesize (1000 bytes)",
            ErrorCode.TOO_LARGE,
            "source exceeds the maximum size",
            False,
        ),
        (
            "ERROR: unable to download webpage: HTTP Error 503: Service Unavailable",
            ErrorCode.SOURCE_UNAVAILABLE,
            "network error",
            True,
        ),
        (
            "ERROR: The read operation timed out",
            ErrorCode.SOURCE_UNAVAILABLE,
            "network error",
            True,
        ),
        (
            "ERROR: Connection reset by peer",
            ErrorCode.SOURCE_UNAVAILABLE,
            "network error",
            True,
        ),
        (
            "ERROR: Unable to download webpage: TLS fingerprint mismatch",
            ErrorCode.SOURCE_UNAVAILABLE,
            "site blocked automated access",
            False,
        ),
        (
            "ERROR: impersonation request failed",
            ErrorCode.SOURCE_UNAVAILABLE,
            "site blocked automated access",
            False,
        ),
        (
            "ERROR: Unable to download webpage: HTTP Error 403: Forbidden",
            ErrorCode.SOURCE_UNAVAILABLE,
            "site blocked automated access",
            False,
        ),
        (
            "ERROR: HTTP Error 403: Forbidden - please sign in to continue",
            ErrorCode.SOURCE_UNAVAILABLE,
            "authentication required",
            False,
        ),
        (
            "ERROR: unexpected explosion at https://example.com/watch?v=abc",
            ErrorCode.SOURCE_UNAVAILABLE,
            "source could not be downloaded",
            False,
        ),
    ],
)
async def test_download_error_mapping_is_short_and_url_free(
    message: str,
    expected_code: ErrorCode,
    expected_message: str,
    retryable: bool,
    make_settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _install(monkeypatch)
    factory.script.append(_YTDLP_UTILS.DownloadError(message))

    with pytest.raises(CaracalError) as excinfo:
        await fetch_metadata(URL, make_settings(), resolver=_resolver())

    assert excinfo.value.code is expected_code
    assert excinfo.value.message == expected_message
    assert excinfo.value.retryable is retryable
    assert "http" not in excinfo.value.message
    assert "https://" not in excinfo.value.message


async def test_extractor_error_is_mapped_too(make_settings: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    factory = _install(monkeypatch)
    factory.script.append(_YTDLP_UTILS.ExtractorError("ERROR: Private video"))
    with pytest.raises(CaracalError) as excinfo:
        await fetch_metadata(URL, make_settings(), resolver=_resolver())
    assert excinfo.value.code is ErrorCode.SOURCE_UNAVAILABLE
    assert excinfo.value.message == "video is private"


def test_curl_cffi_is_installed_for_ytdlp_impersonation() -> None:
    """Verdict 09 carry-forward: yt-dlp uses curl-cffi automatically when present."""
    import curl_cffi

    assert curl_cffi.__version__ == "0.17.0"
