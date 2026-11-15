"""``classify_source_static`` (docs/04 §3): extension-only, no network."""

from __future__ import annotations

import pytest

from caracal.ingest.router import canonical_url, classify_source_static


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/audio.mp3",
        "http://example.com/a/b/talk.m4a",
        "https://example.com/song.AAC",
        "https://example.com/sound.wav",
        "https://example.com/sound.flac?x=1",
        "https://example.com/sound.ogg#frag",
        "https://example.com/sound.opus",
        "https://example.com/clip.webm",
        "https://example.com/clip.mp4",
        "https://example.com/clip.mkv",
        "https://example.com/clip.mov",
    ],
)
def test_media_extensions_are_direct(url: str) -> None:
    assert classify_source_static(url) == "direct"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc",
        "https://example.com/talk",
        "https://example.com/talk.mp3/",
        "https://example.com/talk.mp3.html",
        "https://example.com/audio.mp3.txt",
        "ftp://example.com/audio.mp3",
        "file:///tmp/audio.mp3",
        "not a url",
        "https://example.com/",
    ],
)
def test_everything_else_is_platform(url: str) -> None:
    assert classify_source_static(url) == "platform"


# --- canonical URL (T4.4): YouTube shapes and tracking parameters ---

_VIDEO_ID = "dQw4w9WgXcQ"
_WATCH = f"https://www.youtube.com/watch?v={_VIDEO_ID}"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={_VIDEO_ID}",
        f"https://youtube.com/watch?v={_VIDEO_ID}&t=42s",
        f"https://m.youtube.com/watch?v={_VIDEO_ID}&si=tracking",
        f"https://music.youtube.com/watch?v={_VIDEO_ID}&feature=share",
        f"https://youtu.be/{_VIDEO_ID}",
        f"https://www.youtube.com/shorts/{_VIDEO_ID}",
        f"https://www.youtube.com/embed/{_VIDEO_ID}",
        f"https://www.youtube.com/live/{_VIDEO_ID}?si=tracking",
        f"https://www.youtube.com/v/{_VIDEO_ID}",
        f"HTTPS://WWW.YouTube.COM/watch?v={_VIDEO_ID}#t=9",
    ],
)
def test_youtube_shapes_canonicalise_to_one_watch_url(url: str) -> None:
    assert canonical_url(url) == _WATCH


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "HTTPS://Example.COM/Path/To?utm_source=x&id=7&fbclid=y#frag",
            "https://example.com/Path/To?id=7",
        ),
        ("https://example.com/talk?t=42&start=3", "https://example.com/talk"),
        ("https://example.com/watch?v=abc&gclid=z", "https://example.com/watch?v=abc"),
        ("https://example.com/talk", "https://example.com/talk"),
    ],
)
def test_non_youtube_urls_keep_their_path_and_clean_the_query(url: str, expected: str) -> None:
    assert canonical_url(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://youtu.be/too-short?si=x", "https://youtu.be/too-short"),
        ("https://m.youtube.com/shorts/abc?t=1#frag", "https://m.youtube.com/shorts/abc"),
        ("https://www.youtube.com/watch?v=abc", "https://www.youtube.com/watch?v=abc"),
    ],
)
def test_youtube_url_without_a_valid_id_is_only_cleaned(url: str, expected: str) -> None:
    assert canonical_url(url) == expected


# --- async router: platform hosts, static extensions and the HEAD probe (T1.1) ---

from tests.fakes.resolver import PUBLIC_IP, FakeResolver  # noqa: E402

from caracal.errors import CaracalError, ErrorCode  # noqa: E402
from caracal.ingest.router import classify_source  # noqa: E402


class _Head:
    def __init__(self, content_type: str | None = "audio/mpeg", error: Exception | None = None) -> None:
        self.content_type = content_type
        self.error = error
        self.calls: list[str] = []

    async def __call__(self, url: str) -> str | None:
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return self.content_type


def _resolver() -> FakeResolver:
    resolver = FakeResolver()
    resolver.add("example.com", PUBLIC_IP)
    return resolver


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://vimeo.com/123",
        "https://on.soundcloud.com/x",
        "https://www.tiktok.com/@a/video/1",
        "https://www.instagram.com/reel/x",
        "https://www.facebook.com/watch?v=1",
        "https://fb.watch/abc",
        "https://x.com/a/status/1",
        "https://twitter.com/a/status/1",
        "https://www.twitch.tv/videos/1",
        "https://www.dailymotion.com/video/x",
        "https://www.bilibili.com/video/BV1",
        "https://rumble.com/v1",
    ],
)
async def test_platform_hosts_need_no_network(url: str) -> None:
    head = _Head()
    assert await classify_source(url, resolver=_resolver(), http_head=head) == "platform"
    assert head.calls == []


async def test_static_media_extension_is_direct_without_a_probe() -> None:
    head = _Head()
    assert (
        await classify_source("https://example.com/a.mp3", resolver=_resolver(), http_head=head) == "direct"
    )
    assert head.calls == []


async def test_media_head_response_is_direct() -> None:
    head = _Head("video/mp4; charset=binary")
    assert (
        await classify_source("https://example.com/watch", resolver=_resolver(), http_head=head) == "direct"
    )
    assert head.calls == ["https://example.com/watch"]


async def test_html_head_response_is_platform() -> None:
    head = _Head("text/html")
    assert (
        await classify_source("https://example.com/article", resolver=_resolver(), http_head=head)
        == "platform"
    )


async def test_head_failure_falls_back_to_platform() -> None:
    head = _Head(error=RuntimeError("boom"))
    assert await classify_source("https://example.com/x", resolver=_resolver(), http_head=head) == "platform"


async def test_unavailable_head_falls_back_to_platform() -> None:
    head = _Head(error=CaracalError(ErrorCode.SOURCE_UNAVAILABLE, "dns"))
    assert await classify_source("https://example.com/x", resolver=_resolver(), http_head=head) == "platform"


async def test_blocked_head_propagates() -> None:
    head = _Head(error=CaracalError(ErrorCode.SOURCE_BLOCKED, "blocked"))
    with pytest.raises(CaracalError) as excinfo:
        await classify_source("https://example.com/x", resolver=_resolver(), http_head=head)
    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED


async def test_lookalike_platform_suffix_is_probed() -> None:
    head = _Head("text/html")
    assert (
        await classify_source("https://notyoutube.com/x", resolver=_resolver(), http_head=head) == "platform"
    )
    assert head.calls == ["https://notyoutube.com/x"]


async def test_non_http_scheme_is_platform_without_a_probe() -> None:
    head = _Head()
    assert (
        await classify_source("ftp://example.com/a.mp3", resolver=_resolver(), http_head=head) == "platform"
    )
    assert head.calls == []
