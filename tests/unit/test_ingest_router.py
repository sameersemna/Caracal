"""``classify_source_static`` (docs/04 §3): extension-only, no network."""

from __future__ import annotations

import pytest

from caracal.ingest.router import classify_source_static


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
