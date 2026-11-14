"""Live yt-dlp ingest against one public video (``CARACAL_LIVE_YT_URL``).

Run by the judge, never by the gate (``-m live``). The test exercises the real extractor
and downloader: metadata, platform captions (when the video has them) and the audio file.
stdout must stay empty because it is the MCP stdio channel.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from uuid import uuid4

import pytest

from caracal.config import Settings, get_settings
from caracal.ingest.ssrf import SystemResolver
from caracal.ingest.ytdlp import download_audio, fetch_metadata, fetch_platform_captions
from caracal.storage import JobFiles

pytestmark = pytest.mark.live

_LIVE_URL = os.environ.get("CARACAL_LIVE_YT_URL")


@pytest.mark.skipif(not _LIVE_URL, reason="CARACAL_LIVE_YT_URL is not set")
async def test_live_metadata_captions_and_audio(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    settings: Settings = get_settings()
    resolver = SystemResolver()
    files = JobFiles(data_dir=tmp_path, job_id=uuid4())
    files.ensure()

    meta = await fetch_metadata(_LIVE_URL, settings, resolver=resolver)
    assert meta.title
    assert meta.webpage_url

    tracks = await fetch_platform_captions(_LIVE_URL, meta, files, settings, resolver=resolver)
    for track in tracks:
        assert track.path.exists()
        assert track.path == files.captions(track.lang)
        assert track.path.suffix == ".vtt"
        assert track.path.stem == track.lang

    audio = await download_audio(_LIVE_URL, files, settings, resolver=resolver, cancel=threading.Event())
    assert audio.path.exists()
    assert audio.path.parent == files.source_dir
    assert audio.path.name.startswith("source.")
    assert audio.size_bytes == audio.path.stat().st_size
    assert audio.ext
    assert len(audio.sha256) == 64
    int(audio.sha256, 16)  # hexadecimal
    digest = hashlib.sha256()
    with audio.path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == audio.sha256

    captured = capsys.readouterr()
    assert captured.out == ""
