"""Direct downloads: SSRF pinning, content inspection, caps and redirects (docs/06 S3)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpcore
import httpx
import pytest
from tests.fakes.resolver import PRIVATE_IP, PUBLIC_IP, FakeResolver

from caracal.errors import CaracalError, ErrorCode
from caracal.ingest import direct
from caracal.ingest.direct import PinnedBackend, download_direct
from caracal.ingest.ssrf import PinnedTarget

MEDIA_URL = "https://media.example/audio.mp3"


def _resolver() -> FakeResolver:
    resolver = FakeResolver()
    resolver.add("media.example", PUBLIC_IP)
    return resolver


def _use_mock(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(direct, "_transport_for", lambda _target: httpx.MockTransport(wrapped))
    return requests


async def test_downloads_to_part_then_renames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"hello audio"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=payload)

    _use_mock(monkeypatch, handler)
    dest = tmp_path / "out" / "audio.mp3"
    info = await download_direct(MEDIA_URL, dest, resolver=_resolver(), max_bytes=1024)

    assert dest.read_bytes() == payload
    assert not dest.with_suffix(".mp3.part").exists()
    assert info == direct.DirectInfo(
        final_url=MEDIA_URL,
        content_type="audio/mpeg",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        filename="audio.mp3",
    )


async def test_content_disposition_filename_is_sanitised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "video/mp4",
                "content-disposition": 'attachment; filename="../../My Talk.mp4"',
            },
            content=b"video",
        )

    _use_mock(monkeypatch, handler)
    dest = tmp_path / "clip.mp4"
    info = await download_direct(MEDIA_URL, dest, resolver=_resolver(), max_bytes=1024)
    assert info.filename == "My_Talk.mp4"
    assert dest.read_bytes() == b"video"


async def test_non_media_content_type_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>")

    _use_mock(monkeypatch, handler)
    dest = tmp_path / "page.mp3"
    with pytest.raises(CaracalError) as excinfo:
        await download_direct(MEDIA_URL, dest, resolver=_resolver(), max_bytes=1024)
    assert excinfo.value.code is ErrorCode.UNSUPPORTED_MEDIA
    assert not dest.exists()


async def test_application_octet_stream_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"bytes")

    _use_mock(monkeypatch, handler)
    dest = tmp_path / "blob.bin"
    info = await download_direct(MEDIA_URL, dest, resolver=_resolver(), max_bytes=1024)
    assert info.content_type == "application/octet-stream"
    assert dest.read_bytes() == b"bytes"


class _NoReadStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> Any:
        raise AssertionError("body must not be read when Content-Length exceeds the cap")
        yield b""  # pragma: no cover - makes this an async generator


async def test_declared_length_over_cap_reads_no_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "audio/mpeg", "content-length": "999999"},
            stream=_NoReadStream(),
        )

    _use_mock(monkeypatch, handler)
    dest = tmp_path / "big.mp3"
    with pytest.raises(CaracalError) as excinfo:
        await download_direct(MEDIA_URL, dest, resolver=_resolver(), max_bytes=10)
    assert excinfo.value.code is ErrorCode.TOO_LARGE
    assert not dest.exists()
    assert not dest.with_suffix(".mp3.part").exists()


class _ChunkStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> Any:
        yield b"01234"
        yield b"56789"


async def test_streamed_bytes_over_cap_delete_the_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "audio/mpeg"},
            stream=_ChunkStream(),
        )

    _use_mock(monkeypatch, handler)
    dest = tmp_path / "big.mp3"
    with pytest.raises(CaracalError) as excinfo:
        await download_direct(MEDIA_URL, dest, resolver=_resolver(), max_bytes=7)
    assert excinfo.value.code is ErrorCode.TOO_LARGE
    assert not dest.exists()
    assert not dest.with_suffix(".mp3.part").exists()


async def test_redirect_to_private_address_is_blocked_before_the_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://127.0.0.1:6379/"})

    requests = _use_mock(monkeypatch, handler)
    with pytest.raises(CaracalError) as excinfo:
        await download_direct(MEDIA_URL, tmp_path / "x.mp3", resolver=_resolver(), max_bytes=1024)
    assert excinfo.value.code is ErrorCode.SOURCE_BLOCKED
    assert len(requests) == 1


async def test_redirect_limit_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/next"})

    requests = _use_mock(monkeypatch, handler)
    with pytest.raises(CaracalError) as excinfo:
        await download_direct(
            MEDIA_URL, tmp_path / "x.mp3", resolver=_resolver(), max_bytes=1024, max_redirects=1
        )
    assert excinfo.value.code is ErrorCode.SOURCE_UNAVAILABLE
    assert len(requests) == 2


async def test_non_200_status_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, headers={"content-type": "audio/mpeg"})

    _use_mock(monkeypatch, handler)
    with pytest.raises(CaracalError) as excinfo:
        await download_direct(MEDIA_URL, tmp_path / "x.mp3", resolver=_resolver(), max_bytes=1024)
    assert excinfo.value.code is ErrorCode.SOURCE_UNAVAILABLE
    assert excinfo.value.details["status"] == 404


class _RecordingBackend(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def connect_tcp(self, host: str, port: int, **_: Any) -> Any:
        self.calls.append((host, port))
        return object()

    async def connect_unix_socket(self, path: str, **_: Any) -> Any:
        raise AssertionError("unix sockets must not be used")

    async def sleep(self, seconds: float) -> None:
        return None


async def test_pinned_backend_connects_to_the_ip() -> None:
    recorder = _RecordingBackend()
    backend = PinnedBackend("media.example", PUBLIC_IP, inner=recorder)
    await backend.connect_tcp("media.example", 443)
    assert recorder.calls == [(PUBLIC_IP, 443)]
    assert backend.calls == [("media.example", 443, PUBLIC_IP)]
    assert not any(call[0] == "media.example" for call in recorder.calls)


async def test_pinned_backend_refuses_unexpected_hosts() -> None:
    backend = PinnedBackend("media.example", PUBLIC_IP, inner=_RecordingBackend())
    with pytest.raises(RuntimeError):
        await backend.connect_tcp("other.example", 443)
    with pytest.raises(RuntimeError):
        await backend.connect_unix_socket("unused.sock")


async def test_transport_replaces_the_private_httpx_pool() -> None:
    # Guards the private httpx seam: if ``AsyncHTTPTransport._pool`` disappears this fails.
    transport = direct._pinned_transport(
        PinnedTarget(scheme="https", host="media.example", port=443, ip=PUBLIC_IP)
    )
    pool = transport._pool  # type: ignore[attr-defined]
    assert isinstance(pool, httpcore.AsyncConnectionPool)
    backend = pool._network_backend  # type: ignore[attr-defined]
    assert isinstance(backend, PinnedBackend)
    assert backend.host == "media.example"
    assert backend.ip == PUBLIC_IP
    await transport.aclose()


def test_private_ip_constant_is_not_public() -> None:
    assert PRIVATE_IP.startswith("10.")


async def test_head_content_type_returns_the_media_type(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = _use_mock(
        monkeypatch,
        lambda _request: httpx.Response(200, headers={"content-type": "video/mp4; codecs=avc1"}),
    )
    content_type = await direct.head_content_type("https://media.example/watch", resolver=_resolver())
    assert content_type == "video/mp4"
    assert [request.method for request in requests] == ["HEAD"]


async def test_head_content_type_follows_a_public_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/watch":
            return httpx.Response(302, headers={"location": "/media"})
        return httpx.Response(200, headers={"content-type": "audio/ogg"})

    requests = _use_mock(monkeypatch, handler)
    content_type = await direct.head_content_type("https://media.example/watch", resolver=_resolver())
    assert content_type == "audio/ogg"
    assert [request.url.path for request in requests] == ["/watch", "/media"]
