"""Pinned-connection download of direct media URLs (docs/04 §3, docs/06 S3).

Every hop is validated by :func:`caracal.ingest.ssrf.resolve_and_check` and fetched through
a transport whose TCP connect goes to the *pinned* IP. TLS still verifies the certificate
against the original hostname because httpcore passes ``server_hostname=<origin host>`` to
``start_tls``; only the socket destination is replaced.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import httpcore
import httpx
from python_multipart.multipart import parse_options_header

from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.ssrf import (
    IpPolicy,
    PinnedTarget,
    Resolver,
    default_ip_policy,
    resolve_and_check,
)
from caracal.ingest.upload import sanitize_filename

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MEDIA_PREFIXES = ("audio/", "video/")
_MEDIA_EXACT = frozenset({"application/octet-stream", "application/ogg"})
DEFAULT_HEAD_TIMEOUT_S = 10.0

HttpHead = Callable[[str], Awaitable[str | None]]


@dataclass(frozen=True)
class DirectInfo:
    """Result of a completed direct download."""

    final_url: str
    content_type: str
    size_bytes: int
    sha256: str
    filename: str


class PinnedBackend(httpcore.AsyncNetworkBackend):
    """Network backend that replaces the origin hostname with the pinned IP.

    ``httpcore`` calls ``connect_tcp(host, port)`` with the origin hostname. This backend
    forwards the call to the pinned IP when the hostname matches the validated one and
    refuses every other host, so a proxy-style redirect cannot reach a second address.
    """

    def __init__(
        self,
        host: str,
        ip: str,
        *,
        inner: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self.host = host
        self.ip = ip
        self.calls: list[tuple[str, int, str]] = []
        self._inner = inner if inner is not None else httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore protocol name
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.calls.append((host, port, self.ip))
        if host != self.host:
            raise RuntimeError(f"refusing to connect to unexpected host {host!r}")
        return await self._inner.connect_tcp(
            self.ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 - httpcore protocol name
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise RuntimeError("unix sockets are not used for direct downloads")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def _pinned_transport(target: PinnedTarget) -> httpx.AsyncBaseTransport:
    """Build an httpx transport whose pool uses a :class:`PinnedBackend`.

    httpx 0.28 has no public ``network_backend`` argument on ``AsyncHTTPTransport``, so
    the pool is replaced after construction. ``tests/unit/test_direct.py`` asserts that
    ``_pool`` still exists so an httpx upgrade cannot silently drop the pinning.
    """
    from httpx import AsyncHTTPTransport

    transport = AsyncHTTPTransport()
    # Private on purpose: this is the only seam httpx exposes in 0.28.1. The pool keeps
    # httpx's own SSL context, which preserves certificate verification and SNI.
    pool = transport._pool
    if not isinstance(pool, httpcore.AsyncConnectionPool):
        raise CaracalError(ErrorCode.INTERNAL, "unexpected httpx connection pool")
    transport._pool = httpcore.AsyncConnectionPool(
        ssl_context=pool._ssl_context,
        max_connections=1,
        max_keepalive_connections=0,
        network_backend=PinnedBackend(target.host, target.ip),
    )
    return transport


# Tests replace this seam with ``httpx.MockTransport`` to drive HTTP behaviour without a
# network; production always connects through the pinned backend above.
def _transport_for(target: PinnedTarget) -> httpx.AsyncBaseTransport:
    return _pinned_transport(target)


def _media_type(value: str | None) -> str:
    if not value:
        return ""
    return value.split(";", 1)[0].strip().lower()


def _is_media_type(content_type: str) -> bool:
    return content_type.startswith(_MEDIA_PREFIXES) or content_type in _MEDIA_EXACT


def _declared_length(headers: httpx.Headers) -> int | None:
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _too_large(max_bytes: int) -> CaracalError:
    return CaracalError(ErrorCode.TOO_LARGE, "source exceeds the maximum size", {"max_bytes": max_bytes})


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _unavailable(message: str, host: str, **details: object) -> CaracalError:
    return CaracalError(ErrorCode.SOURCE_UNAVAILABLE, message, {"host": host, **details})


def _filename(url: str, headers: httpx.Headers) -> str:
    candidate = ""
    disposition = headers.get("content-disposition")
    if disposition:
        _kind, params = parse_options_header(disposition)
        raw = params.get(b"filename")
        if raw is not None:
            candidate = raw.decode("utf-8", "replace")
    if not candidate:
        candidate = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    return sanitize_filename(candidate)


async def download_direct(
    url: str,
    dest: Path,
    *,
    resolver: Resolver,
    policy: IpPolicy = default_ip_policy,
    max_bytes: int,
    timeout_s: float = 60,
    max_redirects: int = 5,
) -> DirectInfo:
    """Download ``url`` to ``dest`` through pinned, SSRF-checked connections.

    Redirects are followed manually; ``Location`` is resolved against the current URL and
    revalidated. The body is streamed to ``<dest>.part`` while hashing, with the size cap
    enforced before and during the transfer, then moved into place atomically.
    """
    current = url
    redirects = 0
    while True:
        target = await resolve_and_check(current, resolver=resolver, policy=policy)
        transport = _transport_for(target)
        try:
            async with httpx.AsyncClient(
                transport=transport, timeout=timeout_s, follow_redirects=False
            ) as client:
                async with client.stream("GET", current) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            raise _unavailable("redirect without a Location header", target.host)
                        if redirects >= max_redirects:
                            raise _unavailable("too many redirects", target.host)
                        redirects += 1
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise _unavailable(
                            f"unexpected HTTP status {response.status_code}",
                            target.host,
                            status=response.status_code,
                        )
                    content_type = _media_type(response.headers.get("content-type"))
                    if not _is_media_type(content_type):
                        raise CaracalError(
                            ErrorCode.UNSUPPORTED_MEDIA,
                            "source is not an audio or video file",
                            {"content_type": content_type, "host": target.host},
                        )
                    declared = _declared_length(response.headers)
                    if declared is not None and declared > max_bytes:
                        raise _too_large(max_bytes)

                    filename = _filename(current, response.headers)
                    return await _stream_to_file(
                        response,
                        dest,
                        final_url=current,
                        content_type=content_type,
                        filename=filename,
                        max_bytes=max_bytes,
                    )
        finally:
            await transport.aclose()


async def _stream_to_file(
    response: httpx.Response,
    dest: Path,
    *,
    final_url: str,
    content_type: str,
    filename: str,
    max_bytes: int,
) -> DirectInfo:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(f"{dest.suffix}.part")
    digest = hashlib.sha256()
    size = 0
    try:
        with partial.open("wb") as handle:
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise _too_large(max_bytes)
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, dest)
    except BaseException:
        _unlink_quietly(partial)
        raise
    return DirectInfo(
        final_url=final_url,
        content_type=content_type,
        size_bytes=size,
        sha256=digest.hexdigest(),
        filename=filename,
    )


async def head_content_type(
    url: str,
    *,
    resolver: Resolver,
    policy: IpPolicy = default_ip_policy,
    timeout_s: float = DEFAULT_HEAD_TIMEOUT_S,
    max_redirects: int = 5,
) -> str | None:
    """Return the media type of ``url`` from a pinned ``HEAD`` request, or ``None``.

    Uses the same SSRF validation and pinned transport as :func:`download_direct`.
    """
    current = url
    redirects = 0
    while True:
        target = await resolve_and_check(current, resolver=resolver, policy=policy)
        transport = _transport_for(target)
        try:
            async with httpx.AsyncClient(
                transport=transport, timeout=timeout_s, follow_redirects=False
            ) as client:
                response = await client.head(current)
            if response.status_code in _REDIRECT_STATUSES:
                location = response.headers.get("location")
                if not location or redirects >= max_redirects:
                    return None
                redirects += 1
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                return None
            return _media_type(response.headers.get("content-type")) or None
        finally:
            await transport.aclose()


__all__ = [
    "DirectInfo",
    "HttpHead",
    "PinnedBackend",
    "download_direct",
    "head_content_type",
]
