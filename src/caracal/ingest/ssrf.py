"""SSRF protection for direct media URLs (docs/06 S3).

A URL is only fetched after its host has been resolved and *every* resolved address has
passed an :data:`IpPolicy`. The caller then connects to the pinned IP, never to the
hostname, so a second DNS answer cannot redirect the connection into the private network.

Errors raised here are :class:`~caracal.errors.CaracalError` with ``source_blocked`` for
anything that is not a public HTTP(S) destination and ``source_unavailable`` for DNS
failures. Messages and details carry the host only: query strings may hold credentials.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from caracal.errors import CaracalError, ErrorCode

IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IpPolicy = Callable[[IpAddress], bool]

_NAT64_PREFIX = ipaddress.IPv6Network("64:ff9b::/96")
_SIX_TO_FOUR_PREFIX = ipaddress.IPv6Network("2002::/16")


class Resolver(Protocol):
    """Resolve a hostname to one or more IP address strings."""

    async def resolve(self, host: str, port: int) -> list[str]: ...


class SystemResolver:
    """Default resolver backed by ``loop.getaddrinfo``."""

    async def resolve(self, host: str, port: int) -> list[str]:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        addresses: list[str] = []
        for _family, _type, _proto, _canonname, sockaddr in infos:
            address = str(sockaddr[0])
            if address not in addresses:
                addresses.append(address)
        return addresses


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """Unwrap IPv4 embedded in an IPv6 address (mapped, NAT64, 6to4)."""
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip in _NAT64_PREFIX:
        return ipaddress.IPv4Address(ip.packed[12:])
    if ip in _SIX_TO_FOUR_PREFIX:
        return ipaddress.IPv4Address(ip.packed[2:6])
    return None


def default_ip_policy(ip: IpAddress) -> bool:
    """Allow only globally routable unicast addresses.

    IPv4-mapped, NAT64 and 6to4 IPv6 addresses are unwrapped first so that their embedded
    IPv4 destination is judged (``::ffff:127.0.0.1`` and ``64:ff9b::7f00:1`` are loopback).
    """
    if isinstance(ip, ipaddress.IPv6Address):
        embedded = _embedded_ipv4(ip)
        if embedded is not None:
            ip = embedded
    if not ip.is_global:
        return False
    return not (
        ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or getattr(ip, "is_site_local", False)
    )


@dataclass(frozen=True)
class PinnedTarget:
    """One validated hop: the original hostname plus the IP a client must connect to."""

    scheme: str
    host: str
    port: int
    ip: str


def _blocked(message: str, host: str) -> CaracalError:
    return CaracalError(ErrorCode.SOURCE_BLOCKED, message, {"host": host})


def _unavailable(message: str, host: str) -> CaracalError:
    return CaracalError(ErrorCode.SOURCE_UNAVAILABLE, message, {"host": host})


def parse_ip(host: str) -> IpAddress | None:
    """Parse a host as an IP literal, including inet_aton's decimal/octal/hex forms.

    ``socket.inet_aton`` covers ``2130706433`` (decimal), ``0177.0.0.1`` (octal) and
    ``0x7f.1`` (hex). It never performs DNS resolution, so a hostname simply raises
    ``OSError`` and is left to the resolver.
    """
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(host)
    except OSError:
        return None
    return ipaddress.IPv4Address(packed)


async def resolve_and_check(
    url: str,
    *,
    resolver: Resolver,
    policy: IpPolicy = default_ip_policy,
) -> PinnedTarget:
    """Validate ``url`` and return the first permitted address to connect to.

    Raises ``source_blocked`` for non-HTTP(S) schemes, missing hosts, userinfo, invalid
    ports and any disallowed address. Raises ``source_unavailable`` when DNS fails.
    """
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise _blocked("invalid URL", "") from exc

    host = parts.hostname or ""
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        raise _blocked("only http and https URLs are allowed", host)
    if not host:
        raise _blocked("URL must include a host", host)
    if parts.username is not None or parts.password is not None:
        raise _blocked("URL userinfo is not allowed", host)
    try:
        port = parts.port
    except ValueError as exc:
        raise _blocked("invalid URL port", host) from exc
    if port is None:
        port = 443 if scheme == "https" else 80

    literal = parse_ip(host)
    if literal is not None:
        if not policy(literal):
            raise _blocked("destination address is not allowed", host)
        return PinnedTarget(scheme=scheme, host=host, port=port, ip=str(literal))

    try:
        addresses = await resolver.resolve(host, port)
    except OSError as exc:
        raise _unavailable("could not resolve host", host) from exc
    if not addresses:
        raise _unavailable("could not resolve host", host)

    parsed: list[IpAddress] = []
    for address in addresses:
        try:
            parsed.append(ipaddress.ip_address(address))
        except ValueError as exc:
            raise _blocked("resolver returned an invalid address", host) from exc
    for ip in parsed:
        if not policy(ip):
            raise _blocked("destination address is not allowed", host)
    return PinnedTarget(scheme=scheme, host=host, port=port, ip=str(parsed[0]))


__all__ = [
    "IpAddress",
    "IpPolicy",
    "PinnedTarget",
    "Resolver",
    "SystemResolver",
    "default_ip_policy",
    "parse_ip",
    "resolve_and_check",
]
