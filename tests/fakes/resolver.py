"""Deterministic DNS resolver for tests (docs/07 §4).

Never touches the network: every hostname a test uses must be registered explicitly.
Unregistered hosts raise ``socket.gaierror`` so the ``source_unavailable`` path is
exercised rather than accidentally resolving through the real system resolver.
"""

from __future__ import annotations

import socket

PUBLIC_IP = "93.184.216.34"
PRIVATE_IP = "10.0.0.1"


class FakeResolver:
    """A :class:`caracal.ingest.ssrf.Resolver` backed by a fixed host→addresses map."""

    def __init__(self, mapping: dict[str, list[str]] | None = None) -> None:
        self.mapping: dict[str, list[str]] = {k.lower(): list(v) for k, v in (mapping or {}).items()}
        self.calls: list[tuple[str, int]] = []

    def add(self, host: str, *addresses: str) -> None:
        self.mapping[host.lower()] = list(addresses)

    async def resolve(self, host: str, port: int) -> list[str]:
        self.calls.append((host, port))
        try:
            return list(self.mapping[host.lower()])
        except KeyError:
            raise socket.gaierror(socket.EAI_NONAME, "name or service not known") from None


__all__ = ["PRIVATE_IP", "PUBLIC_IP", "FakeResolver"]
