"""Whisper fake: an ``httpx.MockTransport`` that serves the recorded real fixtures.

The server under test is single-slot and the tests need to assert exactly what was sent, so
the fake records every request, can raise scripted transport errors, and parses the
multipart body back into fields for assertions. No network is used.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import httpx

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "whisper"


def load_fixture(name: str) -> dict[str, Any]:
    """Load one recorded ``verbose_json`` fixture (``"jfk"``, ``"talk_c0"``, ...)."""
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"fixture {name} is not a JSON object")
    return payload


def fixture_response(name: str, *, status_code: int = 200) -> httpx.Response:
    """A 200 JSON response built from a real fixture."""
    return httpx.Response(
        status_code,
        content=json.dumps(load_fixture(name)).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def json_response(payload: object, *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        content=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def html_response(body: str = "<html>not whisper</html>", *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, content=body.encode("utf-8"), headers={"content-type": "text/html"})


@dataclass(frozen=True)
class MultipartPart:
    """One decoded part of a multipart request body."""

    name: str
    filename: str | None
    content_type: str
    content: bytes

    @property
    def text(self) -> str:
        return self.content.decode("utf-8")


def parse_multipart(request: httpx.Request) -> dict[str, MultipartPart]:
    """Decode an ``httpx`` multipart request into ``field name -> part``."""
    request.read()
    content_type = request.headers.get("content-type", "")
    raw = (
        b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.content
    )
    message = BytesParser(policy=policy.default).parsebytes(raw)
    parts: dict[str, MultipartPart] = {}
    if not message.is_multipart():
        return parts
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str) or not name:
            continue
        parts[name] = MultipartPart(
            name=name,
            filename=part.get_filename(),
            content_type=part.get_content_type(),
            content=part.get_payload(decode=True) or b"",
        )
    return parts


class FakeWhisper:
    """A callable ``httpx`` transport serving a scripted list of responses or exceptions."""

    def __init__(
        self,
        script: Sequence[httpx.Response | BaseException],
        *,
        events: list[str] | None = None,
    ) -> None:
        self._script = list(script)
        self._cursor = 0
        self.requests: list[httpx.Request] = []
        self.events = events if events is not None else []

    @classmethod
    def from_fixtures(cls, *names: str, events: list[str] | None = None) -> FakeWhisper:
        return cls([fixture_response(name) for name in names], events=events)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.events.append("request")
        if self._cursor >= len(self._script):
            raise AssertionError(f"unexpected whisper request {len(self.requests)}: no scripted response")
        item = self._script[self._cursor]
        self._cursor += 1
        if isinstance(item, BaseException):
            raise item
        return item

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def parts(self, index: int = -1) -> dict[str, MultipartPart]:
        return parse_multipart(self.requests[index])

    def form(self, index: int = -1) -> dict[str, str]:
        return {name: part.text for name, part in self.parts(index).items() if part.filename is None}


__all__ = [
    "FIXTURES",
    "FakeWhisper",
    "MultipartPart",
    "fixture_response",
    "html_response",
    "json_response",
    "load_fixture",
    "parse_multipart",
]
