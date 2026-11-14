"""Fake Ollama: scripted responses over ``httpx.MockTransport`` (docs/07 §4).

The fake serves the recorded real gemma4 fixtures as well as synthetic tool calls, fenced
JSON, 401/500 responses and transport errors. :class:`AsyncProbeTransport` wraps the same
handler and adds a delay so concurrency limits can be observed.
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ollama"


def load_fixture(name: str) -> dict[str, Any]:
    """Load one recorded response fixture (without the ``.json`` suffix)."""
    payload = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"fixture {name} is not a JSON object")
    return payload


def fixture_response(name: str, *, status_code: int = 200) -> httpx.Response:
    """A response built from a real recorded fixture (body is ``{"message": ...}``)."""
    fixture = load_fixture(name)
    return httpx.Response(
        status_code,
        content=json.dumps(fixture).encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def tool_call_message(
    arguments: object,
    *,
    name: str = "record_analysis",
    call_id: str = "call_test",
) -> dict[str, Any]:
    """Build an Ollama assistant message carrying one tool call."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "function": {"index": 0, "name": name, "arguments": arguments},
            }
        ],
    }


def text_message(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def message_response(message: dict[str, Any], *, status_code: int = 200) -> httpx.Response:
    return httpx.Response(status_code, json={"message": message, "done": True})


def error_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, json={"error": f"status {status_code}"})


class FakeOllama:
    """A callable transport handler serving a scripted list of responses or exceptions."""

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

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.events.append("request")
        if self._cursor >= len(self._script):
            raise AssertionError(f"unexpected Ollama request {len(self.requests)}: no scripted response")
        item = self._script[self._cursor]
        self._cursor += 1
        if isinstance(item, BaseException):
            raise item
        return item

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def body(self, index: int = -1) -> dict[str, Any]:
        request = self.requests[index]
        request.read()
        payload = json.loads(request.content)
        return copy.deepcopy(payload)

    def headers(self, index: int = -1) -> httpx.Headers:
        return self.requests[index].headers


class AsyncProbeTransport(httpx.AsyncBaseTransport):
    """Async transport over a sync handler, with a delay and in-flight accounting."""

    def __init__(self, handler: FakeOllama, *, delay: float = 0.01) -> None:
        self.handler = handler
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            response = self.handler(request)
            response.request = request
            return response
        finally:
            self.in_flight -= 1


__all__ = [
    "FIXTURES",
    "AsyncProbeTransport",
    "FakeOllama",
    "error_response",
    "fixture_response",
    "load_fixture",
    "message_response",
    "text_message",
    "tool_call_message",
]
