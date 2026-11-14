"""Transport guards: auth before body and the streaming body cap (docs/03 §1, §4; docs/06 S1).

These tests drive the ASGI application by hand instead of using ``httpx`` so they can count how
many ``http.request`` chunks the server pulls: a request must never make the app buffer more
than the route limit, and an unauthenticated request must not pull a single chunk at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request

from caracal.app import DEFAULT_V1_BODY_LIMIT_BYTES, create_app
from caracal.errors import ErrorCode
from tests.contract.conftest import AUTH, Harness

CHUNK = 64 * 1024
VALID_BODY = json.dumps({"source": {"url": "https://example.com/audio.mp3"}}).encode()


@dataclass(frozen=True)
class SentResponse:
    """What the ASGI app sent: status, headers and the concatenated body."""

    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)


def _scope(method: str, path: str, headers: dict[str, str]) -> dict[str, Any]:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in headers.items()
    ]
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": raw_headers,
        "client": ("127.0.0.1", 40000),
        "server": ("testserver", 80),
    }


async def _drive(
    app: FastAPI,
    method: str,
    path: str,
    *,
    headers: dict[str, str],
    chunks: Iterable[bytes],
) -> tuple[SentResponse, int, int]:
    """Run one request; return the response and how many chunks/bytes were pulled."""
    sent: list[dict[str, Any]] = []
    iterator = iter(chunks)
    pulled_chunks = 0
    pulled_bytes = 0
    exhausted = False

    async def receive() -> dict[str, Any]:
        nonlocal pulled_chunks, pulled_bytes, exhausted
        if exhausted:
            return {"type": "http.disconnect"}
        pulled_chunks += 1
        try:
            chunk = next(iterator)
        except StopIteration:
            exhausted = True
            return {"type": "http.request", "body": b"", "more_body": False}
        pulled_bytes += len(chunk)
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(_scope(method, path, headers), receive, send)

    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    response = SentResponse(
        status=int(start["status"]),
        headers={name.decode("latin-1"): value.decode("latin-1") for name, value in start.get("headers", [])},
        body=body,
    )
    return response, pulled_chunks, pulled_bytes


async def test_unauthenticated_chunked_body_is_401_before_reading_any_chunk(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    chunks = (b"x" * CHUNK for _ in range(30 * 1024 * 1024 // CHUNK))  # 30 MiB, chunked, no auth

    response, pulled_chunks, pulled_bytes = await _drive(
        h.app, "POST", "/v1/jobs", headers={"transfer-encoding": "chunked"}, chunks=chunks
    )

    assert response.status == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == ErrorCode.UNAUTHORIZED.value
    assert pulled_chunks == 0
    assert pulled_bytes == 0


async def test_unauthenticated_invalid_json_is_401_not_422(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()

    response, pulled_chunks, _bytes = await _drive(
        h.app,
        "POST",
        "/v1/jobs",
        headers={"content-type": "application/json", "transfer-encoding": "chunked"},
        chunks=[b'{"source": not-json'],
    )

    assert response.status == 401
    assert response.json()["error"]["code"] == ErrorCode.UNAUTHORIZED.value
    assert pulled_chunks == 0


async def test_authenticated_chunked_body_over_the_limit_is_413_mid_stream(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    chunks = (b"x" * CHUNK for _ in range(50 * 1024 * 1024 // CHUNK))  # 50 MiB

    response, pulled_chunks, pulled_bytes = await _drive(
        h.app, "POST", "/v1/jobs", headers={**AUTH, "transfer-encoding": "chunked"}, chunks=chunks
    )

    assert response.status == 413
    assert response.json()["error"]["code"] == ErrorCode.TOO_LARGE.value
    # One chunk to fill the cap, one chunk that crosses it, then the stream is abandoned.
    assert pulled_chunks == 2
    assert pulled_bytes == DEFAULT_V1_BODY_LIMIT_BYTES + CHUNK


async def test_declared_content_length_over_the_limit_is_413_without_reading(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()

    response, pulled_chunks, pulled_bytes = await _drive(
        h.app,
        "POST",
        "/v1/jobs",
        headers={**AUTH, "content-type": "application/json", "content-length": "70000"},
        chunks=[b"x" * 10_000],
    )

    assert response.status == 413
    assert response.json()["error"]["code"] == ErrorCode.TOO_LARGE.value
    assert pulled_chunks == 0
    assert pulled_bytes == 0


async def test_understated_content_length_cannot_bypass_the_cap(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()

    response, _pulled_chunks, pulled_bytes = await _drive(
        h.app,
        "POST",
        "/v1/jobs",
        headers={**AUTH, "content-type": "application/json", "content-length": "10"},
        chunks=[b"x" * CHUNK, b"x" * CHUNK],
    )

    assert response.status == 413
    assert response.json()["error"]["code"] == ErrorCode.TOO_LARGE.value
    assert pulled_bytes == DEFAULT_V1_BODY_LIMIT_BYTES + CHUNK


async def test_chunked_body_under_the_limit_reaches_the_route(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    chunks = [VALID_BODY[:15], VALID_BODY[15:40], VALID_BODY[40:]]

    response, _pulled_chunks, pulled_bytes = await _drive(
        h.app,
        "POST",
        "/v1/jobs",
        headers={**AUTH, "content-type": "application/json", "transfer-encoding": "chunked"},
        chunks=chunks,
    )

    assert response.status == 202
    assert response.json()["status"] == "queued"
    assert pulled_bytes == len(VALID_BODY)


async def test_per_route_limits_let_an_upload_route_declare_a_larger_cap(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    upload_cap = 256 * 1024
    app = create_app(h.settings, h.deps, body_limits={("POST", "/v1/_test_upload"): upload_cap})

    @app.post("/v1/_test_upload")
    async def upload(request: Request) -> dict[str, int]:
        return {"size_bytes": len(await request.body())}

    payload = b"y" * (200 * 1024)
    chunks = [payload[offset : offset + 8192] for offset in range(0, len(payload), 8192)]

    response, _pulled_chunks, pulled_bytes = await _drive(
        app, "POST", "/v1/_test_upload", headers={**AUTH, "transfer-encoding": "chunked"}, chunks=chunks
    )

    assert response.status == 200
    assert response.json() == {"size_bytes": len(payload)}
    assert pulled_bytes == len(payload)


async def test_the_default_cap_applies_to_every_v1_route(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()

    @h.app.post("/v1/other")
    async def other(request: Request) -> dict[str, int]:
        return {"size_bytes": len(await request.body())}

    chunks = (b"z" * CHUNK for _ in range(2))

    response, _pulled_chunks, pulled_bytes = await _drive(
        h.app, "POST", "/v1/other", headers={**AUTH, "transfer-encoding": "chunked"}, chunks=chunks
    )

    assert response.status == 413
    assert response.json()["error"]["code"] == ErrorCode.TOO_LARGE.value
    assert pulled_bytes > DEFAULT_V1_BODY_LIMIT_BYTES


async def test_rate_limited_uses_the_error_code_and_retry_after(make_harness: Callable[..., Harness]) -> None:
    h = make_harness(settings_overrides={"CARACAL_RATE_LIMIT_PER_MIN": 1})
    async with h.client() as client:
        first = await client.post("/v1/jobs", headers=AUTH, json=json.loads(VALID_BODY))
        second = await client.post("/v1/jobs", headers=AUTH, json=json.loads(VALID_BODY))

    assert first.status_code == 202
    assert second.status_code == 429
    payload = second.json()
    assert payload["error"]["code"] == ErrorCode.RATE_LIMITED.value
    retry_after = payload["error"]["details"]["retry_after_s"]
    assert isinstance(retry_after, int)
    assert retry_after >= 1
    assert second.headers["retry-after"] == str(retry_after)
