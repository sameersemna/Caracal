"""``POST /v1/uploads`` contract: multipart streaming, caps, auth (docs/03 §2, docs/06 S6)."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path

from tests.contract.conftest import AUTH, Harness

BOUNDARY = "----caracal-test-boundary"


def _uploads_dir(h: Harness) -> Path:
    return h.settings.CARACAL_DATA_DIR / "uploads"


def _files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def _multipart_body(payload: bytes, filename: str = "big.mp3") -> bytes:
    return (
        (
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            "Content-Type: audio/mpeg\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{BOUNDARY}--\r\n".encode()
    )


async def test_happy_path_stores_file_and_creates_row(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    payload = b"ID3\x04\x00\x00\x00\x00\x00\x00fake-mp3-body"
    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            headers=AUTH,
            files={"file": ("a talk.mp3", payload, "audio/mpeg")},
        )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "a_talk.mp3"
    assert body["size_bytes"] == len(payload)
    assert body["sha256"] == hashlib.sha256(payload).hexdigest()

    upload_id = uuid.UUID(body["upload_id"])
    record = await h.repos.uploads.get(upload_id)
    assert record is not None
    assert record.filename == "a_talk.mp3"
    assert record.size_bytes == len(payload)
    assert record.sha256 == body["sha256"]
    assert record.consumed_by is None

    path = Path(record.path)
    assert path.read_bytes() == payload  # noqa: ASYNC240 - test-local file
    assert path.parent.parent == _uploads_dir(h)
    uuid.UUID(path.parent.name)


async def test_traversal_filename_is_sanitised_inside_the_upload_dir(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            headers=AUTH,
            files={"file": ("../../etc/passwd", b"pwned", "application/octet-stream")},
        )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "etc_passwd"
    record = await h.repos.uploads.get(uuid.UUID(body["upload_id"]))
    assert record is not None
    path = Path(record.path)
    assert path.name == "etc_passwd"
    assert _uploads_dir(h) in path.parents
    assert path.read_bytes() == b"pwned"  # noqa: ASYNC240 - test-local file
    assert not (h.settings.CARACAL_DATA_DIR.parent / "etc" / "passwd").exists()


async def test_chunked_oversize_body_is_413_and_leaves_no_file(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness(settings_overrides={"CARACAL_MAX_UPLOAD_MB": 1})
    body = _multipart_body(b"x" * (1024 * 1024 + 1))

    async def stream() -> AsyncIterator[bytes]:
        for start in range(0, len(body), 64 * 1024):
            yield body[start : start + 64 * 1024]

    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            headers={**AUTH, "content-type": f"multipart/form-data; boundary={BOUNDARY}"},
            content=stream(),
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "too_large"
    assert _files_under(_uploads_dir(h)) == []


async def test_no_file_part_is_422(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            headers=AUTH,
            files={"name": (None, "just a field")},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert _files_under(_uploads_dir(h)) == []


async def test_two_file_parts_are_422(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            headers=AUTH,
            files=[
                ("file", ("one.mp3", b"one", "audio/mpeg")),
                ("other", ("two.mp3", b"two", "audio/mpeg")),
            ],
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
    assert _files_under(_uploads_dir(h)) == []


async def test_non_multipart_body_is_415(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            headers={**AUTH, "content-type": "text/plain"},
            content=b"hello",
        )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media"


async def test_unauthenticated_is_401_before_the_body(
    make_harness: Callable[..., Harness],
) -> None:
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            files={"file": ("a.mp3", b"payload", "audio/mpeg")},
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert _files_under(_uploads_dir(h)) == []


async def test_disk_failure_is_500_without_a_path(make_harness: Callable[..., Harness]) -> None:
    h = make_harness()
    _uploads_dir(h).parent.mkdir(parents=True, exist_ok=True)
    _uploads_dir(h).write_text("not a directory")
    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            headers=AUTH,
            files={"file": ("a.mp3", b"payload", "audio/mpeg")},
        )
    assert response.status_code == 500
    payload = response.json()
    assert payload["error"]["code"] == "internal"
    assert str(_uploads_dir(h)) not in payload["error"]["message"]


async def test_upload_directory_is_named_after_the_row_id(
    make_harness: Callable[..., Harness],
) -> None:
    """Verdict 08 carry-forward: the route generates the id before writing the file."""
    h = make_harness()
    async with h.client() as client:
        response = await client.post(
            "/v1/uploads",
            headers=AUTH,
            files={"file": ("talk.mp3", b"payload", "audio/mpeg")},
        )
    assert response.status_code == 201
    upload_id = uuid.UUID(response.json()["upload_id"])

    record = await h.repos.uploads.get(upload_id)
    assert record is not None
    path = Path(record.path)
    assert path.parent.name == str(upload_id)
    assert path.parent.parent == _uploads_dir(h)
    assert path.name == "talk.mp3"
