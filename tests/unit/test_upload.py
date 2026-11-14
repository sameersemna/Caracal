"""Streamed upload storage and filename sanitisation (docs/06 S6)."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.upload import sanitize_filename, save_upload
from caracal.storage import UploadFiles


def _files(tmp_path: Path) -> UploadFiles:
    return UploadFiles(data_dir=tmp_path, upload_id=uuid.UUID("11111111-2222-3333-4444-555555555555"))


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("talk.mp3", "talk.mp3"),
        ("My Talk (final).mp3", "My_Talk__final_.mp3"),
        ("../../etc/passwd", "etc_passwd"),
        ("..\\..\\windows\\system32", "windows_system32"),
        ("/etc/shadow", "etc_shadow"),
        (".hidden", "hidden"),
        ("..", "upload.bin"),
        ("", "upload.bin"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_caps_length() -> None:
    result = sanitize_filename("a" * 200 + ".mp3")
    assert len(result) == 100
    assert result.startswith("a" * 100)


async def test_save_upload_streams_hashes_and_renames(tmp_path: Path) -> None:
    files = _files(tmp_path)
    info = await save_upload(_chunks(b"hello ", b"world"), "../../a talk.mp3", files, max_bytes=1024)

    assert info.upload_id == files.upload_id
    assert info.filename == "a_talk.mp3"
    assert info.size_bytes == 11
    assert info.sha256 == hashlib.sha256(b"hello world").hexdigest()
    assert info.path == tmp_path / "uploads" / str(files.upload_id) / "a_talk.mp3"
    assert info.path.read_bytes() == b"hello world"
    assert not info.path.with_suffix(".mp3.part").exists()


async def test_save_upload_over_cap_deletes_the_partial_file(tmp_path: Path) -> None:
    files = _files(tmp_path)
    with pytest.raises(CaracalError) as excinfo:
        await save_upload(_chunks(b"0123", b"4567"), "clip.mp3", files, max_bytes=5)
    assert excinfo.value.code is ErrorCode.TOO_LARGE
    assert list(files.root.iterdir()) == []
    assert not (files.root / "clip.mp3").exists()


async def test_upload_files_rejects_path_characters(tmp_path: Path) -> None:
    files = _files(tmp_path)
    with pytest.raises(CaracalError) as excinfo:
        files.file("../escape")
    assert excinfo.value.code is ErrorCode.INVALID_INPUT


def test_upload_files_root_is_uuid_keyed(tmp_path: Path) -> None:
    files = _files(tmp_path)
    assert files.root == tmp_path / "uploads" / "11111111-2222-3333-4444-555555555555"


async def test_disk_failure_is_internal_and_hides_the_path(tmp_path: Path) -> None:
    (tmp_path / "uploads").write_text("not a directory")
    files = _files(tmp_path)
    with pytest.raises(CaracalError) as excinfo:
        await save_upload(_chunks(b"data"), "clip.mp3", files, max_bytes=1024)
    assert excinfo.value.code is ErrorCode.INTERNAL
    assert str(tmp_path) not in excinfo.value.message
