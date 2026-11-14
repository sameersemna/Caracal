"""Streamed multipart uploads (docs/06 S6, docs/05 §3, docs/03 §2 ``POST /v1/uploads``).

The HTTP body is parsed as it arrives: a file part's bytes go straight to a ``.part`` file
under ``uploads/<upload_id>/`` while a sha256 is computed, and the file is renamed into
place only after the whole part has been received. Nothing is ever buffered in memory in
full, and no client-supplied path component reaches the filesystem.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from caracal.errors import CaracalError, ErrorCode
from caracal.storage import UploadFiles

if TYPE_CHECKING:
    from python_multipart.multipart import MultipartCallbacks

DEFAULT_UPLOAD_FILENAME = "upload.bin"
_MAX_FILENAME_LENGTH = 100
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class UploadInfo:
    """One stored upload."""

    upload_id: UUID
    filename: str
    size_bytes: int
    sha256: str
    path: Path


def sanitize_filename(filename: str) -> str:
    """Reduce a client-supplied name to a safe basename-like token.

    Path separators are treated as component boundaries: ``..``/``.`` components are
    dropped and the remaining components are joined with ``_``, so ``../../etc/passwd``
    becomes ``etc_passwd``. Everything outside ``[A-Za-z0-9._-]`` becomes ``_``, leading
    dots are stripped, the result is capped at 100 characters and an empty name falls
    back to ``upload.bin``.
    """
    parts = [part for part in filename.replace("\\", "/").split("/") if part not in {"", ".", ".."}]
    name = "_".join(parts)
    name = _UNSAFE_CHARS.sub("_", name)
    name = name.lstrip(".")
    name = name[:_MAX_FILENAME_LENGTH]
    return name or DEFAULT_UPLOAD_FILENAME


def _too_large(max_bytes: int) -> CaracalError:
    return CaracalError(ErrorCode.TOO_LARGE, "upload exceeds the maximum size", {"max_bytes": max_bytes})


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # A broken parent directory is the failure we are already unwinding from.
        return


async def save_upload(
    chunks: AsyncIterator[bytes],
    filename: str,
    files: UploadFiles,
    *,
    max_bytes: int,
) -> UploadInfo:
    """Stream ``chunks`` to ``uploads/<upload_id>/<sanitised name>``.

    The file is written to ``<name>.part``, hashed while it is received, and atomically
    renamed on success. Crossing ``max_bytes`` aborts and removes the partial file.
    A filesystem failure becomes ``internal`` without exposing the path.
    """
    safe_name = sanitize_filename(filename)
    destination = files.file(safe_name)
    partial = destination.with_suffix(f"{destination.suffix}.part")
    digest = hashlib.sha256()
    size = 0
    try:
        files.ensure()
        with partial.open("wb") as handle:
            async for chunk in chunks:
                if not chunk:
                    continue
                size += len(chunk)
                if size > max_bytes:
                    raise _too_large(max_bytes)
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    except CaracalError:
        _unlink_quietly(partial)
        raise
    except OSError as exc:
        _unlink_quietly(partial)
        raise CaracalError(ErrorCode.INTERNAL, "could not store upload") from exc
    except BaseException:
        _unlink_quietly(partial)
        raise
    return UploadInfo(
        upload_id=files.upload_id,
        filename=safe_name,
        size_bytes=size,
        sha256=digest.hexdigest(),
        path=destination,
    )


class MultipartFileStream:
    """Incremental ``multipart/form-data`` parser that streams one file part.

    Feed it body chunks with :meth:`feed`; it returns the file-part bytes that became
    available. Only the *first* part named ``file`` with a filename is streamed; every
    filename-bearing part is counted so the caller can enforce "exactly one".
    """

    def __init__(self, boundary: bytes) -> None:
        from python_multipart.multipart import MultipartParser

        self.file_parts: list[tuple[str | None, str | None]] = []
        self.target_filename: str | None = None
        self.headers_seen = False
        self.finished = False
        self._pending: list[bytes] = []
        self._headers: dict[bytes, bytes] = {}
        self._header_field = b""
        self._header_value = b""
        self._current_name: str | None = None
        self._current_filename: str | None = None
        self._streaming = False
        callbacks: MultipartCallbacks = {
            "on_part_begin": self._on_part_begin,
            "on_header_field": self._on_header_field,
            "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end,
            "on_headers_finished": self._on_headers_finished,
            "on_part_data": self._on_part_data,
            "on_part_end": self._on_part_end,
            "on_end": self._on_end,
        }
        self._parser = MultipartParser(boundary, callbacks)

    def feed(self, data: bytes) -> list[bytes]:
        """Parse one body chunk and return any file bytes it completed."""
        self._pending = []
        self._parser.write(data)
        pending, self._pending = self._pending, []
        return pending

    def finalize(self) -> None:
        """Signal end of body to the parser."""
        self._parser.finalize()

    def _on_part_begin(self) -> None:
        self._headers = {}
        self._header_field = b""
        self._header_value = b""
        self._current_name = None
        self._current_filename = None
        self._streaming = False

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _on_header_end(self) -> None:
        self._headers[self._header_field.strip().lower()] = self._header_value.strip()
        self._header_field = b""
        self._header_value = b""

    def _on_headers_finished(self) -> None:
        from python_multipart.multipart import parse_options_header

        _disposition, params = parse_options_header(self._headers.get(b"content-disposition"))
        name = params.get(b"name")
        filename = params.get(b"filename")
        self._current_name = name.decode("utf-8", "replace") if name is not None else None
        self._current_filename = filename.decode("utf-8", "replace") if filename is not None else None
        if self._current_filename is None:
            return
        self.file_parts.append((self._current_name, self._current_filename))
        if self._current_name == "file" and not self.headers_seen:
            self.headers_seen = True
            self.target_filename = self._current_filename
            self._streaming = True

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._streaming and end > start:
            self._pending.append(bytes(data[start:end]))

    def _on_part_end(self) -> None:
        self._streaming = False

    def _on_end(self) -> None:
        self._streaming = False
        self.finished = True


def _invalid_multipart(message: str) -> CaracalError:
    return CaracalError(ErrorCode.INVALID_INPUT, message)


async def receive_upload(
    body: AsyncIterator[bytes],
    *,
    boundary: bytes,
    files: UploadFiles,
    max_bytes: int,
) -> UploadInfo:
    """Parse one multipart body, stream its single ``file`` part and store it.

    Raises ``invalid_input`` when the body does not contain exactly one file part named
    ``file`` (including no part, a nameless file part, or more than one file part).
    """
    from python_multipart.exceptions import MultipartParseError

    stream = MultipartFileStream(boundary)
    pending: list[bytes] = []
    try:
        async for chunk in body:
            pending.extend(stream.feed(chunk))
            if stream.headers_seen:
                break
        if not stream.headers_seen:
            raise _invalid_multipart("multipart body must contain exactly one file part named 'file'")

        async def file_chunks() -> AsyncIterator[bytes]:
            for buffered in pending:
                yield buffered
            async for chunk in body:
                for data in stream.feed(chunk):
                    yield data
            stream.finalize()

        info = await save_upload(
            file_chunks(),
            stream.target_filename or "",
            files,
            max_bytes=max_bytes,
        )
    except MultipartParseError as exc:
        raise _invalid_multipart("malformed multipart body") from exc

    if len(stream.file_parts) != 1 or stream.file_parts[0][0] != "file" or not stream.finished:
        _unlink_quietly(info.path)
        raise _invalid_multipart("multipart body must contain exactly one file part named 'file'")
    return info


__all__ = [
    "MultipartFileStream",
    "UploadInfo",
    "receive_upload",
    "sanitize_filename",
    "save_upload",
]
