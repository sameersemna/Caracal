"""Live worker crash resilience (card T4.3).

Run by the judge under ``scripts/live-suite.sh`` (real API, Postgres, Redis, whisper). The
test starts two extra ``caracal worker`` processes with fast recovery timings, holds the
whisper single-slot lock so the job's transcribe stage blocks, ``SIGKILL``s the worker that
owns the queue entry, releases the slot, and asserts the other worker reclaims the entry,
finishes the job and leaves a valid report behind.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from redis import Redis

from caracal.config import Settings, get_settings
from caracal.queue import Keys

pytestmark = pytest.mark.live

_AUDIO = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "jfk.wav"
_RECLAIM_AFTER_S = 5
_RECLAIM_EVERY_S = 1


def _base_url() -> str:
    base = os.environ.get("CARACAL_LIVE_BASE_URL")
    if not base:
        pytest.skip("CARACAL_LIVE_BASE_URL is not set")
    return base.rstrip("/")


def _caracal_executable() -> str:
    candidate = Path(sys.executable).parent / "caracal"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("caracal")
    if found is None:
        pytest.skip("the caracal executable is not on PATH")
    return found


def _start_workers(count: int, log_path: Path) -> list[subprocess.Popen[bytes]]:
    """Start ``count`` workers with a fast recovery sweep (defaults stay production)."""
    executable = _caracal_executable()
    command = [
        executable,
        "worker",
        "--reclaim-after-s",
        str(_RECLAIM_AFTER_S),
        "--reclaim-every-s",
        str(_RECLAIM_EVERY_S),
    ]
    log = log_path.open("ab")
    processes: list[subprocess.Popen[bytes]] = []
    for _ in range(count):
        processes.append(
            subprocess.Popen(  # noqa: S603 - fixed argv, no shell, our own executable
                command,
                stdout=subprocess.DEVNULL,
                stderr=log,
                env=os.environ.copy(),
            )
        )
    return processes


def _stop_workers(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _field(fields: dict[bytes, bytes], name: str) -> str | None:
    value = fields.get(name.encode())
    return value.decode("utf-8") if value is not None else None


def _entry_owner(client: Redis, keys: Keys, job_id: str) -> str | None:
    """The consumer that currently owns ``job_id``'s pending stream entry."""
    stream = keys.jobs_stream()
    entry_id: str | None = None
    for raw_id, fields in client.xrange(stream):
        if _field(fields, "job_id") == job_id:
            entry_id = raw_id.decode("utf-8") if isinstance(raw_id, bytes) else str(raw_id)
            break
    if entry_id is None:
        return None
    for row in client.xpending_range(stream, "workers", min="-", max="+", count=100):
        message_id = row["message_id"]
        message_text = message_id.decode("utf-8") if isinstance(message_id, bytes) else str(message_id)
        if message_text == entry_id:
            consumer = row["consumer"]
            return consumer.decode("utf-8") if isinstance(consumer, bytes) else str(consumer)
    return None


def _pid_from_worker_id(worker_id: str) -> int:
    """``<host>-<pid>-<4 hex>`` -> pid (the default consumer name, worker.py)."""
    parts = worker_id.rsplit("-", 2)
    if len(parts) != 3:
        raise AssertionError(f"unexpected worker consumer id: {worker_id!r}")
    return int(parts[1])


async def _submit(client: httpx.AsyncClient, token: str, base: str) -> str:
    with _AUDIO.open("rb") as handle:
        uploaded = await client.post(
            f"{base}/v1/uploads",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (_AUDIO.name, handle, "audio/wav")},
        )
    assert uploaded.status_code == 201, uploaded.text
    created = await client.post(
        f"{base}/v1/jobs",
        headers={"Authorization": f"Bearer {token}", "content-type": "application/json"},
        json={"source": {"upload_id": uploaded.json()["upload_id"]}, "options": {"depth": "quick"}},
    )
    assert created.status_code == 202, created.text
    return str(created.json()["job_id"])


async def _status(client: httpx.AsyncClient, token: str, base: str, job_id: str) -> dict[str, Any]:
    response = await client.get(
        f"{base}/v1/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _transcribing(status: dict[str, Any]) -> bool:
    return any(
        stage.get("name") == "transcribe" and stage.get("status") == "running"
        for stage in status.get("stages", [])
    )


async def _wait_for_transcribe_started(
    client: httpx.AsyncClient, token: str, base: str, job_id: str
) -> dict[str, Any]:
    deadline = time.monotonic() + 180
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = await _status(client, token, base, job_id)
        if _transcribing(latest):
            return latest
        if latest.get("status") in {"failed", "cancelled", "done"}:
            pytest.fail(f"job left running before transcribe was observed: {latest}")
        await asyncio.sleep(0.1)
    pytest.fail(f"transcribe never started for job {job_id}: {latest}")


async def _wait_for_done(client: httpx.AsyncClient, token: str, base: str, job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 240
    latest: dict[str, Any] = {}
    while time.monotonic() < deadline:
        latest = await _status(client, token, base, job_id)
        if latest.get("status") in {"done", "failed", "cancelled"}:
            return latest
        await asyncio.sleep(0.5)
    pytest.fail(f"job {job_id} did not finish: {latest}")


async def test_sigkill_mid_transcribe_is_recovered_by_another_worker(tmp_path: Path) -> None:
    get_settings.cache_clear()
    settings: Settings = get_settings()
    token = settings.CARACAL_API_TOKEN.get_secret_value()
    base = _base_url()
    keys = Keys(settings.REDIS_PREFIX)
    sync_redis = Redis.from_url(settings.REDIS_URL.get_secret_value())
    lock_key = keys.lock_whisper()
    workers = _start_workers(2, tmp_path / "workers.log")
    # Hold the single-slot lock so the job's transcribe stage blocks: the worker we kill
    # is definitely not holding the Redis whisper lock when it dies.
    assert sync_redis.set(lock_key, uuid.uuid4().hex, nx=True, px=120_000) is True
    holder_pid: int | None = None
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            job_id = await _submit(client, token, base)
            await _wait_for_transcribe_started(client, token, base, job_id)

            consumer = _entry_owner(sync_redis, keys, job_id)
            assert consumer is not None, "no consumer owns the pending queue entry"
            holder_pid = _pid_from_worker_id(consumer)
            assert holder_pid != os.getpid()
            os.kill(holder_pid, signal.SIGKILL)

            # The crashed worker never held the whisper lock; free our stand-in slot so the
            # surviving worker can transcribe the reclaimed entry.
            sync_redis.delete(lock_key)

            finished = await _wait_for_done(client, token, base, job_id)
            assert finished["status"] == "done", finished
            # The killed worker cannot ack or write anything, so the finished job and the
            # valid report below were necessarily produced by another worker.
            report = await client.get(
                f"{base}/v1/jobs/{job_id}/report",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert report.status_code == 200, report.text
            payload = report.json()
            assert payload["job_id"] == job_id
            assert payload["schema_version"]
            assert payload["source"]
            assert payload["quality"]
            assert payload["transcript_ref"]
            assert payload["transcript_ref"]["segments"] > 0
            assert payload["quality"]["stages"]["transcribe"] == "done"
            assert payload["quality"]["stages"]["assemble"] == "done"
    finally:
        sync_redis.delete(lock_key)
        sync_redis.close()
        _stop_workers(workers)
