"""Queue worker: consume jobs, run the pipeline, heartbeat and recover dead consumers."""

from __future__ import annotations

import asyncio
import os
import re
import socket
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from redis.asyncio import Redis
from redis.exceptions import RedisError

from caracal.asr.whisper import WhisperClient
from caracal.config import Settings
from caracal.db.repo import JobRepo, ReportRepo, SegmentRepo, SpeakerRepo, UploadRepo
from caracal.db.tables import JOB_TERMINAL_STATUSES
from caracal.errors import CaracalError, ErrorCode
from caracal.ingest.ssrf import Resolver, SystemResolver
from caracal.llm.client import LLMClient
from caracal.log import get_logger
from caracal.pipeline.runner import RunnerDeps, run_pipeline, run_translate
from caracal.pipeline.stages import Repos, Stage, StageFn, m2_registry
from caracal.queue import (
    Delivery,
    JobQueue,
    Keys,
    is_cancelled,
    publish_event,
    redis_lock,
    whisper_lock,
)
from caracal.search import BraveProvider, ExaProvider, SearchProvider, SearchRouter
from caracal.speakers.diarize import Diarizer

logger = get_logger("caracal.worker")

_IDLE_SLEEP_S = 0.01
_READ_ERROR_SLEEP_S = 0.5
_HOSTNAME_RE = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True)
class RepoBundle:
    """The concrete repository bundle handed to :class:`Worker` by the CLI."""

    jobs: JobRepo
    uploads: UploadRepo
    reports: ReportRepo
    segments: SegmentRepo
    speakers: SpeakerRepo


@dataclass(frozen=True)
class WorkerIntervals:
    """Timings for the worker loops; injectable so tests do not wait for production cadence."""

    heartbeat_s: float = 10.0
    touch_s: float = 60.0
    reclaim_s: float = 60.0
    reclaim_min_idle_ms: int = 300_000
    block_ms: int = 5000
    max_deliveries: int = 3
    shutdown_grace_s: float = 30.0


_DEFAULT_INTERVALS = WorkerIntervals()


def default_intervals() -> WorkerIntervals:
    """The production cadence (docs/05 §2)."""
    return _DEFAULT_INTERVALS


def _default_worker_id() -> str:
    host = _HOSTNAME_RE.sub("_", socket.gethostname()).strip("_") or "worker"
    return f"{host}-{os.getpid()}-{uuid.uuid4().hex[:4]}"


class Worker:
    """One worker process: a bounded read loop plus heartbeat and reclaim loops."""

    def __init__(
        self,
        settings: Settings,
        redis: Redis,
        repos: Repos,
        *,
        registry: Mapping[Stage, StageFn] | None = None,
        worker_id: str | None = None,
        intervals: WorkerIntervals = _DEFAULT_INTERVALS,
        http: httpx.AsyncClient | None = None,
        whisper: WhisperClient | None = None,
        llm: LLMClient | None = None,
        resolver: Resolver | None = None,
        search: SearchRouter | None = None,
    ) -> None:
        self.settings = settings
        self.redis = redis
        self.repos = repos
        self.worker_id = worker_id or _default_worker_id()
        self.intervals = intervals
        self.keys = Keys(settings.REDIS_PREFIX)
        self.queue = JobQueue(redis, self.keys)
        self._in_flight: set[asyncio.Task[None]] = set()

        # Process-wide stage clients (docs/01 §4): one HTTP client, one whisper client whose
        # every call takes the Redis single-slot lock, one LLM client sharing the HTTP client,
        # and the DNS resolver used by ingest.
        self._owns_http = http is None
        self._http = (
            http if http is not None else httpx.AsyncClient(timeout=float(settings.WHISPER_TIMEOUT_S))
        )
        self.whisper = (
            whisper
            if whisper is not None
            else WhisperClient(
                settings.WHISPER_URL,
                self._http,
                timeout_s=float(settings.WHISPER_TIMEOUT_S),
                lock=lambda: whisper_lock(redis, self.keys),
            )
        )
        self.llm = llm if llm is not None else LLMClient(settings, self._http)
        self.resolver: Resolver = resolver if resolver is not None else SystemResolver()
        # One search router per process: budget, cache and provider fallback live here
        # (docs/01 §5, docs/04 §8).
        self.search = search if search is not None else self._build_search()
        # The pyannote pipeline is loaded lazily on the first diarized job (docs/04 §5).
        self.diarizer = Diarizer(settings)
        self.registry = registry if registry is not None else m2_registry()

    def _build_search(self) -> SearchRouter:
        """Build the configured providers over the process HTTP client (docs/04 §8)."""
        providers: list[SearchProvider] = []
        if self.settings.BRAVE_API_KEY is not None:
            providers.append(BraveProvider(self.settings.BRAVE_API_KEY.get_secret_value(), self._http))
        if self.settings.EXA_API_KEY is not None:
            providers.append(ExaProvider(self.settings.EXA_API_KEY.get_secret_value(), self._http))
        return SearchRouter(self.settings, self.redis, self.keys, providers)

    @property
    def concurrency(self) -> int:
        return max(1, self.settings.CARACAL_WORKER_CONCURRENCY)

    async def run(self, stop: asyncio.Event) -> None:
        """Run until ``stop`` is set, then drain in-flight jobs within the grace period."""
        await self.queue.ensure_group()
        heartbeat = asyncio.create_task(self._heartbeat())
        reclaim = asyncio.create_task(self._reclaim_loop(stop))
        try:
            await self._consume_loop(stop)
        finally:
            heartbeat.cancel()
            reclaim.cancel()
            await asyncio.gather(heartbeat, reclaim, return_exceptions=True)
            await self._drain()
            if self._owns_http:
                await self._http.aclose()

    async def _consume_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            self._reap()
            capacity = self.concurrency - len(self._in_flight)
            if capacity <= 0:
                await self._wait_for_slot(stop)
                continue
            try:
                deliveries = await self._read_or_stop(stop, capacity)
            except CaracalError as exc:
                if stop.is_set():
                    break
                logger.warning("queue read failed", extra={"error": exc.message})
                await asyncio.sleep(_READ_ERROR_SLEEP_S)
                continue
            if stop.is_set() and not deliveries:
                break
            for delivery in deliveries:
                self._spawn(delivery)

    async def _read_or_stop(self, stop: asyncio.Event, capacity: int) -> list[Delivery]:
        read_task = asyncio.create_task(
            self.queue.read(self.worker_id, count=capacity, block_ms=self.intervals.block_ms)
        )
        stop_task = asyncio.create_task(stop.wait())
        done, pending = await asyncio.wait(
            {read_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if read_task in done:
            return read_task.result()
        return []

    async def _wait_for_slot(self, stop: asyncio.Event) -> None:
        if not self._in_flight:
            await asyncio.sleep(_IDLE_SLEEP_S)
            return
        stop_task = asyncio.create_task(stop.wait())
        waiters: set[asyncio.Task[Any]] = {*self._in_flight, stop_task}
        _done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if task is stop_task:
                task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        self._reap()

    def _spawn(self, delivery: Delivery) -> None:
        task = asyncio.create_task(self._run_delivery(delivery))
        self._in_flight.add(task)

    def _reap(self) -> None:
        for task in list(self._in_flight):
            if not task.done():
                continue
            self._in_flight.discard(task)
            if task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                logger.error("job task crashed", extra={"error": str(error)})

    async def _drain(self) -> None:
        if not self._in_flight:
            return
        _done, pending = await asyncio.wait(
            self._in_flight,
            timeout=self.intervals.shutdown_grace_s,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._reap()

    async def _run_delivery(self, delivery: Delivery) -> None:
        if delivery.times_delivered > self.intervals.max_deliveries:
            await self._fail_repeated(delivery)
            return

        touch = asyncio.create_task(self._touch_entry(delivery.entry_id))
        status: str | None = None
        try:
            if delivery.kind == "translate":
                # Serialise translate runs per job: back-to-back requests must not read the
                # same checkpoint and lose each other's languages. Wait for the lock, never
                # drop the delivery (verdict 35 fix 1).
                async with redis_lock(self.redis, self.keys.lock_translate(delivery.job_id)):
                    status = await run_translate(
                        delivery.job_id,
                        self._deps(delivery.entry_id),
                        self.registry,
                        payload=delivery.payload,
                    )
            else:
                status = await run_pipeline(
                    delivery.job_id,
                    self._deps(delivery.entry_id),
                    self.registry,
                )
        except asyncio.CancelledError:
            raise
        except CaracalError as exc:
            logger.error(
                "pipeline failed unexpectedly",
                extra={
                    "job_id": str(delivery.job_id),
                    "code": exc.code.value,
                    "error": exc.message,
                },
            )
            if exc.code in {ErrorCode.NOT_FOUND, ErrorCode.INVALID_INPUT}:
                await self._ack(delivery.entry_id)
            return
        except Exception as exc:
            logger.error(
                "pipeline crashed",
                extra={"job_id": str(delivery.job_id), "error": str(exc)},
            )
            return
        finally:
            touch.cancel()
            await asyncio.gather(touch, return_exceptions=True)
        if status == "lost":
            # Another consumer reclaimed the entry: it owns the job now, so leave the entry
            # pending for it and write nothing (docs/05 §2).
            logger.warning(
                "job.lost",
                extra={
                    "job_id": str(delivery.job_id),
                    "entry_id": delivery.entry_id,
                    "worker_id": self.worker_id,
                },
            )
            return
        await self._ack(delivery.entry_id)

    async def _touch_entry(self, entry_id: str) -> None:
        while True:
            await asyncio.sleep(self.intervals.touch_s)
            try:
                owned = await self.queue.touch(self.worker_id, entry_id)
            except CaracalError as exc:
                logger.warning(
                    "queue touch failed",
                    extra={"entry_id": entry_id, "error": exc.message},
                )
                continue
            if not owned:
                logger.warning(
                    "queue touch found a new owner",
                    extra={"entry_id": entry_id, "worker_id": self.worker_id},
                )
                return

    async def _fail_repeated(self, delivery: Delivery) -> None:
        job = await self.repos.jobs.get(delivery.job_id)
        if job is not None and job.status not in JOB_TERMINAL_STATUSES:
            await self.repos.jobs.set_status(
                delivery.job_id,
                "failed",
                error={"code": ErrorCode.INTERNAL.value, "message": "worker crashed repeatedly"},
            )
        await self._ack(delivery.entry_id)

    async def _ack(self, entry_id: str) -> None:
        try:
            owned = await self.queue.ack(self.worker_id, entry_id)
        except CaracalError as exc:
            logger.warning("queue ack failed", extra={"entry_id": entry_id, "error": exc.message})
            return
        if not owned:
            logger.warning(
                "queue ack skipped: another consumer owns the entry",
                extra={"entry_id": entry_id, "worker_id": self.worker_id},
            )

    async def _heartbeat(self) -> None:
        while True:
            try:
                await self.redis.set(self.keys.worker(self.worker_id), self.worker_id, ex=30)
            except RedisError as exc:
                logger.warning("worker heartbeat failed", extra={"error": str(exc)})
            await asyncio.sleep(self.intervals.heartbeat_s)

    async def _reclaim_loop(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await asyncio.sleep(self.intervals.reclaim_s)
            if stop.is_set():
                return
            self._reap()
            capacity = self.concurrency - len(self._in_flight)
            if capacity <= 0:
                continue
            try:
                deliveries = await self.queue.reclaim(
                    self.worker_id,
                    min_idle_ms=self.intervals.reclaim_min_idle_ms,
                )
            except CaracalError as exc:
                logger.warning("queue reclaim failed", extra={"error": exc.message})
                continue
            for delivery in deliveries[:capacity]:
                self._spawn(delivery)

    def _deps(self, entry_id: str) -> RunnerDeps:
        async def still_owner() -> bool:
            return await self._still_owner(entry_id)

        return RunnerDeps(
            settings=self.settings,
            repos=self.repos,
            queue_events=self._publish,
            cancel_check=self._cancelled,
            still_owner=still_owner,
            whisper=self.whisper,
            llm=self.llm,
            diarizer=self.diarizer,
            resolver=self.resolver,
            redis=self.redis,
            keys=self.keys,
            search=self.search,
        )

    async def _still_owner(self, entry_id: str) -> bool:
        """True while this worker still owns the queue entry (owner-checked touch)."""
        try:
            return await self.queue.touch(self.worker_id, entry_id)
        except CaracalError as exc:
            # A Redis outage is not proof that another worker claimed the job; keep working
            # (the periodic heartbeat retries) rather than abandoning it mid-stage.
            logger.warning(
                "queue owner check failed",
                extra={"entry_id": entry_id, "error": exc.message},
            )
            return True

    async def _publish(self, job_id: uuid.UUID, event: dict[str, Any]) -> None:
        await publish_event(self.redis, self.keys, job_id, event)

    async def _cancelled(self, job_id: uuid.UUID) -> bool:
        return await is_cancelled(self.redis, self.keys, job_id)
