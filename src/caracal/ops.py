"""Operational tooling: doctor checks, media retention and the metrics registry (docs/01 §3).

Three concerns that belong to running Caracal rather than to analysing audio:

* :func:`run_doctor` — one ``DoctorCheck`` per dependency, with the probes injectable so the
  unit suite can exercise every branch without touching a network (docs/02, docs/00).
* :func:`run_retention` — the documented deletion policy (docs/05 §3): only job media and
  stale unconsumed uploads, never transcripts, subtitles, reports or stage checkpoints.
* :class:`MetricsRegistry` — counters kept in Redis under ``REDIS_PREFIX`` so the API and the
  worker share them, rendered as Prometheus text (no third-party metrics library).
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import json
import math
import shutil
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
import sqlalchemy as sa
from alembic.script import ScriptDirectory
from redis.asyncio import Redis
from redis.exceptions import RedisError

from caracal.config import Settings
from caracal.db.engine import make_engine, session_scope
from caracal.db.repo import JobRecord
from caracal.db.tables import JOB_TERMINAL_STATUSES
from caracal.errors import CaracalError, ErrorCode
from caracal.log import get_logger, redact
from caracal.pipeline.stages import Repos
from caracal.queue import Keys, make_redis
from caracal.storage import JobFiles, UploadFiles

logger = get_logger("caracal.ops")

CheckStatus = Literal["ok", "warn", "fail"]
Probe = tuple[CheckStatus, str]
ProbeFn = Callable[[Settings], Awaitable[Probe]]
SearchProbeFn = Callable[[Settings, bool], Awaitable[Probe]]

_MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_DISK_WARN_BYTES = 20 * 1024 * 1024 * 1024
_UPLOAD_MAX_AGE_HOURS = 24
_RETENTION_PAGE = 100
_FIELD_SEP = "\x1f"
_NOT_MIGRATED = "database not migrated \u2014 run 'caracal migrate'"

_REQUIRED_CHECKS = frozenset({"config", "postgres", "redis", "whisper", "ollama", "ffmpeg"})


@dataclass(frozen=True)
class DoctorCheck:
    """One line of ``caracal doctor`` output."""

    check: str
    status: CheckStatus
    detail: str
    required: bool = False

    def as_dict(self) -> dict[str, str]:
        return {"check": self.check, "status": self.status, "detail": redact(self.detail)}


@dataclass(frozen=True)
class DoctorProbes:
    """The injectable dependency probes behind ``caracal doctor``."""

    postgres: ProbeFn
    redis: ProbeFn
    whisper: ProbeFn
    ollama: ProbeFn
    ffmpeg: ProbeFn
    ytdlp: ProbeFn
    pyannote: ProbeFn
    search: SearchProbeFn
    disk: ProbeFn


def _config_check(settings: Settings) -> DoctorCheck:
    detail = (
        f"valid (host={settings.CARACAL_HOST}, port={settings.CARACAL_PORT}, "
        f"data_dir={settings.CARACAL_DATA_DIR})"
    )
    return DoctorCheck("config", "ok", redact(detail), required=True)


async def _safe_probe(probe: ProbeFn, settings: Settings) -> Probe:
    """Run one probe; any failure becomes a ``fail`` line without leaking a secret."""
    try:
        return await probe(settings)
    except CaracalError as exc:
        return "fail", redact(f"{exc.code.value}: {exc.message}")
    except Exception as exc:  # a doctor line is a status report: never propagate its failure
        return "fail", redact(f"{type(exc).__name__}: {exc}")


async def run_doctor(
    settings: Settings,
    probes: DoctorProbes | None = None,
    *,
    deep: bool = False,
) -> list[DoctorCheck]:
    """Run every documented check (docs/02, docs/00). Required failures fail the command."""
    active = probes if probes is not None else default_probes()
    checks: list[DoctorCheck] = [_config_check(settings)]
    required: tuple[tuple[str, ProbeFn], ...] = (
        ("postgres", active.postgres),
        ("redis", active.redis),
        ("whisper", active.whisper),
        ("ollama", active.ollama),
        ("ffmpeg", active.ffmpeg),
        ("ytdlp", active.ytdlp),
        ("pyannote", active.pyannote),
    )
    for name, probe in required:
        status, detail = await _safe_probe(probe, settings)
        checks.append(DoctorCheck(name, status, detail, required=name in _REQUIRED_CHECKS))
    try:
        search_status, search_detail = await active.search(settings, deep)
    except CaracalError as exc:
        search_status, search_detail = "fail", redact(f"{exc.code.value}: {exc.message}")
    except Exception as exc:
        search_status, search_detail = "fail", redact(f"{type(exc).__name__}: {exc}")
    checks.append(DoctorCheck("search", search_status, search_detail))
    disk_status, disk_detail = await _safe_probe(active.disk, settings)
    checks.append(DoctorCheck("disk", disk_status, disk_detail))
    return checks


def doctor_exit_code(checks: Sequence[DoctorCheck]) -> int:
    """Exit 1 only when a required check failed; warnings and optional failures do not."""
    return 1 if any(check.required and check.status == "fail" for check in checks) else 0


def emit_checks(checks: Sequence[DoctorCheck], *, json_output: bool = False) -> None:
    """Write one line per check to stderr (stdout stays free for the MCP stdio channel)."""
    for check in checks:
        detail = redact(check.detail)
        if json_output:
            line = json.dumps(
                {"check": check.check, "status": check.status, "detail": detail},
                ensure_ascii=False,
            )
        else:
            line = f"{check.check}: {check.status} - {detail}"
        sys.stderr.write(line + "\n")
    sys.stderr.flush()


def default_probes() -> DoctorProbes:
    """The real probes: local processes, datastores and HTTP endpoints."""
    return DoctorProbes(
        postgres=_probe_postgres,
        redis=_probe_redis,
        whisper=_probe_whisper,
        ollama=_probe_ollama,
        ffmpeg=_probe_ffmpeg,
        ytdlp=_probe_ytdlp,
        pyannote=_probe_pyannote,
        search=_probe_search,
        disk=_probe_disk,
    )


def _alembic_head() -> str:
    script = ScriptDirectory(str(_MIGRATIONS_DIR))
    head = script.get_current_head()
    if head is None:
        raise CaracalError(ErrorCode.INTERNAL, "alembic has no head revision")
    return head


async def _probe_postgres(settings: Settings) -> Probe:
    engine = make_engine(settings)
    try:
        try:
            async with session_scope(engine) as connection:
                await connection.execute(sa.text("SELECT 1"))
                vector = (
                    await connection.execute(sa.text("SELECT to_regtype('vector')"))
                ).scalar_one_or_none()
                if vector is None:
                    return "fail", "pgvector type 'vector' is not visible"
                head = await asyncio.to_thread(_alembic_head)
                # TableClause quotes the configured schema instead of interpolating an identifier
                # into SQL; the schema name is validated in config.py.
                version_table = sa.table(
                    "alembic_version",
                    sa.column("version_num"),
                    schema=settings.CARACAL_DB_SCHEMA,
                )
                revision = (
                    await connection.execute(sa.select(version_table.c.version_num))
                ).scalar_one_or_none()
                return _revision_probe(str(revision) if revision is not None else None, head)
        except sa.exc.DBAPIError as exc:
            # The production schema was never migrated: PostgreSQL reports the missing
            # alembic_version table as a DBAPI error. One actionable line is more useful
            # than the raw multi-line SQLAlchemy traceback (verdict 39).
            if "alembic_version" in str(exc).lower():
                return "fail", _NOT_MIGRATED
            raise
    finally:
        await engine.dispose()


def _revision_probe(revision: str | None, head: str) -> Probe:
    """Turn the stored alembic revision into a doctor line (verdict 39)."""
    if revision is None or revision != head:
        return "fail", _NOT_MIGRATED
    return "ok", f"postgres ok; pgvector visible; alembic at {head}"


async def _probe_redis(settings: Settings) -> Probe:
    client = make_redis(settings)
    key = Keys(settings.REDIS_PREFIX).key("doctor", "probe")
    token = uuid.uuid4().hex
    try:
        await client.ping()
        await client.set(key, token, ex=10)
        stored = await client.get(key)
        await client.delete(key)
    finally:
        await client.aclose()
    if stored != token.encode("utf-8"):
        return "fail", "redis write/read/delete returned a different value"
    return "ok", "redis ok; prefixed write/read/delete succeeded"


async def _probe_whisper(settings: Settings) -> Probe:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"{settings.WHISPER_URL.rstrip('/')}/")
    if response.status_code >= 400:
        return "fail", f"whisper HTTP {response.status_code}"
    return "ok", f"whisper reachable at {settings.WHISPER_URL}"


def _model_names(payload: object) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    models = payload.get("models")
    if not isinstance(models, list):
        return set()
    names: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        for field in ("name", "model"):
            value = item.get(field)
            if isinstance(value, str):
                names.add(value)
    return names


def _model_listed(names: set[str], wanted: str) -> bool:
    return wanted in names or f"{wanted}:latest" in names


async def _probe_ollama(settings: Settings) -> Probe:
    """Required check: the cloud path or the local fallback must work (docs/02)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        local_ok = False
        local = await client.get(f"{settings.OLLAMA_LOCAL_URL.rstrip('/')}/api/tags")
        if local.status_code == 200:
            local_ok = _model_listed(_model_names(local.json()), settings.LLM_FALLBACK_MODEL)
        if settings.OLLAMA_CLOUD_API_KEY is None:
            if not local_ok:
                return "fail", f"local ollama does not list {settings.LLM_FALLBACK_MODEL}"
            return "ok", f"local fallback {settings.LLM_FALLBACK_MODEL} listed (local proxy)"
        headers = {"Authorization": f"Bearer {settings.OLLAMA_CLOUD_API_KEY.get_secret_value()}"}
        cloud = await client.get(
            f"{settings.OLLAMA_CLOUD_URL.rstrip('/')}/api/tags",
            headers=headers,
        )
        if cloud.status_code == 200 and local_ok:
            return "ok", f"cloud reachable; local fallback {settings.LLM_FALLBACK_MODEL} listed"
        if cloud.status_code == 200:
            return "warn", f"cloud reachable; local fallback {settings.LLM_FALLBACK_MODEL} is missing"
        if local_ok:
            fallback = settings.LLM_FALLBACK_MODEL
            return "warn", f"cloud HTTP {cloud.status_code}; local fallback {fallback} listed"
        return "fail", f"cloud HTTP {cloud.status_code} and local fallback is unavailable"


async def _run_command(argv: Sequence[str], *, timeout_s: float = 15.0) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    return process.returncode or 0, stdout, stderr


async def _probe_ffmpeg(_settings: Settings) -> Probe:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        return "fail", "ffmpeg and ffprobe must both be on PATH"
    _code, version_out, version_err = await _run_command([ffmpeg, "-hide_banner", "-version"])
    version_lines = (version_out + version_err).decode("utf-8", "replace").splitlines()
    version = version_lines[0] if version_lines else "ffmpeg"
    _code, help_out, help_err = await _run_command([ffmpeg, "-hide_banner", "-h", "full"])
    help_text = (help_out + help_err).decode("utf-8", "replace")
    if "format_whitelist" not in help_text:
        return "fail", f"{version}; -format_whitelist is not supported"
    return "ok", f"{version}; ffprobe present; -format_whitelist supported"


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


async def _probe_ytdlp(_settings: Settings) -> Probe:
    try:
        version = importlib.metadata.version("yt-dlp")
    except importlib.metadata.PackageNotFoundError:
        return "fail", "yt-dlp is not installed"
    node = shutil.which("node")
    if node is None:
        return "fail", f"yt-dlp {version}; node is not on PATH"
    if not _importable("yt_dlp_ejs"):
        return "fail", f"yt-dlp {version}; node found; yt_dlp_ejs is not importable"
    return "ok", f"yt-dlp {version}; node found; yt_dlp_ejs importable"


def _pyannote_flags() -> tuple[bool, bool]:
    pyannote = _importable("pyannote.audio")
    cuda = False
    if _importable("torch"):
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
    return pyannote, cuda


async def _probe_pyannote(settings: Settings) -> Probe:
    pyannote, cuda = await asyncio.to_thread(_pyannote_flags)
    if not pyannote:
        return "warn", "pyannote is not importable (install the speakers extra); diarization is skipped"
    if not cuda:
        return "warn", "pyannote importable but CUDA is unavailable; diarization runs on CPU"
    if settings.HF_TOKEN is None:
        return "warn", "pyannote importable and CUDA available, but HF_TOKEN is not set"
    return "ok", "pyannote importable; CUDA available; HF_TOKEN set"


async def _probe_search(settings: Settings, deep: bool) -> Probe:
    present: list[str] = []
    if settings.BRAVE_API_KEY is not None:
        present.append("brave")
    if settings.EXA_API_KEY is not None:
        present.append("exa")
    if not present:
        return "warn", "no Brave or Exa key; web verification is disabled"
    names = ", ".join(present)
    if not deep:
        return "ok", f"search keys present: {names}"
    failures: list[str] = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for provider in present:
            try:
                status = await _deep_search_one(client, settings, provider)
            except httpx.HTTPError as exc:
                failures.append(f"{provider}: {type(exc).__name__}")
                continue
            if status != 200:
                failures.append(f"{provider}: HTTP {status}")
    if failures:
        return "fail", "; ".join(failures)
    return "ok", f"deep search ok for: {names}"


async def _deep_search_one(client: httpx.AsyncClient, settings: Settings, provider: str) -> int:
    if provider == "brave":
        key = settings.BRAVE_API_KEY
        if key is None:
            raise CaracalError(ErrorCode.INTERNAL, "brave key is not configured")
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": "caracal audio intelligence", "count": 1},
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": key.get_secret_value(),
            },
        )
        return response.status_code
    key = settings.EXA_API_KEY
    if key is None:
        raise CaracalError(ErrorCode.INTERNAL, "exa key is not configured")
    response = await client.post(
        "https://api.exa.ai/search",
        json={"query": "caracal audio intelligence", "numResults": 1},
        headers={"x-api-key": key.get_secret_value()},
    )
    return response.status_code


def _existing_ancestor(path: Path) -> Path | None:
    current = path
    while True:
        if current.exists():
            return current
        if current.parent == current:
            return None
        current = current.parent


async def _probe_disk(settings: Settings) -> Probe:
    anchor = await asyncio.to_thread(_existing_ancestor, settings.CARACAL_DATA_DIR)
    if anchor is None:
        return "fail", f"no existing parent for {settings.CARACAL_DATA_DIR}"
    usage = await asyncio.to_thread(shutil.disk_usage, anchor)
    free_gb = usage.free / (1024**3)
    if usage.free < _DISK_WARN_BYTES:
        return "warn", f"{free_gb:.1f} GiB free at {anchor} (< 20 GiB)"
    return "ok", f"{free_gb:.1f} GiB free at {anchor}"


# -- retention ----------------------------------------------------------------


@dataclass(frozen=True)
class RetentionReport:
    """What one retention pass deleted (or would delete, in a dry run)."""

    dry_run: bool
    jobs: int
    media_files: int
    media_bytes: int
    uploads: int
    upload_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "dry_run": self.dry_run,
            "jobs": self.jobs,
            "media_files": self.media_files,
            "media_bytes": self.media_bytes,
            "uploads": self.uploads,
            "upload_bytes": self.upload_bytes,
        }


def _resolve_inside(path: Path, root: Path) -> Path:
    """Resolve ``path`` and refuse anything that escapes ``CARACAL_DATA_DIR`` (docs/06 S10)."""
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CaracalError(
            ErrorCode.INTERNAL,
            "refusing to delete a path outside CARACAL_DATA_DIR",
            {"name": path.name},
        ) from exc
    return resolved


def _tree_size(directory: Path) -> tuple[int, int]:
    files = 0
    size = 0
    for child in directory.rglob("*"):
        if child.is_file() and not child.is_symlink():
            files += 1
            size += child.stat().st_size
    return files, size


def _remove(path: Path, root: Path, *, dry_run: bool) -> tuple[int, int]:
    """Delete one file or directory tree, returning ``(files, bytes)``; never escape ``root``."""
    target = _resolve_inside(path, root)
    if not target.is_symlink() and not target.exists():
        return 0, 0
    if target.is_dir() and not target.is_symlink():
        files, size = _tree_size(target)
        if not dry_run:
            shutil.rmtree(target)
        return files, size
    try:
        size = target.stat().st_size
    except FileNotFoundError:
        return 0, 0
    if not dry_run:
        target.unlink(missing_ok=True)
    return 1, size


async def _iter_terminal_jobs(repos: Repos) -> AsyncIterator[JobRecord]:
    for status in sorted(JOB_TERMINAL_STATUSES):
        cursor: str | None = None
        while True:
            page, cursor = await repos.jobs.list(status=status, limit=_RETENTION_PAGE, cursor=cursor)
            for record in page:
                yield record
            if cursor is None:
                break


async def run_retention(
    settings: Settings,
    repos: Repos,
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> RetentionReport:
    """Delete job media and stale uploads per docs/05 §3; never touch reports or transcripts."""
    current = now if now is not None else datetime.now(UTC)
    root = settings.CARACAL_DATA_DIR.resolve()
    job_cutoff = current - timedelta(days=max(0, settings.CARACAL_RETENTION_DAYS))
    upload_cutoff = current - timedelta(hours=_UPLOAD_MAX_AGE_HOURS)

    jobs = 0
    media_files = 0
    media_bytes = 0
    async for job in _iter_terminal_jobs(repos):
        if job.finished_at is None or job.finished_at >= job_cutoff:
            continue
        jobs += 1
        files = JobFiles(settings.CARACAL_DATA_DIR, job.id)
        for path in (files.audio_wav, files.source_dir, files.chunks_dir):
            count, size = await asyncio.to_thread(_remove, path, root, dry_run=dry_run)
            media_files += count
            media_bytes += size

    uploads = 0
    upload_bytes = 0
    for record in await repos.uploads.list_unconsumed_older_than(0):
        if record.created_at >= upload_cutoff:
            continue
        upload_files = UploadFiles(settings.CARACAL_DATA_DIR, record.id)
        try:
            path = upload_files.file(record.filename)
        except CaracalError as exc:
            logger.warning(
                "retention skipped an upload with an invalid filename",
                extra={"upload_id": str(record.id), "error": exc.message},
            )
            continue
        count, size = await asyncio.to_thread(_remove, path, root, dry_run=dry_run)
        if count and not dry_run:
            await asyncio.to_thread(_remove_empty_dir, upload_files.root, root)
        if not dry_run:
            await repos.uploads.delete(record.id)
        uploads += 1
        upload_bytes += size

    return RetentionReport(
        dry_run=dry_run,
        jobs=jobs,
        media_files=media_files,
        media_bytes=media_bytes,
        uploads=uploads,
        upload_bytes=upload_bytes,
    )


def _remove_empty_dir(path: Path, root: Path) -> None:
    target = _resolve_inside(path, root)
    try:
        target.rmdir()
    except OSError:
        return


# -- metrics ------------------------------------------------------------------


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _labels_field(*values: str) -> str:
    return _FIELD_SEP.join(values)


def _split_field(value: str, expected: int) -> tuple[str, ...] | None:
    parts = tuple(value.split(_FIELD_SEP))
    return parts if len(parts) == expected else None


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _number(raw: str) -> str:
    try:
        value = float(raw)
    except ValueError:
        return "0"
    if not math.isfinite(value):
        return "0"
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


class MetricsRegistry:
    """Prometheus counters stored in ``<REDIS_PREFIX>metrics:<family>`` hashes."""

    def __init__(self, redis: Redis, keys: Keys) -> None:
        self.redis = redis
        self.keys = keys

    def _hash(self, family: str) -> str:
        return self.keys.key("metrics", family)

    @staticmethod
    def _redis_error(family: str) -> CaracalError:
        return CaracalError(
            ErrorCode.DEPENDENCY_DOWN,
            "redis metrics update failed",
            {"metric": family},
            retryable=True,
        )

    async def _incr(self, family: str, field: str, amount: int = 1) -> None:
        try:
            await self.redis.hincrby(self._hash(family), field, amount)
        except RedisError as exc:
            raise self._redis_error(family) from exc

    async def _add(self, family: str, field: str, amount: float) -> None:
        try:
            await self.redis.hincrbyfloat(self._hash(family), field, amount)
        except RedisError as exc:
            raise self._redis_error(family) from exc

    async def _read(self, family: str) -> dict[str, str]:
        try:
            raw = await self.redis.hgetall(self._hash(family))
        except RedisError as exc:
            raise self._redis_error(family) from exc
        return {_text(key): _text(value) for key, value in raw.items()}

    async def incr_job(self, status: str) -> None:
        await self._incr("jobs_total", status)

    async def observe_stage(self, stage: str, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            return
        await self._incr("stage_seconds_count", stage)
        await self._add("stage_seconds_sum", stage, seconds)

    async def incr_stage_failure(self, stage: str) -> None:
        """Count one failed stage run (``caracal_stage_failures_total{stage}``)."""
        await self._incr("stage_failures_total", stage)

    async def incr_llm_call(self, endpoint: str, mode: str, fallback: bool) -> None:
        await self._incr("llm_calls_total", _labels_field(endpoint, mode, "true" if fallback else "false"))

    async def incr_search_query(self, provider: str, *, cached: bool) -> None:
        await self._incr("search_queries_total", _labels_field(provider, "true" if cached else "false"))

    async def add_search_cost(self, usd: float) -> None:
        if math.isfinite(usd) and usd >= 0:
            await self._add("search_cost_usd_total", "usd", usd)

    async def add_whisper_seconds(self, seconds: float) -> None:
        if math.isfinite(seconds) and seconds >= 0:
            await self._add("whisper_audio_seconds_total", "seconds", seconds)

    async def render(self) -> str:
        """Render every counter in Prometheus text format (one metric per line)."""
        lines: list[str] = []
        for status, value in sorted((await self._read("jobs_total")).items()):
            lines.append(f'caracal_jobs_total{{status="{_label(status)}"}} {_number(value)}')

        counts = await self._read("stage_seconds_count")
        totals = await self._read("stage_seconds_sum")
        for stage, value in sorted(counts.items()):
            lines.append(f'caracal_stage_seconds_count{{stage="{_label(stage)}"}} {_number(value)}')
            lines.append(
                f'caracal_stage_seconds_sum{{stage="{_label(stage)}"}} {_number(totals.get(stage, "0"))}'
            )

        for stage, value in sorted((await self._read("stage_failures_total")).items()):
            lines.append(f'caracal_stage_failures_total{{stage="{_label(stage)}"}} {_number(value)}')

        for field, value in sorted((await self._read("llm_calls_total")).items()):
            parts = _split_field(field, 3)
            if parts is None:
                continue
            endpoint, mode, fallback = parts
            lines.append(
                "caracal_llm_calls_total"
                f'{{endpoint="{_label(endpoint)}",mode="{_label(mode)}",fallback="{_label(fallback)}"}}'
                f" {_number(value)}"
            )

        for field, value in sorted((await self._read("search_queries_total")).items()):
            parts = _split_field(field, 2)
            if parts is None:
                continue
            provider, cached = parts
            lines.append(
                "caracal_search_queries_total"
                f'{{provider="{_label(provider)}",cached="{_label(cached)}"}} {_number(value)}'
            )

        cost = await self._read("search_cost_usd_total")
        lines.append(f"caracal_search_cost_usd_total {_number(cost.get('usd', '0'))}")
        whisper = await self._read("whisper_audio_seconds_total")
        lines.append(f"caracal_whisper_audio_seconds_total {_number(whisper.get('seconds', '0'))}")
        return "\n".join(lines) + "\n"


__all__ = [
    "CheckStatus",
    "DoctorCheck",
    "DoctorProbes",
    "MetricsRegistry",
    "RetentionReport",
    "default_probes",
    "doctor_exit_code",
    "emit_checks",
    "run_doctor",
    "run_retention",
]
