"""Evaluate a *running* Caracal server against the labelled case set (card T4.2).

The harness is deliberately an external client: every case is submitted through the REST
API, so what it measures is the deployed system (ingest → whisper → diarization →
identity → report), not an in-process mock. Reference material (YouTube manual captions
for WER, the JFK/TED splice for DER) is built in a temporary directory and never committed.

Everything is a report, not a test: the exit status is 0 unless ``--strict`` is given and an
expectation is unmet. ``metrics`` (WER/DER/identity/ROC) is pure and unit-tested.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval.metrics import Turn, der, identity_score, wer  # noqa: E402

from caracal.asr.captions import CaptionWord, normalise_word, parse_vtt  # noqa: E402

_DEFAULT_CASES = Path(__file__).with_name("cases.json")
_DEFAULT_ENV_VAR = "CARACAL_API_TOKEN"
_DEFAULT_TIMEOUT_S = 900.0
_DEFAULT_POLL_S = 3.0
_DEFAULT_REQUEST_TIMEOUT_S = 120.0
_COMMAND_TIMEOUT_S = 900.0
_MAX_DETAIL_CHARS = 500
_TERMINAL = frozenset({"done", "failed", "cancelled"})
_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>-?\d+(?:\.\d+)?)$"
)


class EvalError(RuntimeError):
    """A case could not be prepared or submitted; recorded, never swallowed."""


@dataclass(slots=True)
class Check:
    """One expectation and whether the live run met it."""

    name: str
    state: str
    detail: str = ""


@dataclass(slots=True)
class PreparedCase:
    """Everything a case needs before it can be submitted."""

    payload: dict[str, Any] = field(default_factory=dict)
    upload_path: Path | None = None
    ref_turns: list[Turn] = field(default_factory=list)
    reference_captions: Path | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CaseResult:
    """The measured outcome of one case."""

    case_id: str
    status: str = "pending"
    job_id: str | None = None
    wall_s: float | None = None
    error: str | None = None
    wer: tuple[int, int] | None = None
    der: dict[str, float] | None = None
    identity: dict[str, str] = field(default_factory=dict)
    stages: dict[str, float] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    subtitles: dict[str, int] = field(default_factory=dict)
    speakers: list[tuple[str, str | None, float]] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    llm_calls_delta: int = 0
    search_cost_delta: float = 0.0


def _tail(text: str, limit: int = _MAX_DETAIL_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else "…" + text[-limit:]


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _metric_total(values: Mapping[str, float], name: str) -> float:
    return sum(value for key, value in values.items() if key.partition("{")[0] == name)


def _has_metric(values: Mapping[str, float], name: str) -> bool:
    return any(key.partition("{")[0] == name for key in values)


def parse_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text into ``{name{labels}: value}``."""
    values: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        labels = match.group("labels")
        key = f"{name}{{{labels}}}" if labels else name
        values[key] = float(match.group("value"))
    return values


async def _run_command(argv: Sequence[str], *, timeout_s: float = _COMMAND_TIMEOUT_S) -> tuple[int, str]:
    """Run one argv (never a shell) and return ``(exit_code, combined_output)``."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(timeout_s):
            stdout, stderr = await process.communicate()
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise EvalError(f"command timed out after {timeout_s:.0f}s: {argv[0]}") from exc
    output = (stdout + stderr).decode("utf-8", "replace")
    return process.returncode or 0, output


async def _run_ytdlp(argv: Sequence[str], *, attempts: int = 3, delay_s: float = 5.0) -> tuple[int, str]:
    """Run yt-dlp, retrying transient YouTube 403s that a fresh request usually clears."""
    last_code = 1
    last_output = ""
    for attempt in range(attempts):
        last_code, last_output = await _run_command(argv)
        if last_code == 0:
            return last_code, last_output
        if attempt + 1 < attempts:
            print(
                f"  yt-dlp failed (attempt {attempt + 1}/{attempts}), retrying in {delay_s:.0f}s ...",
                file=sys.stderr,
            )
            await asyncio.sleep(delay_s)
    return last_code, last_output


def _ytdlp_argv() -> list[str]:
    argv = [sys.executable, "-m", "yt_dlp", "--no-progress", "--quiet", "--no-warnings", "--no-playlist"]
    if shutil.which("node") is not None:
        argv += ["--js-runtimes", "node"]
    return argv


def _list_files(directory: Path, pattern: str) -> list[Path]:
    return sorted(path for path in directory.glob(pattern) if path.is_file())


async def _download_manual_captions(url: str, work_dir: Path) -> Path | None:
    """Download the *manual* English captions as VTT into ``work_dir`` (never the repo)."""
    target = work_dir / "captions"
    await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
    argv = [
        *_ytdlp_argv(),
        "--skip-download",
        "--write-subs",
        "--no-write-auto-subs",
        "--sub-langs",
        "en",
        "--sub-format",
        "vtt",
        "--output",
        str(target / "%(id)s.%(ext)s"),
        url,
    ]
    _, output = await _run_ytdlp(argv)
    files = await asyncio.to_thread(_list_files, target, "*.vtt")
    if not files:
        print(f"  captions unavailable: {_tail(output, 200)}", file=sys.stderr)
        return None
    return files[0]


async def _probe_duration(ffprobe: str, path: Path) -> float:
    argv = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    code, output = await _run_command(argv, timeout_s=60.0)
    if code != 0:
        raise EvalError(f"ffprobe failed for {path.name}: {_tail(output, 200)}")
    for line in reversed(output.strip().splitlines()):
        try:
            return float(line.strip())
        except ValueError:
            continue
    raise EvalError(f"ffprobe returned no duration for {path.name}")


async def _build_splice(source: Mapping[str, Any], work_dir: Path) -> tuple[Path, list[Turn]]:
    """Concatenate ``jfk.wav`` with two TED ranges and return the file plus reference turns."""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise EvalError("ffmpeg and ffprobe must be on PATH to build the splice case")

    fixture = (_REPO_ROOT / str(source["path"])).resolve()
    if not fixture.is_file():
        raise EvalError(f"splice fixture not found: {fixture}")

    splice_spec = source.get("splice")
    if not isinstance(splice_spec, Mapping) or not isinstance(splice_spec.get("url"), str):
        raise EvalError("splice case needs source.splice.url")
    ranges = splice_spec.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise EvalError("splice case needs source.splice.ranges")

    media_dir = work_dir / "media"
    await asyncio.to_thread(media_dir.mkdir, parents=True, exist_ok=True)
    download_argv = [
        *_ytdlp_argv(),
        "-f",
        "bestaudio/best",
        "--output",
        str(media_dir / "ted.%(ext)s"),
        splice_spec["url"],
    ]
    code, output = await _run_ytdlp(download_argv)
    downloaded = [
        path
        for path in await asyncio.to_thread(_list_files, media_dir, "ted.*")
        if not path.name.endswith(".part")
    ]
    if not downloaded:
        raise EvalError(f"could not download the splice audio: {_tail(output)}")

    cut_paths: list[Path] = []
    for index, pair in enumerate(ranges):
        try:
            start, end = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise EvalError(f"invalid splice range: {pair!r}") from exc
        cut_path = work_dir / f"ted-cut-{index}.wav"
        cut_argv = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(downloaded[0]),
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(cut_path),
        ]
        code, output = await _run_command(cut_argv)
        if code != 0 or not cut_path.is_file():
            raise EvalError(f"ffmpeg could not cut the splice segment: {_tail(output)}")
        cut_paths.append(cut_path)

    splice_path = work_dir / "jfk-ted-splice.wav"
    inputs: list[str] = []
    for path in [fixture, *cut_paths]:
        inputs += ["-i", str(path)]
    filter_inputs = "".join(f"[{index}:a]" for index in range(len(cut_paths) + 1))
    concat_argv = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        *inputs,
        "-filter_complex",
        f"{filter_inputs}concat=n={len(cut_paths) + 1}:v=0:a=1[out]",
        "-map",
        "[out]",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(splice_path),
    ]
    code, output = await _run_command(concat_argv)
    if code != 0 or not splice_path.is_file():
        raise EvalError(f"ffmpeg could not concatenate the splice: {_tail(output)}")

    jfk_duration = await _probe_duration(ffprobe, fixture)
    total_duration = await _probe_duration(ffprobe, splice_path)
    ref_turns = [
        Turn(0.0, jfk_duration, "JFK"),
        Turn(jfk_duration, total_duration, "TED"),
    ]
    return splice_path, ref_turns


async def prepare_case(case: Mapping[str, Any], work_dir: Path) -> PreparedCase:
    """Download/build whatever the case needs, outside the repository."""
    source = case.get("source")
    if not isinstance(source, Mapping):
        raise EvalError("case has no source object")
    expect = case.get("expect") or {}
    prepared = PreparedCase()
    if isinstance(source.get("url"), str):
        prepared.payload = {"url": source["url"]}
        if expect.get("reference_captions") == "manual" and "wer_max" in expect:
            captions = await _download_manual_captions(source["url"], work_dir)
            prepared.reference_captions = captions
            if captions is None:
                prepared.notes.append("manual captions unavailable; WER skipped")
    elif isinstance(source.get("path"), str):
        if "splice" in source:
            splice_path, ref_turns = await _build_splice(source, work_dir)
            prepared.upload_path = splice_path
            prepared.ref_turns = ref_turns
        else:
            prepared.upload_path = (_REPO_ROOT / source["path"]).resolve()
    else:
        raise EvalError("source must contain url or path")
    return prepared


async def _upload(client: httpx.AsyncClient, base_url: str, headers: Mapping[str, str], path: Path) -> str:
    with path.open("rb") as handle:
        response = await client.post(
            f"{base_url}/v1/uploads",
            headers=headers,
            files={"file": (path.name, handle, "audio/wav")},
        )
    if response.status_code != 201:
        raise EvalError(f"POST /v1/uploads -> {response.status_code}: {_tail(response.text)}")
    payload = response.json()
    upload_id = payload.get("upload_id")
    if not isinstance(upload_id, str):
        raise EvalError("upload response has no upload_id")
    return upload_id


async def _submit(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Mapping[str, str],
    case: Mapping[str, Any],
    prepared: PreparedCase,
) -> str:
    source = dict(prepared.payload)
    if prepared.upload_path is not None:
        source = {"upload_id": await _upload(client, base_url, headers, prepared.upload_path)}
    options = {"language": "auto", "depth": "standard", "diarize": "auto", "verify": "auto"}
    options.update(case.get("options") or {})
    response = await client.post(
        f"{base_url}/v1/jobs",
        headers=headers,
        json={"source": source, "options": options},
    )
    if response.status_code not in {200, 202}:
        raise EvalError(f"POST /v1/jobs -> {response.status_code}: {_tail(response.text)}")
    job_id = response.json().get("job_id")
    if not isinstance(job_id, str):
        raise EvalError("job response has no job_id")
    return job_id


async def _wait_for_job(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Mapping[str, str],
    job_id: str,
    *,
    timeout_s: float,
    poll_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while True:
        response = await client.get(f"{base_url}/v1/jobs/{job_id}", headers=headers)
        if response.status_code != 200:
            raise EvalError(f"GET /v1/jobs/{job_id} -> {response.status_code}: {_tail(response.text)}")
        status = response.json()
        if status.get("status") in _TERMINAL:
            return status
        if time.monotonic() >= deadline:
            raise EvalError(f"job {job_id} did not finish within {timeout_s:.0f}s")
        await asyncio.sleep(poll_s)


def _stage_seconds(status: Mapping[str, Any]) -> dict[str, float]:
    seconds: dict[str, float] = {}
    stages = status.get("stages")
    if not isinstance(stages, list):
        return seconds
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        name = stage.get("name")
        started = stage.get("started_at")
        ended = stage.get("ended_at")
        if not isinstance(name, str) or not isinstance(started, str) or not isinstance(ended, str):
            continue
        try:
            duration = (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds()
        except ValueError:
            continue
        if duration >= 0:
            seconds[name] = duration
    return seconds


async def _fetch_json(
    client: httpx.AsyncClient,
    url: str,
    headers: Mapping[str, str],
    *,
    what: str,
) -> dict[str, Any]:
    response = await client.get(url, headers=headers)
    if response.status_code != 200:
        raise EvalError(f"GET {what} -> {response.status_code}: {_tail(response.text)}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise EvalError(f"GET {what} did not return a JSON object")
    return payload


async def _fetch_subtitles(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Mapping[str, str],
    job_id: str,
    status: Mapping[str, Any],
) -> dict[str, int]:
    links = status.get("links")
    subtitles = links.get("subtitles") if isinstance(links, Mapping) else None
    if not isinstance(subtitles, Mapping):
        return {}
    sizes: dict[str, int] = {}
    for name in sorted(subtitles):
        if not isinstance(name, str):
            continue
        response = await client.get(
            f"{base_url}/v1/jobs/{job_id}/subtitles/{quote(name)}",
            headers=headers,
        )
        if response.status_code == 200:
            sizes[name] = len(response.text)
    return sizes


def _turns_from_transcript(transcript: Mapping[str, Any]) -> list[Turn]:
    turns: list[Turn] = []
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        return turns
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        speaker = segment.get("speaker")
        start = segment.get("start")
        end = segment.get("end")
        if not isinstance(speaker, str):
            continue
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            continue
        if not isinstance(end, (int, float)) or isinstance(end, bool):
            continue
        if float(end) > float(start):
            turns.append(Turn(float(start), float(end), speaker))
    return turns


def _caption_tokens(words: Sequence[CaptionWord]) -> list[tuple[float, str]]:
    tokens: list[tuple[float, str]] = []
    for word in words:
        token = normalise_word(word.text)
        if token:
            tokens.append(((word.start + word.end) / 2.0, token))
    return tokens


def _transcript_tokens(transcript: Mapping[str, Any]) -> list[tuple[float, str]]:
    tokens: list[tuple[float, str]] = []
    segments = transcript.get("segments")
    if not isinstance(segments, list):
        return tokens
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, Mapping):
                continue
            text = word.get("text")
            start = word.get("start")
            end = word.get("end")
            if not isinstance(text, str):
                continue
            if not isinstance(start, (int, float)) or isinstance(start, bool):
                continue
            if not isinstance(end, (int, float)) or isinstance(end, bool):
                continue
            token = normalise_word(text)
            if token:
                tokens.append(((float(start) + float(end)) / 2.0, token))
    return tokens


def _overlapping_tokens(
    reference: Sequence[tuple[float, str]],
    hypothesis: Sequence[tuple[float, str]],
) -> tuple[list[str], list[str]]:
    """Restrict both sides to their overlapping time range, like ``caption_agreement``."""
    if not reference or not hypothesis:
        return [], []
    start = max(min(moment for moment, _ in reference), min(moment for moment, _ in hypothesis))
    end = min(max(moment for moment, _ in reference), max(moment for moment, _ in hypothesis))
    ref_words = [token for moment, token in reference if start <= moment <= end]
    hyp_words = [token for moment, token in hypothesis if start <= moment <= end]
    return ref_words, hyp_words


def _identity_detail(scores: Mapping[str, str], actual: Mapping[str, str | None]) -> str:
    parts = []
    for label, state in scores.items():
        parts.append(f"{label}: {state} (actual {actual.get(label)!r})")
    return "; ".join(parts)


async def evaluate_case(
    case: Mapping[str, Any],
    result: CaseResult,
    report: Mapping[str, Any],
    transcript: Mapping[str, Any],
    prepared: PreparedCase,
) -> None:
    """Compute the case metrics and turn each expectation into a :class:`Check`."""
    expect = case.get("expect") or {}
    speakers = report.get("speakers")
    speaker_rows = [row for row in speakers if isinstance(row, Mapping)] if isinstance(speakers, list) else []
    actual: dict[str, str | None] = {}
    for row in speaker_rows:
        label = row.get("label")
        name = row.get("name")
        if isinstance(label, str):
            actual[label] = name if isinstance(name, str) else None
    result.speakers = [
        (
            str(row.get("label")),
            row.get("name") if isinstance(row.get("name"), str) else None,
            float(row.get("confidence") or 0.0),
        )
        for row in speaker_rows
        if isinstance(row.get("label"), str)
    ]
    languages = report.get("languages")
    result.languages = (
        [
            str(row.get("code"))
            for row in languages
            if isinstance(row, Mapping) and isinstance(row.get("code"), str)
        ]
        if isinstance(languages, list)
        else []
    )

    if isinstance(expect.get("identity"), Mapping):
        expected = {
            str(label): (name if isinstance(name, str) else None)
            for label, name in expect["identity"].items()
        }
        scores = identity_score(expected, actual)
        result.identity.update(scores)
        met = bool(scores) and all(state == "correct" for state in scores.values())
        result.checks.append(
            Check("identity (labelled)", "met" if met else "unmet", _identity_detail(scores, actual))
        )

    if isinstance(expect.get("speaker_name"), str):
        dominant = max(speaker_rows, key=lambda row: float(row.get("share") or 0.0), default=None)
        if dominant is None or not isinstance(dominant.get("label"), str):
            result.checks.append(Check("speaker name", "unmet", "the report has no speaker labels"))
        else:
            label = str(dominant["label"])
            scores = identity_score({label: str(expect["speaker_name"])}, actual)
            result.identity.update(scores)
            met = all(state == "correct" for state in scores.values())
            share = float(dominant.get("share") or 0.0)
            detail = f"{_identity_detail(scores, actual)}; dominant share {share:.2f}"
            result.checks.append(
                Check(f"speaker name = {expect['speaker_name']}", "met" if met else "unmet", detail)
            )

    if expect.get("speakers_unnamed") is True:
        named = [label for label, name in actual.items() if name is not None]
        result.checks.append(
            Check(
                "all speaker names abstained",
                "met" if not named else "unmet",
                "no promoted names" if not named else "named: " + ", ".join(named),
            )
        )

    if isinstance(expect.get("no_strong_name"), (int, float)) and not isinstance(
        expect["no_strong_name"], bool
    ):
        limit = float(expect["no_strong_name"])
        offenders = [
            f"{label}={name!r} ({confidence:.2f})"
            for label, name, confidence in result.speakers
            if name is not None and confidence >= limit
        ]
        result.checks.append(
            Check(
                f"no name with confidence >= {limit:.2f}",
                "met" if not offenders else "unmet",
                "none" if not offenders else "; ".join(offenders),
            )
        )

    if isinstance(expect.get("language"), str):
        wanted = expect["language"]
        met = wanted in result.languages
        result.checks.append(
            Check(
                f"language = {wanted}",
                "met" if met else "unmet",
                "detected: " + (", ".join(result.languages) or "none"),
            )
        )

    if isinstance(expect.get("subtitles_include"), list):
        wanted_codes = [str(code) for code in expect["subtitles_include"]]
        missing = [
            code for code in wanted_codes if not any(name.startswith(f"{code}.") for name in result.subtitles)
        ]
        result.checks.append(
            Check(
                "subtitles " + ", ".join(wanted_codes),
                "met" if not missing else "unmet",
                "served: " + (", ".join(sorted(result.subtitles)) or "none"),
            )
        )

    if isinstance(expect.get("music_detected"), bool):
        audio = report.get("audio")
        actual_music = audio.get("music_detected") if isinstance(audio, Mapping) else None
        result.checks.append(
            Check(
                f"music_detected = {expect['music_detected']}",
                "met" if actual_music is expect["music_detected"] else "unmet",
                f"report says {actual_music!r}",
            )
        )

    if isinstance(expect.get("der_max"), (int, float)) and not isinstance(expect["der_max"], bool):
        limit = float(expect["der_max"])
        hypothesis = _turns_from_transcript(transcript)
        if not prepared.ref_turns:
            result.checks.append(Check("DER", "skipped", "no reference turns for this case"))
        elif not hypothesis:
            result.checks.append(Check("DER", "unmet", "the transcript has no speaker turns"))
        else:
            result.der = der(prepared.ref_turns, hypothesis)
            met = result.der["der"] <= limit
            detail = (
                f"DER {_fmt(result.der['der'])} (miss {_fmt(result.der['miss'])}, "
                f"false alarm {_fmt(result.der['false_alarm'])}, confusion {_fmt(result.der['confusion'])})"
            )
            result.checks.append(Check(f"DER <= {limit:.2f}", "met" if met else "unmet", detail))

    if isinstance(expect.get("wer_max"), (int, float)) and not isinstance(expect["wer_max"], bool):
        limit = float(expect["wer_max"])
        if prepared.reference_captions is None:
            result.checks.append(
                Check(f"WER <= {limit:.2f}", "skipped", "manual reference captions unavailable")
            )
        else:
            captions_text = await asyncio.to_thread(prepared.reference_captions.read_text, encoding="utf-8")
            reference = _caption_tokens(parse_vtt(captions_text))
            hypothesis = _transcript_tokens(transcript)
            ref_words, hyp_words = _overlapping_tokens(reference, hypothesis)
            errors, total = wer(ref_words, hyp_words)
            result.wer = (errors, total)
            rate = errors / total if total else None
            met = rate is not None and rate <= limit
            detail = f"{errors}/{total} word errors = {_fmt(rate)}"
            result.checks.append(Check(f"WER <= {limit:.2f}", "met" if met else "unmet", detail))

    if expect.get("jfk_candidate_allowed") is True:
        candidates = [
            str(candidate.get("name"))
            for row in speaker_rows
            for candidate in (row.get("candidates") or [])
            if isinstance(candidate, Mapping) and isinstance(candidate.get("name"), str)
        ]
        kennedy = [name for name in candidates if "kennedy" in name.casefold()]
        result.notes.append(
            "JFK candidates are allowed but not required: " + (", ".join(kennedy) if kennedy else "none seen")
        )


async def execute_case(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Mapping[str, str],
    case: Mapping[str, Any],
    work_dir: Path,
    *,
    timeout_s: float,
    poll_s: float,
) -> CaseResult:
    case_id = str(case.get("id", "unnamed"))
    result = CaseResult(case_id=case_id)
    started = time.monotonic()
    metrics_before = await read_metrics(client, base_url, headers)
    try:
        prepared = await prepare_case(case, work_dir)
        result.notes.extend(prepared.notes)
        result.job_id = await _submit(client, base_url, headers, case, prepared)
        status = await _wait_for_job(
            client, base_url, headers, result.job_id, timeout_s=timeout_s, poll_s=poll_s
        )
        result.status = str(status.get("status"))
        result.stages = _stage_seconds(status)
        if result.status != "done":
            error = status.get("error")
            result.error = _tail(str(error)) if error else f"job finished {result.status}"
            result.checks.append(Check("pipeline completed", "unmet", result.error))
            return result

        report = await _fetch_json(
            client, f"{base_url}/v1/jobs/{result.job_id}/report", headers, what="report"
        )
        transcript = await _fetch_json(
            client, f"{base_url}/v1/jobs/{result.job_id}/transcript?format=json", headers, what="transcript"
        )
        result.subtitles = await _fetch_subtitles(client, base_url, headers, result.job_id, status)
        await evaluate_case(case, result, report, transcript, prepared)
    except (EvalError, httpx.HTTPError, OSError) as exc:
        result.status = "error"
        result.error = _tail(str(exc))
        result.checks.append(Check("pipeline completed", "unmet", result.error))
    finally:
        if result.wall_s is None:
            result.wall_s = time.monotonic() - started
        metrics_after = await read_metrics(client, base_url, headers)
        if metrics_before is not None and metrics_after is not None:
            calls_before = _metric_total(metrics_before, "caracal_llm_calls_total")
            calls_after = _metric_total(metrics_after, "caracal_llm_calls_total")
            result.llm_calls_delta = round(calls_after - calls_before)
            cost_before = _metric_total(metrics_before, "caracal_search_cost_usd_total")
            cost_after = _metric_total(metrics_after, "caracal_search_cost_usd_total")
            result.search_cost_delta = cost_after - cost_before
    return result


async def read_metrics(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Mapping[str, str],
) -> dict[str, float] | None:
    """Read Prometheus counters; telemetry is optional, so an unreachable endpoint is ``None``."""
    try:
        response = await client.get(f"{base_url}/metrics", headers=headers)
    except httpx.HTTPError as exc:
        print(f"  metrics unavailable: {exc}", file=sys.stderr)
        return None
    if response.status_code != 200:
        print(f"  metrics unavailable: HTTP {response.status_code}", file=sys.stderr)
        return None
    return parse_metrics(response.text)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise EvalError(f"{path} must contain a JSON list of case objects")
    return payload


def _select_cases(cases: list[dict[str, Any]], wanted: str | None) -> list[dict[str, Any]]:
    if wanted is None:
        return cases
    identifiers = [item.strip() for item in wanted.split(",") if item.strip()]
    known = {str(case.get("id")) for case in cases}
    unknown = [identifier for identifier in identifiers if identifier not in known]
    if unknown:
        raise EvalError("unknown case id(s): " + ", ".join(unknown))
    selected = [case for case in cases if str(case.get("id")) in identifiers]
    if not selected:
        raise EvalError("no cases selected")
    return selected


def _case_row(result: CaseResult) -> str:
    identity = result.identity
    identity_text = (
        f"{sum(1 for state in identity.values() if state == 'correct')}/{len(identity)}"
        if identity
        else "n/a"
    )
    wer_text = "n/a"
    if result.wer is not None and result.wer[1]:
        wer_text = _fmt(result.wer[0] / result.wer[1])
    der_text = _fmt(result.der["der"]) if result.der is not None else "n/a"
    met = sum(1 for check in result.checks if check.state == "met")
    unmet = sum(1 for check in result.checks if check.state == "unmet")
    skipped = sum(1 for check in result.checks if check.state == "skipped")
    expectation_text = f"{met}/{met + unmet}"
    if skipped:
        expectation_text += f" (+{skipped} skipped)"
    status = result.status if result.error is None else f"{result.status}: {_tail(result.error, 80)}"
    status = status.replace("|", "/")
    return (
        f"| {result.case_id} | {status} | {_fmt(result.wall_s, 1)} | {wer_text} | {der_text} | "
        f"{identity_text} | {expectation_text} |"
    )


def render_report(
    results: Sequence[CaseResult],
    *,
    base_url: str,
    cases_file: Path,
    totals: Mapping[str, float] | None,
    strict: bool,
) -> str:
    """Render the Markdown eval report."""
    met = sum(1 for result in results for check in result.checks if check.state == "met")
    unmet = sum(1 for result in results for check in result.checks if check.state == "unmet")
    skipped = sum(1 for result in results for check in result.checks if check.state == "skipped")

    wer_rates = [
        result.wer[0] / result.wer[1] for result in results if result.wer is not None and result.wer[1]
    ]
    der_rates = [result.der["der"] for result in results if result.der is not None]
    identity_states = [state for result in results for state in result.identity.values()]
    identity_correct = sum(1 for state in identity_states if state == "correct")

    lines: list[str] = []
    lines.append(f"# Caracal eval — {datetime.now(UTC).date().isoformat()}")
    lines.append("")
    lines.append(f"- Base URL: `{base_url}`")
    lines.append(f"- Cases: `{cases_file}`")
    lines.append(f"- Strict: {'yes' if strict else 'no'}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| case | status | wall s | WER | DER | identity | expectations |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    lines.extend(_case_row(result) for result in results)
    lines.append("")
    lines.append(f"**Expectations met: {met}/{met + unmet}**" + (f" ({skipped} skipped)" if skipped else ""))
    lines.append("")
    if wer_rates:
        lines.append(f"- WER: mean {_fmt(sum(wer_rates) / len(wer_rates))} over {len(wer_rates)} case(s)")
    if der_rates:
        lines.append(f"- DER: mean {_fmt(sum(der_rates) / len(der_rates))} over {len(der_rates)} case(s)")
    if identity_states:
        lines.append(f"- Identity labels: {identity_correct}/{len(identity_states)} correct")
    lines.append("")
    lines.append("## Timings and cost")
    lines.append("")
    if totals is None:
        lines.append("`GET /metrics` was unavailable; call and cost totals are not reported.")
    else:
        cost = _metric_total(totals, "caracal_search_cost_usd_total")
        if _has_metric(totals, "caracal_llm_calls_total"):
            calls = _metric_total(totals, "caracal_llm_calls_total")
            lines.append(f"- LLM calls (`caracal_llm_calls_total`): {calls:.0f}")
        else:
            lines.append(
                "- LLM calls: not recorded — `/metrics` exposes no `caracal_llm_calls_total` series "
                "(the counter exists in `ops.MetricsRegistry` but no pipeline client increments it)."
            )
        lines.append(f"- Search API cost (`caracal_search_cost_usd_total`): ${cost:.4f}")
        if _has_metric(totals, "caracal_whisper_audio_seconds_total"):
            whisper = _metric_total(totals, "caracal_whisper_audio_seconds_total")
            suffix = " (no code path increments this counter)" if whisper == 0 else ""
            lines.append(
                f"- Whisper audio processed (`caracal_whisper_audio_seconds_total`): {whisper:.0f}s{suffix}"
            )
        else:
            lines.append(
                "- Whisper audio: not recorded — no `caracal_whisper_audio_seconds_total` series is emitted."
            )
        lines.append(
            "- `/metrics` does not track token usage or LLM spend; the call count is the only LLM "
            "signal it can carry."
        )
    lines.append("")
    lines.append("Per-case wall time and stage durations:")
    lines.append("")
    for result in results:
        stage_text = ", ".join(
            f"{name} {_fmt(seconds, 1)}s" for name, seconds in sorted(result.stages.items())
        )
        lines.append(f"- `{result.case_id}`: wall {_fmt(result.wall_s, 1)}s")
        if stage_text:
            lines.append(f"  - stages: {stage_text}")
        if result.llm_calls_delta or result.search_cost_delta:
            lines.append(
                f"  - LLM calls {result.llm_calls_delta}, search cost ${result.search_cost_delta:.4f}"
            )
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    for result in results:
        lines.append(f"### {result.case_id}")
        lines.append("")
        lines.append(f"- Status: `{result.status}`")
        if result.job_id:
            lines.append(f"- Job: `{result.job_id}`")
        lines.append(f"- Wall time: {_fmt(result.wall_s, 1)}s")
        if result.error:
            lines.append(f"- Error: {result.error}")
        if result.languages:
            lines.append(f"- Languages: {', '.join(result.languages)}")
        if result.speakers:
            speaker_text = "; ".join(
                f"{label} {name!r} ({confidence:.2f})" if name else f"{label} unnamed ({confidence:.2f})"
                for label, name, confidence in result.speakers
            )
            lines.append(f"- Speakers: {speaker_text}")
        if result.subtitles:
            lines.append(
                "- Subtitles: "
                + ", ".join(f"`{name}` {size} bytes" for name, size in sorted(result.subtitles.items()))
            )
        if result.wer is not None:
            errors, total = result.wer
            lines.append(f"- WER: {errors}/{total} = {_fmt(errors / total if total else None)}")
        if result.der is not None:
            lines.append(
                "- DER: {der} (miss {miss}, false alarm {false_alarm}, confusion {confusion})".format(
                    der=_fmt(result.der["der"]),
                    miss=_fmt(result.der["miss"]),
                    false_alarm=_fmt(result.der["false_alarm"]),
                    confusion=_fmt(result.der["confusion"]),
                )
            )
        if result.identity:
            lines.append(
                "- Identity: "
                + ", ".join(f"{label} {state}" for label, state in sorted(result.identity.items()))
            )
        for note in result.notes:
            lines.append(f"- Note: {note}")
        lines.append("- Expectations:")
        for check in result.checks:
            mark = {"met": "x", "unmet": " ", "skipped": "-"}[check.state]
            suffix = f" — {check.detail}" if check.detail else ""
            lines.append(f"  - [{mark}] {check.name}{suffix}")
        lines.append("")
    lines.append("## Voiceprint calibration")
    lines.append("")
    lines.append(
        "Skipped: speaker embeddings are not exposed over the REST API. The report exposes only a "
        "label's similarity to an *enrolled* speaker (`voiceprint_similarity`), which cannot produce "
        "the same/different pairs a threshold calibration needs. `scripts/eval/metrics.py` implements "
        "`roc()` and `best_threshold()` and is unit-tested; wire them to an embedding export when one "
        "exists."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Cases are submitted through the public REST API to a running server, so the numbers include "
        "ingest, ASR, diarization, identity, verification and report assembly."
    )
    lines.append(
        "- WER uses YouTube *manual* captions downloaded with yt-dlp into a temporary directory and the "
        "same normalisation as `caption_agreement`; DER uses the reference turns known from the splice "
        "construction and a 0.25 s collar around reference boundaries."
    )
    lines.append(
        "- Media and reference files live only in a temporary directory; nothing is written into the "
        "repository except this report."
    )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a running Caracal server (card T4.2).")
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:11497")
    parser.add_argument(
        "--token-env",
        default=_DEFAULT_ENV_VAR,
        help=f"environment variable holding the API token (default {_DEFAULT_ENV_VAR})",
    )
    parser.add_argument("--cases", default=None, help="comma-separated case ids (default: all)")
    parser.add_argument("--cases-file", type=Path, default=_DEFAULT_CASES, help="path to cases.json")
    parser.add_argument("--out", type=Path, default=None, help="Markdown output path")
    parser.add_argument("--strict", action="store_true", help="exit 1 when an expectation is unmet")
    parser.add_argument(
        "--timeout", type=float, default=_DEFAULT_TIMEOUT_S, help="per-job timeout in seconds"
    )
    parser.add_argument("--poll", type=float, default=_DEFAULT_POLL_S, help="status poll interval in seconds")
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=_DEFAULT_REQUEST_TIMEOUT_S,
        help="HTTP request timeout in seconds",
    )
    parser.add_argument(
        "--keep-work", action="store_true", help="keep downloaded media in the temp directory"
    )
    return parser.parse_args(argv)


async def _async_main(args: argparse.Namespace) -> int:
    token = os.environ.get(args.token_env)
    if not token:
        print(f"error: environment variable {args.token_env} is not set", file=sys.stderr)
        return 2
    base_url = args.base_url.rstrip("/")
    try:
        cases = _select_cases(_load_cases(args.cases_file), args.cases)
    except (EvalError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    headers = {"Authorization": f"Bearer {token}"}
    out_path = args.out
    if out_path is None:
        out_path = _REPO_ROOT / "docs" / "findings" / f"eval-{datetime.now(UTC).date().isoformat()}.md"

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.keep_work:
        work_dir = Path(tempfile.mkdtemp(prefix="caracal-eval-"))
    else:
        temporary = tempfile.TemporaryDirectory(prefix="caracal-eval-")
        work_dir = Path(temporary.name)

    results: list[CaseResult] = []
    totals: dict[str, float] | None = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(args.request_timeout)) as client:
            for case in cases:
                print(f"eval: {case.get('id')} ...", file=sys.stderr)
                try:
                    result = await execute_case(
                        client,
                        base_url,
                        headers,
                        case,
                        work_dir,
                        timeout_s=args.timeout,
                        poll_s=args.poll,
                    )
                except Exception as exc:  # a report must still be produced when one case crashes
                    message = _tail(f"{type(exc).__name__}: {exc}")
                    result = CaseResult(
                        case_id=str(case.get("id", "unnamed")),
                        status="error",
                        error=message,
                        checks=[Check("pipeline completed", "unmet", message)],
                    )
                results.append(result)
                met = sum(1 for check in result.checks if check.state == "met")
                unmet = sum(1 for check in result.checks if check.state == "unmet")
                print(
                    f"eval: {result.case_id} {result.status} in {_fmt(result.wall_s, 1)}s "
                    f"({met}/{met + unmet} met)",
                    file=sys.stderr,
                )
            totals = await read_metrics(client, base_url, headers)
    finally:
        if temporary is not None:
            temporary.cleanup()

    markdown = render_report(
        results,
        base_url=base_url,
        cases_file=args.cases_file,
        totals=totals,
        strict=args.strict,
    )
    await asyncio.to_thread(out_path.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(out_path.write_text, markdown, encoding="utf-8")
    print(f"eval: report written to {out_path}", file=sys.stderr)
    if args.keep_work:
        print(f"eval: work dir kept at {work_dir}", file=sys.stderr)

    if not args.strict:
        return 0
    failed = any(
        result.error is not None or any(check.state == "unmet" for check in result.checks)
        for result in results
    )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    return asyncio.run(_async_main(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
