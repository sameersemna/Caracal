"""Live ``caracal mcp-stdio`` check (card T4.3).

Run by the judge under ``scripts/live-suite.sh``. Spawns the real stdio server, drives the
JSON-RPC handshake (initialize, tools/list, list_jobs) over stdin, and asserts that stdout
carries nothing but JSON-RPC lines: logs must go to stderr and no stray output may corrupt
the MCP framing.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.live

_PROTOCOL_VERSION = "2026-07-28"


def _caracal_executable() -> str:
    candidate = Path(sys.executable).parent / "caracal"
    if candidate.is_file():
        return str(candidate)
    found = shutil.which("caracal")
    if found is None:
        pytest.skip("the caracal executable is not on PATH")
    return found


async def _request(
    process: asyncio.subprocess.Process,
    message: dict[str, Any],
    lines: list[str],
) -> dict[str, Any]:
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
    await process.stdin.drain()
    line = await asyncio.wait_for(process.stdout.readline(), timeout=60)
    assert line, f"no response to {message.get('method')}"
    text = line.decode("utf-8").strip()
    lines.append(text)
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    assert parsed.get("jsonrpc") == "2.0"
    return parsed


async def test_mcp_stdio_serves_only_json_rpc_on_stdout() -> None:
    process = await asyncio.create_subprocess_exec(
        _caracal_executable(),
        "mcp-stdio",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=os.environ.copy(),
    )
    lines: list[str] = []
    try:
        initialized = await _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "caracal-live", "version": "1"},
                },
            },
            lines,
        )
        assert initialized["result"]["serverInfo"]["name"] == "caracal"

        assert process.stdin is not None
        process.stdin.write(b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
        await process.stdin.drain()

        listed = await _request(
            process,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            lines,
        )
        names = {tool["name"] for tool in listed["result"]["tools"]}
        assert "list_jobs" in names
        assert "ask_about_audio" in names

        called = await _request(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "list_jobs", "arguments": {"limit": 5}},
            },
            lines,
        )
        result = called["result"]
        assert result["isError"] is False, result
        structured = result.get("structuredContent")
        assert isinstance(structured, dict), result
        assert isinstance(structured["items"], list)

        assert process.stdin is not None
        process.stdin.close()
        await asyncio.wait_for(process.wait(), timeout=30)
        assert process.stdout is not None
        remaining = (await process.stdout.read()).decode("utf-8")
        lines.extend(line for line in remaining.splitlines() if line.strip())
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()

    assert lines
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert parsed.get("jsonrpc") == "2.0"
        assert "level" not in parsed
    assert "\n" not in lines[0].strip()
