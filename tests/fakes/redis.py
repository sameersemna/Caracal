"""Fake Redis for queue tests: fakeredis plus a command/channel recorder.

``RecordingRedis`` subclasses the in-memory fakeredis server and records every command that
reaches ``execute_command`` (the funnel for all redis-py commands) as well as every channel
targeted by ``PUBLISH``. Tests use the recording plus a full ``SCAN *`` to prove that the
queue never touches a key or channel outside ``REDIS_PREFIX``.
"""

from __future__ import annotations

from typing import Any

import fakeredis.aioredis


def text(value: object) -> str:
    """Decode a redis value (bytes or str) to ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class RecordingRedis(fakeredis.aioredis.FakeRedis):
    """An in-memory Redis that records commands and published channel names."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commands: list[tuple[str, tuple[object, ...]]] = []
        self.published: list[str] = []

    async def execute_command(self, *args: Any, **options: Any) -> Any:
        name = text(args[0]).upper()
        self.commands.append((name, tuple(args[1:])))
        if name == "PUBLISH" and len(args) > 1:
            self.published.append(text(args[1]))
        return await super().execute_command(*args, **options)

    def command_names(self) -> list[str]:
        return [name for name, _arguments in self.commands]


async def all_keys(redis: RecordingRedis) -> list[str]:
    """Every key in the fake server, via SCAN (never KEYS)."""
    return sorted([text(key) async for key in redis.scan_iter(match="*")])


def make_fake_redis() -> RecordingRedis:
    """A fresh isolated fake Redis server."""
    return RecordingRedis()
