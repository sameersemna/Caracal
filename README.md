# Caracal

Caracal takes an audio/video file or a link (YouTube or any site yt-dlp supports) and returns a
complete, evidence-backed **audio intelligence report**: transcript with word timestamps, languages,
subtitles, speaker identification, context, entities, quotes and cross-verified claims — not just a
transcript. It exposes the same capabilities over a REST API and an MCP server.


Caracal runs as three processes that share one package: `caracal serve` (FastAPI + MCP HTTP),
`caracal worker` (Redis Streams pipeline consumer) and `caracal mcp-stdio` (stdio MCP transport).

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The specification lives in `docs/`.

```bash
uv sync
bash scripts/gate.sh
```

The gate runs ruff (lint + format check), mypy in strict mode, the repository layout/rules checker
and the unit/contract test suite. It makes no network calls; tests that need real services carry the
`live` marker and are excluded.

Configuration is read from the environment or `.env` (see `.env.example`). `CARACAL_API_TOKEN` must be
at least 32 characters. Run `uv run caracal doctor` to validate configuration without contacting any
service.

Voiceprints are biometric data: enrollment is explicit and deleting a speaker removes all of its
embeddings.
