# systemd deployment

Two units, two processes:

| Unit | Command | Notes |
|---|---|---|
| `caracal-api.service` | `.venv/bin/caracal serve` | REST, MCP HTTP and `/metrics`; no GPU needed |
| `caracal-worker.service` | `.venv/bin/caracal worker` | pipeline + daily retention; **needs the GPU**, so it deliberately has no `PrivateDevices=` |

The unit files use the absolute repo path `/home/sameer/Public/Shared/Work/Services/MCP/Caracal`.
If the repository moves, edit `WorkingDirectory`, `EnvironmentFile`, `ExecStart` and
`ReadWritePaths` in both units.

Both units run as `User=sameer`, read `<repo>/.env` (mode 600), restart on failure after 5 s and
allow 60 s for a graceful worker shutdown. `ProtectSystem=strict` makes everything read-only except
`<repo>/data`; `ProtectHome=read-only` keeps the repository readable. `PrivateTmp=true` gives ffmpeg
a private writable `/tmp`.

## Install (owner, with sudo — the worker never runs these)

```bash
cd /home/sameer/Public/Shared/Work/Services/MCP/Caracal
sudo cp deploy/caracal-api.service deploy/caracal-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now caracal-api.service caracal-worker.service
```

Before enabling, make sure the one-shot setup has run as `sameer` (not root):

```bash
.venv/bin/caracal migrate
.venv/bin/caracal doctor
```

## Logs and status

```bash
systemctl status caracal-api.service caracal-worker.service
journalctl -u caracal-api.service -f
journalctl -u caracal-worker.service -f
journalctl -u caracal-worker.service --since "1 hour ago"
```

Restart after changing `.env`:

```bash
sudo systemctl restart caracal-api.service caracal-worker.service
```

Run retention immediately instead of waiting for the daily pass:

```bash
.venv/bin/caracal retention --dry-run   # report only
.venv/bin/caracal retention             # delete expired media
```
