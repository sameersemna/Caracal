#!/bin/sh
# Run as root:  sudo sh deploy/whisper/install.sh
# Causes ~3 s of whisper downtime (one restart to switch it to its own uid).
set -eu
cd "$(dirname "$0")"
install -D -m 644 override.conf           /etc/systemd/system/whisper-server.service.d/override.conf
install    -m 755 whisper-watchdog.sh     /usr/local/sbin/whisper-watchdog
install    -m 644 whisper-watchdog.service whisper-watchdog.timer /etc/systemd/system/
systemctl daemon-reload
systemctl restart whisper-server.service
systemctl enable --now whisper-watchdog.timer
sleep 5
systemctl --no-pager --lines=0 status whisper-server.service
curl -fsS -m 5 http://127.0.0.1:8768/health && echo "  <- whisper healthy"
