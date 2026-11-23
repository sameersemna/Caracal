#!/bin/sh
# Installs restart-always.conf as a drop-in for every long-running service that was killed on 2026-11-21.
#   system units:  sudo sh deploy/host-hardening/install.sh system
#   user units:         sh deploy/host-hardening/install.sh user      (as sameer, no sudo)
# Adding a drop-in + daemon-reload does not restart running services; units that are down get started.
set -eu
CONF="$(cd "$(dirname "$0")" && pwd)/restart-always.conf"
SYSTEM_UNITS="qdrant ocr-server mcp-hybrid-vision mcp-text2image taftish tabby"   # caracal-*: own units; whisper: deploy/whisper
USER_UNITS="kerstin kerstin-tts sunny-brain unlimited-ocr hermes-gateway hermes-dashboard"

case "${1:-}" in
  system) [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
          DIR=/etc/systemd/system; SC="systemctl"; UNITS=$SYSTEM_UNITS ;;
  user)   [ "$(id -u)" != 0 ] || { echo "run WITHOUT sudo"; exit 1; }
          DIR=$HOME/.config/systemd/user; SC="systemctl --user"; UNITS=$USER_UNITS ;;
  *) echo "usage: $0 system|user"; exit 2 ;;
esac

for u in $UNITS; do
  $SC cat "$u.service" >/dev/null 2>&1 || { echo "skip $u (no such unit)"; continue; }
  install -D -m 644 "$CONF" "$DIR/$u.service.d/restart-always.conf"
done
$SC daemon-reload
for u in $UNITS; do
  $SC cat "$u.service" >/dev/null 2>&1 || continue
  $SC is-active --quiet "$u" || { echo "starting $u"; $SC start "$u" || echo "  !! $u failed to start"; }
  printf '%-20s %-8s Restart=%s\n' "$u" "$($SC is-active "$u")" "$($SC show "$u" -p Restart --value)"
done
