#!/bin/sh
# Restart whisper-server if it is down or stops answering /health (a hung process that is still
# "active" is invisible to Restart=always). /health answers in <1 ms even during a long
# transcription (measured 2026-11-23), so "busy" is never mistaken for "dead".
UNIT=whisper-server.service
URL=http://127.0.0.1:8768/health

if ! systemctl is-active --quiet "$UNIT"; then
    state=$(systemctl is-active "$UNIT")
    case "$state" in activating|reloading) exit 0 ;; esac   # restart already in progress
    echo "watchdog: $UNIT is $state, starting it"
    exec systemctl start "$UNIT"
fi

# Grace period after a (re)start: model load takes a few seconds.
since=$(systemctl show -p ActiveEnterTimestampMonotonic --value "$UNIT")
now=$(awk '{printf "%d", $1 * 1000000}' /proc/uptime)
[ $(( (now - since) / 1000000 )) -lt 60 ] && exit 0

for attempt in 1 2 3; do
    curl -fsS -m 10 -o /dev/null "$URL" && exit 0
    sleep 10
done
echo "watchdog: $URL failed 3 times in a row, restarting $UNIT"
exec systemctl restart "$UNIT"
