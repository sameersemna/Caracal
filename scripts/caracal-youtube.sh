#!/usr/bin/env bash
# caracal-youtube.sh — analyse a YouTube video with Caracal and save the results.
#
#   ./youtube/<id>.srt     subtitles in the original language
#   ./youtube/<id>.md      the full report (summary, context, speakers, chapters, claims + verification, …)
#   ./youtube/<id>.en.srt  English subtitles — only when the audio is not English
#
# Usage:
#   scripts/caracal-youtube.sh <youtube-url> [more urls…]
#
# Configuration (environment):
#   CARACAL_URL        API base URL                       (default http://127.0.0.1:11404)
#   CARACAL_API_TOKEN  bearer token                       (default: read from CARACAL_ENV_FILE)
#   CARACAL_ENV_FILE   .env to read the token from        (default: <repo>/.env next to this script)
#   OUT_DIR            output directory                   (default ./youtube)
#   DEPTH              quick | standard | deep            (default standard)
#   TIMEOUT_S          max seconds to wait per video      (default 1800)
#
# Requires: curl, jq.
set -euo pipefail

CARACAL_URL="${CARACAL_URL:-http://127.0.0.1:11404}"
CARACAL_URL="${CARACAL_URL%/}"
OUT_DIR="${OUT_DIR:-./youtube}"
DEPTH="${DEPTH:-standard}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
POLL_S=3

_progress=0   # 1 while a \r progress line is on screen
die()  { [ "$_progress" = 1 ] && printf '\n' >&2; printf 'error: %s\n' "$*" >&2; exit 1; }
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

command -v curl >/dev/null || die "curl is required"
command -v jq   >/dev/null || die "jq is required"
[ $# -ge 1 ] || die "usage: $0 <youtube-url> [more urls…]"
case "$DEPTH" in quick|standard|deep) ;; *) die "DEPTH must be quick, standard or deep" ;; esac

# --- token -----------------------------------------------------------------------------------
if [ -z "${CARACAL_API_TOKEN:-}" ]; then
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  env_file="${CARACAL_ENV_FILE:-$script_dir/../.env}"
  [ -r "$env_file" ] || die "CARACAL_API_TOKEN is not set and $env_file is not readable"
  CARACAL_API_TOKEN="$(grep -E '^CARACAL_API_TOKEN=' "$env_file" | head -1 | cut -d= -f2- | sed -E 's/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/')"
  [ -n "$CARACAL_API_TOKEN" ] || die "CARACAL_API_TOKEN not found in $env_file"
fi

# --- HTTP helpers ----------------------------------------------------------------------------
# api METHOD PATH [JSON-BODY] — prints the body; fails on HTTP >= 400 with the API's error message.
api() {
  local method="$1" path="$2" body="${3:-}" tmp code
  tmp="$(mktemp)"
  if [ -n "$body" ]; then
    code="$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" \
      -H "Authorization: Bearer $CARACAL_API_TOKEN" -H 'Content-Type: application/json' \
      --data "$body" "$CARACAL_URL$path")" || { rm -f "$tmp"; die "cannot reach $CARACAL_URL"; }
  else
    code="$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" \
      -H "Authorization: Bearer $CARACAL_API_TOKEN" "$CARACAL_URL$path")" || { rm -f "$tmp"; die "cannot reach $CARACAL_URL"; }
  fi
  if [ "$code" -ge 400 ]; then
    local msg; msg="$(jq -r '.error | "\(.code): \(.message)"' "$tmp" 2>/dev/null || head -c 300 "$tmp")"
    rm -f "$tmp"; die "$method $path → HTTP $code ($msg)"
  fi
  cat "$tmp"; rm -f "$tmp"
}

# download PATH FILE — writes the response body to FILE atomically.
download() {
  local path="$1" file="$2" code
  code="$(curl -sS -o "$file.part" -w '%{http_code}' -H "Authorization: Bearer $CARACAL_API_TOKEN" "$CARACAL_URL$path")" \
    || { rm -f "$file.part"; die "cannot reach $CARACAL_URL"; }
  [ "$code" = 200 ] || { rm -f "$file.part"; die "GET $path → HTTP $code"; }
  mv -f "$file.part" "$file"
}

# youtube_id URL — the 11-character video id from the common URL shapes.
youtube_id() {
  local url="$1" id=""
  if [[ "$url" =~ [?\&]v=([A-Za-z0-9_-]{11}) ]]; then id="${BASH_REMATCH[1]}"
  elif [[ "$url" =~ youtu\.be/([A-Za-z0-9_-]{11}) ]]; then id="${BASH_REMATCH[1]}"
  elif [[ "$url" =~ youtube\.com/(shorts|embed|live|v)/([A-Za-z0-9_-]{11}) ]]; then id="${BASH_REMATCH[2]}"
  fi
  [ -n "$id" ] || return 1
  printf '%s' "$id"
}

# wait_for JOB_ID JQ-CONDITION — polls GET /v1/jobs/{id} until the condition (on the status JSON) is true.
wait_for() {
  local job="$1" cond="$2" start status pct stage
  start=$(date +%s)
  while :; do
    status="$(api GET "/v1/jobs/$job")"
    if jq -e "$cond" >/dev/null <<<"$status"; then printf '%s' "$status"; return 0; fi
    if jq -e '.status == "failed" or .status == "cancelled"' >/dev/null <<<"$status"; then
      die "job $job $(jq -r '.status' <<<"$status"): $(jq -r '.error // {} | "\(.code // "unknown")\(if .stage then " at " + .stage else "" end): \(.message // "no details")"' <<<"$status")"
    fi
    pct="$(jq -r '.progress.pct // 0' <<<"$status")"; stage="$(jq -r '.progress.stage // "-"' <<<"$status")"
    printf '\r  %s  %5.1f%%  %-12s' "$(jq -r '.status' <<<"$status")" "$pct" "$stage" >&2; _progress=1
    [ $(( $(date +%s) - start )) -lt "$TIMEOUT_S" ] || { echo >&2; die "timed out after ${TIMEOUT_S}s waiting for job $job"; }
    sleep "$POLL_S"
  done
}

# --- main ------------------------------------------------------------------------------------
mkdir -p "$OUT_DIR"

for url in "$@"; do
  vid="$(youtube_id "$url")" || die "not a recognisable YouTube URL: $url"
  log "$vid: submitting $url"

  body="$(jq -nc --arg url "$url" --arg depth "$DEPTH" \
    '{source: {url: $url}, options: {depth: $depth, target_langs: ["en"]}}')"
  created="$(api POST /v1/jobs "$body")"
  job="$(jq -r '.job_id' <<<"$created")"
  [ "$(jq -r '.deduplicated // false' <<<"$created")" = true ] && log "$vid: already analysed recently — reusing job $job"
  log "$vid: job $job"

  status="$(wait_for "$job" '.status == "done"')"; echo >&2

  report="$(api GET "/v1/jobs/$job/report")"
  lang="$(jq -r '[.languages[]?] | max_by(.share) | .code // "und"' <<<"$report")"
  title="$(jq -r '.source.title // "?"' <<<"$report")"
  log "$vid: done — \"$title\" (language: $lang)"

  download "/v1/jobs/$job/subtitles/source.srt" "$OUT_DIR/$vid.srt"
  download "/v1/jobs/$job/report?format=md" "$OUT_DIR/$vid.md"
  log "$vid: wrote $OUT_DIR/$vid.srt and $OUT_DIR/$vid.md"

  # English translation only when the audio is not English (music / unknown language has nothing to translate).
  if [ "$lang" != en ] && [ "$lang" != und ]; then
    if ! jq -e '.links.subtitles["en.srt"]' >/dev/null <<<"$status"; then
      # e.g. a reused job that was created without an English target — request it now.
      log "$vid: requesting English translation"
      api POST "/v1/jobs/$job/translate" '{"target_langs": ["en"]}' >/dev/null
      wait_for "$job" '.links.subtitles["en.srt"] != null' >/dev/null; echo >&2
    fi
    download "/v1/jobs/$job/subtitles/en.srt" "$OUT_DIR/$vid.en.srt"
    log "$vid: wrote $OUT_DIR/$vid.en.srt"
  else
    rm -f "$OUT_DIR/$vid.en.srt"   # stale file from an earlier run, if any
  fi
done
