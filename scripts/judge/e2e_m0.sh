#!/usr/bin/env bash
# Judge M0 demo: real serve + worker processes, throwaway schema/prefix/data dir, kill -9 recovery.
set -uo pipefail
cd /home/sameer/Public/Shared/Work/Services/MCP/Caracal
S="${JUDGE_SCRATCH:-$(mktemp -d)}"
set -a; . ./.env; set +a
T=caracal_t_$(openssl rand -hex 4)
export CARACAL_DB_SCHEMA=$T REDIS_PREFIX="$T:" CARACAL_DATA_DIR=$S/e2e-data/$T CARACAL_PORT=11498 CARACAL_PUBLIC_BASE_URL=http://127.0.0.1:11498 CARACAL_LOG_LEVEL=INFO
PG=${DATABASE_URL/+psycopg/}; TOK=$CARACAL_API_TOKEN; B=http://127.0.0.1:11498; A="Authorization: Bearer $TOK"
LOG=$S/e2e-$T; mkdir -p $LOG
cleanup() {
  kill $(jobs -p) 2>/dev/null; sleep 1
  psql "$PG" -qAtc "DROP SCHEMA IF EXISTS \"$T\" CASCADE" >/dev/null
  uv run -q python -I -c "
import redis,os; r=redis.Redis.from_url(os.environ['REDIS_URL']); ks=list(r.scan_iter(match='$T:*',count=1000))
print('redis keys under prefix at end:', len(ks)); [r.delete(k) for k in ks]
print('caracal keys outside prefix:', [k for k in r.scan_iter(match='caracal*',count=1000) if not k.decode().startswith('$T:')][:5])"
  rm -rf $S/e2e-data/$T
}
trap cleanup EXIT
psql "$PG" -qAtc "CREATE SCHEMA \"$T\"" && .venv/bin/caracal migrate >$LOG/migrate.log 2>&1 && echo "migrate: ok"
.venv/bin/caracal serve >$LOG/serve.log 2>&1 &
for i in $(seq 1 30); do curl -s $B/healthz >/dev/null && break; sleep 0.5; done
echo "healthz: $(curl -s $B/healthz)"
echo "no auth: $(curl -s -o /dev/null -w '%{http_code}' $B/v1/jobs)"
echo "== demo 1: real 'caracal worker' (stub stages)"
WID=w-real .venv/bin/caracal worker >$LOG/worker-real.log 2>&1 & WR=$!
J=$(curl -s -H "$A" -H 'content-type: application/json' -d '{"source":{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}}' $B/v1/jobs); echo "create: $J"
JID=$(echo "$J" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])')
timeout 30 curl -sN -H "$A" $B/v1/jobs/$JID/events > $LOG/sse1.txt; echo "sse events: $(grep -c '^event:' $LOG/sse1.txt) last: $(grep '^event:' $LOG/sse1.txt | tail -1)"
curl -s -H "$A" $B/v1/jobs/$JID | python3 -c 'import json,sys;d=json.load(sys.stdin);print("status:",d["status"],"| stages:",{s["status"] for s in d["stages"]},"| n:",len(d["stages"]))'
kill -TERM $WR; wait $WR 2>/dev/null; echo "real worker SIGTERM exit: $?"
echo "== demo 2: kill -9 mid-job, second worker resumes"
STAGE_S=1 WID=w-A .venv/bin/python scripts/judge/slow_worker.py >$LOG/wA.log 2>&1 & WA=$!
sleep 3
J=$(curl -s -H "$A" -H 'content-type: application/json' -d '{"source":{"url":"https://example.com/talk.mp3"},"options":{"depth":"deep"}}' $B/v1/jobs); JID=$(echo "$J" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])'); echo "create: $J"
sleep 5.5; kill -9 $WA; sleep 0.5; echo "killed A (-9) pid $WA; still alive: $(kill -0 $WA 2>/dev/null && echo YES || echo no)"
curl -s -H "$A" $B/v1/jobs/$JID | python3 -c 'import json,sys;d=json.load(sys.stdin);print("after kill:",d["status"],[s["name"] for s in d["stages"] if s["status"]=="done"])'
STAGE_S=0.2 WID=w-B .venv/bin/python scripts/judge/slow_worker.py >$LOG/wB.log 2>&1 & WB=$!
for i in $(seq 1 40); do st=$(curl -s -H "$A" $B/v1/jobs/$JID | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])'); [ "$st" = done ] && break; sleep 0.5; done
echo "final: $st"
psql "$PG" -qAtc "select name, attempts from \"$T\".job_stages where job_id='$JID' order by started_at" | tr '\n' ' '; echo
python3 - "$CARACAL_DATA_DIR/jobs/$JID/stages" <<'P'
import json,sys,os,collections
d=sys.argv[1]; pids=collections.Counter(json.load(open(os.path.join(d,f)))["output"]["pid"] for f in os.listdir(d)); print("checkpoints by pid:",dict(pids),"files:",len(os.listdir(d)))
P
echo "== attacks"
echo "dedup same url: $(curl -s -H "$A" -H 'content-type: application/json' -d '{"source":{"url":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}}' $B/v1/jobs)"
echo "bad field echo: $(curl -s -H "$A" -H 'content-type: application/json' -d '{"source":{"url":"ftp://SECRET-ECHO-PROBE"}}' $B/v1/jobs | grep -c SECRET-ECHO-PROBE) (0 = not echoed)"
echo "extra field: $(curl -s -o /dev/null -w '%{http_code}' -H "$A" -H 'content-type: application/json' -d '{"source":{"url":"https://a.b/x"},"evil":1}' $B/v1/jobs)"
echo "both sources: $(curl -s -o /dev/null -w '%{http_code}' -H "$A" -H 'content-type: application/json' -d '{"source":{"url":"https://a.b/x","upload_id":"00000000-0000-0000-0000-000000000000"}}' $B/v1/jobs)"
echo "70KB declared: $(head -c 70000 /dev/zero | tr '\0' a | curl -s -o /dev/null -w '%{http_code}' -H "$A" -H 'content-type: application/json' --data-binary @- $B/v1/jobs)"
python3 - <<P
import socket,time
s=socket.create_connection(("127.0.0.1",11498)); t=time.time()
s.sendall(b"POST /v1/jobs HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer $TOK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n")
sent=0; chunk=b"a"*65536
try:
    while sent < 50*1024*1024 and time.time()-t<20:
        s.sendall(b"%x\r\n"%len(chunk)+chunk+b"\r\n"); sent+=len(chunk)
    s.sendall(b"0\r\n\r\n"); s.settimeout(10); resp=s.recv(200).split(b"\r\n")[0]
except Exception as e: resp=repr(e).encode()
print("chunked no-length: server accepted", sent//1048576, "MiB before answering:", resp.decode(errors="replace"))
P
python3 - <<P
import socket,time
s=socket.create_connection(("127.0.0.1",11498)); t=time.time()
s.sendall(b"POST /v1/jobs HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n")
sent=0; chunk=b"a"*65536
try:
    while sent < 30*1024*1024 and time.time()-t<20:
        s.sendall(b"%x\r\n"%len(chunk)+chunk+b"\r\n"); sent+=len(chunk)
    s.sendall(b"0\r\n\r\n"); s.settimeout(10); resp=s.recv(200).split(b"\r\n")[0]
except Exception as e: resp=repr(e).encode()
print("UNAUTH chunked: server accepted", sent//1048576, "MiB before answering:", resp.decode(errors="replace"))
P
echo "unauth valid json: $(curl -s -o /dev/null -w '%{http_code}' -H 'content-type: application/json' -d '{\"source\":{\"url\":\"https://a.b/x\"}}' $B/v1/jobs)"
echo "unauth invalid json: $(printf 'x%.0s' $(seq 1 10) | curl -s -o /dev/null -w '%{http_code}' -H 'Transfer-Encoding: chunked' -H 'content-type: application/json' --data-binary @- $B/v1/jobs)"
echo "path traversal id: $(curl -s -o /dev/null -w '%{http_code}' -H "$A" "$B/v1/jobs/..%2F..%2Fetc")"
echo "500 leak check: $(grep -c Traceback $LOG/serve.log) tracebacks in serve log (expected 0)"
for i in $(seq 1 32); do curl -s -o /dev/null -w '%{http_code} ' -H "$A" -H 'content-type: application/json' -d "{\"source\":{\"url\":\"https://a.b/$i.mp3\"}}" $B/v1/jobs; done; echo "<- 32 creates (limit 30)"
echo "== secrets in logs: $(cat $LOG/*.log | grep -cF "$TOK") token hits, $(cat $LOG/*.log | grep -cF "${REDIS_URL#*:*:}") redis-pw hits"
