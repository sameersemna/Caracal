"""Judge harness: a real worker process whose stub stages each take STAGE_S seconds."""
import asyncio, os, signal, sys
sys.path.insert(0, os.getcwd())
import redis.asyncio as aioredis
from caracal.config import get_settings
from caracal.db.engine import make_engine
from caracal.db.repo import PostgresJobRepo, PostgresUploadRepo, PostgresReportRepo, PostgresSegmentRepo
from caracal.pipeline.stages import Stage, StageResult
from caracal.worker import Worker, WorkerIntervals
from caracal import worker as wmod
STAGE_S = float(os.environ.get("STAGE_S", "1"))
def stub(name):
    async def fn(ctx):
        await asyncio.sleep(STAGE_S); return StageResult("done", output={"stage": name, "pid": os.getpid()})
    return fn
async def main():
    s = get_settings(); r = aioredis.from_url(s.REDIS_URL.get_secret_value()); eng = make_engine(s)
    RepoBundle = wmod.RepoBundle if hasattr(wmod, "RepoBundle") else __import__("caracal.cli", fromlist=["RepoBundle"]).RepoBundle
    repos = RepoBundle(jobs=PostgresJobRepo(eng), uploads=PostgresUploadRepo(eng), reports=PostgresReportRepo(eng), segments=PostgresSegmentRepo(eng))
    iv = WorkerIntervals(heartbeat_s=1, touch_s=1, reclaim_s=1, reclaim_min_idle_ms=3000, block_ms=500, max_deliveries=3, shutdown_grace_s=2)
    w = Worker(s, r, repos, registry={st: stub(st.value) for st in Stage}, worker_id=os.environ["WID"], intervals=iv)
    stop = asyncio.Event(); asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)
    await w.run(stop)
asyncio.run(main())
