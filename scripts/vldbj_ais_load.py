"""Load the prepared AIS trip pool into a new table ``ais_traj``.

Creates only the new table (existing tables are untouched), loads the trips in one
transaction with the same loader as T-Drive/Porto, builds a GiST index, sets the
``traj`` statistics target to 1000 (the primary native setting) and ANALYZEs the
new table. Writes a small metadata JSON next to the pool.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval_module.scalability import bulk_load_trajectories  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = "ais_traj"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--pool", type=Path, default=ROOT / "data/ais_prepared/ais_pool.parquet")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/ais")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(args.pool)
    meta = {"table": TABLE, "pool": str(args.pool), "pool_points": len(pool),
            "pool_trajectories": int(pool.trajectory_id.nunique())}
    with psycopg.connect(args.dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (TABLE,))
        if cur.fetchone()[0] is not None:
            raise SystemExit(f"{TABLE} already exists; refusing to overwrite")
        cur.execute(f"CREATE TABLE {TABLE} (trajectory_id int, traj tgeompoint)")
        started = time.perf_counter()
        n = bulk_load_trajectories(conn, TABLE, pool)
        meta["load_seconds"] = time.perf_counter() - started
        if n != meta["pool_trajectories"]:
            raise RuntimeError(f"loaded {n} trajectories, pool has {meta['pool_trajectories']}")
        started = time.perf_counter()
        cur.execute(f"CREATE INDEX {TABLE}_gist ON {TABLE} USING GIST (traj)")
        meta["index_seconds"] = time.perf_counter() - started
        cur.execute(f"ALTER TABLE {TABLE} ALTER COLUMN traj SET STATISTICS 1000")
        started = time.perf_counter()
        cur.execute(f"ANALYZE {TABLE}")
        meta["analyze_seconds"] = time.perf_counter() - started
        cur.execute(f"SELECT count(*), sum(numInstants(traj)), pg_relation_size('{TABLE}'), "
                    f"pg_relation_size('{TABLE}_gist'), pg_total_relation_size('{TABLE}') FROM {TABLE}")
        rows, points, tbytes, ibytes, total = cur.fetchone()
        meta.update({"rows": rows, "points": int(points), "table_bytes": tbytes, "index_bytes": ibytes,
                     "total_bytes": total, "statistics_target": 1000,
                     "created_utc": datetime.now(timezone.utc).isoformat()})
    (args.output_dir / "load_metadata.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
