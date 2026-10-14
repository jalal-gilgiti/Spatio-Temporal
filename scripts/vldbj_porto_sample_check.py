"""Document which Porto sample porto_traj holds (read-only).

Records porto_traj's trajectory-id range, row count and time span, and its overlap with
the 1M Porto scalability pool (run/real_experiments/scalability/prepared/porto__max_1000000.parquet).
Trip ids cannot be compared directly (both number their trips from 0 independently), so
trips are matched by content: (start epoch second, first longitude, first latitude; the number
of points is NOT used, because MobilityDB normalises sequences on input and can drop
redundant points). Naive pool timestamps are interpreted in the database session time zone, as
when the pool was loaded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "run/real_experiments/scalability/prepared/porto__max_1000000.parquet"
OUT = ROOT / "run/vldbj_rev/porto_sample/porto_sample.json"


def main() -> None:
    with psycopg.connect("postgresql://postgres@localhost:5434/mobilitydb") as conn:
        conn.read_only = True
        cur = conn.cursor()
        cur.execute("SHOW TimeZone")
        tz = ZoneInfo(cur.fetchone()[0])
        cur.execute("SELECT min(trajectory_id), max(trajectory_id), count(*), count(DISTINCT trajectory_id), "
                    "min(startTimestamp(traj)), max(endTimestamp(traj)) FROM porto_traj")
        id_min, id_max, n, n_distinct, t0, t1 = cur.fetchone()
        cur.execute("SELECT trajectory_id, round(extract(epoch from startTimestamp(traj)))::bigint, "
                    "round(ST_X(startValue(traj))::numeric, 6), round(ST_Y(startValue(traj))::numeric, 6), "
                    "numInstants(traj) FROM porto_traj")
        table_sig = pd.DataFrame(cur.fetchall(), columns=["table_id", "t", "x", "y", "n"]).astype(
            {"table_id": "int64", "t": "int64", "x": float, "y": float, "n": "int64"})
        conn.rollback()
    # positive control: porto_traj trips 0 and 1 were located in the pool by hand
    # (pool ids 461221 and 293480, same first point and wall-clock start time)
    control_expect = {0: 461221, 1: 293480}
    pool = pd.read_parquet(POOL).sort_values(["trajectory_id", "timestamp"])
    first = pool.groupby("trajectory_id").head(1).set_index("trajectory_id")
    counts = pool.groupby("trajectory_id").size()
    ts = pd.to_datetime(first["timestamp"])
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts
    epoch_s = ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")
    pool_sig = pd.DataFrame({"t": epoch_s,
                             "x": first["lon"].round(6).astype(float), "y": first["lat"].round(6).astype(float),
                             "n": counts.loc[first.index].astype("int64")}, index=first.index)
    pool_sig["pool_id"] = pool_sig.index
    pool_sig = pool_sig.drop(columns="n").drop_duplicates(["t", "x", "y"])
    merged = table_sig.drop(columns="n").merge(pool_sig, on=["t", "x", "y"], how="left")
    matched = merged["pool_id"].notna()
    ctrl = merged[merged["table_id"].isin(control_expect)]
    ctrl_ok = all(int(r.pool_id) == control_expect[int(r.table_id)] for r in ctrl.itertuples() if pd.notna(r.pool_id))
    ctrl_found = int(ctrl["pool_id"].notna().sum())
    prefix = (merged.loc[matched, "pool_id"] < 100_000)
    out = {
        "porto_traj": {"trajectory_id_min": id_min, "trajectory_id_max": id_max, "rows": n,
                       "distinct_ids": n_distinct, "time_min": t0.isoformat(), "time_max": t1.isoformat()},
        "pool": {"path": str(POOL.relative_to(ROOT)), "trajectories": int(pool_sig.shape[0]),
                 "time_min": str(pool["timestamp"].min()), "time_max": str(pool["timestamp"].max())},
        "content_match": {
            "signature": "start epoch second, first lon/lat (6 dp)",
            "table_trips_found_in_pool": int(matched.sum()),
            "share_found_in_pool": float(matched.mean()),
            "found_with_pool_id_below_100000": int(prefix.sum()),
            "matched_with_same_trajectory_id": int((merged.loc[matched, "pool_id"] == merged.loc[matched, "table_id"]).sum()),
        },
        "positive_control": {
            "description": "porto_traj ids 0 and 1 located in the pool by hand (pool ids 461221, 293480)",
            "found": ctrl_found, "expected": len(control_expect), "pool_ids_correct": bool(ctrl_ok),
        },
        "conclusion": None,
    }
    available = json.loads(POOL.with_suffix(".json").read_text())["available_raw_trajectory_count"]
    out["independence_check"] = {
        "available_csv_trips": available,
        "expected_share_in_pool_if_independent": 1_000_000 / available,
        "expected_in_pool_prefix_100k_if_independent": 100_000 * 100_000 / available,
    }
    share = out["content_match"]["share_found_in_pool"]
    exp = out["independence_check"]
    out["conclusion"] = ("porto_traj is not the 100k prefix of the 1M pool. Its overlap with the pool "
                         f"({share:.1%}; {out['content_match']['found_with_pool_id_below_100000']:,} within the pool's "
                         f"first 100k ids) matches independent sampling from the same CSV "
                         f"({exp['expected_share_in_pool_if_independent']:.1%}; "
                         f"{exp['expected_in_pool_prefix_100k_if_independent']:,.0f} expected)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(out, OUT)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
