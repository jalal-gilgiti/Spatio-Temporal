"""Spatio-temporal join cardinality — native vs sampling vs truth.

Join workload (per dataset, read-only): for each of the first ``--n-windows`` frozen
fixed-extent evaluation windows B, count unordered pairs of distinct trajectories
that both overlap B and whose bounding boxes overlap each other:

    SELECT count(*) FROM t a JOIN t b
      ON a.traj && b.traj AND a.trajectory_id < b.trajectory_id
    WHERE a.traj && B AND b.traj && B

* native   : PostgreSQL/MobilityDB row estimate of the join node (EXPLAIN only). The
             MobilityDB manual states that join selectivity for temporal types is a
             default constant.
* sampling : a deterministic 1% hash sample S of trajectory ids
             (abs(hashint4(id + seed)) % 100 = 0); estimate = pairs within S / p^2.
             Five seeds; the reported estimate is their median.
* truth    : the exact count.
No learned model is involved. q-errors clamp estimates to [1, P] with P = N(N-1)/2.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import query_from_dict  # noqa: E402
from src.data_module.workload_generator import stbox_literal  # noqa: E402
from src.eval_module.accuracy import q_error  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tdrive": ("run/real_experiments/task7b/headline/workloads", "tdrive_traj"),
    "porto": ("run/real_experiments/multiseed_accuracy/workloads", "porto_traj"),
}
SEEDS = (17, 23, 47, 59, 71)
RATE = 100  # 1 in 100 trajectories


def join_sql(table: str, extra: str = "") -> str:
    return (f"SELECT count(*) FROM {table} a JOIN {table} b "
            f"ON a.traj && b.traj AND a.trajectory_id < b.trajectory_id "
            f"WHERE a.traj && %(box)s::stbox AND b.traj && %(box)s::stbox {extra}")


def find_join_rows(node: dict) -> float | None:
    if node.get("Node Type") in ("Nested Loop", "Hash Join", "Merge Join"):
        return float(node["Plan Rows"])
    for child in node.get("Plans", []):
        found = find_join_rows(child)
        if found is not None:
            return found
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/join")
    ap.add_argument("--n-windows", type=int, default=50)
    ap.add_argument("--timeout-s", type=int, default=120, help="per-statement timeout for truths")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with psycopg.connect(args.dsn) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '{args.timeout_s}s'")
            for dataset, (wdir, table) in SOURCES.items():
                cur.execute(f"SELECT count(*) FROM {table}")
                n = int(cur.fetchone()[0])
                pairs_max = n * (n - 1) / 2
                payload = json.loads((ROOT / wdir / f"{dataset}__fixed_extent.json").read_text())
                for i, q in enumerate(payload["evaluation_queries"][: args.n_windows]):
                    box = stbox_literal(query_from_dict(q).predicate)
                    params = {"box": box}
                    cur.execute("EXPLAIN (FORMAT JSON) " + join_sql(table), params)
                    native = find_join_rows(cur.fetchone()[0][0]["Plan"])
                    samples = []
                    for seed in SEEDS:
                        cond = (f"AND abs(hashint4(a.trajectory_id + {seed})) %% {RATE} = 0 "
                                f"AND abs(hashint4(b.trajectory_id + {seed})) %% {RATE} = 0")
                        cur.execute(join_sql(table, cond), params)
                        samples.append(int(cur.fetchone()[0]) * RATE * RATE)
                    started = time.perf_counter()
                    try:
                        cur.execute(join_sql(table), params)
                        truth = int(cur.fetchone()[0])
                    except psycopg.errors.QueryCanceled:
                        conn.rollback()
                        cur.execute(f"SET statement_timeout = '{args.timeout_s}s'")
                        truth = None
                    truth_ms = (time.perf_counter() - started) * 1000
                    cur.execute(f"SELECT count(*) FROM {table} WHERE traj && %(box)s::stbox", params)
                    side = int(cur.fetchone()[0])
                    sampled = float(statistics.median(samples))
                    clamp = lambda e: min(max(e, 1.0), pairs_max)  # noqa: E731
                    rows.append({
                        "dataset": dataset, "window_index": i, "rows_overlapping_window": side,
                        "true_pairs": truth, "truth_ms": truth_ms,
                        "native_join_estimate": native, "sampling_estimate_median": sampled,
                        "sampling_estimates": json.dumps(samples),
                        "native_qerror_clamped": None if truth is None else q_error(truth, clamp(native)),
                        "sampling_qerror_clamped": None if truth is None else q_error(truth, clamp(sampled)),
                    })
            conn.rollback()
    with (args.output_dir / "join_estimates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_results_json({"join": join_sql("<table>"), "windows": "first N frozen fixed-extent eval windows",
                        "sampling": f"hash sample 1/{RATE}, seeds {SEEDS}, median of 5",
                        "q_error": "clamped to [1, N(N-1)/2]", "n_windows": args.n_windows},
                       args.output_dir / "metadata.json")
    for dataset in SOURCES:
        rs = [r for r in rows if r["dataset"] == dataset and r["true_pairs"] is not None]
        if rs:
            print(dataset, len(rs), "windows; median q native",
                  round(statistics.median(r["native_qerror_clamped"] for r in rs), 2),
                  "sampling", round(statistics.median(r["sampling_qerror_clamped"] for r in rs), 2))


if __name__ == "__main__":
    main()
