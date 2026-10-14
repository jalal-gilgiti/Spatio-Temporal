"""Cardinality-stratified mixed-extent evaluation sets.

For each dataset, draw queries from the SAME mixed-extent generator and bounds that
produced its headline mixed workload, label them (read-only), and keep the first
--per-bucket queries per true-cardinality bucket (0, 1-10, 11-100, 101-1,000, >1,000)
in draw order (rejection sampling; estimates are never consulted). Drawing stops when
every bucket is full or --max-draws is reached; buckets the generator cannot reach
are reported with the count obtained.

Generators and bounds, as in the headline runs:
* T-Drive: scalability.generate_predicates("mixed_extent") over the corrected 1M pool's
  bounding box and time range (scripts/task7b_headline.py);
* Porto: workload_generator.generate_workload over the table bounds
  (scripts/multiseed_accuracy_battery.py), called in chunks with derived seeds;
* AIS: scalability.generate_predicates("mixed_extent") over the Danish-waters box and
  the table's time span (scripts/vldbj_ais_battery.py).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import fingerprint, query_dict  # noqa: E402
from src.data_module.workload_generator import generate_workload  # noqa: E402
from src.eval_module.scalability import generate_predicates, label_predicates  # noqa: E402
from src.real_experiment_steps import fetch_table_workload_bounds  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = (("0", 0, 0), ("1-10", 1, 10), ("11-100", 11, 100), ("101-1,000", 101, 1000), (">1,000", 1001, None))
TABLES = {"tdrive": "tdrive_traj", "porto": "porto_traj", "ais": "ais_traj"}
CAPS = {"tdrive": 20_000, "porto": 20_000, "ais": 10_000}
BATCH = 500


def bucket(t: int) -> str:
    for name, lo, hi in BUCKETS:
        if t >= lo and (hi is None or t <= hi):
            return name
    raise ValueError(t)


def draws(conn, dataset: str, seed: int):
    """Yield labelled queries in draw order from the dataset's mixed-extent generator."""
    table = TABLES[dataset]
    if dataset == "porto":
        bbox, times = fetch_table_workload_bounds(conn, table)
        k = 0
        while True:
            for q in generate_workload(conn, table, n_queries=BATCH, seed=seed * 1000 + k, bbox=bbox,
                                       knn_fraction=0.0, time_range=times):
                yield q
            k += 1
    if dataset == "tdrive":
        pool = pd.read_parquet(ROOT / "run/real_experiments/task7_dataset_validity/prepared/tdrive_corrected__max_1000000.parquet",
                               columns=["lon", "lat", "timestamp"])
        bbox = (float(pool.lon.min()), float(pool.lon.max()), float(pool.lat.min()), float(pool.lat.max()))
        times = (pool.timestamp.min().to_pydatetime(), pool.timestamp.max().to_pydatetime())
    else:
        _, times = fetch_table_workload_bounds(conn, table)
        bbox = (3.0, 17.0, 53.0, 59.0)
    preds = generate_predicates("mixed_extent", CAPS[dataset], seed, bbox, times, 0.45, 24.0)
    for start in range(0, len(preds), BATCH):
        labelled, _ = label_predicates(conn, table, preds[start:start + BATCH])
        yield from labelled


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--datasets", nargs="+", default=["tdrive", "porto", "ais"])
    ap.add_argument("--per-bucket", type=int, default=60)
    ap.add_argument("--seed", type=int, default=61001)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/task1e/workloads")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with psycopg.connect(args.dsn) as conn:
        conn.read_only = True
        for dataset in args.datasets:
            path = args.output_dir / f"{dataset}__mixed_stratified.json"
            if path.exists():
                print(f"{dataset}: exists, skipped")
                continue
            kept = {name: [] for name, *_ in BUCKETS}
            n_draws, started = 0, time.perf_counter()
            for q in draws(conn, dataset, args.seed):
                n_draws += 1
                b = bucket(q.true_cardinality)
                if len(kept[b]) < args.per_bucket:
                    kept[b].append(q)
                if all(len(v) >= args.per_bucket for v in kept.values()) or n_draws >= CAPS[dataset]:
                    break
            rows = [dict(query_dict(q), bucket=b, query_id=f"{dataset}:stratified:{b}:{i:02d}")
                    for b, qs in kept.items() for i, q in enumerate(qs)]
            payload = {"dataset": dataset, "table": TABLES[dataset], "workload": "mixed_extent_stratified",
                       "seed": args.seed, "per_bucket_target": args.per_bucket, "draws": n_draws,
                       "max_draws": CAPS[dataset], "label_seconds": time.perf_counter() - started,
                       "per_bucket_obtained": {b: len(v) for b, v in kept.items()},
                       "selection_rule": "first queries per bucket in draw order; estimates never consulted",
                       "evaluation_workload_id": fingerprint(rows), "evaluation_queries": rows}
            write_results_json(payload, path)
            print(dataset, payload["per_bucket_obtained"], "draws", n_draws, f"{payload['label_seconds']:.0f}s")
        conn.rollback()


if __name__ == "__main__":
    main()
