"""Sensitivity of MobilityDB's native estimate to the statistics target.

The originals differ: tdrive_traj.traj has statistics target 1000, porto_traj.traj
the default 100. This script copies both tables into a scratch schema
``vldbj_stats`` (no indexes), and for each copy ANALYZEs at target 100 and at
target 1000 in turn, recording Native-BBox and Native-Served row estimates for all
600 frozen accuracy queries per dataset after each ANALYZE.

* The run at the original target (T-Drive 1000, Porto 100) measures pure ANALYZE
  resampling noise: same target, new random sample.
* The run at the other target measures the effect of the target itself.

Q-error is clamped to [1, N]. Each copy-vs-original comparison is a paired
bootstrap ratio of medians, using the replayed resamples of vldbj_task1_table.py
(10,000, seed 20260911). The original tables are never ANALYZEd: every DDL and
ANALYZE statement names ``vldbj_stats`` explicitly. The schema is dropped at the end.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_multiseed_accuracy import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, bootstrap_median  # noqa: E402
from scripts.multiseed_accuracy_battery import query_from_dict  # noqa: E402
from scripts.vldbj_native_reestimate import FORMS  # noqa: E402
from scripts.vldbj_task1_table import qerrors, resample_indices  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "vldbj_stats"
SOURCES = {
    "tdrive": ("run/real_experiments/task7b/headline/workloads", "tdrive_traj", 1000),
    "porto": ("run/real_experiments/multiseed_accuracy/workloads", "porto_traj", 100),
}
WORKLOADS = ("fixed_extent", "mixed_extent")
TARGETS = (100, 1000)
USED_FORMS = ("native_bbox", "native_served")


def explain_rows(cur, table_qualified: str, where: str) -> float:
    cur.execute(f"EXPLAIN (FORMAT JSON) SELECT * FROM {table_qualified} t WHERE " + where)
    return float(cur.fetchone()[0][0]["Plan"]["Plan Rows"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/stats_target")
    ap.add_argument("--keep-schema", action="store_true")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = list(csv.DictReader((ROOT / "run/vldbj_rev/native/explain_estimates.csv").open()))
    est_rows, analyze_meta = [], {}
    with psycopg.connect(args.dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
        try:
            for dataset, (wdir, table, orig_target) in SOURCES.items():
                copy = f"{SCHEMA}.{table}"
                cur.execute(f"DROP TABLE IF EXISTS {copy}")
                cur.execute(f"CREATE TABLE {copy} AS SELECT * FROM public.{table}")
                cur.execute(f"SELECT count(*) FROM {copy}")
                n_copy = int(cur.fetchone()[0])
                queries = {w: [query_from_dict(q) for q in
                               json.loads((ROOT / wdir / f"{dataset}__{w}.json").read_text())["evaluation_queries"]]
                           for w in WORKLOADS}
                for target in TARGETS:
                    cur.execute(f"ALTER TABLE {copy} ALTER COLUMN traj SET STATISTICS {target}")
                    cur.execute(f"ANALYZE {copy}")
                    cur.execute("SELECT c.reltuples, array_length(s.stanumbers1, 1) FROM pg_class c "
                                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                                "LEFT JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'traj' "
                                "LEFT JOIN pg_statistic s ON s.starelid = c.oid AND s.staattnum = a.attnum "
                                "WHERE n.nspname = %s AND c.relname = %s", (SCHEMA, table))
                    reltuples, hist = cur.fetchone()
                    analyze_meta[f"{dataset}@{target}"] = {"copy_rows": n_copy, "reltuples": reltuples,
                                                           "stanumbers1_length": hist,
                                                           "role": "resampling control" if target == orig_target
                                                           else "target change"}
                    for workload, qs in queries.items():
                        for i, q in enumerate(qs):
                            row = {"dataset": dataset, "workload": workload, "target": target,
                                   "query_index": i, "true_cardinality": q.true_cardinality}
                            for form in USED_FORMS:
                                row[form] = explain_rows(cur, copy, FORMS[form](q.predicate))
                            est_rows.append(row)
        finally:
            if not args.keep_schema:
                cur.execute(f"DROP SCHEMA {SCHEMA} CASCADE")
    with (args.output_dir / "estimates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(est_rows[0])); w.writeheader(); w.writerows(est_rows)

    # clamped q-error and paired bootstrap against the original-table estimates
    summary = []
    for dataset, (_w, table, orig_target) in SOURCES.items():
        n = 100_000.0
        for workload in WORKLOADS:
            orig = sorted((r for r in original if r["dataset"] == dataset and r["workload"] == workload),
                          key=lambda r: int(r["query_index"]))
            truth = np.array([int(r["true_cardinality"]) for r in orig])
            idx = resample_indices(dataset, len(truth))[workload]
            for form in USED_FORMS:
                base = qerrors(truth, np.array([[float(r[form]) for r in orig]]), "planner", n)
                base_boot = bootstrap_median(base, idx)
                for target in TARGETS:
                    new = sorted((r for r in est_rows if r["dataset"] == dataset and r["workload"] == workload
                                  and r["target"] == target), key=lambda r: r["query_index"])
                    if [r["true_cardinality"] for r in new] != truth.tolist():
                        raise RuntimeError("copy estimates not aligned with the original workload")
                    err = qerrors(truth, np.array([[float(r[form]) for r in new]]), "planner", n)
                    boot = bootstrap_median(err, idx)
                    ratio = boot / base_boot
                    lo, hi = np.percentile(ratio, [2.5, 97.5])
                    b = len(ratio)
                    p = min(1.0, 2 * min(np.sum(ratio >= 1) + 1, np.sum(ratio <= 1) + 1) / (b + 1))
                    med = float(np.median(err)); base_med = float(np.median(base))
                    summary.append({
                        "dataset": dataset, "workload": workload, "form": form, "target": target,
                        "original_target": orig_target,
                        "role": "resampling control" if target == orig_target else "target change",
                        "median": med, "p95": float(np.percentile(err, 95)), "max": float(err.max()),
                        "original_median": base_med, "median_ratio_vs_original": med / base_med,
                        "ratio_ci95_lower": float(lo), "ratio_ci95_upper": float(hi), "p_raw": float(p),
                        "estimates_identical_share": float(np.mean(
                            np.array([float(r[form]) for r in new]) == np.array([float(r[form]) for r in orig]))),
                    })
    with (args.output_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    write_results_json({"created_utc": datetime.now(timezone.utc).isoformat(), "schema": SCHEMA,
                        "schema_dropped": not args.keep_schema, "analyze": analyze_meta,
                        "q_error": "clamped to [1, N]",
                        "bootstrap": f"{BOOTSTRAP_RESAMPLES} replayed resamples, seed {BOOTSTRAP_SEED}",
                        "note": "ANALYZE samples rows at random; the original-target run is a resampling control",
                        "originals_analyzed": False}, args.output_dir / "metadata.json")
    for s in summary:
        print(f"{s['dataset']:6s} {s['workload'][:5]} {s['form']:13s} target {s['target']:4d} ({s['role'][:10]}): "
              f"median {s['median']:.3f} vs orig {s['original_median']:.3f}, ratio {s['median_ratio_vs_original']:.3f} "
              f"[{s['ratio_ci95_lower']:.3f}, {s['ratio_ci95_upper']:.3f}] p={s['p_raw']:.4f}")


if __name__ == "__main__":
    main()
