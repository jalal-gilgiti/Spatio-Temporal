"""Decompose the native estimate's error.

For every frozen accuracy query (T-Drive and Porto, fixed and mixed; 1,200 in all)
this counts the exact number of rows satisfying the bounding-box predicate
``traj && B`` (read-only), then splits the Native-BBox error against the
path-intersection truth into two signed parts:

    log10((est+1)/(path+1)) = log10((est+1)/(box+1)) + log10((box+1)/(path+1))
         total error        =   histogram error     +    semantic gap

* histogram error: MobilityDB's estimate of ``traj && B`` against its own exact
  answer. Positive means overestimation.
* semantic gap: trajectories whose bounding box overlaps B minus those whose path
  actually passes through B during the window. Always >= 0, because the path
  test implies the box test.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import query_from_dict  # noqa: E402
from src.data_module.workload_generator import stbox_literal  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tdrive": ("run/real_experiments/task7b/headline/workloads", "tdrive_traj"),
    "porto": ("run/real_experiments/multiseed_accuracy/workloads", "porto_traj"),
    "ais": ("run/vldbj_rev/ais/headline/workloads", "ais_traj"),
}
WORKLOADS = ("fixed_extent", "mixed_extent")
BUCKETS = (("0", 0, 0), ("1-10", 1, 10), ("11-100", 11, 100),
           ("101-1,000", 101, 1000), (">1,000", 1001, None))


def describe(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)), "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
        "share_positive": float(np.mean(values > 1e-12)),
        "share_negative": float(np.mean(values < -1e-12)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--native", type=Path, default=ROOT / "run/vldbj_rev/native/explain_estimates_primary.csv",
                    help="primary: statistics target 1000 on both datasets; explain_estimates.csv = original targets")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--datasets", nargs="+", default=["tdrive", "porto"], choices=list(SOURCES))
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/native")
    args = ap.parse_args()

    box_path = args.output_dir / "box_counts.csv"
    if box_path.exists():
        boxes = {(r["dataset"], r["workload"], int(r["query_index"])): int(r["box_count"])
                 for r in csv.DictReader(box_path.open())}
    else:
        boxes, rows = {}, []
        with psycopg.connect(args.dsn) as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                for dataset, (wdir, table) in SOURCES.items():
                    if dataset not in args.datasets:
                        continue
                    for workload in WORKLOADS:
                        payload = json.loads((ROOT / wdir / f"{dataset}__{workload}.json").read_text())
                        for i, q in enumerate(payload["evaluation_queries"]):
                            p = query_from_dict(q).predicate
                            cur.execute(sql.SQL("SELECT count(*) FROM {} WHERE traj && %s::stbox")
                                        .format(sql.Identifier(table)), (stbox_literal(p),))
                            n = int(cur.fetchone()[0])
                            boxes[(dataset, workload, i)] = n
                            rows.append({"dataset": dataset, "workload": workload,
                                         "query_index": i, "box_count": n})
            conn.rollback()
        with box_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    out = []
    native = list(csv.DictReader(args.native.open()))
    for r_ in native:
        r_["box_count"] = boxes[(r_["dataset"], r_["workload"], int(r_["query_index"]))]
        if r_["box_count"] < int(r_["true_cardinality"]):
            raise RuntimeError(f"box count below path count: {r_}")
    for dataset in args.datasets:
        for workload in WORKLOADS:
            rs = [r for r in native if r["dataset"] == dataset and r["workload"] == workload]
            path = np.array([int(r["true_cardinality"]) for r in rs], float)
            box = np.array([r["box_count"] for r in rs], float)
            est = np.array([float(r["native_bbox"]) for r in rs])
            hist = np.log10((est + 1) / (box + 1))
            gap = np.log10((box + 1) / (path + 1))
            total = np.log10((est + 1) / (path + 1))
            for label, lo, hi in (("all", 0, None),) + BUCKETS:
                m = (path >= lo) & (path <= hi if hi is not None else True)
                if not m.any():
                    continue
                for part, values in (("histogram_error", hist), ("semantic_gap", gap),
                                     ("total_error", total)):
                    out.append({"dataset": dataset, "workload": workload, "bucket": label,
                                "n_queries": int(m.sum()), "component": part,
                                **describe(values[m]),
                                "median_box_to_path_ratio": float(np.median((box[m] + 1) / (path[m] + 1)))})
    with (args.output_dir / f"native_decomposition{args.suffix}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    write_results_json({
        "estimate": "Native-BBox planner row estimate (explain_estimates.csv, column native_bbox)",
        "box_count": "exact SELECT count(*) WHERE traj && B, read-only",
        "decomposition": "log10((e+1)/(p+1)) = log10((e+1)/(b+1)) + log10((b+1)/(p+1))",
        "sign": "positive = overestimate",
        "native_source": args.native.name,
    }, args.output_dir / f"native_decomposition{args.suffix}_metadata.json")
    print("wrote", args.output_dir / f"native_decomposition{args.suffix}.csv")


if __name__ == "__main__":
    main()
