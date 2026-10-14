"""Build the primary native estimates (statistics target 1000 on both taxi datasets).

tdrive_traj.traj already has statistics target 1000, so its rows are taken from
explain_estimates.csv unchanged. porto_traj.traj uses the default target (100); its
Native-BBox and Native-Served estimates are replaced by those measured on the full-table
copy ANALYZEd at target 1000 (stats_target/estimates.csv, rows target=1000). At that
target the ANALYZE sample covers the whole 100,000-row table, so the copy's statistics
are deterministic.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_LABEL = {"tdrive": "original table (statistics target 1000)",
                "porto": "full-table copy, statistics target 1000 (deterministic)"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", type=Path, default=ROOT / "run/vldbj_rev/native/explain_estimates.csv")
    ap.add_argument("--copy", type=Path, default=ROOT / "run/vldbj_rev/stats_target/estimates.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "run/vldbj_rev/native/explain_estimates_primary.csv")
    args = ap.parse_args()
    with args.copy.open() as fh:
        t1000 = {(r["dataset"], r["workload"], r["query_index"]): r for r in csv.DictReader(fh)
                 if r["target"] == "1000" and r["dataset"] == "porto"}
    out = []
    with args.original.open() as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] == "porto":
                s = t1000[(r["dataset"], r["workload"], r["query_index"])]
                if s["true_cardinality"] != r["true_cardinality"]:
                    raise RuntimeError(f"truth mismatch for porto/{r['workload']}/{r['query_index']}")
                r["native_bbox"], r["native_served"] = s["native_bbox"], s["native_served"]
            r["native_source"] = SOURCE_LABEL[r["dataset"]]
            out.append(r)
    with args.output.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
