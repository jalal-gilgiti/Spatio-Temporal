"""Analyze the plan-study re-run with the original plan-study analysis code.

``analyze_plan_sensitive_evaluation`` reads its condition list from module
globals, so they are replaced with the re-run's conditions. Everything else (the
pooled-fingerprint oracle, regret, bad-plan thresholds, speedups relative to
unhinted native, and the query bootstrap) is the published definition.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_plan_sensitive_evaluation as apa  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ("joint", "separable", "no_coupling", "spatial_only")
CONDITIONS = ("native", *LEARNED, *(f"{m}_clamped" for m in LEARNED),
              *(f"{m}_dm" for m in LEARNED), "trajectory_sampling", "true_cardinality_input")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=ROOT / "run/vldbj_rev/plan_rerun")
    ap.add_argument("--dataset", default="tdrive")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = ap.parse_args()
    conditions = tuple(args.conditions)
    apa.CONDITIONS = conditions
    measurements = apa.load_measurements(args.input_dir, (args.dataset,))
    rows = apa.per_query_rows(measurements)
    aggregate = [r for r in apa.aggregate_rows(rows, (args.dataset,)) if r["dataset"] == args.dataset]
    boots = [r for r in apa.bootstrap_rows(rows, (args.dataset,), apa.BOOTSTRAP_RESAMPLES,
                                            apa.BOOTSTRAP_SEED) if r["dataset"] == args.dataset]
    clamped = {name: float(np.mean([m["conditions"][name].get("clamped", False)
                                    for m in measurements])) for name in conditions}
    by_q: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_q.setdefault(r["query_id"], {})[r["condition"]] = r
    for row in aggregate:
        c = row["condition"]
        regrets = [by_q[q][c]["regret"] for q in by_q]
        row["regret_gmean"] = float(np.exp(np.mean(np.log(regrets))))
        # win/tie/loss against native on regret; a tie is the same plan (identical pooled runtime)
        diffs = [by_q[q][c]["regret"] - by_q[q]["native"]["regret"] for q in by_q]
        row["regret_wins_vs_native"] = int(sum(d < -1e-12 for d in diffs))
        row["regret_ties_vs_native"] = int(sum(abs(d) <= 1e-12 for d in diffs))
        row["regret_losses_vs_native"] = int(sum(d > 1e-12 for d in diffs))
        # win/loss on each query's own end-to-end time (median of the 5 runs)
        e2e = [by_q[q][c]["end_to_end_ms"] - by_q[q]["native"]["end_to_end_ms"] for q in by_q]
        row["e2e_faster_than_native"] = int(sum(d < 0 for d in e2e))
        row["e2e_slower_than_native"] = int(sum(d > 0 for d in e2e))
        row["e2e_total_ms"] = float(sum(by_q[q][c]["end_to_end_ms"] for q in by_q))
    for row in aggregate:
        row["share_hints_clamped"] = clamped[row["condition"]]
        row["planning_ms_median"] = float(np.median(
            [r["planning_ms"] for r in rows if r["condition"] == row["condition"]]))
        row["inference_ms_median"] = float(np.median(
            [r["inference_ms"] for r in rows if r["condition"] == row["condition"]]))
    out = args.input_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_query_plan_metrics", rows), ("plan_metrics_summary", aggregate),
                       ("plan_bootstrap_intervals", boots)):
        with (out / f"{name}.csv").open("w", newline="") as fh:
            fields = list(dict.fromkeys(k for r in data for k in r))
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(data)
    write_results_json({"conditions": list(conditions), "dataset": args.dataset, "bootstrap_seed": apa.BOOTSTRAP_SEED,
                        "bootstrap_resamples": apa.BOOTSTRAP_RESAMPLES,
                        "n_queries": len(measurements)}, out / "analysis_metadata.json")
    print(json.dumps({r["condition"]: round(r["regret_median"], 3) for r in aggregate}))


if __name__ == "__main__":
    main()
