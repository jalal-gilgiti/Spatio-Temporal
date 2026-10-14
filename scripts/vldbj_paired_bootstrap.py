"""Paired bootstrap tests on clamped [1, N] q-error, with Holm correction.

Families (each Holm-corrected separately, for each query subset):
  (a) each learned estimator (Joint, Separable, No-coupling, SpatialOnly) vs
      Native-BBox, over the 4 dataset-workload settings: 16 tests;
  (b) Separable vs Joint over the 4 settings: 4 tests.
Subsets: all 300 queries, and non-empty queries only.

The statistic follows ``analyze_multiseed_accuracy.py``: the ratio of the two
estimators' bootstrap medians (mean of per-seed medians) on shared paired-query
resample indices. The resamples are exactly those of ``vldbj_task1_table.py``
(10,000 resamples, seed 20260911): replayed indices for all queries, and the same
fresh draws for the non-empty subset. Two-sided p-value for H0 "ratio = 1":
p = min(1, 2 * min(#{r >= 1} + 1, #{r <= 1} + 1) / (B + 1)).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_multiseed_accuracy import (  # noqa: E402
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, WORKLOADS, bootstrap_median, load_runs,
)
from scripts.vldbj_task1_table import (  # noqa: E402
    SOURCES, prediction_matrix, qerrors, resample_indices,
)
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LEARNED = ("joint", "separable", "no_coupling", "spatial_only")
PATHHIST = ("path_hist_sum", "path_hist_max")
FAMILIES = {
    # family (a): every non-native estimator against MobilityDB's own estimate
    "a_learned_vs_native_bbox": [(m, "native_bbox") for m in LEARNED + PATHHIST],
    "b_separable_vs_joint": [("separable", "joint")],
    "c_path_hist_sum_vs_separable": [("path_hist_sum", "separable")],
    # family (d): the exact box count as an estimator ("perfect box histogram"; an upper
    # bound on the path count, not deployable): headroom of fixing MobilityDB's histogram
    "d_box_oracle": [("box_oracle", "separable"), ("box_oracle", "native_bbox")],
}


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, default=ROOT / "run/vldbj_rev/native/explain_estimates_primary.csv",
                    help="primary: statistics target 1000 on both datasets; explain_estimates.csv = original targets")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/stats")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    args.native = args.native.resolve()
    hist_rows = list(csv.DictReader((ROOT / "run/vldbj_rev/path_histogram/estimates.csv").open()))
    ais_dir = ROOT / "run/vldbj_rev/ais"
    ais_native = (list(csv.DictReader((ais_dir / "native/explain_estimates.csv").open()))
                  if (ais_dir / "native/explain_estimates.csv").exists() else [])
    ais_hist = (list(csv.DictReader((ais_dir / "path_histogram/estimates.csv").open()))
                if (ais_dir / "path_histogram/estimates.csv").exists() else [])
    box = {}
    for path in (ROOT / "run/vldbj_rev/native/box_counts.csv", ais_dir / "native/box_counts.csv"):
        if path.exists():
            box.update({(r["dataset"], r["workload"], int(r["query_index"])): int(r["box_count"])
                        for r in csv.DictReader(path.open())})
    meta = json.loads((args.native.parent / "metadata.json").read_text())["tables"]
    ais_meta = ROOT / "run/vldbj_rev/ais/native/metadata.json"
    if ais_meta.exists():
        meta.update(json.loads(ais_meta.read_text())["tables"])
    native_rows = list(csv.DictReader(args.native.open()))
    rows = []
    for subset in ("all", "nonempty"):
        fresh_rng = np.random.default_rng(BOOTSTRAP_SEED)
        for dataset, (src, _) in SOURCES.items():
            if not (src / "runs").exists():
                continue
            grouped = load_runs(src)
            n_rows = float(meta[f"{dataset}_traj"]["row_count"])
            for workload in WORKLOADS:
                present = [m for m in LEARNED + ("trajectory_sampling",) if (dataset, workload, m) in grouped]
                truth = np.array([p["true_cardinality"]
                                  for p in grouped[(dataset, workload, present[0])][0]["predictions"]])
                preds = {m: prediction_matrix(grouped[(dataset, workload, m)]) for m in present if m in LEARNED}
                nat = sorted((r for r in native_rows + ais_native
                              if r["dataset"] == dataset and r["workload"] == workload),
                             key=lambda r: int(r["query_index"]))
                preds["native_bbox"] = np.array([[float(r["native_bbox"]) for r in nat]])
                hist = sorted((r for r in hist_rows + ais_hist if r["dataset"] == dataset and r["workload"] == workload),
                              key=lambda r: int(r["query_index"]))
                if [int(r["true_cardinality"]) for r in hist] != truth.tolist():
                    raise RuntimeError("path-histogram rows not aligned")
                for form in PATHHIST:
                    preds[form] = np.array([[float(r[form]) for r in hist]])
                keys = [(dataset, workload, i) for i in range(len(truth))]
                if all(k in box for k in keys):
                    preds["box_oracle"] = np.array([[float(box[k]) for k in keys]])
                mask = truth > 0 if subset == "nonempty" else np.ones(len(truth), bool)
                t = truth[mask]
                if subset == "all":
                    indices = resample_indices(dataset, len(truth))[workload]
                else:
                    indices = fresh_rng.integers(0, len(t), size=(BOOTSTRAP_RESAMPLES, len(t)))
                errors = {k: qerrors(t, v[:, mask], "planner", n_rows) for k, v in preds.items()}
                boots = {k: bootstrap_median(e, indices) for k, e in errors.items()}
                point = {k: float(np.mean(np.median(e, axis=1))) for k, e in errors.items()}
                for family, pairs in FAMILIES.items():
                    for left, right in pairs:
                        if left not in boots or right not in boots:
                            continue  # e.g. AIS learned models not trained yet
                        ratio = boots[left] / boots[right]
                        lo, hi = np.percentile(ratio, [2.5, 97.5])
                        b = len(ratio)
                        p = min(1.0, 2 * min(np.sum(ratio >= 1) + 1, np.sum(ratio <= 1) + 1) / (b + 1))
                        rows.append({"family": family, "subset": subset, "dataset": dataset,
                                     "workload": workload, "left": left, "right": right,
                                     "n_queries": int(mask.sum()),
                                     "left_median": point[left], "right_median": point[right],
                                     "median_ratio": point[left] / point[right],
                                     "ratio_ci95_lower": float(lo), "ratio_ci95_upper": float(hi),
                                     "p_raw": float(p)})
    for family in FAMILIES:
        for subset in ("all", "nonempty"):
            idx = [i for i, r in enumerate(rows) if r["family"] == family and r["subset"] == subset]
            for i, adj in zip(idx, holm([rows[i]["p_raw"] for i in idx])):
                rows[i]["p_holm"] = adj
                rows[i]["family_size"] = len(idx)
                rows[i]["significant_holm_0_05"] = adj < 0.05
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"paired_bootstrap_clamped{args.suffix}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_results_json({
        "q_error": "clamped to [1, N]",
        "statistic": "ratio of mean-of-per-seed bootstrap medians, left / right (<1 favours left)",
        "resamples": f"{BOOTSTRAP_RESAMPLES}, seed {BOOTSTRAP_SEED}, identical to vldbj_task1_table.py",
        "p_value": "two-sided, p = min(1, 2*min(#(r>=1)+1, #(r<=1)+1)/(B+1)); smallest possible ~2e-4",
        "multiple_testing": "Holm within each family and subset; family sizes in column family_size",
        "native_source": str(args.native.relative_to(ROOT)),
        "model_source": "saved headline runs (task7b/headline, multiseed_accuracy); rerun after retraining",
    }, args.output_dir / f"paired_bootstrap_clamped{args.suffix}_metadata.json")
    print(f"wrote {len(rows)} comparisons")


if __name__ == "__main__":
    main()
