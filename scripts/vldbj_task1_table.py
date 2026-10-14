"""Combined accuracy table with the re-measured native rows.

Joins the saved per-seed runs (T-Drive: task7b/headline, Porto: multiseed_accuracy)
with ``run/vldbj_rev/native/explain_estimates.csv``. Q-error is recomputed from the
raw predictions, so the estimate can optionally be clamped first:

* ``--clamp none`` (default): estimate floored at 0, as in the original analysis.
  The script checks that the saved medians and median intervals are reproduced
  exactly before it reports anything new.
* ``--clamp planner``: estimate clamped to [1, N], i.e. what PostgreSQL would
  actually use (the planner and pg_hint_plan Rows() never go below one row, and
  a table of N rows cannot return more than N).

``--subset nonempty`` restricts every statistic to queries whose true cardinality
is at least one.

The per-seed aggregation and the paired-query bootstrap are those of
``analyze_multiseed_accuracy.py``. For the full workloads its resample indices are
reproduced by replaying the same RNG draw order. The non-empty subset has a
different size, so its indices are drawn fresh from the same seed, in the same
dataset/workload order.
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
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, WORKLOADS, aggregate_metrics, bootstrap_median,
    load_runs,
)
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# (source directory, dataset order used by the run that produced it)
SOURCES = {
    "tdrive": (ROOT / "run/real_experiments/task7b/headline", ("tdrive",)),
    "porto": (ROOT / "run/real_experiments/multiseed_accuracy", ("tdrive", "porto", "nyc_tlc")),
    # AIS has no earlier analysis to replay; its resamples are drawn fresh from the same seed
    "ais": (ROOT / "run/vldbj_rev/ais/headline", ("ais",)),
}
AIS_NATIVE = ROOT / "run/vldbj_rev/ais/native"
AIS_HIST = ROOT / "run/vldbj_rev/ais/path_histogram/estimates.csv"
SAVED = ("joint", "separable", "no_coupling", "spatial_only", "trajectory_sampling",
         "independent_histogram", "joint_histogram", "minskew_origin", "origin_sampling")
NATIVE = ("pg_default", "native_bbox", "native_served")
PATHHIST = ("path_hist_sum", "path_hist_max")
# exact traj && B count used as an estimate: an upper bound on the path count (box overlap is
# necessary for path intersection) and a "perfect box histogram" oracle, not deployable
BOXORACLE = ("box_oracle",)
ORDER = ("pg_default", "native_bbox", "native_served") + BOXORACLE + SAVED + PATHHIST
DISPLAY = {
    "pg_default": "PostgreSQL default (IS NOT NULL)",
    "native_bbox": "Native-BBox (traj && B)",
    "native_served": "Native-Served (&& AND atStbox)",
    "joint": "Joint", "separable": "Separable", "no_coupling": "No-coupling",
    "spatial_only": "SpatialOnly", "trajectory_sampling": "Trajectory Sampling",
    "path_hist_sum": "Path histogram (sum)", "path_hist_max": "Path histogram (max)",
    "box_oracle": "Box-count oracle (exact traj && B)",
    "independent_histogram": "Independent histogram (origin)", "joint_histogram": "Joint histogram (origin)",
    "minskew_origin": "MinSkew-Origin", "origin_sampling": "Origin sampling",
}
BUCKETS = (("0", 0, 0), ("1-10", 1, 10), ("11-100", 11, 100),
           ("101-1,000", 101, 1000), (">1,000", 1001, None))


def resample_indices(dataset: str, n_queries: int) -> dict[str, np.ndarray]:
    """Replay analyze_multiseed_accuracy's RNG to recover its per-workload indices."""
    _, order = SOURCES[dataset]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out = {}
    for ds in order:
        for workload in WORKLOADS:
            idx = rng.integers(0, n_queries, size=(BOOTSTRAP_RESAMPLES, n_queries))
            if ds == dataset:
                out[workload] = idx
    return out


def bootstrap_p95(errors: np.ndarray, indices: np.ndarray, chunk: int = 500) -> np.ndarray:
    values = np.empty(len(indices))
    for s in range(0, len(indices), chunk):
        sel = indices[s:s + chunk]
        values[s:s + chunk] = np.mean(
            [np.percentile(e[sel], 95, axis=1) for e in errors], axis=0)
    return values


def prediction_matrix(runs: list[dict]) -> np.ndarray:
    """(seeds x queries) raw predictions, seeds ordered as error_matrix orders them."""
    ordered = sorted(runs, key=lambda r: -1 if r["estimator_seed"] is None else r["estimator_seed"])
    return np.array([[p["predicted_cardinality"] for p in r["predictions"]] for r in ordered],
                    dtype=float)


def qerrors(truth: np.ndarray, pred: np.ndarray, clamp: str, n_rows: float) -> np.ndarray:
    est = np.clip(pred, 1.0, n_rows) if clamp == "planner" else np.maximum(pred, 0.0)
    t = np.maximum(truth, 0) + 1.0
    e = est + 1.0
    with np.errstate(over="ignore"):
        return np.maximum(t / e, e / t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, default=ROOT / "run/vldbj_rev/native/explain_estimates_primary.csv",
                    help="primary: statistics target 1000 on both datasets; explain_estimates.csv = original targets")
    ap.add_argument("--suffix", default="", help="appended to every output name, e.g. __origtarget")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/native")
    ap.add_argument("--clamp", choices=("none", "planner"), default="none")
    ap.add_argument("--subset", choices=("all", "nonempty"), default="all")
    args = ap.parse_args()
    args.native = args.native.resolve()
    tag = "" if (args.clamp, args.subset) == ("none", "all") else f"__{args.clamp}_{args.subset}"
    meta_rows = json.loads((args.native.parent / "metadata.json").read_text())["tables"]
    if (AIS_NATIVE / "metadata.json").exists():
        meta_rows.update(json.loads((AIS_NATIVE / "metadata.json").read_text())["tables"])

    native_rows = list(csv.DictReader(args.native.open()))
    hist_rows = list(csv.DictReader((ROOT / "run/vldbj_rev/path_histogram/estimates.csv").open()))
    if (AIS_NATIVE / "explain_estimates.csv").exists():  # AIS is analysed at target 1000 only
        native_rows += list(csv.DictReader((AIS_NATIVE / "explain_estimates.csv").open()))
    if AIS_HIST.exists():
        hist_rows += list(csv.DictReader(AIS_HIST.open()))
    box = {}
    for path in (ROOT / "run/vldbj_rev/native/box_counts.csv", AIS_NATIVE / "box_counts.csv"):
        if path.exists():
            box.update({(r["dataset"], r["workload"], int(r["query_index"])): int(r["box_count"])
                        for r in csv.DictReader(path.open())})
    summary, buckets, shares, checks = [], [], [], []
    fresh_rng = np.random.default_rng(BOOTSTRAP_SEED)
    for dataset, (src, _) in SOURCES.items():
        if not (src / "runs").exists():
            continue
        grouped = load_runs(src)
        n_rows = float(meta_rows[f"{dataset}_traj"]["row_count"])
        final = ({(r["workload"], r["estimator"]): r
                  for r in csv.DictReader((src / "final_summary.csv").open())
                  if r["dataset"] == dataset} if (src / "final_summary.csv").exists() else {})
        for workload in WORKLOADS:
            present = [e for e in SAVED if (dataset, workload, e) in grouped]
            preds = {e: prediction_matrix(grouped[(dataset, workload, e)]) for e in present}
            truth = np.array([p["true_cardinality"]
                              for p in grouped[(dataset, workload, present[0])][0]["predictions"]])
            nat = sorted((r for r in native_rows
                          if r["dataset"] == dataset and r["workload"] == workload),
                         key=lambda r: int(r["query_index"]))
            if nat:  # absent only for a dataset whose native estimates are not computed yet
                if [int(r["true_cardinality"]) for r in nat] != truth.tolist():
                    raise RuntimeError(f"{dataset}/{workload}: native rows not aligned with saved runs")
                for form in NATIVE:
                    preds[form] = np.array([[float(r[form]) for r in nat]])
            hist = sorted((r for r in hist_rows if r["dataset"] == dataset and r["workload"] == workload),
                          key=lambda r: int(r["query_index"]))
            if hist:
                if [int(r["true_cardinality"]) for r in hist] != truth.tolist():
                    raise RuntimeError(f"{dataset}/{workload}: path-histogram rows not aligned")
                for form in PATHHIST:
                    preds[form] = np.array([[float(r[form]) for r in hist]])
            keys = [(dataset, workload, i) for i in range(len(truth))]
            if all(k in box for k in keys):
                b = np.array([box[k] for k in keys])
                if np.any(b < truth):
                    raise RuntimeError(f"{dataset}/{workload}: a box count is below the path count")
                preds["box_oracle"] = b[None, :].astype(float)
            mask = truth > 0 if args.subset == "nonempty" else np.ones(len(truth), bool)
            t = truth[mask]
            if args.subset == "all":
                indices = resample_indices(dataset, len(truth))[workload]
            else:
                indices = fresh_rng.integers(0, len(t), size=(BOOTSTRAP_RESAMPLES, len(t)))
            for est in ORDER:
                if est not in preds:
                    continue
                p = preds[est][:, mask]
                errors = qerrors(t, p, args.clamp, n_rows)
                shares.append({
                    "dataset": dataset, "workload": workload, "estimator": est,
                    "n_predictions": int(p.size),
                    "share_below_1": float(np.mean(p < 1.0)),
                    "share_above_N": float(np.mean(p > n_rows)),
                    "max_prediction": float(p.max()),
                })
                stats = aggregate_metrics(errors)
                med_lo, med_hi = np.percentile(bootstrap_median(errors, indices), [2.5, 97.5])
                p95_lo, p95_hi = np.percentile(bootstrap_p95(errors, indices), [2.5, 97.5])
                if est in SAVED and not tag and final:
                    saved = final[(workload, est)]
                    for ours, theirs in ((stats["median"], saved["median"]),
                                         (med_lo, saved["median_ci95_lower"]),
                                         (med_hi, saved["median_ci95_upper"])):
                        if abs(ours - float(theirs)) > 1e-9:
                            raise RuntimeError(
                                f"{dataset}/{workload}/{est}: saved value {theirs} not "
                                f"reproduced ({ours})")
                    checks.append(f"{dataset}/{workload}/{est}")
                summary.append({
                    "dataset": dataset, "workload": workload, "estimator": est,
                    "display_name": DISPLAY[est], "n_seeds": errors.shape[0],
                    "n_queries": errors.shape[1],
                    "median": stats["median"], "median_ci95_lower": med_lo,
                    "median_ci95_upper": med_hi,
                    "p95": stats["p95"], "p95_ci95_lower": p95_lo, "p95_ci95_upper": p95_hi,
                    "max": stats["max"],
                })
                for label, lo, hi in BUCKETS:
                    bmask = (t >= lo) & (t <= hi if hi is not None else True)
                    if not bmask.any():
                        continue
                    sub = errors[:, bmask]
                    buckets.append({
                        "dataset": dataset, "workload": workload, "estimator": est,
                        "bucket": label, "n_queries": int(bmask.sum()),
                        "median": float(np.mean(np.median(sub, axis=1))),
                        "p95": float(np.mean(np.percentile(sub, 95, axis=1))),
                        "max": float(sub.max()),
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [("task1_accuracy_table", summary), ("task1_bucket_table", buckets)]
    if args.subset == "all":
        outputs.append(("task1_clamp_shares", shares))
    for name, rows in outputs:
        path = args.output_dir / (f"{name}{tag}{args.suffix}.csv" if name != "task1_clamp_shares"
                                  else f"{name}{args.suffix}.csv")
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_results_json({
        "q_error": "max((t+1)/(e+1), (e+1)/(t+1))",
        "clamp": {"none": "e = max(prediction, 0)",
                  "planner": "e = min(max(prediction, 1), N), N = table row count"}[args.clamp],
        "subset": {"all": "all 300 queries", "nonempty": "queries with true cardinality >= 1"}[args.subset],
        "median_and_p95": "mean over estimator seeds of the per-seed statistic",
        "max": "worst q-error over all seeds and queries (no interval)",
        "bootstrap": (f"{BOOTSTRAP_RESAMPLES} paired-query resamples, seed {BOOTSTRAP_SEED}, "
                      + ("indices replayed from analyze_multiseed_accuracy.py" if args.subset == "all"
                         else "fresh indices over the non-empty subset")),
        "reproduced_saved_rows": checks,
        "clamp_shares": "share of raw per-seed predictions below 1 or above N (unclamped inputs)",
        "native_source": str(args.native.relative_to(ROOT)),
    }, args.output_dir / f"task1_table_metadata{tag}{args.suffix}.json")
    print(f"[{args.clamp}/{args.subset}] reproduced {len(checks)} saved median/CI rows exactly")


if __name__ == "__main__":
    main()
