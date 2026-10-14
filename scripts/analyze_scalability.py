"""Validate and analyze the scalability artifacts.

The raw runner deliberately writes one JSON checkpoint per estimator seed.  This
script is the only place those checkpoints become headline tables: it verifies paired
queries, aggregates metrics across seeds using the accuracy-battery convention, bootstraps the
fixed evaluation-query set, and keeps data-size scaling separate from training-budget
scaling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_io import write_results_json  # noqa: E402


DATASETS = ("tdrive", "porto", "nyc_tlc")
WORKLOADS = ("fixed_extent", "mixed_extent")
LEARNED = ("joint", "separable", "no_coupling", "spatial_only")
SAMPLERS = ("trajectory_sampling_fraction_1pct", "trajectory_sampling_count_1000")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 55005


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_runs(output_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted((output_dir / "runs").glob("*/*.json"))
    ]


def error_matrix(runs: list[dict[str, Any]]) -> np.ndarray:
    ordered = sorted(runs, key=lambda row: -1 if row.get("seed") is None else row["seed"])
    if not ordered:
        raise ValueError("cannot build an error matrix from no runs")
    reference_truth = [row["true_cardinality"] for row in ordered[0]["predictions"]]
    reference_indices = [row["query_index"] for row in ordered[0]["predictions"]]
    for run in ordered[1:]:
        truth = [row["true_cardinality"] for row in run["predictions"]]
        indices = [row["query_index"] for row in run["predictions"]]
        if truth != reference_truth or indices != reference_indices:
            raise RuntimeError(
                "unpaired evaluation queries for "
                f"{run['dataset']}/{run['scale']}/{run['workload']}/{run['estimator']}"
            )
    return np.asarray([
        [row["q_error"] for row in run["predictions"]] for run in ordered
    ], dtype=float)


def aggregate_metrics(errors: np.ndarray) -> dict[str, float]:
    """Accuracy-battery convention: report the mean per-seed quantile and global max."""
    per_seed_medians = np.median(errors, axis=1)
    return {
        "median_q_error": float(np.mean(per_seed_medians)),
        "p90_q_error": float(np.mean(np.percentile(errors, 90, axis=1))),
        "p95_q_error": float(np.mean(np.percentile(errors, 95, axis=1))),
        "max_q_error": float(np.max(errors)),
        "median_seed_sd": (
            float(np.std(per_seed_medians, ddof=1)) if len(per_seed_medians) > 1 else 0.0
        ),
    }


def bootstrap_mean_seed_median(
    errors: np.ndarray,
    *,
    resamples: int,
    seed: int,
    chunk_size: int = 500,
) -> tuple[float, float]:
    """Query bootstrap of the aggregate median with fixed RNG seed."""
    rng = np.random.default_rng(seed)
    n_queries = errors.shape[1]
    values = np.empty(resamples, dtype=float)
    for start in range(0, resamples, chunk_size):
        stop = min(resamples, start + chunk_size)
        indices = rng.integers(0, n_queries, size=(stop - start, n_queries))
        per_seed = [np.median(seed_errors[indices], axis=1) for seed_errors in errors]
        values[start:stop] = np.mean(per_seed, axis=0)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper)


def convergence_status(previous: float | None, current: float) -> tuple[float | None, str]:
    if previous is None:
        return None, "initial"
    improvement = 100.0 * (previous - current) / previous
    if improvement >= 5.0:
        return improvement, "undertrained_at_previous_budget"
    if improvement >= 0.0:
        return improvement, "practically_saturated"
    return improvement, "unclear_non_monotone"


def empirical_power_fit(x: list[float], y: list[float]) -> dict[str, float | None]:
    """Descriptive log-log fit; this is not presented as formal complexity."""
    valid = [(a, b) for a, b in zip(x, y) if a > 0 and b > 0 and math.isfinite(b)]
    if len(valid) < 3:
        return {"log_log_slope": None, "r_squared": None}
    lx = np.log(np.asarray([row[0] for row in valid], dtype=float))
    ly = np.log(np.asarray([row[1] for row in valid], dtype=float))
    slope, intercept = np.polyfit(lx, ly, 1)
    predicted = intercept + slope * lx
    denominator = float(np.sum((ly - np.mean(ly)) ** 2))
    r_squared = 1.0 - float(np.sum((ly - predicted) ** 2)) / denominator if denominator else 1.0
    return {"log_log_slope": float(slope), "r_squared": r_squared}


def grouped_runs(runs: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for run in runs:
        key = (
            run["dataset"], run["scale"], run["workload"], run["estimator"],
            run.get("training_budget"),
        )
        grouped.setdefault(key, []).append(run)
    return grouped


def _expected_seed_set(estimator: str, model_seeds: tuple[int, ...], sampling_seeds: tuple[int, ...]):
    if estimator in LEARNED:
        return set(model_seeds)
    if estimator in SAMPLERS:
        return set(sampling_seeds)
    return {None}


def validate_seed_group(
    runs: list[dict[str, Any]], estimator: str,
    model_seeds: tuple[int, ...], sampling_seeds: tuple[int, ...],
) -> None:
    observed = {run.get("seed") for run in runs}
    expected = _expected_seed_set(estimator, model_seeds, sampling_seeds)
    if observed != expected:
        raise RuntimeError(f"incomplete seeds for {estimator}: expected {expected}, got {observed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=[17, 23, 47])
    parser.add_argument("--n-main-train", type=int, default=800)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    datasets = tuple(args.datasets)
    workloads = tuple(args.workloads)
    model_seeds = tuple(args.model_seeds)
    sampling_seeds = tuple(args.sampling_seeds)

    runs = load_runs(args.output_dir)
    grouped = grouped_runs(runs)
    scale_manifests = {
        dataset: json.loads(
            (args.output_dir / "prepared" / f"{dataset}__scales.json").read_text()
        )
        for dataset in datasets
    }

    data_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        scales = tuple(scale_manifests[dataset]["actual_scales"])
        largest = max(scales)
        for scale in scales:
            table_stats = json.loads(
                (args.output_dir / "table_stats" / f"{dataset}__n{scale}.json").read_text()
            )
            resource_rows.append({"resource_kind": "database", **table_stats})
            for workload in workloads:
                label_costs = json.loads(
                    (args.output_dir / "label_costs" / f"{dataset}__n{scale}__{workload}.json").read_text()
                )["training_label_costs"]
                main_label = label_costs[str(args.n_main_train)]
                resource_rows.append({
                    "resource_kind": "training_labels",
                    "dataset": dataset, "scale": scale, "workload": workload,
                    "estimator": None, "training_budget": args.n_main_train,
                    "seconds": main_label["seconds"],
                    "seconds_per_query": main_label["seconds_per_query"],
                })
                estimators = (*LEARNED, *SAMPLERS, "native")
                for estimator in estimators:
                    budget = args.n_main_train if estimator in LEARNED else None
                    key = (dataset, scale, workload, estimator, budget)
                    values = grouped.get(key, [])
                    validate_seed_group(values, estimator, model_seeds, sampling_seeds)
                    errors = error_matrix(values)
                    metrics = aggregate_metrics(errors)
                    ci_lower, ci_upper = bootstrap_mean_seed_median(
                        errors,
                        resamples=args.bootstrap_resamples,
                        seed=args.bootstrap_seed,
                    )
                    row = {
                        "dataset": dataset, "scale": scale, "workload": workload,
                        "estimator": estimator, "n_seeds": len(values),
                        "n_queries": errors.shape[1], **metrics,
                        "median_ci95_lower": ci_lower,
                        "median_ci95_upper": ci_upper,
                    }
                    data_rows.append(row)
                    bootstrap_rows.append({
                        "experiment": "data_size", "dataset": dataset,
                        "scale": scale, "workload": workload, "estimator": estimator,
                        "training_budget": budget,
                        "lower": ci_lower, "upper": ci_upper,
                        "resamples": args.bootstrap_resamples, "seed": args.bootstrap_seed,
                    })
                    for run in values:
                        resource_rows.append({
                            "resource_kind": run["kind"],
                            "dataset": dataset, "scale": scale, "workload": workload,
                            "estimator": estimator, "training_budget": budget,
                            "seed": run.get("seed"),
                            "build_or_training_seconds": run.get(
                                "training_seconds", run.get("sample_build_seconds")
                            ),
                            "storage_bytes": run.get(
                                "serialized_bytes", run.get("sample_storage_bytes")
                            ),
                            "parameter_bytes": run.get("parameter_bytes"),
                            "peak_rss_delta_bytes": run.get(
                                "peak_training_rss_delta_bytes",
                                run.get("peak_build_rss_delta_bytes"),
                            ),
                            "inference_p50_ms": run["summary"]["inference_p50_ms"],
                            "inference_p95_ms": run["summary"]["inference_p95_ms"],
                        })

        for workload in workloads:
            budgets = sorted({
                int(key[4]) for key in grouped
                if key[0] == dataset and key[1] == largest and key[2] == workload
                and key[3] in LEARNED and key[4] is not None
            })
            for estimator in LEARNED:
                previous: float | None = None
                final_status = "initial"
                final_improvement: float | None = None
                for budget in budgets:
                    values = grouped.get((dataset, largest, workload, estimator, budget), [])
                    validate_seed_group(values, estimator, model_seeds, sampling_seeds)
                    errors = error_matrix(values)
                    metrics = aggregate_metrics(errors)
                    improvement, status = convergence_status(
                        previous, metrics["median_q_error"]
                    )
                    ci_lower, ci_upper = bootstrap_mean_seed_median(
                        errors,
                        resamples=args.bootstrap_resamples,
                        seed=args.bootstrap_seed,
                    )
                    label_cost = json.loads(
                        (args.output_dir / "label_costs" /
                         f"{dataset}__n{largest}__{workload}.json").read_text()
                    )["training_label_costs"][str(budget)]
                    training_rows.append({
                        "dataset": dataset, "scale": largest, "workload": workload,
                        "estimator": estimator, "training_budget": budget,
                        "n_seeds": len(values), "n_queries": errors.shape[1], **metrics,
                        "median_ci95_lower": ci_lower, "median_ci95_upper": ci_upper,
                        "improvement_from_previous_pct": improvement,
                        "step_status": status,
                        "label_seconds": label_cost["seconds"],
                        "mean_training_seconds": float(np.mean([
                            run["training_seconds"] for run in values
                        ])),
                    })
                    previous = metrics["median_q_error"]
                    final_status = status
                    final_improvement = improvement
                convergence_rows.append({
                    "dataset": dataset, "workload": workload,
                    "estimator": estimator, "largest_scale": largest,
                    "last_budget": budgets[-1],
                    "last_step_improvement_pct": final_improvement,
                    "status": final_status,
                })

    trend_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        database = [
            row for row in resource_rows
            if row["resource_kind"] == "database" and row["dataset"] == dataset
        ]
        database.sort(key=lambda row: row["scale"])
        for metric in (
            "load_seconds", "index_build_seconds", "analyze_seconds",
            "table_bytes", "index_bytes", "total_bytes",
        ):
            fit = empirical_power_fit(
                [row["scale"] for row in database], [row[metric] for row in database]
            )
            trend_rows.append({
                "dataset": dataset, "metric": metric,
                "n_scales": len(database), **fit,
            })

    outputs = {
        "data_size_accuracy": data_rows,
        "bootstrap_intervals": bootstrap_rows,
        "training_budget_curve": training_rows,
        "convergence": convergence_rows,
        "resources": resource_rows,
        "empirical_trends": trend_rows,
    }
    for name, rows in outputs.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
        write_results_json({"rows": rows}, args.output_dir / f"{name}.json")
    write_results_json({
        "validation": "PASS",
        "datasets": list(datasets),
        "workloads": list(workloads),
        "model_seeds": list(model_seeds),
        "sampling_seeds": list(sampling_seeds),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "counts": {name: len(rows) for name, rows in outputs.items()},
    }, args.output_dir / "validation.json")
    print(f"validated and analyzed scalability artifacts in {args.output_dir}")


if __name__ == "__main__":
    main()
