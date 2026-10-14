"""Analyze the multi-seed accuracy runs with paired query bootstrap uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.results_io import write_results_json  # noqa: E402


DATASETS = ("tdrive", "porto", "nyc_tlc")
WORKLOADS = ("fixed_extent", "mixed_extent")
ESTIMATORS = (
    "joint",
    "separable",
    "no_coupling",
    "spatial_only",
    "trajectory_sampling",
    "minskew_origin",
    "origin_sampling",
    "independent_histogram",
    "joint_histogram",
    "native",
)
DISPLAY_NAMES = {
    "joint": "Joint",
    "separable": "Separable",
    "no_coupling": "No-coupling",
    "spatial_only": "SpatialOnly",
    "trajectory_sampling": "Trajectory Sampling",
    "minskew_origin": "MinSkew-Origin",
    "origin_sampling": "Origin Sampling",
    "independent_histogram": "Independent histogram",
    "joint_histogram": "Joint histogram",
    "native": "Native MobilityDB/PostGIS",
}
MODEL_SEEDS = (1, 2, 3, 4, 5)
SAMPLING_SEEDS = (17, 23, 47, 59, 71)
BOOTSTRAP_SEED = 20260911
BOOTSTRAP_RESAMPLES = 10_000
METRICS = ("median", "p90", "p95", "max")
CENTRAL_PAIRS = (
    ("separable", "joint", "architecture"),
    ("no_coupling", "joint", "architecture"),
    ("separable", "spatial_only", "learned_baseline"),
    ("separable", "trajectory_sampling", "matched_nonlearned"),
    ("joint", "trajectory_sampling", "matched_nonlearned"),
    ("joint", "native", "native"),
    ("separable", "native", "native"),
    ("no_coupling", "native", "native"),
    ("spatial_only", "native", "native"),
)
MATCHED = ("joint", "separable", "no_coupling", "spatial_only", "trajectory_sampling", "native")
PROXIES = ("minskew_origin", "origin_sampling", "independent_histogram", "joint_histogram")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_runs(output_dir: Path) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for path in sorted((output_dir / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        key = (run["dataset"], run["workload"], run["estimator"])
        grouped.setdefault(key, []).append(run)
    return grouped


def expected_seeds(estimator: str, model_seeds: tuple[int, ...], sampling_seeds: tuple[int, ...]):
    if estimator in ("joint", "separable", "no_coupling", "spatial_only", "origin_sampling"):
        return model_seeds
    if estimator == "trajectory_sampling":
        return sampling_seeds
    return (None,)


def validate(
    grouped: dict[tuple[str, str, str], list[dict]],
    datasets: tuple[str, ...],
    workloads: tuple[str, ...],
    model_seeds: tuple[int, ...],
    sampling_seeds: tuple[int, ...],
) -> None:
    for dataset in datasets:
        for workload in workloads:
            reference_ids = None
            reference_truths = None
            reference_workload_id = None
            for estimator in ESTIMATORS:
                runs = grouped.get((dataset, workload, estimator), [])
                observed = tuple(sorted(
                    (run["estimator_seed"] for run in runs),
                    key=lambda value: -1 if value is None else value,
                ))
                expected = tuple(expected_seeds(estimator, model_seeds, sampling_seeds))
                if observed != expected:
                    raise RuntimeError(
                        f"incomplete {dataset}/{workload}/{estimator}: expected {expected}, got {observed}"
                    )
                for run in runs:
                    ids = [row["query_id"] for row in run["predictions"]]
                    truths = [row["true_cardinality"] for row in run["predictions"]]
                    if reference_ids is None:
                        reference_ids = ids
                        reference_truths = truths
                        reference_workload_id = run["workload_id"]
                    if ids != reference_ids or truths != reference_truths:
                        raise RuntimeError(f"unpaired queries in {dataset}/{workload}/{estimator}")
                    if run["workload_id"] != reference_workload_id:
                        raise RuntimeError(f"workload-ID mismatch in {dataset}/{workload}")


def error_matrix(runs: list[dict]) -> np.ndarray:
    ordered = sorted(runs, key=lambda run: -1 if run["estimator_seed"] is None else run["estimator_seed"])
    return np.array(
        [[row["q_error"] for row in run["predictions"]] for run in ordered], dtype=float
    )


def aggregate_metrics(errors: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.mean(np.median(errors, axis=1))),
        "p90": float(np.mean(np.percentile(errors, 90, axis=1))),
        "p95": float(np.mean(np.percentile(errors, 95, axis=1))),
        "max": float(np.max(errors)),
        "mean_tail_sensitive": float(np.mean(errors)),
        "median_seed_sd": float(np.std(np.median(errors, axis=1), ddof=1)) if len(errors) > 1 else 0.0,
    }


def bootstrap_median(errors: np.ndarray, indices: np.ndarray, chunk_size: int = 500) -> np.ndarray:
    values = np.empty(len(indices), dtype=float)
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        chunk = indices[start:stop]
        per_seed = [np.median(seed_errors[chunk], axis=1) for seed_errors in errors]
        values[start:stop] = np.mean(per_seed, axis=0)
    return values


def zero_cardinality_rows(output_dir: Path, datasets: tuple[str, ...], workloads: tuple[str, ...]):
    rows = []
    for dataset in datasets:
        for workload in workloads:
            payload = json.loads(
                (output_dir / "workloads" / f"{dataset}__{workload}.json").read_text()
            )
            truth = np.array(
                [row["true_cardinality"] for row in payload["evaluation_queries"]], dtype=int
            )
            row = {
                "dataset": dataset,
                "workload": workload,
                "workload_id": payload["evaluation_workload_id"],
                "n_queries": len(truth),
            }
            for label, threshold in (("zero", 0), ("le_1", 1), ("le_5", 5)):
                count = int(np.sum(truth <= threshold))
                row[f"n_{label}"] = count
                row[f"pct_{label}"] = 100.0 * count / len(truth)
            rows.append(row)
    return rows


def task2_reproduction(
    output_dir: Path,
    grouped: dict[tuple[str, str, str], list[dict]],
    datasets: tuple[str, ...],
    workloads: tuple[str, ...],
) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    task2 = json.loads((repo_root / "run/real_experiments/trajectory_sampling.json").read_text())
    old_learned = {
        "fixed_extent": json.loads(
            (repo_root / "run/revalidation/separable_real_fixed_extent.json").read_text()
        ),
        "mixed_extent": json.loads(
            (repo_root / "run/revalidation/separable_real_spot_check.json").read_text()
        ),
    }
    checks = []
    for dataset in datasets:
        for workload in workloads:
            old_samples = [
                sample["workloads"][workload]["summary"]["median"]
                for sample in task2["datasets"][dataset]["samples"].values()
                if sample["sample_fraction"] == 0.01 and sample["sample_seed"] in (17, 23, 47)
            ]
            new_samples = [
                run["summary"]["median"]
                for run in grouped[(dataset, workload, "trajectory_sampling")]
                if run["estimator_seed"] in (17, 23, 47)
            ]
            old_row = next(row for row in old_learned[workload]["rows"] if row["dataset"] == dataset)
            for estimator, old_key in (("joint", "joint"), ("separable", "separable")):
                new_values = [
                    run["summary"]["median"]
                    for run in grouped[(dataset, workload, estimator)]
                    if run["estimator_seed"] in (1, 2, 3)
                ]
                checks.append({
                    "dataset": dataset,
                    "workload": workload,
                    "estimator": estimator,
                    "prior_three_seed_mean_median": old_row["mean"][old_key],
                    "recomputed_three_seed_mean_median": statistics.mean(new_values),
                    "absolute_difference": abs(old_row["mean"][old_key] - statistics.mean(new_values)),
                })
            checks.append({
                "dataset": dataset,
                "workload": workload,
                "estimator": "trajectory_sampling",
                "prior_three_seed_mean_median": statistics.mean(old_samples),
                "recomputed_three_seed_mean_median": statistics.mean(new_samples),
                "absolute_difference": abs(statistics.mean(old_samples) - statistics.mean(new_samples)),
            })
    return {
        "comparison": "trajectory-sampling artifact/revalidation summaries versus recomputation",
        "tolerance": 1e-9,
        "all_within_tolerance": all(row["absolute_difference"] <= 1e-9 for row in checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=list(MODEL_SEEDS))
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=list(SAMPLING_SEEDS))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--skip-task2-reproduction", action="store_true",
        help="Do not compare against the trajectory-sampling run (required for replacement datasets).",
    )
    args = parser.parse_args()
    datasets = tuple(args.datasets)
    workloads = tuple(args.workloads)
    model_seeds = tuple(args.model_seeds)
    sampling_seeds = tuple(args.sampling_seeds)
    grouped = load_runs(args.output_dir)
    validate(grouped, datasets, workloads, model_seeds, sampling_seeds)

    final_rows = []
    bootstrap_rows = []
    paired_rows = []
    rng = np.random.default_rng(args.bootstrap_seed)
    for dataset in datasets:
        for workload in workloads:
            matrices = {
                estimator: error_matrix(grouped[(dataset, workload, estimator)])
                for estimator in ESTIMATORS
            }
            n_queries = matrices["joint"].shape[1]
            indices = rng.integers(0, n_queries, size=(args.bootstrap_resamples, n_queries))
            bootstrap_values = {}
            aggregate = {}
            for estimator, errors in matrices.items():
                stats = aggregate_metrics(errors)
                aggregate[estimator] = stats
                boot = bootstrap_median(errors, indices)
                bootstrap_values[estimator] = boot
                lower, upper = np.percentile(boot, [2.5, 97.5])
                final_rows.append({
                    "dataset": dataset,
                    "workload": workload,
                    "estimator": estimator,
                    "display_name": DISPLAY_NAMES[estimator],
                    "n_seeds": errors.shape[0],
                    "n_queries": n_queries,
                    **stats,
                    "median_ci95_lower": float(lower),
                    "median_ci95_upper": float(upper),
                })
                bootstrap_rows.append({
                    "dataset": dataset,
                    "workload": workload,
                    "estimator": estimator,
                    "center_median": stats["median"],
                    "ci95_lower": float(lower),
                    "ci95_upper": float(upper),
                    "n_resamples": args.bootstrap_resamples,
                    "bootstrap_seed": args.bootstrap_seed,
                })

            pairs = (
                list(CENTRAL_PAIRS)
                + [("no_coupling", "separable", "central_architecture")]
                + [(left, right, "matched_vs_proxy") for left in MATCHED for right in PROXIES]
            )
            for left, right, comparison_class in pairs:
                left_per_query = np.mean(matrices[left], axis=0)
                right_per_query = np.mean(matrices[right], axis=0)
                wins = int(np.sum(left_per_query < right_per_query))
                ties = int(np.sum(np.isclose(left_per_query, right_per_query, rtol=0, atol=1e-12)))
                losses = n_queries - wins - ties
                ratio_boot = bootstrap_values[left] / bootstrap_values[right]
                difference_boot = bootstrap_values[left] - bootstrap_values[right]
                ratio_lower, ratio_upper = np.percentile(ratio_boot, [2.5, 97.5])
                diff_lower, diff_upper = np.percentile(difference_boot, [2.5, 97.5])
                paired_rows.append({
                    "dataset": dataset,
                    "workload": workload,
                    "comparison_class": comparison_class,
                    "left_estimator": left,
                    "right_estimator": right,
                    "median_ratio_left_over_right": aggregate[left]["median"] / aggregate[right]["median"],
                    "p90_ratio_left_over_right": aggregate[left]["p90"] / aggregate[right]["p90"],
                    "p95_ratio_left_over_right": aggregate[left]["p95"] / aggregate[right]["p95"],
                    "max_ratio_left_over_right": aggregate[left]["max"] / aggregate[right]["max"],
                    "paired_median_ratio_ci95_lower": float(ratio_lower),
                    "paired_median_ratio_ci95_upper": float(ratio_upper),
                    "paired_median_difference_ci95_lower": float(diff_lower),
                    "paired_median_difference_ci95_upper": float(diff_upper),
                    "left_lower_qerror_queries": wins,
                    "tied_queries": ties,
                    "right_lower_qerror_queries": losses,
                    "left_lower_qerror_pct": 100.0 * wins / n_queries,
                    "tied_pct": 100.0 * ties / n_queries,
                    "right_lower_qerror_pct": 100.0 * losses / n_queries,
                })

    zero_rows = zero_cardinality_rows(args.output_dir, datasets, workloads)
    write_csv(args.output_dir / "final_summary.csv", final_rows)
    write_results_json({"rows": final_rows}, args.output_dir / "final_summary.json")
    write_csv(args.output_dir / "bootstrap_intervals.csv", bootstrap_rows)
    write_results_json({"rows": bootstrap_rows}, args.output_dir / "bootstrap_intervals.json")
    write_csv(args.output_dir / "paired_comparisons.csv", paired_rows)
    write_results_json({"rows": paired_rows}, args.output_dir / "paired_comparisons.json")
    write_csv(args.output_dir / "zero_cardinality.csv", zero_rows)
    write_results_json({"rows": zero_rows}, args.output_dir / "zero_cardinality.json")
    can_reproduce_task2 = (
        not args.skip_task2_reproduction
        and {1, 2, 3}.issubset(model_seeds)
        and {17, 23, 47}.issubset(sampling_seeds)
        and all(
            len(json.loads(
                (args.output_dir / "workloads" / f"{dataset}__{workload}.json").read_text()
            )["evaluation_queries"]) == 300
            for dataset in datasets
            for workload in workloads
        )
    )
    reproduction = (
        task2_reproduction(args.output_dir, grouped, datasets, workloads)
        if can_reproduce_task2
        else {
            "skipped": True,
            "reason": "requires full 300-query workloads and seeds 1/2/3 plus 17/23/47",
        }
    )
    write_results_json(reproduction, args.output_dir / "task2_reproduction.json")
    if can_reproduce_task2 and not reproduction["all_within_tolerance"]:
        raise RuntimeError("the battery did not reproduce the comparable three-seed trajectory-sampling results")
    metadata_path = args.output_dir / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    # Per-run checkpoints are written immediately after each estimator finishes and
    # are not rewritten by consolidation, unlike the aggregate CSV/JSON files.
    raw_checkpoints = list((args.output_dir / "runs").glob("*.json"))
    raw_completed = datetime.fromtimestamp(
        max(path.stat().st_mtime for path in raw_checkpoints), timezone.utc
    )
    raw_started = (
        datetime.fromisoformat(metadata["created_utc"])
        if metadata.get("created_utc")
        else datetime.fromtimestamp(
            min(path.stat().st_mtime for path in raw_checkpoints), timezone.utc
        )
    )
    metadata["raw_battery_completed_utc"] = raw_completed.isoformat()
    metadata["raw_battery_wall_seconds"] = (raw_completed - raw_started).total_seconds()
    metadata["analysis"] = {
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_interval": "95% percentile interval over paired query resamples",
        "reported_median": "mean of per-seed medians on a fixed evaluation workload",
        "reported_p90_p95": "mean of per-seed quantiles on a fixed evaluation workload",
        "reported_max": "worst raw q-error across all estimator seeds and queries",
        "paired_query_summary": "compare mean q-error across estimator seeds for each query",
    }
    write_results_json(metadata, metadata_path)
    write_results_json(
        {
            "partial": False,
            "n_run_files": sum(len(runs) for runs in grouped.values()),
            "n_prediction_rows": sum(
                len(run["predictions"]) for runs in grouped.values() for run in runs
            ),
            "n_final_summary_rows": len(final_rows),
            "n_paired_comparison_rows": len(paired_rows),
            "task2_reproduced": reproduction.get("all_within_tolerance"),
        },
        args.output_dir / "manifest.json",
    )
    print(f"analyzed accuracy battery in {args.output_dir}")


if __name__ == "__main__":
    main()
