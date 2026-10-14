"""Analyze plan-sensitive measurements with paired-query uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval_module.accuracy import q_error  # noqa: E402
from src.results_io import write_results_json  # noqa: E402


DATASETS = ("tdrive", "porto", "nyc_tlc")
CONDITIONS = (
    "native", "joint", "separable", "no_coupling", "spatial_only",
    "trajectory_sampling", "true_cardinality_input",
)
ORACLE_CONDITIONS = (
    "true_cardinality_input", "oracle_nested", "oracle_hash", "oracle_merge",
)
BAD_THRESHOLDS = (1.10, 1.25, 2.0)
BOOTSTRAP_SEED = 20260912
BOOTSTRAP_RESAMPLES = 10_000
JIT_ABOVE_COST = 100_000.0
JIT_AVAILABLE = False


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _mode(values: list[str]) -> str:
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), percentile))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def _execution_fingerprint(plan_tree_fingerprint: str, jit_enabled: bool) -> str:
    """Fingerprint the saved physical tree plus PostgreSQL's JIT decision."""
    payload = f"{plan_tree_fingerprint}:jit={int(jit_enabled)}"
    return hashlib.sha256(payload.encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_repeats(condition: dict[str, Any]) -> dict[str, Any]:
    repeats = condition["repeats"]
    if not repeats:
        raise ValueError("condition has no measured repeats")
    fingerprints = [row["fingerprint"] for row in repeats]
    plan_tree_fingerprint = _mode(fingerprints)
    matching = [row for row in repeats if row["fingerprint"] == plan_tree_fingerprint]
    # The saved PostgreSQL 17 conda build has no LLVM support. New measurements store
    # the observed EXPLAIN flag directly; older rows therefore correctly fall
    # back to false instead of inferring JIT from planner cost alone.
    jit_enabled = bool(_mode([
        str(bool(row.get("jit_enabled", False))) for row in matching
    ]) == "True")
    fingerprint = _execution_fingerprint(plan_tree_fingerprint, jit_enabled)
    inference_ms = float(condition.get("inference_ms", 0.0))
    planning_ms = _median([float(row["planning_ms"]) for row in repeats])
    execution_ms = _median([float(row["execution_ms"]) for row in repeats])
    source_planned_rows = _median([float(row["source_planned_rows"]) for row in repeats])
    buffer_keys = sorted({key for row in repeats for key in row.get("buffers", {})})
    return {
        "fingerprint": fingerprint,
        "plan_tree_fingerprint": plan_tree_fingerprint,
        "jit_enabled": jit_enabled,
        "fingerprint_stability": len(matching) / len(repeats),
        "plan_family": _mode([row["plan_family"] for row in matching])
        + (" + JIT" if jit_enabled else ""),
        "source_scan": _mode([row["source_scan"] for row in matching]),
        "metadata_join": _mode([row["metadata_join"] for row in matching]),
        "join_outer": _mode([row["join_outer"] for row in matching]),
        "source_planned_rows": source_planned_rows,
        "total_cost": _median([float(row["total_cost"]) for row in repeats]),
        "planning_ms": planning_ms,
        "execution_ms": execution_ms,
        "inference_ms": inference_ms,
        "end_to_end_ms": inference_ms + planning_ms + execution_ms,
        "buffers": {
            key: _median([float(row.get("buffers", {}).get(key, 0)) for row in repeats])
            for key in buffer_keys
        },
    }


def pooled_plan_runtime_summary(
    measurement: dict[str, Any], summaries: dict[str, dict[str, Any]]
) -> tuple[dict[str, float], str, str]:
    """Pool repeats by executable plan and return runtimes and empirical oracle."""
    samples: dict[str, list[float]] = {}
    for condition in measurement["conditions"].values():
        for repeat in condition["repeats"]:
            fingerprint = _execution_fingerprint(
                repeat["fingerprint"], bool(repeat.get("jit_enabled", False))
            )
            samples.setdefault(fingerprint, []).append(float(repeat["execution_ms"]))
    runtimes = {fingerprint: _median(values) for fingerprint, values in samples.items()}
    oracle_fingerprint = min(
        runtimes, key=lambda fingerprint: (runtimes[fingerprint], fingerprint)
    )
    oracle_condition = min(
        name for name, summary in summaries.items()
        if summary["fingerprint"] == oracle_fingerprint
    )
    return runtimes, oracle_fingerprint, oracle_condition


def load_measurements(output_dir: Path, datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    measurements = []
    for dataset in datasets:
        paths = sorted((output_dir / "measurements").glob(f"{dataset}__*.json"))
        if len(paths) != 100:
            raise RuntimeError(f"expected 100 {dataset} measurements, found {len(paths)}")
        for path in paths:
            payload = json.loads(path.read_text())
            observed = set(payload["conditions"])
            expected = set(CONDITIONS) | set(ORACLE_CONDITIONS)
            if observed != expected:
                raise RuntimeError(
                    f"condition mismatch in {path}: missing={expected-observed}, extra={observed-expected}"
                )
            repeat_counts = {len(value["repeats"]) for value in payload["conditions"].values()}
            if len(repeat_counts) != 1:
                raise RuntimeError(f"unpaired repeat counts in {path}: {repeat_counts}")
            measurements.append(payload)
    ids = [row["query_id"] for row in measurements]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate plan-study query IDs")
    return measurements


def per_query_rows(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for measurement in measurements:
        summaries = {
            name: summarize_repeats(value)
            for name, value in measurement["conditions"].items()
        }
        plan_runtimes, oracle_fingerprint, oracle_name = pooled_plan_runtime_summary(
            measurement, summaries
        )
        oracle = summaries[oracle_name]
        oracle_execution_ms = plan_runtimes[oracle_fingerprint]
        native = summaries["native"]
        truth = float(measurement["true_cardinality"])
        for name in CONDITIONS:
            summary = summaries[name]
            condition = measurement["conditions"][name]
            prediction = condition.get("predicted_cardinality")
            if name == "native":
                prediction = summary["source_planned_rows"]
            qerr = float(q_error(truth, float(prediction)))
            rows.append({
                "dataset": measurement["dataset"],
                "query_id": measurement["query_id"],
                "profile": measurement["profile"],
                "boundary_position": measurement["boundary_position"],
                "true_cardinality": truth,
                "joined_true_cardinality": measurement["joined_true_cardinality"],
                "boundary_lower": measurement["boundary"]["transition_lower"],
                "boundary_upper": measurement["boundary"]["transition_upper"],
                "boundary_geomean": measurement["boundary"]["boundary_geomean"],
                "condition": name,
                "predicted_cardinality": float(prediction),
                "q_error": qerr,
                "fingerprint": summary["fingerprint"],
                "plan_tree_fingerprint": summary["plan_tree_fingerprint"],
                "jit_enabled": summary["jit_enabled"],
                "plan_family": summary["plan_family"],
                "source_scan": summary["source_scan"],
                "metadata_join": summary["metadata_join"],
                "join_outer": summary["join_outer"],
                "fingerprint_stability": summary["fingerprint_stability"],
                "planner_cost": summary["total_cost"],
                "planning_ms": summary["planning_ms"],
                "inference_ms": summary["inference_ms"],
                "database_predicate_ms": condition.get("database_predicate_ms"),
                "execution_ms": summary["execution_ms"],
                "end_to_end_ms": summary["end_to_end_ms"],
                "plan_changed_from_native": summary["fingerprint"] != native["fingerprint"],
                "oracle_condition": oracle_name,
                "oracle_fingerprint": oracle_fingerprint,
                "oracle_plan_family": oracle["plan_family"],
                "oracle_execution_ms": oracle_execution_ms,
                "oracle_agreement": summary["fingerprint"] == oracle_fingerprint,
                "plan_runtime_ms": plan_runtimes[summary["fingerprint"]],
                "regret": _safe_ratio(
                    plan_runtimes[summary["fingerprint"]], oracle_execution_ms
                ),
                "execution_speedup_vs_native": _safe_ratio(
                    native["execution_ms"], summary["execution_ms"]
                ),
                "end_to_end_speedup_vs_native": _safe_ratio(
                    native["end_to_end_ms"], summary["end_to_end_ms"]
                ),
                **{
                    "buffer_" + key.lower().replace(" ", "_"): value
                    for key, value in summary["buffers"].items()
                },
            })
    return rows


def aggregate_rows(rows: list[dict[str, Any]], datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    output = []
    for dataset in (*datasets, "overall"):
        for condition in CONDITIONS:
            chosen = [
                row for row in rows
                if row["condition"] == condition
                and (dataset == "overall" or row["dataset"] == dataset)
            ]
            regrets = [row["regret"] for row in chosen]
            qerrors = [row["q_error"] for row in chosen]
            exec_speedups = [row["execution_speedup_vs_native"] for row in chosen]
            e2e_speedups = [row["end_to_end_speedup_vs_native"] for row in chosen]
            execution = [row["execution_ms"] for row in chosen]
            end_to_end = [row["end_to_end_ms"] for row in chosen]
            output.append({
                "dataset": dataset,
                "condition": condition,
                "n_queries": len(chosen),
                "plan_change_rate": float(np.mean([row["plan_changed_from_native"] for row in chosen])),
                "oracle_agreement_rate": float(np.mean([row["oracle_agreement"] for row in chosen])),
                "regret_median": _median(regrets),
                "regret_p90": _percentile(regrets, 90),
                "regret_p95": _percentile(regrets, 95),
                "regret_max": max(regrets),
                "bad_plan_rate_gt_1_10": float(np.mean(np.asarray(regrets) > 1.10)),
                "bad_plan_rate_gt_1_25": float(np.mean(np.asarray(regrets) > 1.25)),
                "bad_plan_rate_gt_2": float(np.mean(np.asarray(regrets) > 2.0)),
                "q_error_median": _median(qerrors),
                "q_error_p95": _percentile(qerrors, 95),
                "q_error_max": max(qerrors),
                "execution_ms_median": _median(execution),
                "end_to_end_ms_median": _median(end_to_end),
                "execution_speedup_median": _median(exec_speedups),
                "end_to_end_speedup_median": _median(e2e_speedups),
                "ratio_of_median_execution_speedup": _safe_ratio(
                    _median([row["execution_ms"] for row in chosen if row["condition"] == "native"])
                    if condition == "native" else
                    _median([
                        native["execution_ms"]
                        for row in chosen
                        for native in rows
                        if native["query_id"] == row["query_id"] and native["condition"] == "native"
                    ]),
                    _median(execution),
                ),
            })
    return output


def bootstrap_rows(
    rows: list[dict[str, Any]], datasets: tuple[str, ...], n_resamples: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output = []
    metrics: dict[str, Callable[[np.ndarray], float]] = {
        "regret_median": lambda values: float(np.median(values)),
        "oracle_agreement_rate": lambda values: float(np.mean(values)),
        "execution_speedup_median": lambda values: float(np.median(values)),
        "end_to_end_speedup_median": lambda values: float(np.median(values)),
    }
    columns = {
        "regret_median": "regret",
        "oracle_agreement_rate": "oracle_agreement",
        "execution_speedup_median": "execution_speedup_vs_native",
        "end_to_end_speedup_median": "end_to_end_speedup_vs_native",
    }
    for dataset in (*datasets, "overall"):
        query_ids = sorted({
            row["query_id"] for row in rows
            if dataset == "overall" or row["dataset"] == dataset
        })
        indices = rng.integers(0, len(query_ids), size=(n_resamples, len(query_ids)))
        for condition in CONDITIONS:
            by_id = {
                row["query_id"]: row for row in rows
                if row["condition"] == condition
                and (dataset == "overall" or row["dataset"] == dataset)
            }
            for metric, function in metrics.items():
                values = np.asarray([by_id[query_id][columns[metric]] for query_id in query_ids], dtype=float)
                estimates = np.asarray([function(values[index]) for index in indices], dtype=float)
                lower, upper = np.percentile(estimates, [2.5, 97.5])
                output.append({
                    "dataset": dataset, "condition": condition, "metric": metric,
                    "point_estimate": function(values), "ci_lower": float(lower),
                    "ci_upper": float(upper), "n_queries": len(query_ids),
                    "n_resamples": n_resamples, "bootstrap_seed": seed,
                })
    return output


def relationship_rows(rows: list[dict[str, Any]], datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    output = []
    for dataset in (*datasets, "overall"):
        for condition in CONDITIONS:
            chosen = [
                row for row in rows if row["condition"] == condition
                and (dataset == "overall" or row["dataset"] == dataset)
            ]
            qerrors = np.asarray([row["q_error"] for row in chosen], dtype=float)
            regrets = np.asarray([row["regret"] for row in chosen], dtype=float)
            rho, pvalue = spearmanr(np.log10(qerrors), np.log10(regrets))
            output.append({
                "dataset": dataset, "condition": condition, "n_queries": len(chosen),
                "spearman_rho_log_qerror_log_regret": None if math.isnan(rho) else float(rho),
                "spearman_pvalue": None if math.isnan(pvalue) else float(pvalue),
                "high_qerror_low_regret_n": int(np.sum((qerrors > 10) & (regrets <= 1.10))),
                "low_qerror_high_regret_n": int(np.sum((qerrors <= 2) & (regrets > 1.25))),
            })
    return output


def boundary_examples(rows: list[dict[str, Any]], datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    examples = []
    for dataset in datasets:
        query_ids = sorted({row["query_id"] for row in rows if row["dataset"] == dataset})
        candidates = []
        for query_id in query_ids:
            query_rows = [row for row in rows if row["query_id"] == query_id]
            changed = sum(row["plan_changed_from_native"] for row in query_rows)
            best_gain = max(row["execution_speedup_vs_native"] for row in query_rows)
            proximity = abs(math.log10(
                (query_rows[0]["true_cardinality"] + 1.0)
                / (query_rows[0]["boundary_geomean"] + 1.0)
            ))
            candidates.append((changed > 0, best_gain, -proximity, query_id, query_rows))
        for _, _, _, query_id, query_rows in sorted(candidates, reverse=True)[:3]:
            examples.append({
                "dataset": dataset, "query_id": query_id,
                "true_cardinality": query_rows[0]["true_cardinality"],
                "boundary": [query_rows[0]["boundary_lower"], query_rows[0]["boundary_upper"]],
                "conditions": {
                    row["condition"]: {
                        "prediction": row["predicted_cardinality"],
                        "q_error": row["q_error"], "plan": row["plan_family"],
                        "execution_ms": row["execution_ms"], "regret": row["regret"],
                    }
                    for row in query_rows
                },
            })
    return examples


def overhead_summary(output_dir: Path, datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        payload = json.loads((output_dir / "predictions" / f"{dataset}.json").read_text())
        for condition, details in payload["overhead"].items():
            if condition == "trajectory_sampling":
                samples = details["samples"]
                rows.append({
                    "dataset": dataset, "condition": condition,
                    "fit_or_build_ms_median": _median([row["sample_build_ms"] for row in samples]),
                    "cold_load_ms_median": None,
                    "artifact_bytes_total": sum(row["storage_bytes"] for row in samples),
                    "memory_bytes_total": sum(row["storage_bytes"] for row in samples),
                    "cached_batch_size": None,
                    "cached_batch_inference_ms_median": None,
                    "cached_batch_per_query_ms_median": None,
                    "policy": details["deployment_policy"],
                })
            else:
                models = details["models"]
                fit_values = [row["fit_ms"] for row in models if row.get("fit_ms") is not None]
                rows.append({
                    "dataset": dataset, "condition": condition,
                    "fit_or_build_ms_median": _median(fit_values) if fit_values else None,
                    "cold_load_ms_median": _median([row["cold_deserialize_ms"] for row in models]),
                    "artifact_bytes_total": sum(row["serialized_bytes"] for row in models),
                    "memory_bytes_total": sum(row["torch_parameter_bytes"] for row in models),
                    "cached_batch_size": models[0].get("batch_size"),
                    "cached_batch_inference_ms_median": (
                        _median([row["cached_batch_inference_ms"] for row in models])
                        if "cached_batch_inference_ms" in models[0] else None
                    ),
                    "cached_batch_per_query_ms_median": (
                        _median([row["cached_batch_per_query_ms"] for row in models])
                        if "cached_batch_per_query_ms" in models[0] else None
                    ),
                    "policy": details["deployment_policy"],
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    datasets = tuple(args.datasets)

    measurements = load_measurements(args.output_dir, datasets)
    query_rows = per_query_rows(measurements)
    aggregates = aggregate_rows(query_rows, datasets)
    bootstraps = bootstrap_rows(
        query_rows, datasets, args.bootstrap_resamples, args.bootstrap_seed
    )
    relationships = relationship_rows(query_rows, datasets)
    examples = boundary_examples(query_rows, datasets)
    overhead = overhead_summary(args.output_dir, datasets)

    write_csv(args.output_dir / "per_query_plan_metrics.csv", query_rows)
    write_csv(args.output_dir / "plan_metrics_summary.csv", aggregates)
    write_csv(args.output_dir / "plan_bootstrap_intervals.csv", bootstraps)
    write_csv(args.output_dir / "qerror_regret_relationship.csv", relationships)
    write_csv(args.output_dir / "serving_overhead.csv", overhead)
    write_results_json({"examples": examples}, args.output_dir / "boundary_examples.json")
    write_results_json({
        "datasets": list(datasets), "conditions": list(CONDITIONS),
        "oracle_candidates": list(ORACLE_CONDITIONS),
        "oracle_definition": (
            "fastest pooled-median execution among every distinct valid measured "
            "physical-plan tree plus JIT-decision fingerprint"
        ),
        "jit_fingerprint": {
            "jit": "on", "jit_above_cost": JIT_ABOVE_COST,
            "jit_available": JIT_AVAILABLE,
            "source": (
                "settings audited with postgres -C; pg_config lacks --with-llvm and "
                "manual EXPLAIN reruns contain no JIT block"
            ),
        },
        "repeat_policy": "one warm-up then five measured randomized warm-cache rounds",
        "bootstrap": {
            "method": "paired query percentile bootstrap", "seed": args.bootstrap_seed,
            "resamples": args.bootstrap_resamples, "confidence": 0.95,
        },
        "aggregate": aggregates, "bootstrap_intervals": bootstraps,
        "qerror_regret_relationship": relationships, "serving_overhead": overhead,
        "boundary_examples": examples,
    }, args.output_dir / "analysis.json")
    print(f"analyzed {len(measurements)} queries into {args.output_dir}")


if __name__ == "__main__":
    main()
