"""Evaluate exact-predicate trajectory sampling on saved-workload seeds.

The predefined headline budget is 1% of each 100,000-row trajectory table. Headline
sampling seeds are 17, 23, and 47. A fraction curve at 0.1%, 0.5%, 1%, and 5% uses
seed 17. Both mixed- and fixed-extent evaluation workloads are reconstructed with
seed 43, matching the existing seed-42 experiments' ``seed + 1`` convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_coupling_advantage import fixed_extent_workload
from src.data_module.workload_generator import STPredicate, WorkloadQuery, generate_workload
from src.estimator_module.trajectory_sampling import TrajectorySamplingEstimator
from src.eval_module.accuracy import q_error
from src.real_experiment_steps import fetch_table_workload_bounds
from src.results_io import write_results_json
from src.utils.db import get_connection


LOGGER = logging.getLogger(__name__)
DSN = "postgresql://postgres@localhost:5434/mobilitydb"
WORKLOAD_SEED = 43
N_EVAL = 300
FRACTIONS = (0.001, 0.005, 0.01, 0.05)
HEADLINE_FRACTION = 0.01
CURVE_SEED = 17
HEADLINE_SEEDS = (17, 23, 47)
DATASETS = {
    "tdrive": {"table": "tdrive_traj", "extent": 0.45, "window_hours": 24.0},
    "porto": {"table": "porto_traj", "extent": 0.15, "window_hours": 336.0},
    "nyc_tlc": {"table": "nyc_tlc_traj", "extent": 0.10, "window_hours": 96.0},
}


def predicate_to_dict(predicate: STPredicate) -> dict:
    return {
        "min_lon": predicate.min_lon,
        "max_lon": predicate.max_lon,
        "min_lat": predicate.min_lat,
        "max_lat": predicate.max_lat,
        "start_time": predicate.start_time.isoformat(),
        "end_time": predicate.end_time.isoformat(),
        "k": predicate.k,
        "knn_center": predicate.knn_center,
    }


def query_to_dict(query: WorkloadQuery) -> dict:
    return {
        "predicate": predicate_to_dict(query.predicate),
        "true_cardinality": query.true_cardinality,
    }


def query_from_dict(payload: dict) -> WorkloadQuery:
    raw = payload["predicate"]
    predicate = STPredicate(
        min_lon=raw["min_lon"],
        max_lon=raw["max_lon"],
        min_lat=raw["min_lat"],
        max_lat=raw["max_lat"],
        start_time=datetime.fromisoformat(raw["start_time"]),
        end_time=datetime.fromisoformat(raw["end_time"]),
        k=raw.get("k"),
        knn_center=tuple(raw["knn_center"]) if raw.get("knn_center") else None,
    )
    return WorkloadQuery(predicate, int(payload["true_cardinality"]))


def workload_payload(queries: list[WorkloadQuery], kind: str, spec: dict) -> dict:
    serialized = [query_to_dict(query) for query in queries]
    fingerprint = hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "kind": kind,
        "seed": WORKLOAD_SEED,
        "n_queries": len(queries),
        "extent": spec.get("extent") if kind == "fixed_extent" else None,
        "window_hours": spec.get("window_hours") if kind == "fixed_extent" else None,
        "sha256": fingerprint,
        "queries": serialized,
    }


def build_workloads(conn, table: str, spec: dict) -> dict[str, dict]:
    bbox, time_range = fetch_table_workload_bounds(conn, table)
    mixed = generate_workload(
        conn,
        table,
        n_queries=N_EVAL,
        seed=WORKLOAD_SEED,
        bbox=bbox,
        knn_fraction=0.0,
        time_range=time_range,
    )
    fixed = fixed_extent_workload(
        conn,
        table,
        n_queries=N_EVAL,
        seed=WORKLOAD_SEED,
        extent=spec["extent"],
        window_hours=spec["window_hours"],
        bbox=bbox,
        time_range=time_range,
    )
    return {
        "mixed_extent": workload_payload(mixed, "mixed_extent", spec),
        "fixed_extent": workload_payload(fixed, "fixed_extent", spec),
    }


def summarize(raw: list[dict]) -> dict:
    errors = np.array([row["q_error"] for row in raw], dtype=float)
    latencies = np.array([row["latency_ms"] for row in raw], dtype=float)
    return {
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p90_latency_ms": float(np.percentile(latencies, 90)),
    }


def evaluate_workload(
    estimator: TrajectorySamplingEstimator, workload: dict
) -> dict:
    queries = [query_from_dict(payload) for payload in workload["queries"]]
    raw = []
    for index, query in enumerate(queries):
        details = estimator.estimate_with_details(query.predicate)
        raw.append(
            {
                "query_index": index,
                "true_cardinality": query.true_cardinality,
                "sample_hits": details.sample_hits,
                "estimated_cardinality": details.estimated_cardinality,
                "q_error": q_error(query.true_cardinality, details.estimated_cardinality),
                "latency_ms": details.latency_ms,
            }
        )
    return {"summary": summarize(raw), "per_query": raw}


def experiment_configs() -> list[tuple[float, int, str]]:
    configs = [(fraction, CURVE_SEED, "fraction_curve") for fraction in FRACTIONS]
    configs.extend(
        (HEADLINE_FRACTION, seed, "headline_repeat")
        for seed in HEADLINE_SEEDS
        if seed != CURVE_SEED
    )
    return configs


def checkpoint(payload: dict, output: Path) -> None:
    payload["wall_seconds"] = round(time.time() - payload["started_epoch"], 1)
    write_results_json(payload, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.out.exists():
        payload = json.loads(args.out.read_text())
        payload["started_epoch"] = time.time() - float(payload.get("wall_seconds", 0.0))
        LOGGER.info("resuming %s", args.out)
    else:
        payload = {
            "schema_version": 1,
            "partial": True,
            "started_epoch": time.time(),
            "definition": "N * h / n; h uses the same atStbox full-path predicate as ground truth",
            "run_config": {
                "workload_seed": WORKLOAD_SEED,
                "n_eval": N_EVAL,
                "fractions": list(FRACTIONS),
                "headline_fraction": HEADLINE_FRACTION,
                "curve_seed": CURVE_SEED,
                "headline_sample_seeds": list(HEADLINE_SEEDS),
                "datasets": DATASETS,
            },
            "datasets": {},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(args.dsn)
    try:
        for dataset in args.datasets:
            spec = DATASETS[dataset]
            dataset_result = payload["datasets"].setdefault(
                dataset, {"table": spec["table"], "workloads": {}, "samples": {}}
            )
            if not dataset_result["workloads"]:
                LOGGER.info("%s: reconstructing paired evaluation workloads", dataset)
                dataset_result["workloads"] = build_workloads(conn, spec["table"], spec)
                checkpoint(payload, args.out)

            for fraction, seed, role in experiment_configs():
                key = f"fraction={fraction:.3f};seed={seed}"
                if key in dataset_result["samples"]:
                    continue
                LOGGER.info("%s: fitting trajectory sample %s", dataset, key)
                estimator = TrajectorySamplingEstimator(sample_fraction=fraction, seed=seed)
                estimator.fit(conn, spec["table"])
                try:
                    sample_result = {
                        "sample_fraction": fraction,
                        "sample_seed": seed,
                        "role": role,
                        "n_total": estimator.n_total,
                        "n_sample": estimator.n_sample,
                        "storage_bytes": estimator.storage_bytes,
                        "sample_ids": estimator.sample_ids,
                        "workloads": {},
                    }
                    for workload_name, workload in dataset_result["workloads"].items():
                        LOGGER.info("%s %s: evaluating %s", dataset, key, workload_name)
                        sample_result["workloads"][workload_name] = evaluate_workload(
                            estimator, workload
                        )
                    dataset_result["samples"][key] = sample_result
                    checkpoint(payload, args.out)
                finally:
                    estimator.close()
        payload["partial"] = any(
            len(payload["datasets"].get(dataset, {}).get("samples", {}))
            < len(experiment_configs())
            for dataset in args.datasets
        )
        checkpoint(payload, args.out)
    finally:
        conn.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
