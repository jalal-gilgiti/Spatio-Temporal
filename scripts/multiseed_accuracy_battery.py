"""Reproducible, paired multi-seed headline accuracy battery.

Evaluation queries are read verbatim from the trajectory-sampling run's saved workloads.
Training queries use the existing seed-42 workload construction and are generated once,
serialized, and then reused by every trainable estimator.  Each completed estimator
configuration is atomically checkpointed under ``<output>/runs`` so interrupted runs
resume without changing workloads or repeating completed training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import psycopg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_coupling_advantage import fixed_extent_workload  # noqa: E402
from src.data_module.workload_generator import (  # noqa: E402
    STPredicate,
    WorkloadQuery,
    generate_workload,
)
from src.estimator_module.coupled_learned.model import CoupledLearnedEstimator  # noqa: E402
from src.estimator_module.histograms import (  # noqa: E402
    IndependentHistogramEstimator,
    JointHistogramEstimator,
)
from src.estimator_module.minskew_origin import MinSkewOriginHistogramEstimator  # noqa: E402
from src.estimator_module.native_mobilitydb import NativeMobilityDBEstimator  # noqa: E402
from src.estimator_module.sampling import SamplingEstimator  # noqa: E402
from src.estimator_module.spatial_only_ml import SpatialOnlyMLEstimator  # noqa: E402
from src.estimator_module.trajectory_sampling import TrajectorySamplingEstimator  # noqa: E402
from src.eval_module.accuracy import q_error  # noqa: E402
from src.real_experiment_steps import fetch_table_workload_bounds  # noqa: E402
from src.results_io import write_results_json  # noqa: E402
from src.utils.db import get_connection  # noqa: E402


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DSN = "postgresql://postgres@localhost:5434/mobilitydb"
TASK2_ARTIFACT = REPO_ROOT / "run/real_experiments/trajectory_sampling.json"
MODEL_SEEDS = (1, 2, 3, 4, 5)
SAMPLING_SEEDS = (17, 23, 47, 59, 71)
TRAIN_SEED = 42
N_TRAIN = 800
N_EVAL = 300
EPOCHS = 1000
DATASETS = {
    "tdrive": {"table": "tdrive_traj", "extent": 0.45, "window_hours": 24.0},
    "porto": {"table": "porto_traj", "extent": 0.15, "window_hours": 336.0},
    "nyc_tlc": {"table": "nyc_tlc_traj", "extent": 0.10, "window_hours": 96.0},
}
WORKLOADS = ("fixed_extent", "mixed_extent")
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
LEARNED = ("joint", "separable", "no_coupling", "spatial_only")


def predicate_dict(predicate: STPredicate) -> dict:
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


def query_dict(query: WorkloadQuery) -> dict:
    return {"predicate": predicate_dict(query.predicate), "true_cardinality": query.true_cardinality}


def query_from_dict(payload: dict) -> WorkloadQuery:
    raw = payload["predicate"]
    predicate = STPredicate(
        min_lon=float(raw["min_lon"]),
        max_lon=float(raw["max_lon"]),
        min_lat=float(raw["min_lat"]),
        max_lat=float(raw["max_lat"]),
        start_time=datetime.fromisoformat(raw["start_time"]),
        end_time=datetime.fromisoformat(raw["end_time"]),
        k=raw.get("k"),
        knn_center=tuple(raw["knn_center"]) if raw.get("knn_center") else None,
    )
    return WorkloadQuery(predicate, int(payload["true_cardinality"]))


def fingerprint(serialized: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_or_create_workload(
    conn: psycopg.Connection,
    output_dir: Path,
    task2: dict,
    dataset: str,
    workload: str,
    spec: dict,
    n_train: int,
    n_eval: int,
) -> tuple[list[WorkloadQuery], list[WorkloadQuery], dict]:
    path = output_dir / "workloads" / f"{dataset}__{workload}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        train = [query_from_dict(row) for row in payload["training_queries"]]
        evaluation = [query_from_dict(row) for row in payload["evaluation_queries"]]
        return train, evaluation, payload

    task2_workload = task2["datasets"][dataset]["workloads"][workload]
    evaluation_rows = task2_workload["queries"][:n_eval]
    if n_eval == task2_workload["n_queries"]:
        evaluation_id = task2_workload["sha256"]
    else:
        evaluation_id = fingerprint(evaluation_rows)
    evaluation = [query_from_dict(row) for row in evaluation_rows]

    bbox, time_range = fetch_table_workload_bounds(conn, spec["table"])
    if workload == "fixed_extent":
        train = fixed_extent_workload(
            conn,
            spec["table"],
            n_queries=n_train,
            seed=TRAIN_SEED,
            extent=spec["extent"],
            window_hours=spec["window_hours"],
            bbox=bbox,
            time_range=time_range,
        )
    else:
        train = generate_workload(
            conn,
            spec["table"],
            n_queries=n_train,
            seed=TRAIN_SEED,
            bbox=bbox,
            knn_fraction=0.0,
            time_range=time_range,
        )
    training_rows = [query_dict(query) for query in train]
    payload = {
        "dataset": dataset,
        "table": spec["table"],
        "workload": workload,
        "training_seed": TRAIN_SEED,
        "training_workload_id": fingerprint(training_rows),
        "evaluation_seed": task2_workload["seed"],
        "evaluation_workload_id": evaluation_id,
        "task2_evaluation_workload_id": task2_workload["sha256"],
        "n_train": len(train),
        "n_eval": len(evaluation),
        "bbox": list(bbox),
        "time_range": [value.isoformat() for value in time_range],
        "training_queries": training_rows,
        "evaluation_queries": evaluation_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return train, evaluation, payload


def estimator_configs(
    model_seeds: tuple[int, ...], sampling_seeds: tuple[int, ...], epochs: int
):
    factories: dict[str, Callable[[int], object]] = {
        "joint": lambda seed: CoupledLearnedEstimator(
            seed=seed,
            epochs=epochs,
            use_coupling=True,
            coupling_kind="joint",
            include_interactions=True,
        ),
        "separable": lambda seed: CoupledLearnedEstimator(
            seed=seed,
            epochs=epochs,
            use_coupling=True,
            coupling_kind="separable",
            include_interactions=False,
        ),
        "no_coupling": lambda seed: CoupledLearnedEstimator(
            seed=seed,
            epochs=epochs,
            use_coupling=False,
            coupling_kind="joint",
            include_interactions=False,
        ),
        "spatial_only": lambda seed: SpatialOnlyMLEstimator(seed=seed),
        "origin_sampling": lambda seed: SamplingEstimator(sample_rate=0.1, seed=seed),
        "trajectory_sampling": lambda seed: TrajectorySamplingEstimator(
            sample_fraction=0.01, seed=seed
        ),
    }
    for name in LEARNED:
        for seed in model_seeds:
            yield name, seed, factories[name]
    for seed in sampling_seeds:
        yield "trajectory_sampling", seed, factories["trajectory_sampling"]
    for seed in model_seeds:
        yield "origin_sampling", seed, factories["origin_sampling"]
    deterministic = {
        "minskew_origin": lambda _: MinSkewOriginHistogramEstimator(),
        "independent_histogram": lambda _: IndependentHistogramEstimator(),
        "joint_histogram": lambda _: JointHistogramEstimator(),
        "native": lambda _: NativeMobilityDBEstimator(),
    }
    for name, factory in deterministic.items():
        yield name, None, factory


def summarize(predictions: list[dict]) -> dict:
    errors = np.array([row["q_error"] for row in predictions], dtype=float)
    return {
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
        "mean_tail_sensitive": float(np.mean(errors)),
    }


def run_estimator(
    conn: psycopg.Connection,
    table: str,
    train: list[WorkloadQuery],
    evaluation: list[WorkloadQuery],
    workload_meta: dict,
    dataset: str,
    workload: str,
    estimator_name: str,
    estimator_seed: int | None,
    factory: Callable[[int], object],
) -> dict:
    estimator = factory(0 if estimator_seed is None else estimator_seed)
    started = time.perf_counter()
    estimator.fit(conn, table, train_queries=train)
    fit_seconds = time.perf_counter() - started
    predictions = []
    try:
        for index, query in enumerate(evaluation):
            predicted = float(estimator.estimate(query.predicate))
            error = float(q_error(query.true_cardinality, predicted))
            if not np.isfinite(predicted) or not np.isfinite(error):
                raise RuntimeError(
                    f"non-finite result: {dataset}/{workload}/{estimator_name}/"
                    f"{estimator_seed}/query-{index}"
                )
            predictions.append(
                {
                    "query_id": f"{dataset}:{workload}:{index:03d}",
                    "query_index": index,
                    "true_cardinality": int(query.true_cardinality),
                    "predicted_cardinality": predicted,
                    "q_error": error,
                }
            )
    finally:
        close = getattr(estimator, "close", None)
        if callable(close):
            close()
    return {
        "dataset": dataset,
        "table": table,
        "workload": workload,
        "workload_id": workload_meta["evaluation_workload_id"],
        "training_workload_id": workload_meta["training_workload_id"],
        "estimator": estimator_name,
        "display_name": DISPLAY_NAMES[estimator_name],
        "estimator_seed": estimator_seed,
        "fit_seconds": fit_seconds,
        "summary": summarize(predictions),
        "predictions": predictions,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def consolidate(output_dir: Path) -> None:
    runs = [json.loads(path.read_text()) for path in sorted((output_dir / "runs").glob("*.json"))]
    prediction_rows = []
    summary_rows = []
    for run in runs:
        base = {
            key: run[key]
            for key in ("dataset", "workload", "workload_id", "estimator", "estimator_seed")
        }
        for row in run["predictions"]:
            prediction_rows.append({**base, **row})
        summary_rows.append(
            {
                **base,
                "n_queries": len(run["predictions"]),
                "fit_seconds": run["fit_seconds"],
                **run["summary"],
            }
        )
    write_csv(
        output_dir / "predictions.csv",
        [
            "dataset", "workload", "workload_id", "query_id", "query_index",
            "estimator", "estimator_seed", "true_cardinality",
            "predicted_cardinality", "q_error",
        ],
        prediction_rows,
    )
    write_results_json({"rows": prediction_rows}, output_dir / "predictions.json")
    write_csv(
        output_dir / "per_query_qerrors.csv",
        [
            "dataset", "workload", "workload_id", "query_id", "query_index",
            "estimator", "estimator_seed", "true_cardinality", "q_error",
        ],
        [
            {key: row[key] for key in (
                "dataset", "workload", "workload_id", "query_id", "query_index",
                "estimator", "estimator_seed", "true_cardinality", "q_error",
            )}
            for row in prediction_rows
        ],
    )
    write_results_json(
        {
            "rows": [
                {key: row[key] for key in (
                    "dataset", "workload", "workload_id", "query_id", "query_index",
                    "estimator", "estimator_seed", "true_cardinality", "q_error",
                )}
                for row in prediction_rows
            ]
        },
        output_dir / "per_query_qerrors.json",
    )
    write_csv(output_dir / "seed_summary.csv", list(summary_rows[0]), summary_rows)
    write_results_json({"rows": summary_rows}, output_dir / "seed_summary.json")


def metadata(conn: psycopg.Connection, args: argparse.Namespace, task2: dict) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        database_version = cur.fetchone()[0]
        cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY extname")
        extensions = {name: version for name, version in cur.fetchall()}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        commit = None
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "psycopg", "pandas"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "task": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_worktree_dirty": bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "database_version": database_version,
        "database_extensions": extensions,
        "datasets": DATASETS,
        "workloads": list(args.workloads),
        "task2_artifact": str(TASK2_ARTIFACT.relative_to(REPO_ROOT)),
        "task2_workload_ids": {
            dataset: {
                workload: task2["datasets"][dataset]["workloads"][workload]["sha256"]
                for workload in args.workloads
            }
            for dataset in args.datasets
        },
        "model_seeds": list(args.model_seeds),
        "trajectory_sampling_seeds": list(args.sampling_seeds),
        "training_seed": TRAIN_SEED,
        "n_train": args.n_train,
        "n_eval": args.n_eval,
        "model_hyperparameters": {
            "joint": {"epochs": args.epochs, "lr": 0.01, "coupling_kind": "joint", "include_interactions": True},
            "separable": {"epochs": args.epochs, "lr": 0.01, "coupling_kind": "separable", "include_interactions": False},
            "no_coupling": {"epochs": args.epochs, "lr": 0.01, "use_coupling": False, "include_interactions": False},
            "spatial_only": {
                **SpatialOnlyMLEstimator(seed=0).model.get_params(),
                "random_state": "estimator_seed",
            },
            "trajectory_sampling": {"sample_fraction": 0.01},
            "origin_sampling": {"sample_rate": 0.1},
            "minskew_origin": {"n_buckets": 64},
            "independent_histogram": {"n_bins": 20},
            "joint_histogram": {"n_bins_per_dimension": 10},
        },
        "q_error": "max((true+1)/(estimate+1), (estimate+1)/(true+1)); negative values floored at zero",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=list(MODEL_SEEDS))
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=list(SAMPLING_SEEDS))
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.model_seeds = tuple(args.model_seeds)
    args.sampling_seeds = tuple(args.sampling_seeds)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    task2 = json.loads(TASK2_ARTIFACT.read_text())
    if task2.get("partial"):
        raise RuntimeError("trajectory-sampling artifact is partial")

    conn = get_connection(args.dsn)
    try:
        write_results_json(metadata(conn, args, task2), args.output_dir / "experiment_metadata.json")
        for dataset in args.datasets:
            spec = DATASETS[dataset]
            for workload in args.workloads:
                LOGGER.info("%s/%s: loading fixed paired workloads", dataset, workload)
                train, evaluation, workload_meta = load_or_create_workload(
                    conn, args.output_dir, task2, dataset, workload, spec,
                    args.n_train, args.n_eval,
                )
                for name, seed, factory in estimator_configs(
                    args.model_seeds, args.sampling_seeds, args.epochs
                ):
                    seed_label = "deterministic" if seed is None else str(seed)
                    run_path = args.output_dir / "runs" / f"{dataset}__{workload}__{name}__{seed_label}.json"
                    if run_path.exists():
                        continue
                    LOGGER.info("%s/%s: fitting %s seed=%s", dataset, workload, name, seed_label)
                    run = run_estimator(
                        conn, spec["table"], train, evaluation, workload_meta,
                        dataset, workload, name, seed, factory,
                    )
                    write_results_json(run, run_path)
                    LOGGER.info(
                        "%s/%s/%s/%s median=%.4f p95=%.4f max=%.4f",
                        dataset, workload, name, seed_label,
                        run["summary"]["median"], run["summary"]["p95"],
                        run["summary"]["max"],
                    )
        consolidate(args.output_dir)
    finally:
        conn.close()
    print(f"wrote raw accuracy battery to {args.output_dir}")


if __name__ == "__main__":
    main()
