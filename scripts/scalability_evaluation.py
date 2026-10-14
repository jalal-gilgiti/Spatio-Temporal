"""Checkpointed data-size and training-workload scalability study."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import logging
import os
import pickle
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psycopg
import torch
from omegaconf import OmegaConf
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.load_real_dataset import (  # noqa: E402
    load_nyc_tlc,
    load_porto,
    load_tdrive,
)
from src.data_module.tdrive_sampling import (  # noqa: E402
    TDRIVE_DATASET_SEED,
    TDRIVE_SELECTION_POLICY,
)
from src.estimator_module.coupled_learned.model import CoupledLearnedEstimator  # noqa: E402
from src.estimator_module.native_mobilitydb import NativeMobilityDBEstimator  # noqa: E402
from src.estimator_module.spatial_only_ml import SpatialOnlyMLEstimator  # noqa: E402
from src.estimator_module.trajectory_sampling import TrajectorySamplingEstimator  # noqa: E402
from src.eval_module.scalability import (  # noqa: E402
    PeakRSSMonitor,
    bulk_load_trajectories,
    evaluate_estimator,
    fingerprint,
    generate_predicates,
    label_predicates,
    make_nested_pool,
    normalized_bounds,
    predicate_dict,
    predicate_from_dict,
)
from src.results_io import write_results_json  # noqa: E402
from src.utils.db import get_connection  # noqa: E402


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DSN = "postgresql://postgres@localhost:5434/mobilitydb"
DATA_SAMPLE_SEED = TDRIVE_DATASET_SEED
TRAIN_PREDICATE_SEED = 42
EVAL_PREDICATE_SEED = 43
MODEL_SEEDS = (1, 2, 3)
SAMPLING_SEEDS = (17, 23, 47)
SCALES = (100_000, 250_000, 500_000, 1_000_000)
TRAIN_BUDGETS = (100, 200, 400, 800, 1_600)
N_MAIN_TRAIN = 800
N_EVAL = 300
EPOCHS = 1_000
STATS_TARGET = 1_000
WORKLOADS = ("fixed_extent", "mixed_extent")
DATASETS = {
    "tdrive": {"extent": 0.45, "window_hours": 24.0},
    "porto": {"extent": 0.15, "window_hours": 336.0},
    "nyc_tlc": {"extent": 0.10, "window_hours": 96.0},
}
LEARNED = ("joint", "separable", "no_coupling", "spatial_only")
TABLE = "task5_scaling_traj"
INDEX = "task5_scaling_traj_gist"
PLAN_METADATA_TABLE = "task5_scaling_trip_metadata"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _portable_artifact_path(path: Path) -> str:
    """Prefer repository-relative paths, but support isolated /tmp smoke runs."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(resolved)


def _hardware() -> dict[str, Any]:
    cpu_model = None
    with open("/proc/cpuinfo") as handle:
        for line in handle:
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return {
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "gpu_available_to_torch": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "storage": "workspace on network-mounted filesystem; PostgreSQL data under project .pgdata",
    }


def experiment_metadata(conn: psycopg.Connection, args: argparse.Namespace) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        database_version = cur.fetchone()[0]
        cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY extname")
        extensions = dict(cur.fetchall())
        cur.execute(
            "SELECT name, setting, unit, source FROM pg_settings "
            "WHERE name = ANY(%s) ORDER BY name",
            (["shared_buffers", "effective_cache_size", "work_mem", "random_page_cost",
              "seq_page_cost", "max_parallel_workers_per_gather", "jit"],),
        )
        settings = {
            name: {"setting": setting, "unit": unit, "source": source}
            for name, setting, unit, source in cur.fetchall()
        }
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "psycopg"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "task": 5, "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(), "python": sys.version,
        "platform": platform.platform(), "packages": packages,
        "hardware": _hardware(), "database_version": database_version,
        "extensions": extensions, "planner_settings": settings,
        "datasets": args.datasets, "requested_scales": args.scales,
        "model_seeds": args.model_seeds, "sampling_seeds": args.sampling_seeds,
        "workloads": args.workloads, "n_main_train": args.n_main_train,
        "n_eval": args.n_eval, "training_budgets": args.training_budgets,
        "epochs": args.epochs, "statistics_target": args.statistics_target,
        "data_sample_seed": DATA_SAMPLE_SEED,
        "workload_seeds": {"training": TRAIN_PREDICATE_SEED, "evaluation": EVAL_PREDICATE_SEED},
        "scale_policy": (
            "one fixed maximum-size uniformly sampled pool per dataset; deterministic "
            "whole-trajectory rank prefixes make every smaller scale a strict nested subset"
        ),
        "tdrive_source_selection_policy": TDRIVE_SELECTION_POLICY,
        "cache_policy": "warm process/database cache; no cache eviction",
    }


def prepare_pool(output_dir: Path, dataset: str, max_scale: int) -> tuple[pd.DataFrame, dict]:
    prepared_dir = output_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    path = prepared_dir / f"{dataset}__max_{max_scale}.parquet"
    metadata_path = prepared_dir / f"{dataset}__max_{max_scale}.json"
    if path.exists() and metadata_path.exists():
        return pd.read_parquet(path), json.loads(metadata_path.read_text())

    corrected_tdrive = (
        REPO_ROOT / "run/real_experiments/task7_dataset_validity/prepared/"
        "tdrive_corrected__max_1000000.parquet"
    )
    if dataset == "tdrive" and max_scale <= 1_000_000 and corrected_tdrive.exists():
        started = time.perf_counter()
        pool = pd.read_parquet(corrected_tdrive)
        pool = pool[pool["trajectory_id"] < max_scale].copy()
        pool.to_parquet(path, index=False, compression="zstd")
        bbox, time_range = normalized_bounds(pool)
        payload = {
            "dataset": dataset, "requested_max_scale": max_scale,
            "available_raw_trajectory_count": 1_000_000,
            "availability_basis": "validated corrected randomized T-Drive pool",
            "prepared_trajectory_count": int(pool["trajectory_id"].nunique()),
            "prepared_point_count": len(pool),
            "preparation_seconds": time.perf_counter() - started,
            "sampling_seed": TDRIVE_DATASET_SEED,
            "nested_rank_policy": "validated dense rank prefixes",
            "bbox": list(bbox),
            "time_range": [value.isoformat() for value in time_range],
            "cache_file": _portable_artifact_path(path),
            "cache_bytes": path.stat().st_size,
            "tdrive_source_selection": {
                "selected_source_taxis": 10_047,
                "contributing_source_taxis": 9_992,
                "canonical_source": _portable_artifact_path(corrected_tdrive),
            },
        }
        write_results_json(payload, metadata_path)
        return pool, payload

    cfg = OmegaConf.load(REPO_ROOT / "conf" / "dataset" / f"{dataset}.yaml")
    started = time.perf_counter()
    if dataset == "tdrive":
        raw, source_meta = load_tdrive(
            cfg, max_scale, DATA_SAMPLE_SEED, return_metadata=True
        )
        available = int(raw["trajectory_id"].nunique())
        availability_basis = (
            "post-cleaning, post-segmentation trip segments from a seeded random source-taxi sample"
        )
    elif dataset == "porto":
        raw = load_porto(cfg, max_scale, DATA_SAMPLE_SEED)
        available = 1_710_670
        availability_basis = "official CSV data rows; prepared pool excludes missing/empty/out-of-bbox data"
    else:
        raw = load_nyc_tlc(cfg, max_scale, DATA_SAMPLE_SEED)
        available = 2_964_624
        availability_basis = "Parquet rows before documented date/duration/zone validity filters"
    prepared_count = int(raw["trajectory_id"].nunique())
    if prepared_count < max_scale:
        LOGGER.warning("%s supports only %d prepared trajectories", dataset, prepared_count)
    pool = make_nested_pool(raw, DATA_SAMPLE_SEED)
    del raw
    pool.to_parquet(path, index=False, compression="zstd")
    bbox, time_range = normalized_bounds(pool)
    payload = {
        "dataset": dataset, "requested_max_scale": max_scale,
        "available_raw_trajectory_count": available,
        "availability_basis": availability_basis,
        "prepared_trajectory_count": prepared_count,
        "prepared_point_count": len(pool),
        "preparation_seconds": time.perf_counter() - started,
        "sampling_seed": DATA_SAMPLE_SEED,
        "nested_rank_policy": "one seeded permutation, remapped to dense rank; scale N uses rank < N",
        "bbox": list(bbox), "time_range": [value.isoformat() for value in time_range],
        "cache_file": _portable_artifact_path(path),
        "cache_bytes": path.stat().st_size,
    }
    if dataset == "tdrive":
        payload["tdrive_source_selection"] = source_meta
    write_results_json(payload, metadata_path)
    return pool, payload


def load_or_create_predicates(
    output_dir: Path,
    dataset: str,
    pool_meta: dict[str, Any],
    spec: dict[str, Any],
    workload: str,
    n_training: int,
    n_eval: int,
) -> dict[str, Any]:
    path = output_dir / "predicates" / f"{dataset}__{workload}.json"
    if path.exists():
        return json.loads(path.read_text())
    bbox = tuple(pool_meta["bbox"])
    time_range = tuple(datetime.fromisoformat(value) for value in pool_meta["time_range"])
    training = generate_predicates(
        workload, n_training, TRAIN_PREDICATE_SEED, bbox, time_range,
        spec["extent"], spec["window_hours"],
    )
    evaluation = generate_predicates(
        workload, n_eval, EVAL_PREDICATE_SEED, bbox, time_range,
        spec["extent"], spec["window_hours"],
    )
    payload = {
        "dataset": dataset, "workload": workload,
        "training_seed": TRAIN_PREDICATE_SEED, "evaluation_seed": EVAL_PREDICATE_SEED,
        "bbox": list(bbox), "time_range": pool_meta["time_range"],
        "n_training": len(training), "n_evaluation": len(evaluation),
        "training_predicates": [predicate_dict(value) for value in training],
        "evaluation_predicates": [predicate_dict(value) for value in evaluation],
    }
    payload["training_predicate_id"] = fingerprint(payload["training_predicates"])
    payload["evaluation_predicate_id"] = fingerprint(payload["evaluation_predicates"])
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return payload


def build_scale_table(
    conn: psycopg.Connection,
    pool: pd.DataFrame,
    dataset: str,
    scale: int,
    statistics_target: int,
) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(TABLE)))
        cur.execute(sql.SQL("CREATE TABLE {} (trajectory_id int, traj tgeompoint)").format(
            sql.Identifier(TABLE)
        ))
    subset = pool[pool["trajectory_id"] < scale]
    started = time.perf_counter()
    inserted = bulk_load_trajectories(conn, TABLE, subset)
    load_seconds = time.perf_counter() - started
    if inserted != min(scale, int(pool["trajectory_id"].nunique())):
        raise RuntimeError(f"{dataset}/{scale}: inserted {inserted} trajectories")
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE INDEX {} ON {} USING GIST (traj)").format(
            sql.Identifier(INDEX), sql.Identifier(TABLE)
        ))
    index_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql.SQL(
            "ALTER TABLE {} ALTER COLUMN traj SET STATISTICS {}"
        ).format(sql.Identifier(TABLE), sql.Literal(statistics_target)))
        cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(TABLE)))
    analyze_seconds = time.perf_counter() - started
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_relation_size(%s::regclass), pg_indexes_size(%s::regclass), "
            "pg_total_relation_size(%s::regclass)", (TABLE, TABLE, TABLE),
        )
        table_bytes, index_bytes, total_bytes = map(int, cur.fetchone())
    return {
        "dataset": dataset, "scale": inserted, "table": TABLE,
        "index": INDEX, "statistics_target": statistics_target,
        "point_rows": len(subset), "load_seconds": load_seconds,
        "index_build_seconds": index_seconds, "analyze_seconds": analyze_seconds,
        "table_bytes": table_bytes, "index_bytes": index_bytes,
        "total_bytes": total_bytes,
    }


def _query_dict(query) -> dict[str, Any]:
    return {
        "predicate": predicate_dict(query.predicate),
        "true_cardinality": int(query.true_cardinality),
    }


def _queries_from_label_payload(payload: dict[str, Any]):
    from src.data_module.workload_generator import WorkloadQuery

    return [
        WorkloadQuery(predicate_from_dict(row["predicate"]), int(row["true_cardinality"]))
        for row in payload["queries"]
    ]


def load_or_label(
    conn: psycopg.Connection,
    output_dir: Path,
    dataset: str,
    scale: int,
    workload: str,
    split: str,
    predicates: list,
) -> tuple[list, dict[str, Any]]:
    """Label a frozen predicate prefix once and retain every query latency."""
    path = output_dir / "labels" / f"{dataset}__n{scale}__{workload}__{split}.json"
    expected_id = fingerprint([predicate_dict(value) for value in predicates])
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["predicate_id"] != expected_id:
            raise RuntimeError(f"predicate mismatch in {path}")
        return _queries_from_label_payload(payload), payload

    queries, latencies = label_predicates(conn, TABLE, predicates)
    payload = {
        "dataset": dataset,
        "scale": scale,
        "workload": workload,
        "split": split,
        "predicate_id": expected_id,
        "n_queries": len(queries),
        "total_label_seconds": float(sum(latencies) / 1_000.0),
        "mean_label_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_label_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "per_query_label_ms": latencies,
        "queries": [_query_dict(query) for query in queries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return queries, payload


def learned_factory(name: str, seed: int, epochs: int):
    if name == "joint":
        return CoupledLearnedEstimator(
            seed=seed, epochs=epochs, use_coupling=True,
            coupling_kind="joint", include_interactions=True,
        )
    if name == "separable":
        return CoupledLearnedEstimator(
            seed=seed, epochs=epochs, use_coupling=True,
            coupling_kind="separable", include_interactions=False,
        )
    if name == "no_coupling":
        return CoupledLearnedEstimator(
            seed=seed, epochs=epochs, use_coupling=False,
            coupling_kind="joint", include_interactions=False,
        )
    if name == "spatial_only":
        return SpatialOnlyMLEstimator(seed=seed)
    raise ValueError(f"unknown learned estimator {name!r}")


def _model_parameter_bytes(estimator: object) -> int:
    model = getattr(estimator, "model", None)
    if hasattr(model, "parameters"):
        return int(sum(value.numel() * value.element_size() for value in model.parameters()))
    model = getattr(estimator, "model", None)
    if hasattr(model, "estimators_"):
        # scikit-learn's Python object graph is best represented by its artifact size.
        return 0
    return 0


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def learned_run_path(
    output_dir: Path, dataset: str, scale: int, workload: str,
    budget: int, estimator: str, seed: int,
) -> Path:
    return output_dir / "runs" / "learned" / (
        f"{dataset}__n{scale}__{workload}__b{budget}__{estimator}__s{seed}.json"
    )


def model_path_for(run_path: Path) -> Path:
    return run_path.parents[2] / "models" / f"{run_path.stem}.pkl"


def run_learned_estimator(
    conn: psycopg.Connection,
    output_dir: Path,
    dataset: str,
    scale: int,
    workload: str,
    budget: int,
    estimator_name: str,
    seed: int,
    epochs: int,
    train_queries: list,
    eval_queries: list,
) -> dict[str, Any]:
    path = learned_run_path(
        output_dir, dataset, scale, workload, budget, estimator_name, seed
    )
    if path.exists():
        return json.loads(path.read_text())

    estimator = learned_factory(estimator_name, seed, epochs)
    started = time.perf_counter()
    with PeakRSSMonitor() as memory:
        estimator.fit(conn, TABLE, train_queries=train_queries[:budget])
    training_seconds = time.perf_counter() - started
    predictions, summary = evaluate_estimator(estimator, eval_queries)
    artifact = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    model_path = model_path_for(path)
    _atomic_write_bytes(model_path, artifact)
    payload = {
        "kind": "learned",
        "dataset": dataset,
        "scale": scale,
        "workload": workload,
        "training_budget": budget,
        "estimator": estimator_name,
        "seed": seed,
        "epochs": epochs,
        "training_seconds": training_seconds,
        "peak_rss_start_bytes": memory.result.start_rss_bytes,
        "peak_rss_bytes": memory.result.peak_rss_bytes,
        "peak_training_rss_delta_bytes": memory.result.delta_bytes,
        "serialized_bytes": len(artifact),
        "parameter_bytes": _model_parameter_bytes(estimator),
        "model_artifact": _portable_artifact_path(model_path),
        "summary": summary,
        "predictions": predictions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return payload


def sampler_run_path(
    output_dir: Path, dataset: str, scale: int, workload: str,
    policy: str, seed: int,
) -> Path:
    return output_dir / "runs" / "sampling" / (
        f"{dataset}__n{scale}__{workload}__{policy}__s{seed}.json"
    )


def run_trajectory_sampler(
    conn: psycopg.Connection,
    output_dir: Path,
    dataset: str,
    scale: int,
    workload: str,
    policy: str,
    seed: int,
    eval_queries: list,
) -> dict[str, Any]:
    path = sampler_run_path(output_dir, dataset, scale, workload, policy, seed)
    if path.exists():
        return json.loads(path.read_text())
    if policy == "fraction_1pct":
        estimator = TrajectorySamplingEstimator(sample_fraction=0.01, seed=seed)
    elif policy == "count_1000":
        estimator = TrajectorySamplingEstimator(sample_count=1_000, seed=seed)
    else:
        raise ValueError(f"unknown sampling policy {policy!r}")
    started = time.perf_counter()
    with PeakRSSMonitor() as memory:
        estimator.fit(conn, TABLE)
    build_seconds = time.perf_counter() - started
    try:
        predictions, summary = evaluate_estimator(estimator, eval_queries)
        payload = {
            "kind": "trajectory_sampling",
            "dataset": dataset,
            "scale": scale,
            "workload": workload,
            "estimator": f"trajectory_sampling_{policy}",
            "sampling_policy": policy,
            "seed": seed,
            "sample_rows": estimator.n_sample,
            "sample_fraction_actual": (
                estimator.n_sample / estimator.n_total if estimator.n_total else 0.0
            ),
            "sample_build_seconds": build_seconds,
            "sample_storage_bytes": estimator.storage_bytes,
            "peak_build_rss_delta_bytes": memory.result.delta_bytes,
            "summary": summary,
            "predictions": predictions,
        }
    finally:
        estimator.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return payload


def native_run_path(output_dir: Path, dataset: str, scale: int, workload: str) -> Path:
    return output_dir / "runs" / "native" / f"{dataset}__n{scale}__{workload}.json"


def run_native_estimator(
    conn: psycopg.Connection,
    output_dir: Path,
    dataset: str,
    scale: int,
    workload: str,
    eval_queries: list,
) -> dict[str, Any]:
    path = native_run_path(output_dir, dataset, scale, workload)
    if path.exists():
        return json.loads(path.read_text())
    estimator = NativeMobilityDBEstimator()
    estimator.fit(conn, TABLE)
    predictions, summary = evaluate_estimator(estimator, eval_queries)
    payload = {
        "kind": "native",
        "dataset": dataset,
        "scale": scale,
        "workload": workload,
        "estimator": "native",
        "seed": None,
        "summary": summary,
        "predictions": predictions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return payload


def _prefix_label_cost(label_payload: dict[str, Any], budget: int) -> dict[str, float]:
    values = label_payload["per_query_label_ms"][:budget]
    return {
        "n_queries": len(values),
        "seconds": float(sum(values) / 1_000.0),
        "seconds_per_query": float(sum(values) / 1_000.0 / len(values)) if values else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def consolidate(output_dir: Path) -> dict[str, Any]:
    run_paths = sorted((output_dir / "runs").glob("*/*.json"))
    runs = [json.loads(path.read_text()) for path in run_paths]
    rows = []
    for run in runs:
        row = {
            "kind": run["kind"],
            "dataset": run["dataset"],
            "scale": run["scale"],
            "workload": run["workload"],
            "estimator": run["estimator"],
            "seed": run.get("seed"),
            "training_budget": run.get("training_budget"),
            **run["summary"],
            "training_seconds": run.get("training_seconds"),
            "sample_build_seconds": run.get("sample_build_seconds"),
            "storage_bytes": run.get("serialized_bytes", run.get("sample_storage_bytes")),
            "peak_rss_delta_bytes": run.get(
                "peak_training_rss_delta_bytes", run.get("peak_build_rss_delta_bytes")
            ),
        }
        rows.append(row)
    write_csv(output_dir / "raw_run_summary.csv", rows)
    payload = {"n_runs": len(runs), "rows": rows}
    write_results_json(payload, output_dir / "raw_run_summary.json")
    return payload


def _expected_scale_paths(
    output_dir: Path,
    dataset: str,
    scale: int,
    workloads: tuple[str, ...],
    model_seeds: tuple[int, ...],
    sampling_seeds: tuple[int, ...],
    n_main_train: int,
    training_budgets: tuple[int, ...],
    largest_scale: int,
) -> list[Path]:
    expected: list[Path] = []
    budgets = {n_main_train}
    if scale == largest_scale:
        budgets.update(training_budgets)
    for workload in workloads:
        expected.extend([
            output_dir / "labels" / f"{dataset}__n{scale}__{workload}__training.json",
            output_dir / "labels" / f"{dataset}__n{scale}__{workload}__evaluation.json",
            native_run_path(output_dir, dataset, scale, workload),
        ])
        for budget in budgets:
            for estimator in LEARNED:
                for seed in model_seeds:
                    expected.append(learned_run_path(
                        output_dir, dataset, scale, workload, budget, estimator, seed
                    ))
        for policy in ("fraction_1pct", "count_1000"):
            for seed in sampling_seeds:
                expected.append(sampler_run_path(
                    output_dir, dataset, scale, workload, policy, seed
                ))
    return expected


def run_scale(
    conn: psycopg.Connection,
    output_dir: Path,
    dataset: str,
    pool: pd.DataFrame,
    pool_meta: dict[str, Any],
    scale: int,
    largest_scale: int,
    args: argparse.Namespace,
) -> None:
    expected = _expected_scale_paths(
        output_dir, dataset, scale, args.workloads, args.model_seeds,
        args.sampling_seeds, args.n_main_train, args.training_budgets, largest_scale,
    )
    if all(path.exists() for path in expected):
        LOGGER.info("%s/%d already complete", dataset, scale)
        return

    LOGGER.info("%s/%d: loading scale table", dataset, scale)
    table_stats = build_scale_table(conn, pool, dataset, scale, args.statistics_target)
    table_path = output_dir / "table_stats" / f"{dataset}__n{scale}.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if not table_path.exists():
        write_results_json(table_stats, table_path)

    for workload in args.workloads:
        max_budget = max(args.training_budgets) if scale == largest_scale else args.n_main_train
        predicate_payload = load_or_create_predicates(
            output_dir, dataset, pool_meta, DATASETS[dataset], workload,
            max(max(args.training_budgets), args.n_main_train), args.n_eval,
        )
        training_predicates = [
            predicate_from_dict(value) for value in predicate_payload["training_predicates"]
        ]
        evaluation_predicates = [
            predicate_from_dict(value) for value in predicate_payload["evaluation_predicates"]
        ]
        train, label_meta = load_or_label(
            conn, output_dir, dataset, scale, workload, "training",
            training_predicates[:max_budget],
        )
        evaluation, _ = load_or_label(
            conn, output_dir, dataset, scale, workload, "evaluation", evaluation_predicates
        )

        budgets = {args.n_main_train}
        if scale == largest_scale:
            budgets.update(args.training_budgets)
        for budget in sorted(budgets):
            if budget > len(train):
                raise RuntimeError(
                    f"{dataset}/{scale}/{workload}: budget {budget} exceeds {len(train)} labels"
                )
            for name in LEARNED:
                for seed in args.model_seeds:
                    run = run_learned_estimator(
                        conn, output_dir, dataset, scale, workload, budget, name, seed,
                        args.epochs, train, evaluation,
                    )
                    LOGGER.info(
                        "%s/%d/%s/b%d/%s/s%d median=%.4f p95=%.4f",
                        dataset, scale, workload, budget, name, seed,
                        run["summary"]["median_q_error"], run["summary"]["p95_q_error"],
                    )
        label_cost_path = output_dir / "label_costs" / f"{dataset}__n{scale}__{workload}.json"
        label_cost_path.parent.mkdir(parents=True, exist_ok=True)
        costs = {
            str(budget): _prefix_label_cost(label_meta, budget)
            for budget in sorted(budgets)
        }
        write_results_json({
            "dataset": dataset, "scale": scale, "workload": workload,
            "training_label_costs": costs,
        }, label_cost_path)

        for policy in ("fraction_1pct", "count_1000"):
            for seed in args.sampling_seeds:
                run = run_trajectory_sampler(
                    conn, output_dir, dataset, scale, workload, policy, seed, evaluation
                )
                LOGGER.info(
                    "%s/%d/%s/%s/s%d median=%.4f p95=%.4f",
                    dataset, scale, workload, policy, seed,
                    run["summary"]["median_q_error"], run["summary"]["p95_q_error"],
                )
        run_native_estimator(conn, output_dir, dataset, scale, workload, evaluation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "run/real_experiments/scalability",
    )
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--scales", nargs="+", type=int, default=list(SCALES))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=list(MODEL_SEEDS))
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=list(SAMPLING_SEEDS))
    parser.add_argument("--n-main-train", type=int, default=N_MAIN_TRAIN)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--training-budgets", nargs="+", type=int, default=list(TRAIN_BUDGETS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--statistics-target", type=int, default=STATS_TARGET)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.datasets = tuple(args.datasets)
    args.scales = tuple(sorted(set(args.scales)))
    args.workloads = tuple(args.workloads)
    args.model_seeds = tuple(args.model_seeds)
    args.sampling_seeds = tuple(args.sampling_seeds)
    args.training_budgets = tuple(sorted(set(args.training_budgets)))
    if args.n_main_train > max(args.training_budgets):
        raise ValueError("n_main_train must not exceed the maximum training budget")
    if min(args.scales) <= 0 or min(args.training_budgets) <= 0:
        raise ValueError("scales and training budgets must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(args.dsn)
    try:
        metadata_path = args.output_dir / "experiment_metadata.json"
        if not metadata_path.exists():
            write_results_json(experiment_metadata(conn, args), metadata_path)
        for dataset in args.datasets:
            LOGGER.info("%s: preparing maximum nested pool", dataset)
            pool, pool_meta = prepare_pool(args.output_dir, dataset, max(args.scales))
            available = int(pool_meta["prepared_trajectory_count"])
            actual_scales = tuple(sorted({min(value, available) for value in args.scales}))
            if not actual_scales or actual_scales[0] <= 0:
                raise RuntimeError(f"{dataset}: no valid prepared trajectories")
            scale_manifest = {
                "dataset": dataset,
                "requested_scales": list(args.scales),
                "actual_scales": list(actual_scales),
                "prepared_trajectory_count": available,
                "shortfall": available < max(args.scales),
            }
            manifest_path = args.output_dir / "prepared" / f"{dataset}__scales.json"
            write_results_json(scale_manifest, manifest_path)
            largest = max(actual_scales)
            for scale in actual_scales:
                run_scale(
                    conn, args.output_dir, dataset, pool, pool_meta,
                    scale, largest, args,
                )
            del pool
        consolidate(args.output_dir)
    finally:
        conn.close()
    print(f"wrote raw scalability battery to {args.output_dir}")


if __name__ == "__main__":
    main()
