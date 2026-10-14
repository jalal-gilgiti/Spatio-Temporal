"""Plan-sensitive PostgreSQL/MobilityDB experiment.

This creates only the plan-study GiST indexes and deterministic trip-metadata dimensions.  It
does not alter source rows or the accuracy-battery workloads.  The normal estimator conditions use
pg_hint_plan cardinality injection because an in-process MobilityDB selectivity hook
would require a compiled PostgreSQL/MobilityDB extension; metadata states that limit
explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import pickle
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psycopg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multiseed_accuracy_battery import predicate_dict, query_from_dict  # noqa: E402
from src.data_module.workload_generator import STPredicate, WorkloadQuery  # noqa: E402
from src.estimator_module.coupled_learned.model import CoupledLearnedEstimator  # noqa: E402
from src.estimator_module.spatial_only_ml import SpatialOnlyMLEstimator  # noqa: E402
from src.estimator_module.trajectory_sampling import TrajectorySamplingEstimator  # noqa: E402
from src.eval_module.accuracy import q_error  # noqa: E402
from src.eval_module.plan_sensitive import (  # noqa: E402
    build_plan_sensitive_query,
    exact_indexable_predicate,
    normalized_plan_tree,
    plan_summary,
    root_buffer_counts,
)
from src.results_io import write_results_json  # noqa: E402
from src.real_experiment_steps import fetch_table_workload_bounds  # noqa: E402
from src.utils.db import get_connection  # noqa: E402


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DSN = "postgresql://postgres@localhost:5434/mobilitydb"
TASK3_DIR = REPO_ROOT / "run/real_experiments/task7b/headline"
DATASETS = {
    "tdrive": {"table": "tdrive_traj", "extent": 0.45, "window_hours": 24.0},
    "porto": {"table": "porto_traj", "extent": 0.15, "window_hours": 336.0},
    "nyc_tlc": {"table": "nyc_tlc_traj", "extent": 0.10, "window_hours": 96.0},
}
MODEL_SEEDS = (1, 2, 3, 4, 5)
SAMPLING_SEEDS = (17, 23, 47, 59, 71)
ESTIMATE_GRID = (1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000)
PROFILES = {
    "highly_selective": (0.02, 0.20),
    "selective": (0.15, 0.70),
    "boundary_scale": (0.50, 2.00),
    "moderate": (1.50, 5.00),
    "broad": (4.00, 20.00),
}
CONDITIONS = (
    "native", "joint", "separable", "no_coupling", "spatial_only",
    "trajectory_sampling", "true_cardinality_input",
)
ORACLE_SETTINGS = {
    "oracle_nested": {
        "enable_nestloop": "on", "enable_hashjoin": "off", "enable_mergejoin": "off",
    },
    "oracle_hash": {
        "enable_nestloop": "off", "enable_hashjoin": "on", "enable_mergejoin": "off",
    },
    "oracle_merge": {
        "enable_nestloop": "off", "enable_hashjoin": "off", "enable_mergejoin": "on",
    },
}


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def prepare_database(conn: psycopg.Connection, datasets: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    """Create reproducible indexes and plausible derived trip dimensions."""
    objects = {}
    with conn.cursor() as cur:
        for dataset in (datasets or tuple(DATASETS)):
            spec = DATASETS[dataset]
            table = spec["table"]
            gist = f"task4_{dataset}_traj_gist"
            metadata = f"task4_{dataset}_trip_metadata"
            metadata_pk = f"{metadata}_pk"
            cur.execute(f"CREATE INDEX IF NOT EXISTS {gist} ON {table} USING GIST (traj)")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {metadata} AS "
                f"SELECT DISTINCT trajectory_id, "
                f"(abs(hashint4(trajectory_id)) % 16)::smallint AS operator_id, "
                f"((abs(hashint4(trajectory_id + 7919)) % 20) <> 0) AS active "
                f"FROM {table}"
            )
            cur.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {metadata_pk} "
                f"ON {metadata}(trajectory_id)"
            )
            cur.execute(f"ANALYZE {table}")
            cur.execute(f"ANALYZE {metadata}")
            cur.execute(f"SELECT count(*) FROM {table}")
            n_rows = int(cur.fetchone()[0])
            cur.execute(f"SELECT count(*) FROM {metadata}")
            n_metadata = int(cur.fetchone()[0])
            if n_metadata != n_rows:
                raise RuntimeError(f"{metadata} has {n_metadata} rows for {n_rows} source rows")
            objects[dataset] = {
                "table": table, "row_count": n_rows, "gist_index": gist,
                "metadata_table": metadata, "metadata_index": metadata_pk,
                "metadata_semantics": (
                    "one deterministic row per trip; operator_id and active are derived "
                    "dimension attributes, not observed source fields"
                ),
            }
    return objects


def database_metadata(conn: psycopg.Connection, objects: dict[str, Any]) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY extname")
        extensions = {name: value for name, value in cur.fetchall()}
        settings = (
            "random_page_cost", "seq_page_cost", "cpu_tuple_cost",
            "cpu_index_tuple_cost", "effective_cache_size", "shared_buffers",
            "work_mem", "enable_seqscan", "enable_indexscan", "enable_bitmapscan",
            "enable_nestloop", "enable_hashjoin", "enable_mergejoin",
            "max_parallel_workers_per_gather", "jit", "jit_above_cost",
            "jit_inline_above_cost", "jit_optimize_above_cost",
        )
        cur.execute(
            "SELECT name, setting, unit, source FROM pg_settings WHERE name = ANY(%s) "
            "ORDER BY name", (list(settings),)
        )
        planner_settings = {
            name: {"setting": setting, "unit": unit, "source": source}
            for name, setting, unit, source in cur.fetchall()
        }
        for dataset, obj in objects.items():
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE schemaname=current_schema() AND tablename = ANY(%s) ORDER BY indexname",
                ([obj["table"], obj["metadata_table"]],),
            )
            obj["indexes"] = [dict(zip(("name", "definition"), row)) for row in cur.fetchall()]
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "psycopg"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "task": 4,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "database_version": version,
        "extensions": extensions,
        "planner_settings": planner_settings,
        "database_objects": objects,
        "native_selectivity_hook": {
            "implemented": False,
            "reason": (
                "PostgreSQL restriction selectivity callbacks are compiled extension hooks; "
                "the Python estimators cannot be registered as MobilityDB operator selectivity "
                "functions within the scope of this study. pg_hint_plan Rows injection is used "
                "on a downstream metadata join and is not described as native integration."
            ),
        },
        "cache_policy": (
            "warm shared/OS cache; no cache eviction; one warm-up per condition followed by "
            "randomized measured rounds"
        ),
    }


def _random_predicate(
    rng: np.random.Generator,
    bbox: tuple[float, float, float, float],
    time_range: tuple[datetime, datetime],
    base_extent: float,
    base_hours: float,
    profile: str,
) -> STPredicate:
    low, high = PROFILES[profile]
    spatial_multiplier = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    temporal_multiplier = float(np.exp(rng.uniform(np.log(low), np.log(high))))
    min_lon, max_lon, min_lat, max_lat = bbox
    width = min(max_lon - min_lon, base_extent * spatial_multiplier)
    height = min(max_lat - min_lat, base_extent * spatial_multiplier)
    lon0 = float(rng.uniform(min_lon, max_lon - width)) if width < max_lon - min_lon else min_lon
    lat0 = float(rng.uniform(min_lat, max_lat - height)) if height < max_lat - min_lat else min_lat
    full_seconds = (time_range[1] - time_range[0]).total_seconds()
    seconds = min(full_seconds, base_hours * 3600.0 * temporal_multiplier)
    offset = float(rng.uniform(0, full_seconds - seconds)) if seconds < full_seconds else 0.0
    start = time_range[0] + timedelta(seconds=offset)
    return STPredicate(lon0, lon0 + width, lat0, lat0 + height, start, start + timedelta(seconds=seconds))


def _truth(conn: psycopg.Connection, table: str, metadata: str, predicate: STPredicate) -> tuple[int, int]:
    query = build_plan_sensitive_query(table, metadata, predicate, None)
    with conn.cursor() as cur:
        # ST truth is kept separate from the active-metadata join output.
        cur.execute(
            f"SELECT count(*) FROM {table} t WHERE ({exact_indexable_predicate(predicate)})"
        )
        st_count = int(cur.fetchone()[0])
        cur.execute(query)
        joined_count = int(cur.fetchone()[0])
    return st_count, joined_count


def _explain(
    conn: psycopg.Connection,
    query: str,
    source_table: str,
    metadata_table: str,
    *,
    analyze: bool,
    settings: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with conn.transaction():
        with conn.cursor() as cur:
            for name, value in (settings or {}).items():
                cur.execute(f"SET LOCAL {name} = {value}")
            options = "ANALYZE, BUFFERS, TIMING OFF, FORMAT JSON" if analyze else "FORMAT JSON"
            cur.execute(f"EXPLAIN ({options}) {query}")
            plan = cur.fetchone()[0][0]
    return plan, plan_summary(plan, source_table, metadata_table)


def _boundary_probe(
    conn: psycopg.Connection, table: str, metadata: str, predicate: STPredicate
) -> dict[str, Any] | None:
    probes = []
    for estimate in ESTIMATE_GRID:
        query = build_plan_sensitive_query(table, metadata, predicate, estimate)
        _, summary = _explain(conn, query, table, metadata, analyze=False)
        probes.append({"estimate": estimate, **summary})
    joins = [row["metadata_join"] for row in probes]
    first = joins[0]
    change = next((i for i, value in enumerate(joins[1:], 1) if value != first), None)
    if change is None:
        return None
    threshold = math.sqrt(ESTIMATE_GRID[change - 1] * ESTIMATE_GRID[change])
    return {
        "low_estimate_join": first,
        "high_estimate_join": joins[change],
        "transition_lower": ESTIMATE_GRID[change - 1],
        "transition_upper": ESTIMATE_GRID[change],
        "boundary_geomean": threshold,
        "probes": probes,
    }


def _select_candidates(
    candidates: list[dict[str, Any]], n_queries: int
) -> list[dict[str, Any]]:
    """Select unique predicates across scale profiles and boundary positions."""
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    target_per_profile = n_queries // len(PROFILES)

    def take(rows: list[dict[str, Any]], limit: int) -> None:
        for row in rows:
            key = _sha256(row["predicate"])
            if key in used:
                continue
            selected.append(dict(row))
            used.add(key)
            if limit and sum(
                candidate["profile"] == row["profile"] for candidate in selected
            ) >= limit:
                break

    for profile in PROFILES:
        available = sorted(
            (row for row in candidates if row["profile"] == profile),
            key=lambda row: math.log10(row["true_to_boundary_ratio"]),
        )
        low_count = min(target_per_profile // 3, len(available))
        take(available[:low_count], target_per_profile)
        remaining = [row for row in available if _sha256(row["predicate"]) not in used]
        high_count = min(target_per_profile // 3, len(remaining))
        take(remaining[-high_count:] if high_count else [], target_per_profile)
        remaining = [row for row in available if _sha256(row["predicate"]) not in used]
        remaining.sort(key=lambda row: abs(math.log10(row["true_to_boundary_ratio"])))
        take(remaining, target_per_profile)

    remaining = [row for row in candidates if _sha256(row["predicate"]) not in used]
    remaining.sort(key=lambda row: abs(math.log10(row["true_to_boundary_ratio"])))
    for row in remaining:
        if len(selected) >= n_queries:
            break
        selected.append(dict(row))
        used.add(_sha256(row["predicate"]))
    if len(selected) < n_queries:
        raise RuntimeError(f"only {len(selected)} unique plan-sensitive candidates")
    return selected[:n_queries]


def build_workload(
    conn: psycopg.Connection,
    output_dir: Path,
    dataset: str,
    spec: dict[str, Any],
    objects: dict[str, Any],
    n_candidates: int,
    n_queries: int,
    seed: int,
) -> dict[str, Any]:
    path = output_dir / "workloads" / f"{dataset}.json"
    if path.exists():
        return json.loads(path.read_text())
    task3 = json.loads((TASK3_DIR / "workloads" / f"{dataset}__mixed_extent.json").read_text())
    if "bbox" in task3 and "time_range" in task3:
        bbox = tuple(float(value) for value in task3["bbox"])
        time_range = tuple(datetime.fromisoformat(value) for value in task3["time_range"])
    else:
        bbox, time_range = fetch_table_workload_bounds(conn, spec["table"])
    rng = np.random.default_rng(seed)
    candidates = []
    profiles = tuple(PROFILES)
    per_profile = math.ceil(n_candidates / len(profiles))
    for profile in profiles:
        for _ in range(per_profile):
            predicate = _random_predicate(
                rng, bbox, time_range, spec["extent"], spec["window_hours"], profile
            )
            truth, joined_truth = _truth(
                conn, objects["table"], objects["metadata_table"], predicate
            )
            boundary = _boundary_probe(
                conn, objects["table"], objects["metadata_table"], predicate
            )
            if boundary is None:
                continue
            ratio = (truth + 1.0) / (boundary["boundary_geomean"] + 1.0)
            candidates.append({
                "profile": profile,
                "predicate": predicate_dict(predicate),
                "true_cardinality": truth,
                "joined_true_cardinality": joined_truth,
                "true_to_boundary_ratio": ratio,
                "boundary_position": (
                    "below" if ratio < 0.5 else "above" if ratio > 2.0 else "near"
                ),
                "boundary": boundary,
            })
    if len(candidates) < n_queries:
        raise RuntimeError(f"only {len(candidates)} plan-sensitive candidates for {dataset}")

    selected = _select_candidates(candidates, n_queries)
    # Stable query IDs are assigned after selection, never used to select candidates.
    for index, row in enumerate(selected):
        row["query_id"] = f"{dataset}:plan-sensitive:{index:03d}"
    payload = {
        "dataset": dataset,
        "seed": seed,
        "selection_rule": (
            "up to 20 unique predicates per predefined geometry/time scale profile, "
            "balancing lowest, closest, and highest true-cardinality/boundary ratios, "
            "then filling scarce profiles with the closest unused candidates; estimator "
            "predictions unused"
        ),
        "n_candidates_requested": n_candidates,
        "n_plan_sensitive_candidates": len(candidates),
        "n_queries": len(selected),
        "workload_id": _sha256([
            {"predicate": row["predicate"], "truth": row["true_cardinality"]}
            for row in selected
        ]),
        "queries": selected,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return payload


def _model_factories(epochs: int) -> dict[str, Callable[[int], Any]]:
    return {
        "joint": lambda seed: CoupledLearnedEstimator(
            seed=seed, epochs=epochs, use_coupling=True,
            coupling_kind="joint", include_interactions=True,
        ),
        "separable": lambda seed: CoupledLearnedEstimator(
            seed=seed, epochs=epochs, use_coupling=True,
            coupling_kind="separable", include_interactions=False,
        ),
        "no_coupling": lambda seed: CoupledLearnedEstimator(
            seed=seed, epochs=epochs, use_coupling=False,
            coupling_kind="joint", include_interactions=False,
        ),
        "spatial_only": lambda seed: SpatialOnlyMLEstimator(seed=seed),
    }


def predict_estimators(
    conn: psycopg.Connection,
    output_dir: Path,
    dataset: str,
    spec: dict[str, Any],
    workload: dict[str, Any],
    epochs: int,
) -> dict[str, Any]:
    path = output_dir / "predictions" / f"{dataset}.json"
    if path.exists():
        return json.loads(path.read_text())
    training_payload = json.loads(
        (TASK3_DIR / "workloads" / f"{dataset}__mixed_extent.json").read_text()
    )
    train = [query_from_dict(row) for row in training_payload["training_queries"]]
    queries = [query_from_dict({"predicate": row["predicate"], "true_cardinality": row["true_cardinality"]})
               for row in workload["queries"]]
    predictions = {row["query_id"]: {} for row in workload["queries"]}
    overhead: dict[str, Any] = {}
    model_dir = output_dir / "models" / dataset
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, factory in _model_factories(epochs).items():
        models = []
        model_rows = []
        for seed in MODEL_SEEDS:
            model_path = model_dir / f"{name}__{seed}.pkl"
            if model_path.exists():
                payload = model_path.read_bytes()
                estimator = pickle.loads(payload)
                fit_ms = None
                reused_checkpoint = True
            else:
                estimator = factory(seed)
                started = time.perf_counter()
                estimator.fit(conn, spec["table"], train_queries=train)
                fit_ms = (time.perf_counter() - started) * 1000.0
                payload = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
                model_path.write_bytes(payload)
                reused_checkpoint = False
            started = time.perf_counter()
            pickle.loads(payload)
            load_ms = (time.perf_counter() - started) * 1000.0
            parameter_bytes = 0
            model = getattr(estimator, "model", None)
            if hasattr(model, "parameters"):
                parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            models.append(estimator)
            model_rows.append({
                "seed": seed, "fit_ms": fit_ms, "reused_checkpoint": reused_checkpoint,
                "serialized_bytes": len(payload),
                "cold_deserialize_ms": load_ms, "torch_parameter_bytes": parameter_bytes,
                "artifact": str(model_path.resolve().relative_to(REPO_ROOT)),
            })
        if name in {"joint", "separable", "no_coupling"}:
            for estimator, model_row in zip(models, model_rows):
                assert estimator.featurizer is not None and estimator.model is not None
                batch = torch.from_numpy(np.stack([
                    estimator.featurizer.transform(query.predicate) for query in queries
                ]))
                estimator.model.eval()
                with torch.no_grad():
                    estimator.model(batch)
                    started = time.perf_counter()
                    estimator.model(batch)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                model_row["batch_size"] = len(queries)
                model_row["cached_batch_inference_ms"] = elapsed_ms
                model_row["cached_batch_per_query_ms"] = elapsed_ms / len(queries)
        for query_row, query in zip(workload["queries"], queries):
            # Warm cached inference before timing the predefined five-seed median policy.
            for estimator in models:
                estimator.estimate(query.predicate)
            started = time.perf_counter()
            seed_values = [float(estimator.estimate(query.predicate)) for estimator in models]
            latency_ms = (time.perf_counter() - started) * 1000.0
            predictions[query_row["query_id"]][name] = {
                "seed_predictions": dict(zip(map(str, MODEL_SEEDS), seed_values)),
                "predicted_cardinality": float(statistics.median(seed_values)),
                "inference_ms": latency_ms,
            }
        overhead[name] = {"models": model_rows, "deployment_policy": "median of five seeds"}

    samplers = []
    sampler_rows = []
    for seed in SAMPLING_SEEDS:
        estimator = TrajectorySamplingEstimator(sample_fraction=0.01, seed=seed)
        started = time.perf_counter()
        estimator.fit(conn, spec["table"])
        fit_ms = (time.perf_counter() - started) * 1000.0
        samplers.append(estimator)
        sampler_rows.append({
            "seed": seed, "sample_build_ms": fit_ms,
            "sample_rows": estimator.n_sample, "storage_bytes": estimator.storage_bytes,
        })
    for query_row, query in zip(workload["queries"], queries):
        started = time.perf_counter()
        details = [estimator.estimate_with_details(query.predicate) for estimator in samplers]
        wall_ms = (time.perf_counter() - started) * 1000.0
        seed_values = [detail.estimated_cardinality for detail in details]
        predictions[query_row["query_id"]]["trajectory_sampling"] = {
            "seed_predictions": dict(zip(map(str, SAMPLING_SEEDS), seed_values)),
            "sample_hits": dict(zip(map(str, SAMPLING_SEEDS), [d.sample_hits for d in details])),
            "predicted_cardinality": float(statistics.median(seed_values)),
            "inference_ms": wall_ms,
            "database_predicate_ms": sum(d.latency_ms for d in details),
        }
    for estimator in samplers:
        estimator.close()
    overhead["trajectory_sampling"] = {
        "samples": sampler_rows, "deployment_policy": "median of five 1% samples"
    }
    payload = {
        "dataset": dataset,
        "training_workload_id": training_payload["training_workload_id"],
        "training_workload": "accuracy-battery mixed-extent, 800 queries",
        "model_seeds": list(MODEL_SEEDS),
        "sampling_seeds": list(SAMPLING_SEEDS),
        "predictions": predictions,
        "overhead": overhead,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return payload


def _measure_once(
    conn: psycopg.Connection,
    query: str,
    table: str,
    metadata: str,
    settings: dict[str, str] | None,
) -> dict[str, Any]:
    plan, summary = _explain(
        conn, query, table, metadata, analyze=True, settings=settings
    )
    return {
        **summary,
        "planning_ms": float(plan["Planning Time"]),
        "execution_ms": float(plan["Execution Time"]),
        "buffers": root_buffer_counts(plan),
        "normalized_plan": normalized_plan_tree(plan["Plan"]),
        "jit": plan.get("JIT"),
        "jit_enabled": bool(plan.get("JIT")),
    }


def execute_workload(
    conn: psycopg.Connection,
    output_dir: Path,
    dataset: str,
    objects: dict[str, Any],
    workload: dict[str, Any],
    predictions: dict[str, Any],
    repeats: int,
    seed: int,
) -> None:
    measured_dir = output_dir / "measurements"
    measured_dir.mkdir(parents=True, exist_ok=True)
    table, metadata = objects["table"], objects["metadata_table"]
    rng = random.Random(seed)
    for row in workload["queries"]:
        path = measured_dir / f"{dataset}__{row['query_id'].rsplit(':', 1)[1]}.json"
        if path.exists():
            continue
        predicate = query_from_dict(
            {"predicate": row["predicate"], "true_cardinality": row["true_cardinality"]}
        ).predicate
        conditions: dict[str, tuple[str, dict[str, str] | None]] = {}
        for name in CONDITIONS:
            if name == "native":
                estimate = None
            elif name == "true_cardinality_input":
                estimate = row["true_cardinality"]
            else:
                estimate = predictions["predictions"][row["query_id"]][name]["predicted_cardinality"]
            conditions[name] = (
                build_plan_sensitive_query(table, metadata, predicate, estimate), None
            )
        for name, settings in ORACLE_SETTINGS.items():
            conditions[name] = (
                build_plan_sensitive_query(
                    table, metadata, predicate, row["true_cardinality"]
                ),
                settings,
            )
        names = list(conditions)
        warm_order = names[:]
        rng.shuffle(warm_order)
        for name in warm_order:
            query, settings = conditions[name]
            _measure_once(conn, query, table, metadata, settings)
        measurements = {name: [] for name in names}
        for _ in range(repeats):
            order = names[:]
            rng.shuffle(order)
            for name in order:
                query, settings = conditions[name]
                measurements[name].append(
                    _measure_once(conn, query, table, metadata, settings)
                )
        output = {
            "dataset": dataset,
            "query_id": row["query_id"],
            "workload_id": workload["workload_id"],
            "predicate": row["predicate"],
            "profile": row["profile"],
            "boundary_position": row["boundary_position"],
            "boundary": row["boundary"],
            "true_cardinality": row["true_cardinality"],
            "joined_true_cardinality": row["joined_true_cardinality"],
            "conditions": {},
        }
        for name, values in measurements.items():
            estimate = (
                None if name == "native"
                else row["true_cardinality"] if name.startswith("oracle_") or name == "true_cardinality_input"
                else predictions["predictions"][row["query_id"]][name]["predicted_cardinality"]
            )
            inference_ms = (
                predictions["predictions"][row["query_id"]][name]["inference_ms"]
                if name in predictions["predictions"][row["query_id"]] else 0.0
            )
            database_predicate_ms = (
                predictions["predictions"][row["query_id"]][name].get(
                    "database_predicate_ms"
                )
                if name in predictions["predictions"][row["query_id"]] else None
            )
            output["conditions"][name] = {
                "predicted_cardinality": estimate,
                "q_error": None if estimate is None else float(q_error(row["true_cardinality"], estimate)),
                "inference_ms": inference_ms,
                "database_predicate_ms": database_predicate_ms,
                "repeats": values,
            }
        write_results_json(output, path)
        LOGGER.info("measured %s", row["query_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "run/real_experiments/plan_sensitive",
    )
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--n-candidates", type=int, default=400)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--workload-seed", type=int, default=40404)
    parser.add_argument("--execution-seed", type=int, default=44004)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workload-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(args.dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("LOAD 'pg_hint_plan'")
        objects = prepare_database(conn, args.datasets)
        metadata_path = args.output_dir / "experiment_metadata.json"
        if not metadata_path.exists():
            meta = database_metadata(conn, objects)
            meta["run_config"] = {
                "datasets": args.datasets, "n_candidates": args.n_candidates,
                "n_queries": args.n_queries, "workload_seed": args.workload_seed,
                "execution_seed": args.execution_seed, "epochs": args.epochs,
                "warmups": 1, "measured_repeats": args.repeats,
                "estimate_grid": list(ESTIMATE_GRID),
                "profiles": PROFILES,
                "bad_plan_thresholds": [1.10, 1.25, 2.0],
            }
            write_results_json(meta, metadata_path)
        for dataset in args.datasets:
            workload = build_workload(
                conn, args.output_dir, dataset, DATASETS[dataset], objects[dataset],
                args.n_candidates, args.n_queries, args.workload_seed,
            )
            if args.workload_only:
                continue
            predictions = predict_estimators(
                conn, args.output_dir, dataset, DATASETS[dataset], workload, args.epochs
            )
            execute_workload(
                conn, args.output_dir, dataset, objects[dataset], workload,
                predictions, args.repeats, args.execution_seed,
            )
    finally:
        conn.close()
    print(f"wrote plan-study raw measurements to {args.output_dir}")


if __name__ == "__main__":
    main()
