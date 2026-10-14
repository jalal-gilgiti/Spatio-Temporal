"""Re-run the T-Drive plan-sensitive study with clamped and
distribution-matched learned estimates.

Uses the frozen 100-query workload and saved predictions from
``run/real_experiments/task7b/plan_sensitive``. Every condition is measured in one
fresh connection with the original protocol (one warm-up plus five randomized
EXPLAIN ANALYZE repeats per condition; execution seed 44004):

* ``native``, ``true_cardinality_input``, ``trajectory_sampling`` and the three
  join-method oracles, as in the original study;
* ``<model>``: the original 800-label mixed-extent predictions, unclamped
  (a re-measurement of the published conditions);
* ``<model>_clamped``: the same predictions clamped to [0, N] before injection;
* ``<model>_dm``: models retrained on 800 labels drawn from the plan study's five
  scale profiles (disjoint from the 100 plan queries), clamped to [0, N].

The connection is read-only and this script never calls ``prepare_database``, so no
ANALYZE runs and the existing statistics are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import predicate_dict, query_from_dict  # noqa: E402
from scripts.plan_sensitive_evaluation import (  # noqa: E402
    DATASETS, MODEL_SEEDS, ORACLE_SETTINGS, PROFILES, _measure_once, _model_factories,
    _random_predicate,
)
from src.data_module.workload_generator import WorkloadQuery  # noqa: E402
from src.eval_module.accuracy import q_error  # noqa: E402
from src.eval_module.plan_sensitive import (  # noqa: E402
    build_plan_sensitive_query, exact_indexable_predicate,
)
from src.real_experiment_steps import fetch_table_workload_bounds  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "run/real_experiments/task7b/plan_sensitive"
DATASET = "tdrive"
METADATA_TABLE = "task4_tdrive_trip_metadata"
LEARNED = ("joint", "separable", "no_coupling", "spatial_only")


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_dm_training(conn, out: Path, plan_workload: dict, n_train: int, seed: int) -> dict:
    """Label n_train predicates drawn evenly from the plan study's five profiles."""
    path = out / "workloads" / f"{DATASET}__dm_train.json"
    if path.exists():
        return json.loads(path.read_text())
    spec = DATASETS[DATASET]
    bbox, time_range = fetch_table_workload_bounds(conn, spec["table"])
    excluded = {_sha(q["predicate"]) for q in plan_workload["queries"]}
    rng = np.random.default_rng(seed)
    rows, label_ms, skipped = [], 0.0, 0
    per_profile = n_train // len(PROFILES)
    with conn.cursor() as cur:
        for profile in PROFILES:
            made = 0
            while made < per_profile:
                predicate = _random_predicate(rng, bbox, time_range, spec["extent"],
                                              spec["window_hours"], profile)
                pd = predicate_dict(predicate)
                if _sha(pd) in excluded:
                    skipped += 1
                    continue
                excluded.add(_sha(pd))
                started = time.perf_counter()
                cur.execute(f"SELECT count(*) FROM {spec['table']} t WHERE "
                            f"({exact_indexable_predicate(predicate)})")
                truth = int(cur.fetchone()[0])
                label_ms += (time.perf_counter() - started) * 1000.0
                rows.append({"profile": profile, "predicate": pd, "true_cardinality": truth})
                made += 1
    payload = {
        "dataset": DATASET, "seed": seed, "n_train": len(rows),
        "profiles": PROFILES, "per_profile": per_profile,
        "bbox": list(bbox), "time_range": [t.isoformat() for t in time_range],
        "disjoint_from_workload_id": plan_workload["workload_id"],
        "skipped_duplicates": skipped, "label_ms_total": label_ms,
        "training_workload_id": _sha([{"p": r["predicate"], "t": r["true_cardinality"]} for r in rows]),
        "zero_count": sum(r["true_cardinality"] == 0 for r in rows),
        "training_queries": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return payload


def predict_dm(conn, out: Path, plan_workload: dict, train_payload: dict, epochs: int) -> dict:
    """Train the four learned models on the DM workload; median-of-five-seeds policy."""
    path = out / "predictions" / f"{DATASET}__dm.json"
    if path.exists():
        return json.loads(path.read_text())
    table = DATASETS[DATASET]["table"]
    train = [WorkloadQuery(query_from_dict(r).predicate, r["true_cardinality"])
             for r in train_payload["training_queries"]]
    queries = [query_from_dict({"predicate": r["predicate"], "true_cardinality": r["true_cardinality"]})
               for r in plan_workload["queries"]]
    predictions = {r["query_id"]: {} for r in plan_workload["queries"]}
    overhead = {}
    model_dir = out / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, factory in _model_factories(epochs).items():
        models, rows = [], []
        for seed in MODEL_SEEDS:
            model_path = model_dir / f"{name}__{seed}.pkl"
            if model_path.exists():
                estimator, fit_ms = pickle.loads(model_path.read_bytes()), None
            else:
                estimator = factory(seed)
                started = time.perf_counter()
                estimator.fit(conn, table, train_queries=train)
                fit_ms = (time.perf_counter() - started) * 1000.0
                model_path.write_bytes(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
            models.append(estimator)
            rows.append({"seed": seed, "fit_ms": fit_ms})
            logging.info("dm %s seed=%s fit_ms=%s", name, seed, fit_ms)
        for query_row, query in zip(plan_workload["queries"], queries):
            for estimator in models:  # warm, as in the original timing policy
                estimator.estimate(query.predicate)
            started = time.perf_counter()
            values = [float(e.estimate(query.predicate)) for e in models]
            latency_ms = (time.perf_counter() - started) * 1000.0
            predictions[query_row["query_id"]][name] = {
                "seed_predictions": dict(zip(map(str, MODEL_SEEDS), values)),
                "predicted_cardinality": float(statistics.median(values)),
                "inference_ms": latency_ms,
            }
        overhead[name] = {"models": rows, "deployment_policy": "median of five seeds"}
    payload = {"dataset": DATASET, "training_workload_id": train_payload["training_workload_id"],
               "training_workload": f"{train_payload['n_train']} labels from the five plan-study profiles",
               "predictions": predictions, "overhead": overhead}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(payload, path)
    return payload


def conditions_for(row: dict, original: dict, dm: dict, n_rows: int) -> dict:
    """condition -> (estimate or None, inference_ms, database_predicate_ms, clamped flag)."""
    qid = row["query_id"]
    orig = original["predictions"][qid]
    clamp = lambda v: min(max(float(v), 0.0), float(n_rows))  # noqa: E731
    out = {
        "native": (None, 0.0, None, False),
        "true_cardinality_input": (float(row["true_cardinality"]), 0.0, None, False),
        "trajectory_sampling": (orig["trajectory_sampling"]["predicted_cardinality"],
                                orig["trajectory_sampling"]["inference_ms"],
                                orig["trajectory_sampling"].get("database_predicate_ms"), False),
    }
    for name in (LEARNED if dm is not None else ()):
        raw = orig[name]["predicted_cardinality"]
        out[name] = (raw, orig[name]["inference_ms"], None, False)
        out[f"{name}_clamped"] = (clamp(raw), orig[name]["inference_ms"], None, clamp(raw) != raw)
        dm_raw = dm["predictions"][qid][name]["predicted_cardinality"]
        out[f"{name}_dm"] = (clamp(dm_raw), dm["predictions"][qid][name]["inference_ms"], None,
                             clamp(dm_raw) != dm_raw)
    return out


def measure(conn, out: Path, plan_workload: dict, original: dict, dm: dict,
            n_rows: int, repeats: int, seed: int) -> None:
    table = DATASETS[DATASET]["table"]
    measured = out / "measurements"
    measured.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    for row in plan_workload["queries"]:
        path = measured / f"{DATASET}__{row['query_id'].rsplit(':', 1)[1]}.json"
        conds = conditions_for(row, original, dm, n_rows)
        queries = {name: (build_plan_sensitive_query(table, METADATA_TABLE,
                                                     query_from_dict({"predicate": row["predicate"],
                                                                      "true_cardinality": 0}).predicate,
                                                     est), None)
                   for name, (est, *_rest) in conds.items()}
        for name, settings in ORACLE_SETTINGS.items():
            queries[name] = (build_plan_sensitive_query(
                table, METADATA_TABLE,
                query_from_dict({"predicate": row["predicate"], "true_cardinality": 0}).predicate,
                row["true_cardinality"]), settings)
        names = list(queries)
        if path.exists():  # keep the shared RNG stream identical on resume
            rng.shuffle(names[:])
            for _ in range(repeats):
                rng.shuffle(names[:])
            continue
        order = names[:]
        rng.shuffle(order)
        for name in order:
            _measure_once(conn, queries[name][0], table, METADATA_TABLE, queries[name][1])
        results = {name: [] for name in names}
        for _ in range(repeats):
            order = names[:]
            rng.shuffle(order)
            for name in order:
                results[name].append(_measure_once(conn, queries[name][0], table, METADATA_TABLE,
                                                   queries[name][1]))
        payload = {"dataset": DATASET}
        payload.update({k: row[k] for k in ("query_id", "predicate", "profile",
                                            "boundary_position", "boundary", "true_cardinality",
                                            "joined_true_cardinality")})
        payload["workload_id"] = plan_workload["workload_id"]
        payload["conditions"] = {}
        for name in names:
            if name in conds:
                est, inf_ms, db_ms, clamped = conds[name]
            else:
                est, inf_ms, db_ms, clamped = float(row["true_cardinality"]), 0.0, None, False
            payload["conditions"][name] = {
                "predicted_cardinality": est,
                "q_error": None if est is None else float(q_error(row["true_cardinality"], est)),
                "inference_ms": inf_ms, "database_predicate_ms": db_ms,
                "clamped": clamped, "repeats": results[name],
            }
        write_results_json(payload, path)
        logging.info("measured %s", row["query_id"])


def main() -> None:
    global DATASET, METADATA_TABLE, SOURCE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/plan_rerun")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--dm-seed", type=int, default=70001)
    ap.add_argument("--execution-seed", type=int, default=44004)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dataset", choices=("tdrive", "porto"), default="tdrive")
    ap.add_argument("--source-dir", type=Path, default=SOURCE,
                    help="directory holding the frozen plan workload and saved predictions")
    ap.add_argument("--no-learned", action="store_true",
                    help="measure only native, true-cardinality, trajectory-sampling and oracle conditions")
    ap.add_argument("--retime-from", type=Path, default=None,
                    help="re-time only: reuse this run's distribution-matched predictions")
    ap.add_argument("--retime-n", type=int, default=10)
    ap.add_argument("--retime-seed", type=int, default=81001)
    args = ap.parse_args()
    DATASET, SOURCE = args.dataset, args.source_dir.resolve()
    METADATA_TABLE = f"task4_{DATASET}_trip_metadata"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_workload = json.loads((SOURCE / "workloads" / f"{DATASET}.json").read_text())
    original = json.loads((SOURCE / "predictions" / f"{DATASET}.json").read_text())
    conn = psycopg.connect(args.dsn, autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
            cur.execute("LOAD 'pg_hint_plan'")
            cur.execute(f"SELECT count(*) FROM {DATASETS[DATASET]['table']}")
            n_rows = int(cur.fetchone()[0])
        meta_path = args.output_dir / "experiment_metadata.json"
        if not meta_path.exists():
            write_results_json({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": str(SOURCE.relative_to(ROOT)), "n_rows": n_rows,
                "clamp": "[0, N] before injection; pg_hint_plan Rows() also floors at 1",
                "run_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "analyze_run": False,
            }, meta_path)
        if args.no_learned:
            with conn.cursor() as cur:  # confirm the table still matches the frozen labels
                for row in plan_workload["queries"][::10]:
                    pred = query_from_dict({"predicate": row["predicate"], "true_cardinality": 0}).predicate
                    cur.execute(f"SELECT count(*) FROM {DATASETS[DATASET]['table']} t WHERE "
                                f"({exact_indexable_predicate(pred)})")
                    got = int(cur.fetchone()[0])
                    if got != row["true_cardinality"]:
                        raise RuntimeError(f"{row['query_id']}: truth {got} != saved {row['true_cardinality']}")
            logging.info("truth re-check passed on %d queries", len(plan_workload["queries"][::10]))
            measure(conn, args.output_dir, plan_workload, original, None, n_rows,
                    args.repeats, args.execution_seed)
            return
        if args.retime_from is not None:
            dm = json.loads((args.retime_from / "predictions" / f"{DATASET}__dm.json").read_text())
            chosen = sorted(random.Random(args.retime_seed).sample(
                range(len(plan_workload["queries"])), args.retime_n))
            plan_workload = dict(plan_workload, queries=[plan_workload["queries"][i] for i in chosen])
            logging.info("re-timing %d queries: %s", len(chosen), chosen)
            measure(conn, args.output_dir, plan_workload, original, dm, n_rows,
                    args.repeats, args.execution_seed)
            return
        train_payload = build_dm_training(conn, args.output_dir, plan_workload, args.n_train, args.dm_seed)
        logging.info("dm training labelled: %d queries, %.1f s", train_payload["n_train"],
                     train_payload["label_ms_total"] / 1000)
        dm = predict_dm(conn, args.output_dir, plan_workload, train_payload, args.epochs)
        measure(conn, args.output_dir, plan_workload, original, dm, n_rows,
                args.repeats, args.execution_seed)
    finally:
        conn.close()
    print(f"wrote plan re-run to {args.output_dir}")


if __name__ == "__main__":
    main()
