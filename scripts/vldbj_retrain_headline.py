"""Deterministically retrain the headline learned models and check
that they reproduce the saved headline predictions.

The original accuracy runs (T-Drive: task7b/headline; Porto: multiseed_accuracy)
did not save their models. This script retrains Joint, Separable, No-coupling and
SpatialOnly with the original seeds (1-5), epochs and frozen training workloads.
It runs single-threaded with deterministic torch algorithms, saves every model,
writes runs in the original format, and compares each run with the saved one
query by query.

The originals were trained with 4 torch threads, so bitwise equality is not
guaranteed; the reproduction report gives exact-match shares and the largest
log-ratio differences.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import (  # noqa: E402
    DISPLAY_NAMES, estimator_configs, query_from_dict, summarize,
)
from src.eval_module.accuracy import q_error  # noqa: E402
from src.results_io import write_results_json  # noqa: E402
from src.utils.db import get_connection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tdrive": (ROOT / "run/real_experiments/task7b/headline", "tdrive_traj"),
    "porto": (ROOT / "run/real_experiments/multiseed_accuracy", "porto_traj"),
}
WORKLOADS = ("fixed_extent", "mixed_extent")
LEARNED = ("joint", "separable", "no_coupling", "spatial_only")
MODEL_SEEDS = (1, 2, 3, 4, 5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/headline_retrain")
    ap.add_argument("--epochs", type=int, default=1000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for sub in ("models", "runs"):
        (args.output_dir / sub).mkdir(parents=True, exist_ok=True)
    factories = {name: f for name, _seed, f in estimator_configs(MODEL_SEEDS, (), args.epochs)
                 if name in LEARNED}
    report = []
    conn = get_connection(args.dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
        for dataset, (src, table) in SOURCES.items():
            for workload in WORKLOADS:
                payload = json.loads((src / "workloads" / f"{dataset}__{workload}.json").read_text())
                train = [query_from_dict(q) for q in payload["training_queries"]]
                evaluation = [query_from_dict(q) for q in payload["evaluation_queries"]]
                for name in LEARNED:
                    for seed in MODEL_SEEDS:
                        stem = f"{dataset}__{workload}__{name}__{seed}"
                        run_path = args.output_dir / "runs" / f"{stem}.json"
                        model_path = args.output_dir / "models" / f"{stem}.pkl"
                        if run_path.exists():
                            run = json.loads(run_path.read_text())
                        else:
                            estimator = factories[name](seed)
                            started = time.perf_counter()
                            estimator.fit(conn, table, train_queries=train)
                            fit_seconds = time.perf_counter() - started
                            model_path.write_bytes(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
                            preds = []
                            for i, q in enumerate(evaluation):
                                p = float(estimator.estimate(q.predicate))
                                preds.append({"query_id": f"{dataset}:{workload}:{i:03d}", "query_index": i,
                                              "true_cardinality": int(q.true_cardinality),
                                              "predicted_cardinality": p,
                                              "q_error": float(q_error(q.true_cardinality, p))})
                            run = {"dataset": dataset, "table": table, "workload": workload,
                                   "workload_id": payload["evaluation_workload_id"],
                                   "training_workload_id": payload["training_workload_id"],
                                   "estimator": name, "display_name": DISPLAY_NAMES[name],
                                   "estimator_seed": seed, "fit_seconds": fit_seconds,
                                   "torch_threads": 1, "deterministic_algorithms": True,
                                   "model_artifact": str(model_path.relative_to(ROOT)),
                                   "summary": summarize(preds), "predictions": preds}
                            write_results_json(run, run_path)
                            logging.info("%s fit %.1fs", stem, fit_seconds)
                        saved = json.loads((src / "runs" / f"{stem}.json").read_text())["predictions"]
                        old = np.array([p["predicted_cardinality"] for p in saved])
                        new = np.array([p["predicted_cardinality"] for p in run["predictions"]])
                        diff = np.abs(np.log10((np.maximum(new, 0) + 1) / (np.maximum(old, 0) + 1)))
                        report.append({"dataset": dataset, "workload": workload, "estimator": name,
                                       "seed": seed, "exact_match_share": float(np.mean(new == old)),
                                       "max_abs_log10_diff": float(diff.max()),
                                       "median_abs_log10_diff": float(np.median(diff)),
                                       "old_median_qerror": saved and float(np.median([p["q_error"] for p in saved])),
                                       "new_median_qerror": float(np.median([p["q_error"] for p in run["predictions"]]))})
    finally:
        conn.close()
    with (args.output_dir / "reproduction.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(report[0])); w.writeheader(); w.writerows(report)
    exact = sum(r["exact_match_share"] == 1.0 for r in report)
    print(f"runs reproduced bit-exactly: {exact}/{len(report)}; "
          f"worst max |log10 diff| = {max(r['max_abs_log10_diff'] for r in report):.3g}")


if __name__ == "__main__":
    main()
