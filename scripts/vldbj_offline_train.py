"""Train the learned estimators off-server (no database needed) and write run files.

The learned estimators (Joint, Separable, No-coupling, SpatialOnly) derive their
featurizer bounds from the training queries and never read the database, so they
can be trained from the frozen labelled workloads alone.

Usage (from the repository root):

    python scripts/vldbj_offline_train.py --manifest offline/manifest.json --out offline_out

The manifest is a JSON list of jobs:

    [{"dataset": "tdrive", "workload": "mixed_extent", "table": "tdrive_traj",
      "training_file": "run/real_experiments/task7b/headline/workloads/tdrive__mixed_extent.json",
      "eval_files": {"headline": "run/real_experiments/task7b/headline/workloads/tdrive__mixed_extent.json"}}]

A training file must contain ``training_queries`` (predicate + true_cardinality).
Each evaluation file must contain ``evaluation_queries`` or ``queries``.
Output: ``<out>/models/<dataset>__<workload>__<model>__<seed>.pkl``, plus one run file
per evaluation set in the format of the saved headline runs:
``<out>/runs/<eval_name>/<dataset>__<workload>__<model>__<seed>.json``.
Runs single-threaded with deterministic torch algorithms; finished runs are skipped,
so the script can be restarted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import (  # noqa: E402
    DISPLAY_NAMES, estimator_configs, query_from_dict, summarize,
)
from src.eval_module.accuracy import q_error  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

LEARNED = ("joint", "separable", "no_coupling", "spatial_only")
MODEL_SEEDS = (1, 2, 3, 4, 5)


def _queries(payload: dict, key_order=("evaluation_queries", "queries")) -> list[dict]:
    for key in key_order:
        if key in payload:
            return payload[key]
    raise KeyError(f"no query list among {key_order}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--models", nargs="+", default=list(LEARNED), choices=LEARNED)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(__file__).resolve().parents[1]
    factories = {n: f for n, _s, f in estimator_configs(MODEL_SEEDS, (), args.epochs) if n in LEARNED}
    (args.out / "models").mkdir(parents=True, exist_ok=True)
    env = {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
           "torch_threads": 1, "deterministic_algorithms": True}
    for job in json.loads(args.manifest.read_text()):
        train_payload = json.loads((root / job["training_file"]).read_text())
        train = [query_from_dict(q) for q in train_payload["training_queries"]]
        evals = {name: json.loads((root / path).read_text()) for name, path in job["eval_files"].items()}
        for model in args.models:
            for seed in MODEL_SEEDS:
                stem = f"{job['dataset']}__{job['workload']}__{model}__{seed}"
                model_path = args.out / "models" / f"{stem}.pkl"
                todo = [n for n in evals if not (args.out / "runs" / n / f"{stem}.json").exists()]
                if not todo:
                    continue
                if model_path.exists():
                    estimator, fit_seconds = pickle.loads(model_path.read_bytes()), None
                else:
                    estimator = factories[model](seed)
                    started = time.perf_counter()
                    estimator.fit(None, job["table"], train_queries=train)
                    fit_seconds = time.perf_counter() - started
                    model_path.write_bytes(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
                    logging.info("%s fit %.1fs", stem, fit_seconds)
                model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
                for name in todo:
                    rows = []
                    for i, q in enumerate(_queries(evals[name])):
                        wq = query_from_dict(q)
                        p = float(estimator.estimate(wq.predicate))
                        rows.append({"query_id": q.get("query_id", f"{job['dataset']}:{name}:{i:03d}"),
                                     "query_index": i, "true_cardinality": int(wq.true_cardinality),
                                     "predicted_cardinality": p,
                                     "q_error": float(q_error(wq.true_cardinality, p))})
                    run = {"dataset": job["dataset"], "table": job["table"], "workload": job["workload"],
                           "eval_set": name,
                           "workload_id": evals[name].get("evaluation_workload_id", evals[name].get("workload_id")),
                           "training_workload_id": train_payload.get("training_workload_id"),
                           "estimator": model, "display_name": DISPLAY_NAMES[model],
                           "estimator_seed": seed, "fit_seconds": fit_seconds,
                           "model_artifact": str(model_path), "model_sha256": model_sha,
                           "environment": env, "summary": summarize(rows), "predictions": rows}
                    out = args.out / "runs" / name / f"{stem}.json"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    write_results_json(run, out)
    print(f"done; outputs in {args.out}")


if __name__ == "__main__":
    main()
