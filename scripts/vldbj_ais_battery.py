"""AIS workloads and the non-learned estimator battery.

Mirrors scripts/task7b_headline.py for the new ``ais_traj`` table:
* workloads: training seed 42 (800 fixed-extent / 6,400 mixed-extent labels),
  evaluation seed 43 (300 queries each), same JSON format as the headline workloads;
* bounds: the Danish-waters box lon 3-17, lat 53-59 (98.9% of all AIS points; the raw
  table extent is inflated by a few outlying positions), and the table's time span;
* fixed extent: 1.5 degree box, 24 h window: the smallest box with <= 15% empty
  queries in a 300-query pilot (1.0 deg: 19% empty);
* estimators run now: Trajectory Sampling (seeds 17, 23, 47, 59, 71), origin sampling
  (seeds 1-5), independent / joint origin histograms and MinSkew-Origin. The learned
  models wait for the training decision.
Run files use the headline format, so the existing table and bootstrap scripts can read
them. Restart-safe: finished workloads and runs are reused.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import (  # noqa: E402
    consolidate, estimator_configs, fingerprint, query_dict, query_from_dict, run_estimator,
)
from src.eval_module.scalability import generate_predicates, label_predicates  # noqa: E402
from src.real_experiment_steps import fetch_table_workload_bounds  # noqa: E402
from src.results_io import write_results_json  # noqa: E402
from src.utils.db import get_connection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = "ais_traj"
BBOX = (3.0, 17.0, 53.0, 59.0)
SPECS = {"fixed_extent": (800, 1.5, 24.0), "mixed_extent": (6400, 1.5, 24.0)}
TRAIN_SEED, EVAL_SEED, N_EVAL = 42, 43, 300
MODEL_SEEDS = (1, 2, 3, 4, 5)
SAMPLE_SEEDS = (17, 23, 47, 59, 71)
NON_LEARNED = ("trajectory_sampling", "origin_sampling", "independent_histogram",
               "joint_histogram", "minskew_origin")


def workload(conn, out: Path, name: str, times, ntrain: int, extent: float, hours: float):
    path = out / "workloads" / f"ais__{name}.json"
    if path.exists():
        p = json.loads(path.read_text())
        return ([query_from_dict(x) for x in p["training_queries"]],
                [query_from_dict(x) for x in p["evaluation_queries"]], p)
    tp = generate_predicates(name, ntrain, TRAIN_SEED, BBOX, times, extent, hours)
    ep = generate_predicates(name, N_EVAL, EVAL_SEED, BBOX, times, extent, hours)
    train, tms = label_predicates(conn, TABLE, tp)
    ev, ems = label_predicates(conn, TABLE, ep)
    tr = [query_dict(x) for x in train]
    er = [query_dict(x) for x in ev]
    p = {"dataset": "ais", "table": TABLE, "workload": name, "bbox": list(BBOX),
         "time_range": [t.isoformat() for t in times], "extent_degrees": extent, "window_hours": hours,
         "training_seed": TRAIN_SEED, "evaluation_seed": EVAL_SEED, "n_train": len(tr), "n_eval": len(er),
         "training_workload_id": fingerprint(tr), "evaluation_workload_id": fingerprint(er),
         "training_queries": tr, "evaluation_queries": er,
         "training_label_ms": tms, "evaluation_label_ms": ems,
         "training_zero_count": sum(x["true_cardinality"] == 0 for x in tr),
         "evaluation_zero_count": sum(x["true_cardinality"] == 0 for x in er)}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(p, path)
    return train, ev, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/ais/headline")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (args.output_dir / "runs").mkdir(parents=True, exist_ok=True)
    conn = get_connection(args.dsn)
    try:
        _, times = fetch_table_workload_bounds(conn, TABLE)
        write_results_json({"task": "AIS battery", "dataset": "AIS (DMA), 2025-01-06..2025-02-19, 100k trips",
                            "table": TABLE, "bbox": list(BBOX), "specs": SPECS,
                            "training_seed": TRAIN_SEED, "evaluation_seed": EVAL_SEED,
                            "model_seeds": list(MODEL_SEEDS), "sampling_seeds": list(SAMPLE_SEEDS),
                            "estimators_run": list(NON_LEARNED), "learned": "pending training decision",
                            "pilot": "extent chosen on a 300-query pilot drawn with the evaluation seed 43"},
                           args.output_dir / "experiment_metadata.json")
        for name, (budget, extent, hours) in SPECS.items():
            train, ev, meta = workload(conn, args.output_dir, name, times, budget, extent, hours)
            logging.info("%s: %d train / %d eval, eval zeros %d", name, len(train), len(ev),
                         meta["evaluation_zero_count"])
            for est, seed, factory in estimator_configs(MODEL_SEEDS, SAMPLE_SEEDS, 1000):
                if est not in NON_LEARNED:
                    continue
                label = "deterministic" if seed is None else str(seed)
                rp = args.output_dir / "runs" / f"ais__{name}__{est}__{label}.json"
                if rp.exists():
                    continue
                logging.info("%s %s seed=%s", name, est, label)
                write_results_json(run_estimator(conn, TABLE, train, ev, meta, "ais", name, est, seed, factory), rp)
        consolidate(args.output_dir)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
