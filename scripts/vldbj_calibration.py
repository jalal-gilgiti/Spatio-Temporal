"""Fixed-extent calibration rule (empty share < 15% AND median true count >= 10).

* AIS pilot: label 300 fixed-extent queries (evaluation seed 43, Danish-waters bounds,
  24 h window) for box sizes 1.0, 1.5 and 2.0 degrees (read-only). The chosen extent is
  the smallest that satisfies the rule.
* All datasets: check the frozen fixed-extent evaluation and training workloads
  (T-Drive 0.45 deg / 24 h, Porto 0.15 deg / 336 h, AIS 1.5 deg / 24 h) against the
  same rule. T-Drive and Porto extents predate the rule (checked post hoc).
Writes run/vldbj_rev/calibration/calibration.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval_module.scalability import generate_predicates, label_predicates  # noqa: E402
from src.real_experiment_steps import fetch_table_workload_bounds  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RULE = {"max_empty_share": 0.15, "min_median_true": 10}
WORKLOADS = {
    "tdrive": ("run/real_experiments/task7b/headline/workloads/tdrive__fixed_extent.json", 0.45, 24.0),
    "porto": ("run/real_experiments/multiseed_accuracy/workloads/porto__fixed_extent.json", 0.15, 336.0),
    "ais": ("run/vldbj_rev/ais/headline/workloads/ais__fixed_extent.json", 1.5, 24.0),
}


def summary(truths: list[int]) -> dict:
    empty = sum(t == 0 for t in truths) / len(truths)
    med = statistics.median(truths)
    return {"n": len(truths), "empty_share": empty, "median_true": med,
            "satisfies_rule": empty < RULE["max_empty_share"] and med >= RULE["min_median_true"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output", type=Path, default=ROOT / "run/vldbj_rev/calibration/calibration.json")
    args = ap.parse_args()
    out = {"rule": RULE, "pilot_ais": {}, "frozen_workloads": {}}
    with psycopg.connect(args.dsn) as conn:
        conn.read_only = True
        _, times = fetch_table_workload_bounds(conn, "ais_traj")
        for extent in (1.0, 1.5, 2.0):
            qs, _ = label_predicates(conn, "ais_traj", generate_predicates(
                "fixed_extent", 300, 43, (3.0, 17.0, 53.0, 59.0), times, extent, 24.0))
            out["pilot_ais"][f"{extent:.1f}"] = summary([q.true_cardinality for q in qs])
        conn.rollback()
    ok = [float(e) for e, s in out["pilot_ais"].items() if s["satisfies_rule"]]
    out["pilot_ais_chosen_extent"] = min(ok) if ok else None
    out["pilot_note"] = "pilot queries use the evaluation seed 43; the 1.5 deg pilot set is the AIS evaluation set"
    for ds, (path, extent, hours) in WORKLOADS.items():
        w = json.loads((ROOT / path).read_text())
        out["frozen_workloads"][ds] = {
            "extent_degrees": extent, "window_hours": hours, "rule_applied": "chosen by the rule" if ds == "ais" else "post hoc",
            "evaluation": summary([q["true_cardinality"] for q in w["evaluation_queries"]]),
            "training": summary([q["true_cardinality"] for q in w["training_queries"]]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_results_json(out, args.output)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
