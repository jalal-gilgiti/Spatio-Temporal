"""Compare the low-load re-timing of 10 plan queries with the same queries in the main run.

Both runs are analysed with the published definitions (scripts/analyze_plan_sensitive_evaluation.py):
pooled plan fingerprints, an empirical oracle per query within each run, regret, and
end-to-end speedup against native. Reports, per condition: whether the chosen plan is
the same in both runs, execution-time ratio (low / high load), regret in each run,
and whether the per-query oracle plan is the same.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scripts.analyze_plan_sensitive_evaluation as apa  # noqa: E402
from scripts.vldbj_plan_rerun_analysis import CONDITIONS  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def per_query(paths: list[Path]) -> dict[str, dict[str, dict]]:
    apa.CONDITIONS = CONDITIONS
    rows = apa.per_query_rows([json.loads(p.read_text()) for p in paths])
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        out.setdefault(r["query_id"], {})[r["condition"]] = r
    return out


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--high", type=Path, default=ROOT / "run/vldbj_rev/plan_rerun")
    ap.add_argument("--low", type=Path, default=ROOT / "run/vldbj_rev/plan_retime_lowload")
    args = ap.parse_args()
    low_paths = sorted((args.low / "measurements").glob("*.json"))
    high_paths = [args.high / "measurements" / p.name for p in low_paths]
    low, high = per_query(low_paths), per_query(high_paths)
    qids = sorted(low)
    rows = []
    for c in CONDITIONS:
        same_plan = sum(low[q][c]["plan_tree_fingerprint"] == high[q][c]["plan_tree_fingerprint"] for q in qids)
        ratio = [low[q][c]["execution_ms"] / high[q][c]["execution_ms"] for q in qids]
        reg_l = [low[q][c]["regret"] for q in qids]
        reg_h = [high[q][c]["regret"] for q in qids]
        bad_l = sum(x > 1.25 for x in reg_l)
        bad_h = sum(x > 1.25 for x in reg_h)
        sp_l = [low[q][c]["end_to_end_speedup_vs_native"] for q in qids]
        sp_h = [high[q][c]["end_to_end_speedup_vs_native"] for q in qids]
        rows.append({"condition": c, "n_queries": len(qids), "same_plan_both_runs": same_plan,
                     "exec_ratio_low_over_high_median": statistics.median(ratio),
                     "regret_median_high": statistics.median(reg_h), "regret_median_low": statistics.median(reg_l),
                     "regret_gmean_high": gmean(reg_h), "regret_gmean_low": gmean(reg_l),
                     "bad_gt_1_25_high": bad_h, "bad_gt_1_25_low": bad_l,
                     "e2e_speedup_median_high": statistics.median(sp_h),
                     "e2e_speedup_median_low": statistics.median(sp_l)})
    oracle_same = sum(low[q]["native"]["oracle_fingerprint"] == high[q]["native"]["oracle_fingerprint"] for q in qids)
    native_exec = [(high[q]["native"]["execution_ms"], low[q]["native"]["execution_ms"]) for q in qids]
    out = args.low / "comparison"
    out.mkdir(exist_ok=True)
    with (out / "retime_vs_main.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rank = lambda key: [r["condition"] for r in sorted(rows, key=lambda r: r[key])]  # noqa: E731
    summary = {"queries": qids, "oracle_plan_same_both_runs": oracle_same,
               "native_exec_ms_high_low": native_exec,
               "rank_by_gmean_regret_high": rank("regret_gmean_high"),
               "rank_by_gmean_regret_low": rank("regret_gmean_low")}
    write_results_json(summary, out / "summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "native_exec_ms_high_low"}, indent=1))
    for r in rows:
        print(f"{r['condition']:22s} same plan {r['same_plan_both_runs']}/10  exec low/high {r['exec_ratio_low_over_high_median']:.2f}  "
              f"regret gm {r['regret_gmean_high']:.3f}->{r['regret_gmean_low']:.3f}  bad {r['bad_gt_1_25_high']}->{r['bad_gt_1_25_low']}  "
              f"e2e sp {r['e2e_speedup_median_high']:.3f}->{r['e2e_speedup_median_low']:.3f}")


if __name__ == "__main__":
    main()
