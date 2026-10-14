"""Does MobilityDB's space-time independence assumption explain the
undercount of its && estimate? (read-only)

MobilityDB 1.3.0 estimates the restriction selectivity of tgeompoint && stbox as the
PRODUCT of a spatial selectivity (geo_sel: 2-D N-D histogram) and a temporal
selectivity (temporal_sel_tstzspan: time-span histogram); see temporal_sel() in
mobilitydb/src/temporal/temporal_selfuncs.c, lines 666-697.

For every fixed-extent evaluation query of T-Drive, Porto and AIS this script records:
  J  exact joint count       traj && STBOX XT(box, window)   (from box_counts.csv)
  S  exact spatial-only count traj && STBOX X(box)
  T  exact temporal-only count traj && tstzspan(window)
  eS, eT, eJ  the planner's estimates for the same three predicates
and decomposes the histogram error log10((eJ+1)/(J+1)) into
  independence error  log10((S*T/N + 1) / (J + 1))   the product of the EXACT marginals vs truth
  marginal error      log10((eS*eT/N + 1) / (S*T/N + 1))
  combination residual log10((eJ+1) / (eS*eT/N + 1))  (about 0 if the estimate is the product)
Negative independence error = positive space-time dependence that the product misses.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import query_from_dict  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tdrive": ("run/real_experiments/task7b/headline/workloads/tdrive__fixed_extent.json", "tdrive_traj",
               "run/vldbj_rev/native/box_counts.csv", "run/vldbj_rev/native/explain_estimates_primary.csv"),
    "porto": ("run/real_experiments/multiseed_accuracy/workloads/porto__fixed_extent.json", "porto_traj",
              "run/vldbj_rev/native/box_counts.csv", "run/vldbj_rev/native/explain_estimates_primary.csv"),
    "ais": ("run/vldbj_rev/ais/headline/workloads/ais__fixed_extent.json", "ais_traj",
            "run/vldbj_rev/ais/native/box_counts.csv", "run/vldbj_rev/ais/native/explain_estimates.csv"),
}


def explain_rows(cur, sql: str, params) -> float:
    cur.execute("EXPLAIN (FORMAT JSON) " + sql, params)
    return float(cur.fetchone()[0][0]["Plan"]["Plan Rows"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/independence")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    with psycopg.connect(args.dsn) as conn:
        conn.read_only = True
        cur = conn.cursor()
        cur.execute("SHOW TimeZone")
        tz = cur.fetchone()[0]
        for ds, (wpath, table, bpath, npath) in SOURCES.items():
            cur.execute(f"SELECT count(*) FROM {table}")
            n = float(cur.fetchone()[0])
            box = {int(r["query_index"]): int(r["box_count"]) for r in csv.DictReader((ROOT / bpath).open())
                   if r["dataset"] == ds and r["workload"] == "fixed_extent"}
            nat = {int(r["query_index"]): float(r["native_bbox"]) for r in csv.DictReader((ROOT / npath).open())
                   if r["dataset"] == ds and r["workload"] == "fixed_extent"}
            queries = json.loads((ROOT / wpath).read_text())["evaluation_queries"]
            for i, q in enumerate(queries):
                p = query_from_dict(q).predicate
                sbox = f"STBOX X(({p.min_lon},{p.min_lat}),({p.max_lon},{p.max_lat}))"
                span = f"[{p.start_time.isoformat()}, {p.end_time.isoformat()})"
                s_sql = f"SELECT * FROM {table} WHERE traj && %s::stbox"
                t_sql = f"SELECT * FROM {table} WHERE traj && %s::tstzspan"
                cur.execute(s_sql.replace("*", "count(*)", 1), (sbox,)); s = int(cur.fetchone()[0])
                cur.execute(t_sql.replace("*", "count(*)", 1), (span,)); t = int(cur.fetchone()[0])
                es, et = explain_rows(cur, s_sql, (sbox,)), explain_rows(cur, t_sql, (span,))
                j, ej = box[i], nat[i]
                prod_exact, prod_est = s * t / n, es * et / n
                rows.append({"dataset": ds, "query_index": i, "true_path": q["true_cardinality"],
                             "J_exact": j, "S_exact": s, "T_exact": t, "eJ": ej, "eS": es, "eT": et,
                             "hist_error": np.log10((ej + 1) / (j + 1)),
                             "independence_error": np.log10((prod_exact + 1) / (j + 1)),
                             "marginal_error": np.log10((prod_est + 1) / (prod_exact + 1)),
                             "combination_residual": np.log10((ej + 1) / (prod_est + 1))})
            conn.rollback()
            d = [r for r in rows if r["dataset"] == ds]
            med = lambda k: float(statistics.median(r[k] for r in d))  # noqa: E731
            share_neg = float(np.mean([r["independence_error"] < 0 for r in d]))
            summary.append({"dataset": ds, "n_queries": len(d), "N": n,
                            "median_hist_error": med("hist_error"),
                            "median_independence_error": med("independence_error"),
                            "median_marginal_error": med("marginal_error"),
                            "median_combination_residual": med("combination_residual"),
                            "share_independence_underestimates": share_neg})
    for name, data in (("per_query", rows), ("summary", summary)):
        with (args.output_dir / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    write_results_json({"source": "MobilityDB v1.3.0 temporal_sel(), temporal_selfuncs.c:666-697 (geo_sel x temporal_sel_tstzspan)",
                        "session_timezone": tz, "workloads": "fixed-extent evaluation sets (300 each)",
                        "decomposition": __doc__.split("decomposes")[1].strip()}, args.output_dir / "metadata.json")
    for s in summary:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
