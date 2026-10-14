"""Re-measure the native PostgreSQL/MobilityDB row estimates.

For every frozen 300-query evaluation workload this records the planner's row
estimate (``EXPLAIN (FORMAT JSON)``, never executed) for three predicate forms:

* ``pg_default``  -- ``atStbox(traj, B) IS NOT NULL`` alone. Reproduces the old
  "Native" row, which is PostgreSQL's DEFAULT_NOT_UNK_SEL fallback.
* ``native_bbox`` -- ``traj && B`` alone: MobilityDB's own selectivity estimator.
* ``native_served`` -- ``traj && B AND atStbox(traj, B) IS NOT NULL``: the
  predicate actually served in the plan-sensitive study.

It also re-counts the exact truth for a subset of queries to confirm that the
tables still hold the data the saved labels were computed on. Every statement runs
in a read-only transaction; ANALYZE is never run, so existing statistics are used.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import query_from_dict  # noqa: E402
from src.data_module.workload_generator import st_predicate_sql, stbox_literal  # noqa: E402
from src.eval_module.plan_sensitive import exact_indexable_predicate  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tdrive": ("run/real_experiments/task7b/headline/workloads", "tdrive_traj"),
    "porto": ("run/real_experiments/multiseed_accuracy/workloads", "porto_traj"),
    "ais": ("run/vldbj_rev/ais/headline/workloads", "ais_traj"),
}
WORKLOADS = ("fixed_extent", "mixed_extent")
FORMS = {
    "pg_default": lambda p: st_predicate_sql(p),
    "native_bbox": lambda p: f"traj && stbox '{stbox_literal(p)}'",
    "native_served": lambda p: exact_indexable_predicate(p, alias="t"),
}


def explain_rows(cur: psycopg.Cursor, table: str, where: str) -> float:
    cur.execute(
        sql.SQL("EXPLAIN (FORMAT JSON) SELECT * FROM {} t WHERE " + where).format(
            sql.Identifier(table)
        )
    )
    return float(cur.fetchone()[0][0]["Plan"]["Plan Rows"])


def table_metadata(cur: psycopg.Cursor, table: str) -> dict:
    cur.execute(
        "SELECT c.reltuples, c.relpages, s.last_analyze, s.last_autoanalyze "
        "FROM pg_class c LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid "
        "WHERE c.oid = %s::regclass",
        (table,),
    )
    reltuples, relpages, last_analyze, last_autoanalyze = cur.fetchone()
    cur.execute(
        "SELECT stakind1, stakind2, stakind3, stakind4, stakind5 FROM pg_statistic "
        "WHERE starelid = %s::regclass AND staattnum = "
        "(SELECT attnum FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'traj')",
        (table, table),
    )
    kinds = cur.fetchone()
    cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
    row_count = cur.fetchone()[0]
    cur.execute(
        "SELECT a.attstattarget, array_length(s.stanumbers1, 1), "
        "current_setting('default_statistics_target')::int "
        "FROM pg_attribute a LEFT JOIN pg_statistic s "
        "ON s.starelid = a.attrelid AND s.staattnum = a.attnum "
        "WHERE a.attrelid = %s::regclass AND a.attname = 'traj'",
        (table,),
    )
    target, hist_numbers, default_target = cur.fetchone()
    cur.execute(
        "SELECT n_live_tup, analyze_count, autoanalyze_count FROM pg_stat_user_tables "
        "WHERE relid = %s::regclass", (table,))
    n_live_tup, analyze_count, autoanalyze_count = cur.fetchone()
    return {
        "reltuples": reltuples,
        "relpages": relpages,
        "row_count": row_count,
        "reltuples_equals_row_count": reltuples == row_count,
        "last_analyze": str(last_analyze),
        "last_autoanalyze": str(last_autoanalyze),
        "traj_statistic_kinds": list(kinds) if kinds else None,
        "traj_statistics_target": target if target is not None else f"default ({default_target})",
        "traj_stanumbers1_length": hist_numbers,
        "activity_counters": {"n_live_tup": n_live_tup, "analyze_count": analyze_count,
                              "autoanalyze_count": autoanalyze_count},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/native")
    ap.add_argument("--datasets", nargs="+", default=["tdrive", "porto"], choices=list(SOURCES))
    ap.add_argument("--truth-checks", type=int, default=20,
                    help="evenly spaced queries per workload whose truth is re-counted")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, checks, meta = [], [], {"tables": {}, "workloads": {}}
    with psycopg.connect(args.dsn) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            meta["database_version"] = cur.fetchone()[0]
            cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1")
            meta["extensions"] = dict(cur.fetchall())
            for dataset, (wdir, table) in SOURCES.items():
                if dataset not in args.datasets:
                    continue
                meta["tables"][table] = table_metadata(cur, table)
                for workload in WORKLOADS:
                    path = ROOT / wdir / f"{dataset}__{workload}.json"
                    payload = json.loads(path.read_text())
                    if payload["table"] != table:
                        raise RuntimeError(f"{path} was labelled on {payload['table']}, not {table}")
                    meta["workloads"][f"{dataset}/{workload}"] = {
                        "path": str(path.relative_to(ROOT)),
                        "evaluation_workload_id": payload["evaluation_workload_id"],
                        "n_eval": payload["n_eval"],
                    }
                    queries = [query_from_dict(q) for q in payload["evaluation_queries"]]
                    step = max(1, len(queries) // args.truth_checks)
                    check_idx = set(range(0, len(queries), step)[: args.truth_checks])
                    for i, q in enumerate(queries):
                        row = {"dataset": dataset, "workload": workload, "query_index": i,
                               "true_cardinality": q.true_cardinality}
                        for form, build in FORMS.items():
                            row[form] = explain_rows(cur, table, build(q.predicate))
                        rows.append(row)
                        if i in check_idx:
                            start = time.perf_counter()
                            cur.execute(
                                sql.SQL("SELECT count(*) FROM {} t WHERE "
                                        + exact_indexable_predicate(q.predicate, alias="t")
                                        ).format(sql.Identifier(table))
                            )
                            recount = cur.fetchone()[0]
                            checks.append({"dataset": dataset, "workload": workload,
                                           "query_index": i, "saved_truth": q.true_cardinality,
                                           "recounted_truth": recount,
                                           "match": recount == q.true_cardinality,
                                           "count_ms": (time.perf_counter() - start) * 1000})
            conn.rollback()

    with (args.output_dir / "explain_estimates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (args.output_dir / "truth_recheck.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(checks[0])); w.writeheader(); w.writerows(checks)
    meta.update({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "forms": {"pg_default": "atStbox(traj, B) IS NOT NULL",
                  "native_bbox": "traj && B",
                  "native_served": "traj && B AND atStbox(traj, B) IS NOT NULL"},
        "estimate": "top-level Plan Rows of EXPLAIN (FORMAT JSON) SELECT * FROM table t WHERE <form>",
        "analyze_run": False,
        "statistics_note": (
            "pg_statistic holds full traj statistics for both tables. porto_traj shows "
            "last_analyze=None and analyze_count=0 only because the cumulative activity "
            "counters (pg_stat_user_tables) were lost; PostgreSQL discards them after an "
            "unclean shutdown, and stale postmaster.pid files show unclean shutdowns on "
            "2026-09-14. The statistics themselves persist in pg_statistic. tdrive_traj.traj "
            "uses statistics target 1000 (set by task7b_headline.py); porto_traj.traj uses "
            "the default target, so Native has a smaller histogram on Porto. ANALYZE was "
            "not run."),
        "truth_checks_matched": sum(c["match"] for c in checks),
        "truth_checks_total": len(checks),
    })
    write_results_json(meta, args.output_dir / "metadata.json")
    print(json.dumps({k: meta[k] for k in ("tables", "truth_checks_matched", "truth_checks_total")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
