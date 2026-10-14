"""Recompute the exact answers of the evaluation and training workloads from a loaded table.

T-Drive answers are not distributed (the T-Drive licence forbids redistributing derived
files). The T-Drive workload files under ``run/`` contain the query definitions only.
After loading ``tdrive_traj`` from the original download (see README), this script
recomputes every answer, writes one CSV per workload to ``answers/tdrive/`` and checks
each CSV against ``data/tdrive_answers.sha256``. With ``--restore`` it also writes the
answers back into the workload files, so the experiment and analysis scripts can run.

For Porto the answers are distributed; the script recomputes them and compares them with
the values in the workload files.

    python scripts/rebuild_answers.py tdrive [--restore] [--prepare-join-table]
    python scripts/rebuild_answers.py porto [--table porto_traj]

Answer CSV format: ``list,index,true_cardinality[,joined_true_cardinality]``, one line per
query in file order. The joined count is the plan study's trip-metadata join
(``scripts/plan_sensitive_evaluation.py``); it needs the table
``task4_<dataset>_trip_metadata``, which ``--prepare-join-table`` creates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import query_from_dict  # noqa: E402
from src.eval_module.plan_sensitive import build_plan_sensitive_query, exact_indexable_predicate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "tdrive": [
        "run/real_experiments/task7b/headline/workloads/tdrive__fixed_extent.json",
        "run/real_experiments/task7b/headline/workloads/tdrive__mixed_extent.json",
        "run/real_experiments/task7b/plan_sensitive/workloads/tdrive.json",
        "run/vldbj_rev/plan_rerun/workloads/tdrive__dm_train.json",
        "run/vldbj_rev/task1e/workloads/tdrive__mixed_stratified.json",
    ],
    "porto": [
        "run/real_experiments/multiseed_accuracy/workloads/porto__fixed_extent.json",
        "run/real_experiments/multiseed_accuracy/workloads/porto__mixed_extent.json",
        "run/real_experiments/plan_sensitive/workloads/porto.json",
        "run/vldbj_rev/task1e/workloads/porto__mixed_stratified.json",
    ],
}
LISTS = ("training_queries", "evaluation_queries", "queries")
JOINED = {"plan_sensitive/workloads/"}  # workloads that also carry the joined count


def bucket(n: int) -> str:
    return "0" if n == 0 else "1-10" if n <= 10 else "11-100" if n <= 100 else "101-1,000" if n <= 1000 else ">1,000"


def answer_name(rel: str) -> str:
    return rel.replace("run/", "", 1).replace("/", "__").replace(".json", ".csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=tuple(WORKLOADS))
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--table", default=None, help="default: <dataset>_traj")
    ap.add_argument("--metadata-table", default=None, help="default: task4_<dataset>_trip_metadata")
    ap.add_argument("--prepare-join-table", action="store_true")
    ap.add_argument("--restore", action="store_true", help="write the answers into the workload files")
    args = ap.parse_args()
    table = args.table or f"{args.dataset}_traj"
    metadata = args.metadata_table or f"task4_{args.dataset}_trip_metadata"
    out_dir = ROOT / "answers" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {}
    if args.dataset == "tdrive":
        for line in (ROOT / "data/tdrive_answers.sha256").read_text().splitlines():
            h, name = line.split()
            expected[Path(name).name] = h
    failures = 0
    with psycopg.connect(args.dsn) as conn:
        if args.prepare_join_table:
            from scripts.plan_sensitive_evaluation import prepare_database
            prepare_database(conn, [args.dataset])
            conn.commit()
        with conn.cursor() as cur:
            for rel in WORKLOADS[args.dataset]:
                path = ROOT / rel
                payload = json.loads(path.read_text())
                joined = any(k in rel for k in JOINED)
                lines = ["list,index,true_cardinality" + (",joined_true_cardinality" if joined else "")]
                mismatches = 0
                for key in LISTS:
                    for i, q in enumerate(payload.get(key, [])):
                        p = query_from_dict({"true_cardinality": -1, **q}).predicate  # only the predicate is used
                        cur.execute(f"SELECT count(*) FROM {table} t WHERE ({exact_indexable_predicate(p)})")
                        n = int(cur.fetchone()[0])
                        line = f"{key},{i},{n}"
                        if joined:
                            cur.execute(build_plan_sensitive_query(table, metadata, p, None))
                            j = int(cur.fetchone()[0])
                            line += f",{j}"
                        lines.append(line)
                        if "true_cardinality" in q and int(q["true_cardinality"]) != n:
                            mismatches += 1
                        if joined and "joined_true_cardinality" in q and int(q["joined_true_cardinality"]) != j:
                            mismatches += 1
                        if args.restore:
                            q["true_cardinality"] = n
                            if joined:
                                q["joined_true_cardinality"] = j
                            if "query_id" in q and "stratified" in rel:
                                q["bucket"] = bucket(n)
                conn.rollback()
                data = ("\n".join(lines) + "\n").encode()
                name = answer_name(rel)
                (out_dir / name).write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                if args.dataset == "tdrive":
                    ok = expected.get(name) == digest
                    status = "hash matches" if ok else "HASH MISMATCH"
                else:
                    ok = mismatches == 0
                    status = "all answers match the workload file" if ok else f"{mismatches} MISMATCHES"
                failures += not ok
                print(f"{rel}: {len(lines) - 1} queries, {status}")
                if args.restore:
                    path.write_text(json.dumps(payload, indent=1) + "\n")
    if failures:
        raise SystemExit(f"{failures} workload file(s) did not match")


if __name__ == "__main__":
    main()
