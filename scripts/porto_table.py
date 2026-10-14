"""Export or load the Porto trajectory table used in the paper (``porto_traj``).

The table holds 100,000 Porto taxi trips (ECML/PKDD 2015 challenge data, CC BY 4.0) as
MobilityDB ``tgeompoint`` sequences. It is stored in ``data/porto/`` as parquet files with
one row per trip: ``trajectory_id`` and ``traj_wkb``, the trip in MobilityDB's binary
format (``asBinary``), so a reload is bit-identical to the original table.

    python scripts/porto_table.py load               # creates porto_traj, as in the paper
    python scripts/porto_table.py load --table t2    # load under another name
    python scripts/porto_table.py export             # writes data/porto/ from porto_traj

``load`` also builds the GiST index and ANALYZEs the new table at the default statistics
target, as the original table was. The paper's primary native estimates use target 1000
on a copy (see ``scripts/vldbj_stats_target.py``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg import sql

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/porto"
PARTS = 4
N_TRIPS = 100_000


def export(conn: psycopg.Connection, table: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_part = N_TRIPS // PARTS
    with conn.cursor() as cur:
        for k in range(PARTS):
            cur.execute(sql.SQL("SELECT trajectory_id, asBinary(traj) FROM {} "
                                "WHERE trajectory_id >= %s AND trajectory_id < %s ORDER BY trajectory_id")
                        .format(sql.Identifier(table)), (k * per_part, (k + 1) * per_part))
            rows = cur.fetchall()
            pq.write_table(pa.table({"trajectory_id": pa.array([r[0] for r in rows], pa.int32()),
                                     "traj_wkb": pa.array([bytes(r[1]) for r in rows], pa.binary())}),
                           OUT / f"porto_traj_part{k}.parquet", compression="zstd", compression_level=19)
            print(f"part {k}: {len(rows)} trips")


def load(conn: psycopg.Connection, table: str) -> None:
    ident = sql.Identifier(*table.split("."))
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE TABLE {} (trajectory_id integer, traj tgeompoint)").format(ident))
        n = 0
        for part in sorted(OUT.glob("porto_traj_part*.parquet")):
            t = pq.read_table(part)
            cur.executemany(sql.SQL("INSERT INTO {} (trajectory_id, traj) VALUES (%s, tgeompointFromBinary(%s))")
                            .format(ident),
                            list(zip(t["trajectory_id"].to_pylist(), t["traj_wkb"].to_pylist())))
            n += t.num_rows
        cur.execute(sql.SQL("CREATE INDEX ON {} USING GIST (traj)").format(ident))
        cur.execute(sql.SQL("ANALYZE {}").format(ident))
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(ident))
        loaded = cur.fetchone()[0]
    conn.commit()
    if loaded != N_TRIPS or n != N_TRIPS:
        raise RuntimeError(f"expected {N_TRIPS} trips, loaded {loaded}")
    print(f"loaded {loaded} trips into {table}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("load", "export"))
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--table", default="porto_traj")
    args = ap.parse_args()
    with psycopg.connect(args.dsn) as conn:
        if args.action == "export":
            conn.read_only = True
            export(conn, args.table)
        else:
            load(conn, args.table)


if __name__ == "__main__":
    main()
