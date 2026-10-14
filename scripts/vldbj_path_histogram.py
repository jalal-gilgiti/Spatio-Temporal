"""Path-aware space-time histogram baseline.

Estimand: the number of distinct trajectories whose (linearly interpolated) path
passes through the query box during its time window, as for the learned models.

Build:
* Read every trajectory's instants from the database (read-only), using the same
  extraction SQL as ``src/estimator_module/_point_utils.py``.
* Lay a regular grid of GX x GY x GT cells over the table's spatial bounds and time
  span. The default 16 x 16 x 18 = 4,608 int32 counters is comparable to Joint's
  4,673 parameters.
* Rasterize each linear segment exactly: split it at every grid-plane crossing and
  mark each piece's cell. n_c is the number of distinct trajectories touching
  cell c.

Combine (both reported): with f_c the fraction of cell c's volume inside the query box,
* sum rule  est = sum_c f_c * n_c   (a trajectory crossing several cells is counted
  once per cell, so this is biased upward when query boxes span cells);
* max rule  est = max_c f_c * n_c   (biased downward: it ignores trajectories that are
  not in the busiest cell).

Validation: for randomly chosen cells, n_c is compared with an exact
``atStbox(traj, cell) IS NOT NULL`` count (the label semantics). Boundary effects
make small mismatches possible; the report gives the agreement rate.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import query_from_dict  # noqa: E402
from src.data_module.workload_generator import STPredicate, st_predicate_sql  # noqa: E402
from src.real_experiment_steps import fetch_table_workload_bounds  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "tdrive": ("run/real_experiments/task7b/headline/workloads", "tdrive_traj"),
    "porto": ("run/real_experiments/multiseed_accuracy/workloads", "porto_traj"),
    "ais": ("run/vldbj_rev/ais/headline/workloads", "ais_traj"),
}
WORKLOADS = ("fixed_extent", "mixed_extent")


def fetch_points(conn, table: str):
    """Return trajectory ids, lon, lat, epoch seconds, ordered by (id, time)."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL(
            "SELECT trajectory_id, ST_X(g), ST_Y(g), extract(epoch from t) "
            "FROM {t}, LATERAL unnest(instants({t}.traj)) i, "
            "LATERAL (SELECT getValue(i) AS g, getTimestamp(i) AS t) sub "
            "ORDER BY trajectory_id, t").format(t=sql.Identifier(table)))
        rows = np.array(cur.fetchall(), dtype=float)
    return rows[:, 0].astype(np.int64), rows[:, 1], rows[:, 2], rows[:, 3]


class PathHistogram:
    def __init__(self, bounds, shape, tz):
        self.tz = tz  # naive datetimes in workloads are in the DB session time zone
        (self.x0, self.x1, self.y0, self.y1, self.t0, self.t1) = bounds
        self.shape = shape
        self.counts = np.zeros(shape, dtype=np.int64)

    def _norm(self, x, y, t):
        gx, gy, gt = self.shape
        return ((x - self.x0) / (self.x1 - self.x0) * gx,
                (y - self.y0) / (self.y1 - self.y0) * gy,
                (t - self.t0) / (self.t1 - self.t0) * gt)

    def _clip_cell(self, u, n):
        return np.clip(np.floor(u).astype(np.int64), 0, n - 1)

    def build(self, ids, x, y, t):
        gx, gy, gt = self.shape
        u, v, w = self._norm(x, y, t)
        cell_pairs = []  # (trajectory id, flat cell)
        # single-instant trajectories and every instant
        flat = np.ravel_multi_index((self._clip_cell(u, gx), self._clip_cell(v, gy),
                                     self._clip_cell(w, gt)), self.shape)
        cell_pairs.append(np.stack([ids, flat], axis=1))
        same = ids[1:] == ids[:-1]
        a = np.nonzero(same)[0]
        b = a + 1
        crosses = flat[a] != flat[b]
        for i, j in zip(a[crosses], b[crosses]):
            # split the segment at every grid-plane crossing; mark each piece's midpoint
            params = [0.0, 1.0]
            for p0, p1 in ((u[i], u[j]), (v[i], v[j]), (w[i], w[j])):
                if p1 != p0:
                    lo, hi = sorted((p0, p1))
                    for k in range(int(np.floor(lo)) + 1, int(np.ceil(hi))):
                        params.append((k - p0) / (p1 - p0))
            params = np.unique(np.clip(params, 0.0, 1.0))
            mids = (params[:-1] + params[1:]) / 2
            mu, mv, mw = (u[i] + mids * (u[j] - u[i]), v[i] + mids * (v[j] - v[i]),
                          w[i] + mids * (w[j] - w[i]))
            cells = np.ravel_multi_index((self._clip_cell(mu, gx), self._clip_cell(mv, gy),
                                          self._clip_cell(mw, gt)), self.shape)
            cell_pairs.append(np.stack([np.full(len(cells), ids[i]), cells], axis=1))
        pairs = np.unique(np.concatenate(cell_pairs), axis=0)
        np.add.at(self.counts.reshape(-1), pairs[:, 1], 1)

    def estimate(self, p: STPredicate) -> tuple[float, float]:
        gx, gy, gt = self.shape
        lo = self._norm(p.min_lon, p.min_lat, self._epoch(p.start_time))
        hi = self._norm(p.max_lon, p.max_lat, self._epoch(p.end_time))
        overlaps = []
        for n, a, b in zip(self.shape, lo, hi):
            k = np.arange(n)
            overlaps.append(np.clip(np.minimum(k + 1, b) - np.maximum(k, a), 0, 1))
        frac = overlaps[0][:, None, None] * overlaps[1][None, :, None] * overlaps[2][None, None, :]
        weighted = frac * self.counts
        return float(weighted.sum()), float(weighted.max())

    def _epoch(self, dt: datetime) -> float:
        return (dt if dt.tzinfo else dt.replace(tzinfo=self.tz)).timestamp()

    def cell_predicate(self, flat: int) -> STPredicate:
        ix, iy, it = np.unravel_index(flat, self.shape)
        gx, gy, gt = self.shape
        dx, dy, dt = (self.x1 - self.x0) / gx, (self.y1 - self.y0) / gy, (self.t1 - self.t0) / gt
        return STPredicate(self.x0 + ix * dx, self.x0 + (ix + 1) * dx, self.y0 + iy * dy,
                           self.y0 + (iy + 1) * dy,
                           datetime.fromtimestamp(self.t0 + it * dt, self.tz).replace(tzinfo=None),
                           datetime.fromtimestamp(self.t0 + (it + 1) * dt, self.tz).replace(tzinfo=None))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/path_histogram")
    ap.add_argument("--shape", nargs=3, type=int, default=(16, 16, 18))
    ap.add_argument("--validate-cells", type=int, default=50)
    ap.add_argument("--seed", type=int, default=90001)
    ap.add_argument("--datasets", nargs="+", default=["tdrive", "porto"], choices=list(SOURCES))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(args.shape)
    rows, meta = [], {"shape": shape, "counters": int(np.prod(shape)),
                      "bytes_int32": int(np.prod(shape)) * 4, "datasets": {}}
    rng = np.random.default_rng(args.seed)
    with psycopg.connect(args.dsn) as conn:
        conn.read_only = True
        for dataset, (wdir, table) in SOURCES.items():
            if dataset not in args.datasets:
                continue
            with conn.cursor() as cur:
                cur.execute("SHOW TimeZone")
                tz = ZoneInfo(cur.fetchone()[0])
            (x0, x1, y0, y1), (ts, te) = fetch_table_workload_bounds(conn, table)
            to_epoch = lambda d: (d if d.tzinfo else d.replace(tzinfo=tz)).timestamp()  # noqa: E731
            hist = PathHistogram((x0, x1, y0, y1, to_epoch(ts), to_epoch(te)), shape, tz)
            started = time.perf_counter()
            ids, x, y, t = fetch_points(conn, table)
            read_s = time.perf_counter() - started
            started = time.perf_counter()
            hist.build(ids, x, y, t)
            build_s = time.perf_counter() - started
            np.save(args.output_dir / f"{dataset}_counts.npy", hist.counts)
            # validation against exact atStbox counts on random non-empty cells
            nonzero = np.flatnonzero(hist.counts.reshape(-1))
            sample = rng.choice(nonzero, size=min(args.validate_cells, len(nonzero)), replace=False)
            checks = []
            with conn.cursor() as cur:
                for flat in sample:
                    pred = hist.cell_predicate(int(flat))
                    cur.execute(sql.SQL("SELECT count(*) FROM {} WHERE " + st_predicate_sql(pred))
                                .format(sql.Identifier(table)))
                    exact = int(cur.fetchone()[0])
                    checks.append({"cell": int(flat), "histogram": int(hist.counts.reshape(-1)[flat]),
                                   "exact": exact})
            agree = [c["histogram"] == c["exact"] for c in checks]
            ratios = [(c["histogram"] + 1) / (c["exact"] + 1) for c in checks]
            meta["datasets"][dataset] = {
                "table": table, "n_points": int(len(ids)), "n_trajectories": int(len(np.unique(ids))),
                "bounds": [x0, x1, y0, y1, ts.isoformat(), te.isoformat()],
                "read_seconds": read_s, "build_seconds": build_s,
                "nonzero_cells": int(len(nonzero)),
                "validation": {"cells": len(checks), "exact_agreement": float(np.mean(agree)),
                               "median_ratio_hist_over_exact": float(np.median(ratios)),
                               "max_abs_log10_ratio": float(np.max(np.abs(np.log10(ratios)))),
                               "rows": checks},
            }
            for workload in WORKLOADS:
                payload = json.loads((ROOT / wdir / f"{dataset}__{workload}.json").read_text())
                for i, q in enumerate(payload["evaluation_queries"]):
                    wq = query_from_dict(q)
                    s, m = hist.estimate(wq.predicate)
                    rows.append({"dataset": dataset, "workload": workload, "query_index": i,
                                 "true_cardinality": wq.true_cardinality,
                                 "path_hist_sum": s, "path_hist_max": m})
            conn.rollback()
    with (args.output_dir / "estimates.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    meta["created_utc"] = datetime.now(timezone.utc).isoformat()
    write_results_json(meta, args.output_dir / "metadata.json")
    print(json.dumps({d: {k: v for k, v in m.items() if k in ("n_points", "build_seconds", "validation")}
                      | {"validation": {k: v for k, v in m["validation"].items() if k != "rows"}}
                      for d, m in meta["datasets"].items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
