"""Non-learned estimators on the cardinality-stratified evaluation sets.

Evaluates Native-BBox, Native-Served (planner EXPLAIN estimates), Trajectory
Sampling (1% samples, seeds 17/23/47/59/71; connection-local temp tables) and the path
histogram (sum and max rules, rebuilt from the saved grid counts) on the
cardinality-stratified mixed-extent sets in run/vldbj_rev/task1e/workloads/.

Native uses the primary statistics (target 1000): the original tables for T-Drive and AIS
(read-only), and for Porto a full-table copy in the scratch schema ``vldbj_t1e`` ANALYZEd at
target 1000 (deterministic: the sample covers the whole table), dropped at the end. The
original tables are never ANALYZEd. Before scoring, native estimates must reproduce the
primary headline estimates exactly.

Bounded q-error: estimates clamped to [1, N]. Per bucket, the median and p95 are the mean
over seeds of the per-seed statistic, as in the main tables. Before scoring, the rebuilt
path histograms must reproduce the saved headline estimates
(run/vldbj_rev/{,ais/}path_histogram/estimates.csv) exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.multiseed_accuracy_battery import query_from_dict  # noqa: E402
from scripts.vldbj_native_reestimate import FORMS  # noqa: E402
from scripts.vldbj_stats_target import explain_rows  # noqa: E402
from scripts.vldbj_path_histogram import PathHistogram  # noqa: E402
from src.estimator_module.trajectory_sampling import TrajectorySamplingEstimator  # noqa: E402
from src.results_io import write_results_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = {"tdrive": "tdrive_traj", "porto": "porto_traj", "ais": "ais_traj"}
HIST = {"tdrive": ("run/vldbj_rev/path_histogram", "tdrive_counts.npy"),
        "porto": ("run/vldbj_rev/path_histogram", "porto_counts.npy"),
        "ais": ("run/vldbj_rev/ais/path_histogram", "ais_counts.npy")}
HEADLINE = {"tdrive": "run/real_experiments/task7b/headline/workloads/tdrive__{w}.json",
            "porto": "run/real_experiments/multiseed_accuracy/workloads/porto__{w}.json",
            "ais": "run/vldbj_rev/ais/headline/workloads/ais__{w}.json"}
SAMPLE_SEEDS = (17, 23, 47, 59, 71)
BUCKETS = ("0", "1-10", "11-100", "101-1,000", ">1,000")
N = 100_000.0
SCHEMA = "vldbj_t1e"
NATIVE_REF = {"tdrive": "run/vldbj_rev/native/explain_estimates_primary.csv",
              "porto": "run/vldbj_rev/native/explain_estimates_primary.csv",
              "ais": "run/vldbj_rev/ais/native/explain_estimates.csv"}


def qerr(t: np.ndarray, e: np.ndarray) -> np.ndarray:
    e = np.clip(e, 1.0, N) + 1.0
    t = t + 1.0
    return np.maximum(t / e, e / t)


def load_hist(ds: str, tz: ZoneInfo) -> PathHistogram:
    d, f = HIST[ds]
    meta = json.loads((ROOT / d / "metadata.json").read_text())
    x0, x1, y0, y1, ts, te = meta["datasets"][ds]["bounds"]
    ep = lambda s: datetime.fromisoformat(s).replace(tzinfo=tz).timestamp()  # noqa: E731
    h = PathHistogram((x0, x1, y0, y1, ep(ts), ep(te)), tuple(meta["shape"]), tz)
    h.counts = np.load(ROOT / d / f)
    # validation: reproduce the saved headline estimates exactly
    saved = [r for r in csv.DictReader((ROOT / d / "estimates.csv").open()) if r["dataset"] == ds]
    for w in ("fixed_extent", "mixed_extent"):
        qs = json.loads((ROOT / HEADLINE[ds].format(w=w)).read_text())["evaluation_queries"]
        ref = {int(r["query_index"]): (float(r["path_hist_sum"]), float(r["path_hist_max"]))
               for r in saved if r["workload"] == w}
        for i, q in enumerate(qs):
            s, m = h.estimate(query_from_dict(q).predicate)
            if abs(s - ref[i][0]) > 1e-6 * max(1, ref[i][0]) or abs(m - ref[i][1]) > 1e-6 * max(1, ref[i][1]):
                raise RuntimeError(f"{ds}/{w}/{i}: rebuilt histogram does not reproduce the saved estimate")
    return h


def check_native(cur, ds: str, table: str) -> None:
    ref = [r for r in csv.DictReader((ROOT / NATIVE_REF[ds]).open()) if r["dataset"] == ds]
    qs = {w: json.loads((ROOT / HEADLINE[ds].format(w=w)).read_text())["evaluation_queries"]
          for w in ("fixed_extent", "mixed_extent")}
    for r in ref:
        p = query_from_dict(qs[r["workload"]][int(r["query_index"])]).predicate
        for form in ("native_bbox", "native_served"):
            if explain_rows(cur, table, FORMS[form](p)) != float(r[form]):
                raise RuntimeError(f"{ds}/{r['workload']}/{r['query_index']}/{form}: native does not reproduce the primary estimate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", default="postgresql://postgres@localhost:5434/mobilitydb")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "run/vldbj_rev/task1e/nolearn")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_query, summary = [], []
    with psycopg.connect(args.dsn) as ro, psycopg.connect(args.dsn, autocommit=True) as rw:
        ro.read_only = True
        cur = ro.cursor()
        cur.execute("SHOW TimeZone")
        tz = ZoneInfo(cur.fetchone()[0])
        wcur = rw.cursor()
        wcur.execute(f"CREATE SCHEMA {SCHEMA}")
        try:
            wcur.execute(f"CREATE TABLE {SCHEMA}.porto_traj AS SELECT * FROM public.porto_traj")
            wcur.execute(f"ALTER TABLE {SCHEMA}.porto_traj ALTER COLUMN traj SET STATISTICS 1000")
            wcur.execute(f"ANALYZE {SCHEMA}.porto_traj")
            native_cur = {"tdrive": (cur, "tdrive_traj"), "porto": (wcur, f"{SCHEMA}.porto_traj"),
                          "ais": (cur, "ais_traj")}
            for ds, (c, t) in native_cur.items():
                check_native(c, ds, t)
            print("native reproduces the primary headline estimates")
            native = {}
            for ds in TABLES:
                c, t = native_cur[ds]
                qs = json.loads((ROOT / f"run/vldbj_rev/task1e/workloads/{ds}__mixed_stratified.json").read_text())["evaluation_queries"]
                native[ds] = {form: np.array([[explain_rows(c, t, FORMS[form](query_from_dict(q).predicate)) for q in qs]])
                              for form in ("native_bbox", "native_served")}
            ro.rollback()
        finally:
            wcur.execute(f"DROP SCHEMA {SCHEMA} CASCADE")
        for ds, table in TABLES.items():
            w = json.loads((ROOT / f"run/vldbj_rev/task1e/workloads/{ds}__mixed_stratified.json").read_text())
            qs = w["evaluation_queries"]
            preds = [query_from_dict(q).predicate for q in qs]
            truth = np.array([q["true_cardinality"] for q in qs], float)
            bucket = [q["bucket"] for q in qs]
            est: dict[str, np.ndarray] = dict(native[ds])
            h = load_hist(ds, tz)
            hs = np.array([h.estimate(p) for p in preds])
            est["path_hist_sum"], est["path_hist_max"] = hs[None, :, 0], hs[None, :, 1]
            ts_rows = []
            for seed in SAMPLE_SEEDS:
                sampler = TrajectorySamplingEstimator(sample_fraction=0.01, seed=seed)
                sampler.fit(rw, table)
                try:
                    ts_rows.append([float(sampler.estimate(p)) for p in preds])
                finally:
                    sampler.close()
            est["trajectory_sampling"] = np.array(ts_rows)
            ro.rollback()
            for name, mat in est.items():
                q = qerr(truth[None, :], mat)
                for i in range(len(qs)):
                    per_query.append({"dataset": ds, "estimator": name, "query_id": qs[i]["query_id"],
                                      "bucket": bucket[i], "true_cardinality": int(truth[i]),
                                      "estimate_mean_over_seeds": float(mat[:, i].mean()),
                                      "qerror_mean_over_seeds": float(q[:, i].mean())})
                for b in BUCKETS:
                    m = np.array([x == b for x in bucket])
                    row = {"dataset": ds, "estimator": name, "bucket": b, "n_queries": int(m.sum()),
                           "n_seeds": mat.shape[0]}
                    if m.any():
                        row["median"] = float(np.mean(np.median(q[:, m], axis=1)))
                        row["p95"] = float(np.mean(np.percentile(q[:, m], 95, axis=1)))
                        row["max"] = float(q[:, m].max())
                    summary.append(row)
            print(ds, "done", {b: int(sum(x == b for x in bucket)) for b in BUCKETS})
    for name, data in (("per_query", per_query), ("bucket_summary", summary)):
        with (args.output_dir / f"{name}.csv").open("w", newline="") as fh:
            fields = list(dict.fromkeys(k for r in data for k in r))
            wtr = csv.DictWriter(fh, fieldnames=fields); wtr.writeheader(); wtr.writerows(data)
    write_results_json({"q_error": "bounded to [1, N], N = 100,000",
                        "aggregation": "per bucket: mean over seeds of the per-seed median / p95; max over seeds and queries",
                        "trajectory_sampling": f"1% samples, seeds {list(SAMPLE_SEEDS)}",
                        "path_histogram": "rebuilt from saved grid counts; reproduces the saved headline estimates exactly",
                        "native": "EXPLAIN row estimates at statistics target 1000: original tables (tdrive, ais); porto from a full-table copy in a dropped scratch schema; reproduces the primary headline estimates exactly"},
                       args.output_dir / "metadata.json")


if __name__ == "__main__":
    main()
