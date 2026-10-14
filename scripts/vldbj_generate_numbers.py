"""Generate paper/numbers.tex from the result files.

Every number in the paper is a LaTeX macro. paper/macros.txt lists the macros in output
order, and each has a function below that computes its value from the named source file.
The script fails on missing sources and warns about listed macros it does not implement.

    python scripts/vldbj_generate_numbers.py            # writes paper/numbers.tex
    python scripts/vldbj_generate_numbers.py --check    # print values, write nothing
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACRO_LIST = ROOT / "paper/macros.txt"
OUT = ROOT / "paper/numbers.tex"
NATIVE = ROOT / "run/vldbj_rev/native"
STATS = ROOT / "run/vldbj_rev/stats"


# ---------------------------------------------------------------- formatting
def dp(x: float, n: int) -> str:
    return f"{x:.{n}f}"


def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def sci2(x: float) -> str:
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** exp
    if round(mant, 1) >= 10:
        mant, exp = mant / 10, exp + 1
    return rf"\ensuremath{{{mant:.1f}\times10^{{{exp}}}}}"


RAW_P_FLOOR = 2 / 10001  # smallest attainable two-sided bootstrap p with B = 10,000


def fmt_p(p: float, floor: float = RAW_P_FLOOR) -> str:
    """p at its resolution floor prints as '$\\le$' + floor (4dp); otherwise 3dp."""
    return f"$\\le${dp(floor, 4)}" if p <= floor * (1 + 1e-9) else dp(p, 3)


# ---------------------------------------------------------------- sources
def rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def one(table: list[dict], **where) -> dict:
    hit = [r for r in table if all(r[k] == v for k, v in where.items())]
    if len(hit) != 1:
        raise RuntimeError(f"expected exactly one row for {where}, found {len(hit)}")
    return hit[0]


def acc(clamp: str = "planner_all") -> list[dict]:
    name = "task1_accuracy_table.csv" if clamp == "unclamped" else f"task1_accuracy_table__{clamp}.csv"
    return rows(NATIVE / name)


DISPLAY_SETTING = {("tdrive", "fixed_extent"): "T-Drive & Fixed", ("tdrive", "mixed_extent"): "T-Drive & Mixed",
                   ("porto", "fixed_extent"): "Porto & Fixed", ("porto", "mixed_extent"): "Porto & Mixed",
                   ("ais", "fixed_extent"): "AIS & Fixed", ("ais", "mixed_extent"): "AIS & Mixed"}
DISPLAY_EST = {"joint": "Joint", "separable": "Separable", "no_coupling": "No-coupling",
               "spatial_only": "SpatialOnly", "native_bbox": "Native-BBox",
               "path_hist_sum": "Path-hist (sum)", "path_hist_max": "Path-hist (max)",
               "box_oracle": "Box oracle"}


def paired_rows(subset: str) -> str:
    out = []
    for r in rows(STATS / "paired_bootstrap_clamped.csv"):
        if r["subset"] != subset:
            continue
        sig = r"$^{*}$" if r["significant_holm_0_05"] == "True" else ""
        out.append(f"{DISPLAY_SETTING[(r['dataset'], r['workload'])]} & {DISPLAY_EST[r['left']]} vs "
                   f"{DISPLAY_EST[r['right']]} & {r['n_queries']} & {dp(float(r['median_ratio']), 3)} "
                   f"[{dp(float(r['ratio_ci95_lower']), 3)}, {dp(float(r['ratio_ci95_upper']), 3)}] & "
                   f"{fmt_p(float(r['p_raw']))} & {fmt_p(float(r['p_holm']), RAW_P_FLOOR * int(r['family_size']))}{sig} \\\\")
    return "\n".join(out)


def n_significant(family: str, subset: str) -> str:
    hits = [r for r in rows(STATS / "paired_bootstrap_clamped.csv")
            if r["family"].startswith(family) and r["subset"] == subset]
    return fmt_int(sum(r["significant_holm_0_05"] == "True" for r in hits))


def table_rows_common() -> str:
    tables = json.loads((NATIVE / "metadata.json").read_text())["tables"]
    a, b = tables["tdrive_traj"]["row_count"], tables["porto_traj"]["row_count"]
    if a != b:
        raise RuntimeError(f"row counts differ: {a} vs {b}")
    return fmt_int(a)


def pg_default_rows() -> str:
    values = {r["pg_default"] for r in rows(NATIVE / "explain_estimates.csv")}
    if len(values) != 1:
        raise RuntimeError(f"pg_default is not constant: {values}")
    return fmt_int(float(values.pop()))


def syn_ratio(fn) -> str:
    data = json.loads((ROOT / "run/revalidation/separable_grid_control.json").read_text())
    return dp(fn(r["ratio_over_joint"]["separable"] for r in data["rows"]), 3)


def one_m_label_seconds() -> str:
    hits = [float(r["label_seconds"]) for r in rows(ROOT / "run/real_experiments/task7b/scalability/training_budget_curve.csv")
            if r["scale"] == "1000000" and r["training_budget"] == "6400"]
    if not hits:
        raise RuntimeError("no 1M/6,400-label rows")
    return fmt_int(min(hits))


def pct0(x: float) -> str:
    return f"{round(100 * x)}\\%"


def rows_tsv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def budget_rows() -> list[dict]:
    hits = [r for r in rows(ROOT / "run/real_experiments/task7b/scalability/training_budget_curve.csv")
            if r["scale"] == "1000000" and r["training_budget"] == "6400"]
    if not hits:
        raise RuntimeError("no 1M/6,400-label rows")
    return hits


def transfer_medians() -> list[float]:
    data = json.loads((ROOT / "run/real_experiments/task7b/transfer_sanity.json").read_text())
    return [statistics.median(v["transfer"]["per_query"]) for v in data["pairs"].values()]


def plan_row(condition: str) -> dict:
    return one(rows(ROOT / "run/vldbj_rev/plan_rerun/analysis/plan_metrics_summary.csv"),
               dataset="tdrive", condition=condition)


def plan_overhead_range() -> str:
    vals = [round(100 * (1 - float(plan_row(c)["end_to_end_speedup_median"])))
            for c in ("joint_dm", "separable_dm", "no_coupling_dm", "spatial_only_dm")]
    lo, hi = min(vals), max(vals)
    return f"{lo}\\%" if lo == hi else f"{lo}--{hi}\\%"


def stats_row(dataset: str, workload: str, target: int) -> dict:
    return one(rows(ROOT / "run/vldbj_rev/stats_target/summary.csv"), dataset=dataset, workload=workload,
               form="native_bbox", target=str(target))


def jfile(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def td_manifest() -> dict:
    return jfile("run/real_experiments/task7_dataset_validity/tdrive_corrected_sampling_manifest.json")


def zero_count(rel: str) -> str:
    return fmt_int(sum(q["true_cardinality"] == 0 for q in jfile(rel)["evaluation_queries"]))


PROFILE = ROOT / "run/vldbj_rev/native/table_profile.json"


def table_profile() -> dict:
    """Read-only per-table profile, snapshotted once so the generator rarely needs the DB."""
    if not PROFILE.exists():
        import psycopg
        out = {}
        with psycopg.connect("postgresql://postgres@localhost:5434/mobilitydb") as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                for table in ("tdrive_traj", "porto_traj"):
                    cur.execute(
                        f"SELECT sum(numInstants(traj)), "
                        f"percentile_cont(0.5) WITHIN GROUP (ORDER BY numInstants(traj)), "
                        f"percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch from duration(traj)) / 60), "
                        f"count(*) FROM {table}")
                    total, med_pts, med_dur, n = cur.fetchone()
                    out[table] = {"sum_instants": int(total), "median_instants": float(med_pts),
                                  "median_duration_min": float(med_dur), "rows": int(n)}
            conn.rollback()
        out["created_utc"] = datetime.now(timezone.utc).isoformat()
        PROFILE.write_text(json.dumps(out, indent=1))
    return json.loads(PROFILE.read_text())


def sep_natbox_tdmix() -> dict:
    return one(rows(STATS / "paired_bootstrap_clamped.csv"), family="a_learned_vs_native_bbox", subset="all",
               dataset="tdrive", workload="mixed_extent", left="separable", right="native_bbox")


# ---------------------------------------------------------------- table bodies and batch-3 rows
SETTINGS = (("tdrive", "fixed_extent", "T-Drive, fixed"), ("tdrive", "mixed_extent", "T-Drive, mixed"),
            ("porto", "fixed_extent", "Porto, fixed"), ("porto", "mixed_extent", "Porto, mixed"))
ACC_ORDER = (("pg_default", "PostgreSQL default"), ("native_bbox", "Native-BBox"), ("native_served", "Native-Served"),
             ("trajectory_sampling", "Trajectory Sampling"), ("path_hist_sum", "Path histogram (sum)"),
             ("path_hist_max", "Path histogram (max)"), ("joint", "Joint"), ("separable", "Separable"),
             ("no_coupling", "No-coupling"), ("spatial_only", "SpatialOnly"))
ORIGIN_ORDER = (("independent_histogram", "Independent histogram"), ("joint_histogram", "Joint histogram"),
                ("minskew_origin", "MinSkew-Origin"), ("origin_sampling", "Origin sampling"))
LEARNED_EST = ("joint", "separable", "no_coupling", "spatial_only")


def acc_body(table: list[dict], order, sci_learned_max: bool = False) -> str:
    blocks = []
    for ds, wl, label in SETTINGS:
        lines = []
        for k, (est, name) in enumerate(order):
            r = one(table, dataset=ds, workload=wl, estimator=est)
            med, lo, hi, p95, mx = (float(r[c]) for c in ("median", "median_ci95_lower", "median_ci95_upper", "p95", "max"))
            big = lambda x, n: fmt_int(x) if x >= 1000 else dp(x, n)  # noqa: E731
            if est == "pg_default":
                cells = f"{fmt_int(med)} [{fmt_int(lo)}, {fmt_int(hi)}] & {fmt_int(p95)} & {fmt_int(mx)}"
            else:
                mx_s = sci2(mx) if (sci_learned_max and est in LEARNED_EST and mx >= 1e4) else big(mx, 1)
                cells = f"{big(med, 2)} [{big(lo, 2)}, {big(hi, 2)}] & {big(p95, 1)} & {mx_s}"
            lines.append(f"{label if k == 0 else ''} & {name} & {cells} \\\\")
        blocks.append("\n".join(lines))
    return "\n\\addlinespace\n".join(blocks)


PLAN_ORDER = (("native", "Native"), ("true_cardinality_input", "True cardinalities"),
              ("trajectory_sampling", "Trajectory Sampling"))
PLAN_MODELS = (("joint", "Joint"), ("separable", "Separable"), ("no_coupling", "No-coupling"),
               ("spatial_only", "SpatialOnly"))


def plan_body(run_dir: str, with_learned: bool) -> str:
    base = ROOT / run_dir / "analysis"
    summ = {r["condition"]: r for r in rows(base / "plan_metrics_summary.csv")}
    perq = rows(base / "per_query_plan_metrics.csv")
    by_q: dict[str, dict[str, dict]] = {}
    for r in perq:
        by_q.setdefault(r["query_id"], {})[r["condition"]] = r
    order = list(PLAN_ORDER)
    if with_learned:
        for key, name in PLAN_MODELS:
            order += [(key, f"{name} (original)"), (f"{key}_clamped", f"{name} (clamped)"), (f"{key}_dm", f"{name} (DM)")]
    out = []
    for cond, name in order:
        r = summ[cond]
        regs = [float(by_q[q][cond]["regret"]) for q in by_q]
        gm = math.exp(sum(math.log(x) for x in regs) / len(regs))
        w = t = l = 0
        for q in by_q:
            a, n = by_q[q][cond], by_q[q]["native"]
            if a["plan_tree_fingerprint"] == n["plan_tree_fingerprint"]:
                t += 1
            elif float(a["regret"]) < float(n["regret"]):
                w += 1
            else:
                l += 1
        out.append(f"{name} & {pct0(float(r['plan_change_rate']))} & {pct0(float(r['oracle_agreement_rate']))} & "
                   f"{dp(float(r['regret_median']), 3)} & {dp(gm, 3)} & {pct0(float(r['bad_plan_rate_gt_1_25']))} & "
                   f"{w}/{t}/{l} & {dp(float(r['end_to_end_ms_median']), 1)} & "
                   f"{dp(float(r['end_to_end_speedup_median']), 3)} & {dp(float(r['inference_ms_median']), 1)} \\\\")
    return "\n".join(out)


def budget_body() -> str:
    import glob
    import numpy as np
    n = 1_000_000.0
    out = []
    for est, name in PLAN_MODELS:
        cells = []
        for b in (800, 1600, 3200, 6400):
            meds, p95s = [], []
            files = sorted(glob.glob(str(ROOT / f"run/real_experiments/task7b/scalability/runs/learned/"
                                            f"tdrive__n1000000__mixed_extent__b{b}__{est}__s*.json")))
            if len(files) != 3:
                raise RuntimeError(f"expected 3 seeds for b{b} {est}, found {len(files)}")
            for f in files:
                preds = json.loads(Path(f).read_text())["predictions"]
                t = np.array([p["true_cardinality"] for p in preds], float) + 1
                e = np.clip([p["predicted_cardinality"] for p in preds], 1, n) + 1
                q = np.maximum(t / e, e / t)
                meds.append(float(np.median(q))); p95s.append(float(np.percentile(q, 95)))
            cells.append(dp(statistics.mean(meds), 3))
            last_p95 = statistics.mean(p95s)
        out.append(f"{name} & {' & '.join(cells)} & {dp(last_p95, 2)} \\\\")
    return "\n".join(out)


def pb(**where) -> dict:
    return one(rows(STATS / "paired_bootstrap_clamped.csv"), **where)


def fam_size(prefix: str) -> int:
    sizes = {r["family_size"] for r in rows(STATS / "paired_bootstrap_clamped.csv") if r["family"].startswith(prefix)}
    if len(sizes) != 1:
        raise RuntimeError(f"family {prefix} sizes differ by subset: {sizes}")
    return int(sizes.pop())


def plan_summary(run_dir: str, cond: str) -> dict:
    return one(rows(ROOT / run_dir / "analysis/plan_metrics_summary.csv"), condition=cond)


def spo_rows() -> list[dict]:
    return [r for r in rows(ROOT / "run/vldbj_rev/plan_rerun/analysis/per_query_plan_metrics.csv")
            if r["condition"] == "spatial_only"]


def onem_db() -> dict:
    return one(rows(ROOT / "run/real_experiments/task7b/scalability/resources.csv"),
               resource_kind="database", scale="1000000")


def onem_infer(fn) -> str:
    # resources.csv has learned rows only at 800 labels; the 6,400-label runs carry inference_p50_ms
    import glob
    per_est = {}
    for f in glob.glob(str(ROOT / "run/real_experiments/task7b/scalability/runs/learned/tdrive__n1000000__*__b6400__*__s*.json")):
        d = json.loads(Path(f).read_text())
        per_est.setdefault(d["estimator"], []).append(float(d["summary"]["inference_p50_ms"]))
    if len(per_est) != 4:
        raise RuntimeError(f"expected 4 learned estimators at 1M/6,400, found {sorted(per_est)}")
    return dp(fn(statistics.median(v) for v in per_est.values()), 2)


def retrain_max_dev() -> str:
    devs = []
    for new in sorted((ROOT / "run/vldbj_rev/headline_retrain/runs").glob("*.json")):
        d = new.name.split("__")[0]
        old = ROOT / ("run/real_experiments/task7b/headline/runs" if d == "tdrive" else
                      "run/real_experiments/multiseed_accuracy/runs") / new.name
        a, b = json.loads(new.read_text())["summary"], json.loads(old.read_text())["summary"]
        devs += [100 * abs(a[k] / b[k] - 1) for k in ("median", "p90", "p95", "max")]
    return dp(max(devs), 2) + "\\%"


def sep_best_settings() -> str:
    table = acc()
    n = 0
    for ds, wl, _ in SETTINGS:
        meds = {e: float(one(table, dataset=ds, workload=wl, estimator=e)["median"]) for e in LEARNED_EST}
        n += all(meds["separable"] < v for e, v in meds.items() if e != "separable")
    return fmt_int(n)


def syn() -> dict:
    return jfile("run/revalidation/separable_grid_control.json")


def syn_ratio_stats(row: dict, arm: str) -> tuple[float, float]:
    ratios = [a / j for a, j in zip(row["per_seed"][arm], row["per_seed"]["joint"], strict=True)]
    return statistics.mean(ratios), statistics.stdev(ratios)


def syn_rows() -> str:
    out = []
    for r in syn()["rows"]:
        stats = {arm: syn_ratio_stats(r, arm) for arm in ("no_coupling", "separable", "separable_ifeat")}
        best = min(stats, key=lambda a: stats[a][0])
        cells = []
        for arm in ("no_coupling", "separable", "separable_ifeat"):
            m, sd = stats[arm]
            txt = f"{dp(m, 3)} $\\pm$ {dp(sd, 3)}"
            cells.append(f"\\textbf{{{txt}}}" if arm == best else txt)
        out.append(f"{dp(r['correlation_strength'], 2)} & {dp(r['measured_mi'], 4)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(out)


def syn_slope() -> str:
    xs, ys = [], []
    for r in syn()["rows"]:
        xs.append(r["correlation_strength"]); ys.append(syn_ratio_stats(r, "separable")[0])
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return f"${'+' if slope >= 0 else '-'}${dp(abs(slope), 3)}"


def ridge_rows() -> str:
    out = []
    for r in syn()["rows"]:
        h = r["interaction_headroom_r2"]
        out.append(f"{dp(r['correlation_strength'], 2)} & {dp(r['separable_r2_heldout'], 3)} & "
                   f"{dp(r['joint_r2_heldout'], 3)} & ${'+' if h >= 0 else '-'}${dp(abs(h), 3)} \\\\")
    return "\n".join(out)



def sig2(x: float) -> str:
    return f"{x:.2g}" if abs(x) >= 1e-4 else f"{x:.1e}"


def budget_cells() -> dict[str, list[float]]:
    import glob
    import numpy as np
    out = {}
    for est, _ in PLAN_MODELS:
        vals = []
        for b in (800, 1600, 3200, 6400):
            meds = []
            for f in sorted(glob.glob(str(ROOT / f"run/real_experiments/task7b/scalability/runs/learned/"
                                           f"tdrive__n1000000__mixed_extent__b{b}__{est}__s*.json"))):
                preds = json.loads(Path(f).read_text())["predictions"]
                t = np.array([p["true_cardinality"] for p in preds], float) + 1
                e = np.clip([p["predicted_cardinality"] for p in preds], 1, 1_000_000) + 1
                meds.append(float(np.median(np.maximum(t / e, e / t))))
            vals.append(statistics.mean(meds))
        out[est] = vals
    return out


def syn_ifeat() -> list[tuple[float, float]]:
    return [syn_ratio_stats(r, "separable_ifeat") for r in syn()["rows"]]


def neural_same_shares() -> list[float]:
    perq = rows(ROOT / "run/vldbj_rev/plan_rerun/analysis/per_query_plan_metrics.csv")
    by_q: dict[str, dict[str, dict]] = {}
    for r in perq:
        by_q.setdefault(r["query_id"], {})[r["condition"]] = r
    conds = [f"{m}{suf}" for m in ("joint", "separable", "no_coupling") for suf in ("", "_clamped", "_dm")]
    return [sum(by_q[q][c]["plan_tree_fingerprint"] == by_q[q]["native"]["plan_tree_fingerprint"] for q in by_q) / len(by_q)
            for c in conds]


def same_plan_overhead_range() -> str:
    perq = rows(ROOT / "run/vldbj_rev/plan_rerun/analysis/per_query_plan_metrics.csv")
    by_q: dict[str, dict[str, dict]] = {}
    for r in perq:
        by_q.setdefault(r["query_id"], {})[r["condition"]] = r
    vals = []
    for c in ("joint_dm", "separable_dm", "no_coupling_dm", "spatial_only_dm"):
        d = [100 * (float(by_q[q][c]["end_to_end_ms"]) / float(by_q[q]["native"]["end_to_end_ms"]) - 1)
             for q in by_q if by_q[q][c]["plan_tree_fingerprint"] == by_q[q]["native"]["plan_tree_fingerprint"]]
        vals.append(round(statistics.median(d)))
    lo, hi = min(vals), max(vals)
    return f"{lo}\\%" if lo == hi else f"{lo}--{hi}\\%"


def plan_q(run_dir: str, cond: str) -> list[dict]:
    return [r for r in rows(ROOT / run_dir / "analysis/per_query_plan_metrics.csv") if r["condition"] == cond]


def join_rows(ds: str) -> list[dict]:
    return [r for r in rows(ROOT / "run/vldbj_rev/join/join_estimates.csv") if r["dataset"] == ds]


def join_nonempty(ds: str) -> list[dict]:
    return [r for r in join_rows(ds) if r["true_pairs"] and int(r["true_pairs"]) > 0]


def retime_cmp() -> list[dict]:
    return rows(ROOT / "run/vldbj_rev/plan_retime_lowload/comparison/retime_vs_main.csv")


def retime_load_med() -> str:
    checks = rows_tsv_noheader(ROOT / "run/vldbj_rev/plan_retime_lowload/load_checks.tsv")
    trigger = [c for c in checks if float(c[1]) < 8 and float(c[2]) < 8]
    return fmt_int(statistics.median(float(c[1]) for c in trigger))


def rows_tsv_noheader(path: Path) -> list[list[str]]:
    return [line.split("\t") for line in path.read_text().splitlines() if line.strip()]


def p_holm_fmt(row: dict, family_prefix: str) -> str:
    floor = 2 / 10001 * fam_size(family_prefix)
    p = float(row["p_holm"])
    return f"$\\le${dp(floor, 3)}" if p <= floor + 1e-12 else dp(p, 3)



def signed2(x: float) -> str:
    if round(abs(x), 2) == 0:
        return "$+$0.00"  # a value that rounds to zero gets "+" (avoids a misleading "-0.00")
    return f"${'+' if x >= 0 else '-'}${dp(abs(x), 2)}"


def ais_dec(workload: str, bucket: str, component: str) -> dict:
    return one(rows(ROOT / "run/vldbj_rev/ais/native/native_decomposition.csv"), dataset="ais",
               workload=workload, bucket=bucket, component=component)



def calib() -> dict:
    return jfile("run/vldbj_rev/calibration/calibration.json")


def med_fmt(x: float) -> str:
    return dp(x, 1)



INDEP = ROOT / "run/vldbj_rev/independence"


def indep_summary(ds: str) -> dict:
    return one(rows(INDEP / "summary.csv"), dataset=ds)


def marginal_split(ds: str) -> dict:
    d = [r for r in rows(INDEP / "per_query.csv") if r["dataset"] == ds]
    sp = [math.log10((float(r["eS"]) + 1) / (int(r["S_exact"]) + 1)) for r in d]
    tm = [math.log10((float(r["eT"]) + 1) / (int(r["T_exact"]) + 1)) for r in d]
    return {"spatial": statistics.median(sp), "temporal": statistics.median(tm),
            "spatial_under": sum(x < 0 for x in sp) / len(sp)}



BOX_SETTINGS = SETTINGS + (("ais", "fixed_extent", "AIS, fixed"), ("ais", "mixed_extent", "AIS, mixed"))


def box_oracle_rows() -> str:
    a, ne = acc(), acc("planner_nonempty")
    out = []
    for ds, wl, label in BOX_SETTINGS:
        cells = []
        for t in (a, ne):
            r = one(t, dataset=ds, workload=wl, estimator="box_oracle")
            big = lambda x, n: fmt_int(x) if x >= 1000 else dp(x, n)  # noqa: E731
            cells.append(f"{dp(float(r['median']), 2)} [{dp(float(r['median_ci95_lower']), 2)}, "
                         f"{dp(float(r['median_ci95_upper']), 2)}] & {big(float(r['p95']), 1)} & {big(float(r['max']), 1)}")
        out.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
    return "\n".join(out)


def fam_d(subset: str, right: str) -> list[dict]:
    return [r for r in rows(STATS / "paired_bootstrap_clamped.csv")
            if r["family"] == "d_box_oracle" and r["subset"] == subset and r["right"] == right]



STRAT = ROOT / "run/vldbj_rev/task1e"
STRAT_BUCKETS = ("0", "1-10", "11-100", "101-1,000", ">1,000")
STRAT_ORDER = (("native_bbox", "Native-BBox"), ("native_served", "Native-Served"),
               ("trajectory_sampling", "Trajectory Sampling"), ("path_hist_sum", "Path histogram (sum)"),
               ("path_hist_max", "Path histogram (max)"))


def strat_nolearn_rows(stat: str) -> str:
    """Stratified sets, non-learned estimators: per dataset and estimator, bounded q-error `stat` per bucket
    (mean over seeds of the per-seed statistic); "--" where the stratified set has no queries."""
    t = rows(STRAT / "nolearn/bucket_summary.csv")
    blocks = []
    for ds, label in (("tdrive", "T-Drive"), ("porto", "Porto"), ("ais", "AIS")):
        lines = []
        for k, (est, name) in enumerate(STRAT_ORDER):
            cells = []
            for b in STRAT_BUCKETS:
                r = one(t, dataset=ds, estimator=est, bucket=b)
                cells.append(dp(float(r[stat]), 2) if int(r["n_queries"]) else "--")
            lines.append(f"{label if k == 0 else ''} & {name} & {' & '.join(cells)} \\\\")
        blocks.append("\n".join(lines))
    return "\n\\addlinespace\n".join(blocks)


def strat_per_bucket() -> str:
    """n_queries of the filled buckets (all filled buckets must have the same count)."""
    ns = {int(r["n_queries"]) for r in rows(STRAT / "nolearn/bucket_summary.csv") if int(r["n_queries"])}
    if len(ns) != 1:
        raise RuntimeError(f"filled stratified buckets differ in size: {sorted(ns)}")
    return fmt_int(ns.pop())


def strat_cell(ds: str, est: str, bucket: str) -> float:
    return float(one(rows(STRAT / "nolearn/bucket_summary.csv"), dataset=ds, estimator=est, bucket=bucket)["median"])


def pct1(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def indep_share(ds: str, est: str, ref: str) -> float:
    """Share of the per-query independence rows of `ds` with est < ref."""
    d = [r for r in rows(INDEP / "per_query.csv") if r["dataset"] == ds]
    if len(d) != 300:
        raise RuntimeError(f"expected 300 fixed-extent rows for {ds}, found {len(d)}")
    return sum(float(r[est]) < float(r[ref]) for r in d) / len(d)


def taxi_max(est: str) -> str:
    """Max of column `max` over the four taxi rows (T-Drive, Porto x fixed, mixed); 1dp, int from 1,000."""
    v = max(float(r["max"]) for r in acc() if r["estimator"] == est and r["dataset"] in ("tdrive", "porto"))
    return fmt_int(v) if v >= 1000 else dp(v, 1)



def final_step_improvement(est: str) -> float:
    """Saturation rule of scripts/analyze_scalability.py:convergence_status on the BOUNDED 1M
    mixed curve: 100 x (median@3,200 - median@6,400) / median@3,200."""
    v = budget_cells()[est]
    return 100.0 * (v[2] - v[3]) / v[2]


def saturation_status(est: str) -> str:
    imp = final_step_improvement(est)
    return "undertrained" if imp >= 5.0 else "saturated" if imp >= 0.0 else "non-monotone"


def signed1(x: float) -> str:
    return "$+$0.0" if round(abs(x), 1) == 0 else f"${'+' if x >= 0 else '-'}${dp(abs(x), 1)}"


MACROS = {
    "NumTDFixNatBoxMed": lambda: dp(float(one(acc(), dataset="tdrive", workload="fixed_extent", estimator="native_bbox")["median"]), 2),
    "NumTDFixSepMed": lambda: dp(float(one(acc(), dataset="tdrive", workload="fixed_extent", estimator="separable")["median"]), 2),
    "NumPOFixNatBoxMed": lambda: dp(float(one(acc(), dataset="porto", workload="fixed_extent", estimator="native_bbox")["median"]), 2),
    "NumPOFixSepMed": lambda: dp(float(one(acc(), dataset="porto", workload="fixed_extent", estimator="separable")["median"]), 2),
    "NumUnclampedLearnedMaxQerr": lambda: sci2(max(float(r["max"]) for r in acc("unclamped")
                                                   if r["estimator"] in ("joint", "separable", "no_coupling", "spatial_only"))),
    "NumTableRows": table_rows_common,
    "NumPgDefEstRows": pg_default_rows,
    "NumTDFixPgDefMed": lambda: fmt_int(float(one(acc(), dataset="tdrive", workload="fixed_extent", estimator="pg_default")["median"])),
    "NumTDMixPgDefMed": lambda: fmt_int(float(one(acc(), dataset="tdrive", workload="mixed_extent", estimator="pg_default")["median"])),
    "NumBoxPathRatioMedMax": lambda: dp(max(float(r["median_box_to_path_ratio"]) for r in rows(NATIVE / "native_decomposition.csv")
                                            if r["bucket"] == "all" and r["component"] == "total_error"), 2),
    "NumTDFixTSPNinetyFive": lambda: dp(float(one(acc(), dataset="tdrive", workload="fixed_extent", estimator="trajectory_sampling")["p95"]), 1),
    "NumTDFixSepPNinetyFive": lambda: dp(float(one(acc(), dataset="tdrive", workload="fixed_extent", estimator="separable")["p95"]), 2),
    "NumPOMixClampedLearnedMax": lambda: fmt_int(max(float(r["max"]) for r in acc()
                                                     if r["dataset"] == "porto" and r["workload"] == "mixed_extent"
                                                     and r["estimator"] in ("joint", "separable"))),
    "NumSynSepJointRatioMin": lambda: syn_ratio(min),
    "NumSynSepJointRatioMax": lambda: syn_ratio(max),
    "NumOneMLabelSecMin": one_m_label_seconds,
    "NumOneMLabelSecMax": lambda: fmt_int(max(float(r["label_seconds"]) for r in budget_rows())),
    "NumOneMTrainSecMax": lambda: dp(max(float(r["mean_training_seconds"]) for r in budget_rows()), 1),
    "NumTransferMedMin": lambda: dp(min(transfer_medians()), 1),
    "NumTransferMedMax": lambda: dp(max(transfer_medians()), 1),
    "NumPlanTrueBad": lambda: pct0(float(plan_row("true_cardinality_input")["bad_plan_rate_gt_1_25"])),
    "NumPlanNativeBad": lambda: pct0(float(plan_row("native")["bad_plan_rate_gt_1_25"])),
    "NumPlanLearnedOverheadRange": plan_overhead_range,
    "NumPlanLoadMed": lambda: fmt_int(statistics.median(
        float(r["load1"]) for r in rows_tsv(ROOT / "run/vldbj_rev/plan_rerun/load_readings.tsv"))),
    "NumTDMixSepNatBoxRatio": lambda: dp(float(sep_natbox_tdmix()["median_ratio"]), 3),
    "NumTDMixSepNatBoxRatioLo": lambda: dp(float(sep_natbox_tdmix()["ratio_ci95_lower"]), 3),
    "NumTDMixSepNatBoxRatioHi": lambda: dp(float(sep_natbox_tdmix()["ratio_ci95_upper"]), 3),
    "NumTDMixSepNatBoxPHolm": lambda: dp(float(sep_natbox_tdmix()["p_holm"]), 3),
    # dataset preprocessing statistics
    "NumTDSourceFiles": lambda: fmt_int(td_manifest()["matched_file_count"]),
    "NumTDValidFiles": lambda: fmt_int(td_manifest()["valid_file_count"]),
    "NumTDSelectedFiles": lambda: fmt_int(td_manifest()["selected_file_count"]),
    "NumTDContribTaxis": lambda: fmt_int(td_manifest()["contributing_source_taxi_count"]),
    "NumTDOneMPoints": lambda: fmt_int(jfile("run/real_experiments/task7b/scalability/prepared/tdrive__max_1000000.json")["prepared_point_count"]),
    "NumPOAvailTrips": lambda: fmt_int(jfile("run/real_experiments/scalability/prepared/porto__max_1000000.json")["available_raw_trajectory_count"]),
    "NumPOOneMPoints": lambda: fmt_int(jfile("run/real_experiments/scalability/prepared/porto__max_1000000.json")["prepared_point_count"]),
    "NumTDHundredKPoints": lambda: fmt_int(table_profile()["tdrive_traj"]["sum_instants"]),
    "NumPOHundredKPoints": lambda: fmt_int(table_profile()["porto_traj"]["sum_instants"]),
    "NumTDHundredKMedPoints": lambda: fmt_int(table_profile()["tdrive_traj"]["median_instants"]),
    "NumPOHundredKMedPoints": lambda: fmt_int(table_profile()["porto_traj"]["median_instants"]),
    "NumTDHundredKMedDurMin": lambda: dp(table_profile()["tdrive_traj"]["median_duration_min"], 1),
    "NumPOHundredKMedDurMin": lambda: dp(table_profile()["porto_traj"]["median_duration_min"], 1),
    "NumTDFixZero": lambda: zero_count("run/real_experiments/task7b/headline/workloads/tdrive__fixed_extent.json"),
    "NumTDMixZero": lambda: zero_count("run/real_experiments/task7b/headline/workloads/tdrive__mixed_extent.json"),
    "NumPOFixZero": lambda: zero_count("run/real_experiments/multiseed_accuracy/workloads/porto__fixed_extent.json"),
    "NumPOMixZero": lambda: zero_count("run/real_experiments/multiseed_accuracy/workloads/porto__mixed_extent.json"),
    # statistics-target sensitivity
    "NumStatsTDMixNatBoxPNinetyFiveTargetHundred": lambda: dp(float(stats_row("tdrive", "mixed_extent", 100)["p95"]), 1),
    "NumStatsTDMixNatBoxPNinetyFiveTargetThousand": lambda: dp(float(stats_row("tdrive", "mixed_extent", 1000)["p95"]), 1),
    "NumStatsPOFixNatBoxResampleRatio": lambda: dp(float(stats_row("porto", "fixed_extent", 100)["median_ratio_vs_original"]), 2),
    "NumStatsPOFixNatBoxResampleRatioLo": lambda: dp(float(stats_row("porto", "fixed_extent", 100)["ratio_ci95_lower"]), 2),
    "NumStatsPOFixNatBoxResampleRatioHi": lambda: dp(float(stats_row("porto", "fixed_extent", 100)["ratio_ci95_upper"]), 2),
    "NumStatsPOFixNatBoxTargetThousandMed": lambda: dp(float(stats_row("porto", "fixed_extent", 1000)["median"]), 2),

    "NumAccRowsAll": lambda: acc_body(acc(), ACC_ORDER),
    "NumAccRowsNonEmpty": lambda: acc_body(acc("planner_nonempty"), ACC_ORDER),
    "NumAccRowsUnbounded": lambda: acc_body(acc("unclamped"), ACC_ORDER, sci_learned_max=True),
    "NumOriginRows": lambda: acc_body(acc(), ORIGIN_ORDER),
    "NumFamASize": lambda: fmt_int(fam_size("a_")),
    "NumFamBSize": lambda: fmt_int(fam_size("b_")),
    "NumHolmFloorA": lambda: dp(2 / 10001 * fam_size("a_"), 4),
    "NumHolmFloorB": lambda: dp(2 / 10001 * fam_size("b_"), 4),
    "NumTDFixSpONatBoxRatio": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="all", dataset="tdrive", workload="fixed_extent", left="spatial_only", right="native_bbox")["median_ratio"]), 3),
    "NumTDFixSpONatBoxRatioLo": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="all", dataset="tdrive", workload="fixed_extent", left="spatial_only", right="native_bbox")["ratio_ci95_lower"]), 3),
    "NumTDFixSpONatBoxRatioHi": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="all", dataset="tdrive", workload="fixed_extent", left="spatial_only", right="native_bbox")["ratio_ci95_upper"]), 3),
    "NumTDFixSpONatBoxPHolm": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="all", dataset="tdrive", workload="fixed_extent", left="spatial_only", right="native_bbox")["p_holm"]), 3),
    "NumLearnedShareBelowOneMax": lambda: pct0(max(float(r["share_below_1"]) for r in rows(NATIVE / "task1_clamp_shares.csv") if r["estimator"] in LEARNED_EST)),
    "NumLearnedShareAboveNMax": lambda: dp(100 * max(float(r["share_above_N"]) for r in rows(NATIVE / "task1_clamp_shares.csv") if r["estimator"] in LEARNED_EST), 1) + "\\%",
    "NumPOFixTSPNinetyFive": lambda: dp(float(one(acc(), dataset="porto", workload="fixed_extent", estimator="trajectory_sampling")["p95"]), 1),
    "NumPOFixSepPNinetyFive": lambda: dp(float(one(acc(), dataset="porto", workload="fixed_extent", estimator="separable")["p95"]), 2),
    "NumPathHistSumMaxMax": lambda: taxi_max("path_hist_sum"),
    "NumNatBoxTaxiMax": lambda: taxi_max("native_bbox"),
    "NumTSTaxiMax": lambda: taxi_max("trajectory_sampling"),
    "NumSpOTaxiMax": lambda: taxi_max("spatial_only"),
    "NumPOFixPathHistSumMed": lambda: dp(float(one(acc(), dataset="porto", workload="fixed_extent", estimator="path_hist_sum")["median"]), 2),
    "NumPOFixPathHistMaxMed": lambda: dp(float(one(acc(), dataset="porto", workload="fixed_extent", estimator="path_hist_max")["median"]), 2),
    "NumPOMixNEPathHistSumMed": lambda: dp(float(one(acc("planner_nonempty"), dataset="porto", workload="mixed_extent", estimator="path_hist_sum")["median"]), 2),
    "NumPOMixNEBestLearnedMed": lambda: dp(min(float(one(acc("planner_nonempty"), dataset="porto", workload="mixed_extent", estimator=e)["median"]) for e in LEARNED_EST), 2),
    "NumPlanRowsTD": lambda: plan_body("run/vldbj_rev/plan_rerun", True),
    "NumPlanRowsPO": lambda: plan_body("run/vldbj_rev/plan_porto_nolearned", False),
    "NumPOPlanLoadMed": lambda: fmt_int(statistics.median(float(r["load1"]) for r in rows_tsv(ROOT / "run/vldbj_rev/plan_porto_nolearned/load_readings.tsv"))),
    "NumPlanExecSpread": lambda: pct0((lambda v: max(v) / min(v) - 1)([float(r["execution_ms_median"]) for r in rows(ROOT / "run/vldbj_rev/plan_rerun/analysis/plan_metrics_summary.csv")])),
    "NumPOPlanTrueBad": lambda: pct0(float(plan_summary("run/vldbj_rev/plan_porto_nolearned", "true_cardinality_input")["bad_plan_rate_gt_1_25"])),
    "NumPOPlanNativeBad": lambda: pct0(float(plan_summary("run/vldbj_rev/plan_porto_nolearned", "native")["bad_plan_rate_gt_1_25"])),
    "NumPOPlanTSInferMs": lambda: dp(float(plan_summary("run/vldbj_rev/plan_porto_nolearned", "trajectory_sampling")["inference_ms_median"]), 1),
    "NumPOPlanTSSpeedup": lambda: dp(float(plan_summary("run/vldbj_rev/plan_porto_nolearned", "trajectory_sampling")["end_to_end_speedup_median"]), 3),
    "NumPlanClampedQerrMin": lambda: dp(min(float(plan_summary("run/vldbj_rev/plan_rerun", c)["q_error_median"]) for c in ("joint_clamped", "separable_clamped", "no_coupling_clamped")), 1),
    "NumPlanClampedQerrMax": lambda: dp(max(float(plan_summary("run/vldbj_rev/plan_rerun", c)["q_error_median"]) for c in ("joint_clamped", "separable_clamped", "no_coupling_clamped")), 1),
    "NumPlanSpOUnderFactor": lambda: fmt_int(statistics.median((float(r["true_cardinality"]) + 1) / (float(r["predicted_cardinality"]) + 1) for r in spo_rows())),
    "NumPlanSpOAgree": lambda: fmt_int(sum(r["oracle_agreement"] == "True" for r in spo_rows())),
    "NumPlanSpOAgreeMedTrue": lambda: fmt_int(statistics.median(float(r["true_cardinality"]) for r in spo_rows() if r["oracle_agreement"] == "True")),
    "NumPlanSpODisagreeMedTrue": lambda: fmt_int(statistics.median(float(r["true_cardinality"]) for r in spo_rows() if r["oracle_agreement"] != "True")),
    "NumPlanSpODisagreeNLOracle": lambda: fmt_int(sum("Nested Loop" in r["oracle_plan_family"] for r in spo_rows() if r["oracle_agreement"] != "True")),
    "NumPlanSpOOrigEndToEnd": lambda: dp(float(plan_summary("run/vldbj_rev/plan_rerun", "spatial_only")["end_to_end_speedup_median"]), 3),
    "NumTDOneMLoadSec": lambda: fmt_int(float(onem_db()["load_seconds"])),
    "NumTDOneMGistSec": lambda: dp(float(onem_db()["index_build_seconds"]), 1),
    "NumTDOneMAnalyzeSec": lambda: dp(float(onem_db()["analyze_seconds"]), 1),
    "NumTDOneMTotalMB": lambda: fmt_int(float(onem_db()["total_bytes"]) / 1e6),
    "NumOneMInferMsMin": lambda: onem_infer(min),
    "NumOneMInferMsMax": lambda: onem_infer(max),
    "NumBudgetRows": budget_body,
    "NumStatsPOMixNatBoxMaxTargetThousand": lambda: fmt_int(float(one(rows(ROOT / "run/vldbj_rev/stats_target/summary.csv"), dataset="porto", workload="mixed_extent", form="native_bbox", target="1000")["max"])),
    "NumStatsPOMixNatBoxMaxTargetHundred": lambda: fmt_int(float(one(rows(ROOT / "run/vldbj_rev/stats_target/summary.csv"), dataset="porto", workload="mixed_extent", form="native_bbox", target="100")["max"])),
    "NumPOMixEvalMaxTrue": lambda: fmt_int(max(q["true_cardinality"] for q in jfile("run/real_experiments/multiseed_accuracy/workloads/porto__mixed_extent.json")["evaluation_queries"])),
    "NumRetrainMaxDevPct": retrain_max_dev,
    "NumSepBestLearnedSettings": sep_best_settings,
    "NumSynRows": syn_rows,
    "NumSynSeeds": lambda: fmt_int(len(syn()["run_config"]["model_seeds"])),
    "NumSynSepSlope": syn_slope,
    "NumRidgeRows": ridge_rows,
    "NumRidgeHeadroomMax": lambda: dp(max(syn()["rows"], key=lambda r: r["correlation_strength"])["interaction_headroom_r2"], 2),

    "NumFamCSize": lambda: fmt_int(fam_size("c_")),
    "NumHolmFloorC": lambda: dp(2 / 10001 * fam_size("c_"), 4),
    "NumFamCHistWorseAll": lambda: fmt_int(sum(r["significant_holm_0_05"] == "True" and float(r["median_ratio"]) > 1
                                               for r in rows(STATS / "paired_bootstrap_clamped.csv")
                                               if r["family"] == "c_path_hist_sum_vs_separable" and r["subset"] == "all")),
    "NumPOMixNEHistSepRatio": lambda: dp(float(pb(family="c_path_hist_sum_vs_separable", subset="nonempty", dataset="porto", workload="mixed_extent")["median_ratio"]), 3),
    "NumPOMixNEHistSepPHolm": lambda: p_holm_fmt(pb(family="c_path_hist_sum_vs_separable", subset="nonempty", dataset="porto", workload="mixed_extent"), "c_"),
    "NumSynSepIfeatRatioMin": lambda: dp(min(m for m, _ in syn_ifeat()), 3),
    "NumSynSepIfeatRatioMax": lambda: dp(max(m for m, _ in syn_ifeat()), 3),
    "NumSynSepIfeatRatioMaxSD": lambda: dp(max(syn_ifeat(), key=lambda t: t[0])[1], 3),
    "NumBudgetSixtyFourMin": lambda: dp(min(v[3] for v in budget_cells().values()), 3),
    "NumBudgetSixtyFourMax": lambda: dp(max(v[3] for v in budget_cells().values()), 3),
    "NumBudgetSepThirtyTwo": lambda: dp(budget_cells()["separable"][2], 3),
    "NumBudgetSepSixtyFour": lambda: dp(budget_cells()["separable"][3], 3),
    "NumPOFixNatUnderCount": lambda: fmt_int(round(float((lambda r: r["share_negative"])(one(rows(NATIVE / "native_decomposition.csv"), dataset="porto", workload="fixed_extent", bucket="all", component="total_error"))) * 300)),
    "NumTDFixHistMaxNatRatio": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="all", dataset="tdrive", workload="fixed_extent", left="path_hist_max")["median_ratio"]), 3),
    "NumPOFixHistSumNatRatio": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="all", dataset="porto", workload="fixed_extent", left="path_hist_sum")["median_ratio"]), 3),
    "NumTDMixNEHistSumNatRatio": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="nonempty", dataset="tdrive", workload="mixed_extent", left="path_hist_sum")["median_ratio"]), 3),
    "NumPlanNeuralSameMin": lambda: pct0(min(neural_same_shares())),
    "NumPlanNeuralSameMax": lambda: pct0(max(neural_same_shares())),
    "NumPlanSamePlanOverheadRange": same_plan_overhead_range,
    "NumPlanTrueFasterTD": lambda: fmt_int(sum(float(r["end_to_end_speedup_vs_native"]) > 1 for r in plan_q("run/vldbj_rev/plan_rerun", "true_cardinality_input"))),
    "NumPlanTrueFasterPO": lambda: fmt_int(sum(float(r["end_to_end_speedup_vs_native"]) > 1 for r in plan_q("run/vldbj_rev/plan_porto_nolearned", "true_cardinality_input"))),
    "NumPlanTrueSameTD": lambda: fmt_int(sum(r["plan_changed_from_native"] == "False" for r in plan_q("run/vldbj_rev/plan_rerun", "true_cardinality_input"))),
    "NumPlanTrueSamePO": lambda: fmt_int(sum(r["plan_changed_from_native"] == "False" for r in plan_q("run/vldbj_rev/plan_porto_nolearned", "true_cardinality_input"))),
    "NumRetimeQueries": lambda: fmt_int(int(jfile("run/vldbj_rev/plan_retime_lowload/experiment_metadata.json")["run_config"]["retime_n"])),
    "NumRetimeLoadMed": retime_load_med,
    "NumRetimeFasterMin": lambda: pct0(1 - max(float(r["exec_ratio_low_over_high_median"]) for r in retime_cmp())),
    "NumRetimeFasterMax": lambda: pct0(1 - min(float(r["exec_ratio_low_over_high_median"]) for r in retime_cmp())),
    "NumRetimeOracleSame": lambda: fmt_int(jfile("run/vldbj_rev/plan_retime_lowload/comparison/summary.json")["oracle_plan_same_both_runs"]),
    "NumJoinWindows": lambda: fmt_int(jfile("run/vldbj_rev/join/metadata.json")["n_windows"]),
    "NumJoinTDNonEmpty": lambda: fmt_int(len(join_nonempty("tdrive"))),
    "NumJoinPONonEmpty": lambda: fmt_int(len(join_nonempty("porto"))),
    "NumJoinTDNatUnder": lambda: fmt_int(sum(float(r["native_join_estimate"]) < float(r["true_pairs"]) for r in join_rows("tdrive") if r["true_pairs"])),
    "NumJoinPONatUnder": lambda: fmt_int(sum(float(r["native_join_estimate"]) < float(r["true_pairs"]) for r in join_rows("porto") if r["true_pairs"])),
    "NumJoinTDNatRatioMed": lambda: sig2(statistics.median(float(r["native_join_estimate"]) / float(r["true_pairs"]) for r in join_nonempty("tdrive"))),
    "NumJoinPONatRatioMed": lambda: sig2(statistics.median(float(r["native_join_estimate"]) / float(r["true_pairs"]) for r in join_nonempty("porto"))),
    "NumJoinTDNatQMed": lambda: dp(statistics.median(float(r["native_qerror_clamped"]) for r in join_rows("tdrive") if r["native_qerror_clamped"]), 1),
    "NumJoinPONatQMed": lambda: dp(statistics.median(float(r["native_qerror_clamped"]) for r in join_rows("porto") if r["native_qerror_clamped"]), 1),
    "NumJoinTDSampZero": lambda: fmt_int(sum(float(r["sampling_estimate_median"]) == 0 for r in join_nonempty("tdrive"))),
    "NumJoinPOSampZero": lambda: fmt_int(sum(float(r["sampling_estimate_median"]) == 0 for r in join_nonempty("porto"))),
    "NumJoinTDSampQMed": lambda: dp(statistics.median(float(r["sampling_qerror_clamped"]) for r in join_rows("tdrive") if r["sampling_qerror_clamped"]), 2),
    "NumJoinPOSampQMed": lambda: dp(statistics.median(float(r["sampling_qerror_clamped"]) for r in join_rows("porto") if r["sampling_qerror_clamped"]), 2),
    # AIS (optional rows; values from the load metadata and the read-only table profile)
    "NumAISTrips": lambda: fmt_int(jfile("run/vldbj_rev/ais/load_metadata.json")["rows"]),
    "NumAISPoints": lambda: fmt_int(jfile("run/vldbj_rev/ais/load_metadata.json")["points"]),
    "NumAISMedSamplingSec": lambda: fmt_int(jfile("run/vldbj_rev/native/table_profile.json")["ais_traj"]["median_gap_seconds"]),
    "NumAISHundredKPoints": lambda: fmt_int(jfile("run/vldbj_rev/native/table_profile.json")["ais_traj"]["sum_instants"]),
    "NumAISHundredKMedPoints": lambda: fmt_int(jfile("run/vldbj_rev/native/table_profile.json")["ais_traj"]["median_instants"]),
    "NumAISHundredKMedDurMin": lambda: dp(jfile("run/vldbj_rev/native/table_profile.json")["ais_traj"]["median_duration_min"], 1),
    "NumAISBoxPathRatioMed": lambda: dp(float(one(rows(ROOT / "run/vldbj_rev/ais/native/native_decomposition.csv"),
                                                 dataset="ais", workload="fixed_extent", bucket="all",
                                                 component="total_error")["median_box_to_path_ratio"]), 2),
    "NumAISFixHistLogErr": lambda: signed2(float(ais_dec("fixed_extent", "all", "histogram_error")["median"])),
    "NumAISFixTotalLogErr": lambda: signed2(float(ais_dec("fixed_extent", "all", "total_error")["median"])),
    "NumAISMixHistLogErr": lambda: signed2(float(ais_dec("mixed_extent", "all", "histogram_error")["median"])),
    "NumAISMixGapLogErr": lambda: signed2(float(ais_dec("mixed_extent", "all", "semantic_gap")["median"])),
    "NumAISMixTotalLogErr": lambda: signed2(float(ais_dec("mixed_extent", "all", "total_error")["median"])),
    "NumAISMixBoxPathRatioMed": lambda: dp(float(ais_dec("mixed_extent", "all", "total_error")["median_box_to_path_ratio"]), 2),
    "NumAISFixZero": lambda: fmt_int(int(ais_dec("fixed_extent", "0", "total_error")["n_queries"])),
    "NumAISMixZero": lambda: fmt_int(int(ais_dec("mixed_extent", "0", "total_error")["n_queries"])),
    "NumAISSmallBoxPathRatioMin": lambda: dp(min(float(ais_dec(w, "1-10", "semantic_gap")["median_box_to_path_ratio"]) for w in ("fixed_extent", "mixed_extent")), 1),
    "NumAISSmallBoxPathRatioMax": lambda: dp(max(float(ais_dec(w, "1-10", "semantic_gap")["median_box_to_path_ratio"]) for w in ("fixed_extent", "mixed_extent")), 1),
    "NumAISFixHistSumNatRatio": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="all", dataset="ais", workload="fixed_extent", left="path_hist_sum")["median_ratio"]), 3),
    "NumAISMixHistSumNatRatio": lambda: dp(float(pb(family="a_learned_vs_native_bbox", subset="all", dataset="ais", workload="mixed_extent", left="path_hist_sum")["median_ratio"]), 3),
    # fixed-extent calibration rule (empty < 15% and median true >= 10)
    "NumCalibTDEmpty": lambda: pct0(calib()["frozen_workloads"]["tdrive"]["evaluation"]["empty_share"]),
    "NumCalibTDMed": lambda: med_fmt(calib()["frozen_workloads"]["tdrive"]["evaluation"]["median_true"]),
    "NumCalibPOEmpty": lambda: pct0(calib()["frozen_workloads"]["porto"]["evaluation"]["empty_share"]),
    "NumCalibPOMed": lambda: med_fmt(calib()["frozen_workloads"]["porto"]["evaluation"]["median_true"]),
    "NumCalibAISEmpty": lambda: pct0(calib()["frozen_workloads"]["ais"]["evaluation"]["empty_share"]),
    "NumCalibAISMed": lambda: med_fmt(calib()["frozen_workloads"]["ais"]["evaluation"]["median_true"]),
    "NumAISPilotOneDegEmpty": lambda: pct0(calib()["pilot_ais"]["1.0"]["empty_share"]),
    "NumAISPilotOneDegMed": lambda: med_fmt(calib()["pilot_ais"]["1.0"]["median_true"]),
    "NumAISFixExtentDeg": lambda: dp(calib()["pilot_ais_chosen_extent"], 1),
    # root cause: MobilityDB && = geo_sel x temporal_sel_tstzspan (independence) and geo_sel undercounts overlap
    "NumIndepTDErr": lambda: signed2(float(indep_summary("tdrive")["median_independence_error"])),
    "NumIndepPOErr": lambda: signed2(float(indep_summary("porto")["median_independence_error"])),
    "NumIndepAISErr": lambda: signed2(float(indep_summary("ais")["median_independence_error"])),
    "NumIndepAISUnderShare": lambda: pct0(float(indep_summary("ais")["share_independence_underestimates"])),
    "NumMargTDErr": lambda: signed2(float(indep_summary("tdrive")["median_marginal_error"])),
    "NumMargPOErr": lambda: signed2(float(indep_summary("porto")["median_marginal_error"])),
    "NumMargAISErr": lambda: signed2(float(indep_summary("ais")["median_marginal_error"])),
    "NumSpatMargTDErr": lambda: signed2(marginal_split("tdrive")["spatial"]),
    "NumSpatMargPOErr": lambda: signed2(marginal_split("porto")["spatial"]),
    "NumSpatMargAISErr": lambda: signed2(marginal_split("ais")["spatial"]),
    "NumSpatMargUnderMin": lambda: pct0(min(marginal_split(d)["spatial_under"] for d in ("tdrive", "porto", "ais"))),
    "NumTempMargAbsMax": lambda: dp(max(abs(marginal_split(d)["temporal"]) for d in ("tdrive", "porto", "ais")), 3),
    "NumCombResidAbsMax": lambda: dp(max(abs(float(indep_summary(d)["median_combination_residual"])) for d in ("tdrive", "porto", "ais")), 3),
    # family (d): exact box count as an estimator (upper bound; not deployable)
    "NumBoxOracleRows": box_oracle_rows,
    "NumBoxOracleTDFixMed": lambda: dp(float(one(acc(), dataset="tdrive", workload="fixed_extent", estimator="box_oracle")["median"]), 2),
    "NumBoxOraclePOFixMed": lambda: dp(float(one(acc(), dataset="porto", workload="fixed_extent", estimator="box_oracle")["median"]), 2),
    "NumBoxOracleAISFixMed": lambda: dp(float(one(acc(), dataset="ais", workload="fixed_extent", estimator="box_oracle")["median"]), 2),
    "NumFamDSize": lambda: fmt_int(fam_size("d_")),
    "NumHolmFloorD": lambda: dp(2 / 10001 * fam_size("d_"), 4),
    "NumFamDSigAll": lambda: fmt_int(sum(r["significant_holm_0_05"] == "True" for r in rows(STATS / "paired_bootstrap_clamped.csv")
                                         if r["family"] == "d_box_oracle" and r["subset"] == "all")),
    "NumBoxSepRatioMinTaxi": lambda: dp(min(float(r["median_ratio"]) for r in fam_d("all", "separable") if float(r["median_ratio"]) != 1.0), 3),
    "NumBoxSepRatioMaxTaxi": lambda: dp(max(float(r["median_ratio"]) for r in fam_d("all", "separable") if float(r["median_ratio"]) != 1.0), 3),
    "NumAISMixBoxNatRatio": lambda: dp(float(one(fam_d("all", "native_bbox"), dataset="ais", workload="mixed_extent")["median_ratio"]), 3),
    # stratified mixed workloads, non-learned estimators
    "NumStratNoLearnRows": lambda: strat_nolearn_rows("median"),
    "NumStratNoLearnTailRows": lambda: strat_nolearn_rows("p95"),
    "NumStratPerBucket": strat_per_bucket,
    "NumStratPONatMid": lambda: dp(strat_cell("porto", "native_bbox", "11-100"), 2),
    "NumStratPOHistSumMid": lambda: dp(strat_cell("porto", "path_hist_sum", "11-100"), 2),
    "NumStratAISHistSumMid": lambda: dp(strat_cell("ais", "path_hist_sum", "11-100"), 2),
    "NumStratAISHistSumHigh": lambda: dp(strat_cell("ais", "path_hist_sum", "101-1,000"), 2),
    "NumStratTSMidMin": lambda: dp(min(strat_cell(d, "trajectory_sampling", "11-100") for d in ("tdrive", "porto", "ais")), 2),
    "NumStratTSMidMax": lambda: dp(max(strat_cell(d, "trajectory_sampling", "11-100") for d in ("tdrive", "porto", "ais")), 2),
    # per-dataset shares from the independence check (fixed extent, 300 queries each)
    "NumSpatMargUnderTD": lambda: pct1(indep_share("tdrive", "eS", "S_exact")),
    "NumSpatMargUnderPO": lambda: pct1(indep_share("porto", "eS", "S_exact")),
    "NumSpatMargUnderAIS": lambda: pct1(indep_share("ais", "eS", "S_exact")),
    "NumEstUnderTD": lambda: pct1(indep_share("tdrive", "eJ", "true_path")),
    "NumEstUnderPO": lambda: pct1(indep_share("porto", "eJ", "true_path")),
    "NumEstUnderAIS": lambda: pct1(indep_share("ais", "eJ", "true_path")),
    # saturation rule recomputed on the bounded learning curve (1M T-Drive, mixed extent)
    "NumSatJointStep": lambda: signed1(final_step_improvement("joint")),
    "NumSatSepStep": lambda: signed1(final_step_improvement("separable")),
    "NumSatNoCStep": lambda: signed1(final_step_improvement("no_coupling")),
    "NumSatSpOStep": lambda: signed1(final_step_improvement("spatial_only")),
    "NumSatCount": lambda: fmt_int(sum(saturation_status(e) == "saturated" for e in ("joint", "separable", "no_coupling", "spatial_only"))),
    # paired bootstrap
    "NumPairedRowsAll": lambda: paired_rows("all"),
    "NumPairedRowsNonEmpty": lambda: paired_rows("nonempty"),
    "NumFamASigAll": lambda: n_significant("a_", "all"),
    "NumFamASigNonEmpty": lambda: n_significant("a_", "nonempty"),
    "NumFamBSigAll": lambda: n_significant("b_", "all"),
    "NumFamBSigNonEmpty": lambda: n_significant("b_", "nonempty"),
}
NOTES = {
    "NumAISBoxPathRatioMed": "AIS decomposition rows are in run/vldbj_rev/ais/native/native_decomposition.csv",
    "NumRetimeLoadMed": "load_checks.tsv has one reading inside the timed window (the trigger reading, load1 5.73); later readings were not taken",
    "NumJoinTDNatRatioMed": "over windows with a non-empty true result",
    "NumJoinPONatRatioMed": "over windows with a non-empty true result",
    "NumSynRows": "UNCLAMPED: the synthetic control saved no per-query predictions",
    "NumSynSepSlope": "least squares on per-strength means of per-seed Separable/Joint ratios (stored summary uses ratio of means: +0.124)",
    "NumOneMInferMsMin": "from run-file summaries (task7b/scalability/runs/learned, b6400); resources.csv has learned rows only at 800 labels",
    "NumOneMInferMsMax": "from run-file summaries (task7b/scalability/runs/learned, b6400); resources.csv has learned rows only at 800 labels",
    "NumBudgetRows": "all budgets from run/real_experiments/task7b/scalability (corrected T-Drive; matches published Table 7)",
    "NumStatsPOMixNatBoxMaxTargetHundred": "target=100 row is a re-ANALYZEd copy (fresh 30,000-row sample), not the original table's statistics",
    "NumOriginRows": "T-Drive origin rows from the task7b headline runs (corrected sample); Porto from multiseed_accuracy",
    "NumTransferMedMin": "UNCLAMPED: transfer_sanity.json stores per-query q-errors only, not predictions",
    "NumTransferMedMax": "UNCLAMPED: transfer_sanity.json stores per-query q-errors only, not predictions",
    "NumSynSepJointRatioMin": "UNCLAMPED: the synthetic control saved no per-query predictions",
    "NumSynSepJointRatioMax": "UNCLAMPED: the synthetic control saved no per-query predictions",
}


def macro_list() -> list[str]:
    return [line.strip() for line in MACRO_LIST.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print the values, write nothing")
    args = ap.parse_args()
    names = macro_list()
    missing = [m for m in names if m not in MACROS]
    values = {m: MACROS[m]() for m in names if m in MACROS}
    lines = ["% Generated by scripts/vldbj_generate_numbers.py from the result files. Do not edit by hand."]
    for m, v in values.items():
        if m in NOTES:
            lines.append(f"% {m}: {NOTES[m]}")
        lines.append(f"\\newcommand{{\\{m}}}{{{v}}}")
    text = "\n".join(lines) + "\n"
    for m, v in values.items():
        more = " ..." if len(v.splitlines()) > 1 else ""
        print(f"{m:28s} {v.splitlines()[0][:90]}{more}")
    if missing:
        print("WARNING: macros without a generator function:", ", ".join(missing))
    if not args.check:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(ROOT)} ({len(values)} macros)")


if __name__ == "__main__":
    main()
