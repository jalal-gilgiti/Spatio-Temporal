"""Generate the paper's figures.

All q-errors bound estimates to [1, N]; native estimates use statistics target 1000.
Every figure reads the same result files as the numbers generator, so plotted values
equal the corresponding macros. Output: vector PDFs at exact page widths, fonts embedded
(TrueType, Type 42), 8-9 pt text, one fixed Okabe-Ito colour, marker and dash pattern
per estimator.

    python scripts/vldbj_figures.py                 # all figures (AIS rows included)
    python scripts/vldbj_figures.py --no-ais        # taxi datasets only
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_multiseed_accuracy import load_runs  # noqa: E402
from scripts.vldbj_task1_table import SOURCES, prediction_matrix, qerrors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
NATIVE = ROOT / "run/vldbj_rev/native"
MM = 1 / 25.4
COL_W, TEXT_W = 84.0, 174.0

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6,
    "lines.linewidth": 1.1, "lines.markersize": 4.5, "legend.frameon": False,
    "savefig.dpi": 600,
})

STYLE = {  # paper name, colour, marker, dash
    "native_bbox": ("Native-BBox", "#000000", "o", "-"),
    "native": ("Native", "#000000", "o", "-"),
    "trajectory_sampling": ("Trajectory Sampling", "#E69F00", "^", "--"),
    "path_hist_sum": ("Path histogram (sum)", "#56B4E9", "s", "-."),
    "path_hist_max": ("Path histogram (max)", "#F0E442", "h", (0, (1, 3))),
    "joint": ("Joint", "#009E73", "D", (0, (5, 1))),
    "separable": ("Separable", "#0072B2", "p", (0, (3, 1, 1, 1))),
    "no_coupling": ("No-coupling", "#D55E00", "v", (0, (1, 1.5))),
    "spatial_only": ("SpatialOnly", "#CC79A7", "x", (0, (4, 2, 1, 2, 1, 2))),
    "true_cardinality_input": ("True cardinalities", "#999999", "*", (0, (8, 2, 1, 2))),
}
STYLE["joint_dm"] = ("Joint (DM)",) + STYLE["joint"][1:]
STYLE["separable_dm"] = ("Separable (DM)",) + STYLE["separable"][1:]
STYLE["no_coupling_dm"] = ("No-coupling (DM)",) + STYLE["no_coupling"][1:]
STYLE["spatial_only_dm"] = ("SpatialOnly (DM)",) + STYLE["spatial_only"][1:]

DATASETS = {"tdrive": "T-Drive", "porto": "Porto", "ais": "AIS"}
WORKLOADS = {"fixed_extent": "fixed extent", "mixed_extent": "mixed extent"}
ACC_EST = ("native_bbox", "trajectory_sampling", "path_hist_sum", "joint", "separable",
           "no_coupling", "spatial_only")
BUCKETS = ("0", "1-10", "11-100", "101-1,000", ">1,000")
BUCKET_LABELS = ("0", "1–10", "11–100", "101–1k", ">1k")
MIN_BUCKET_N = 20  # buckets with fewer queries are drawn faded
FADE = 0.35
REGRET_TICKS = (1, 1.25, 1.5, 2, 3)


def plain_log(ax, axis: str = "y", decades_only: bool = False) -> None:
    """Log axis with plain-number tick labels (1, 2, 5, 10, ...) instead of 2x10^0."""
    fmt = FuncFormatter(lambda v, _: (f"{v:,.0f}" if v >= 1000 else f"{v:g}") if v >= 1 else "")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0,) if decades_only else (1.0, 2.0, 5.0)))
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


def top_legend(fig, handles, labels, ncol: int) -> None:
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, handlelength=2.2,
               columnspacing=1.0, borderaxespad=0.2)


def rows(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def save(fig, name: str, width_mm: float, height_mm: float) -> Path:
    fig.set_size_inches(width_mm * MM, height_mm * MM)
    path = OUT / name
    # no bbox_inches: keep the exact page width; no creation date, so reruns are byte-identical
    fig.savefig(path, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)
    return path


def table(name: str) -> dict[tuple[str, str, str], dict]:
    return {(r["dataset"], r["workload"], r["estimator"]): r for r in rows(NATIVE / name)}


# ---------------------------------------------------------------- acc_main
def acc_main(datasets: list[str]) -> Path:
    all_q, ne_q = table("task1_accuracy_table__planner_all.csv"), table("task1_accuracy_table__planner_nonempty.csv")
    fig, axes = plt.subplots(len(datasets), 2, sharey="row", layout="constrained")
    axes = np.atleast_2d(axes)
    for i, ds in enumerate(datasets):
        for j, wl in enumerate(WORKLOADS):
            ax = axes[i, j]
            # estimators the dataset has (AIS: no learned models); same x positions within a row
            ests = [e for e in ACC_EST if any((ds, w, e) in all_q for w in WORKLOADS)]
            for k, est in enumerate(ests):
                name, colour, marker, _ = STYLE[est]
                for src, dx, filled in ((all_q, -0.14, True), (ne_q, 0.14, False)):
                    r = src.get((ds, wl, est))
                    if r is None:
                        continue
                    med, lo, hi = float(r["median"]), float(r["median_ci95_lower"]), float(r["median_ci95_upper"])
                    ax.errorbar(k + dx, med, yerr=[[med - lo], [hi - med]], fmt=marker, color=colour,
                                mfc=colour if filled else "white", mec=colour, ms=4.5, elinewidth=0.8, capsize=1.5)
            ax.set_xticks(range(len(ests)))
            if i == len(datasets) - 1 or len(ests) != len(ACC_EST):
                few = len(ests) <= 3  # e.g. AIS: unrotated, wrapped onto two lines
                names = [STYLE[e][0].replace(" ", "\n", 1) if few else STYLE[e][0] for e in ests]
                ax.set_xticklabels(names, rotation=0 if few else 35, ha="center" if few else "right")
            else:
                ax.set_xticklabels([])
            ax.set_xlim(-0.6, len(ests) - 0.4)
            ax.set_yscale("log")
            plain_log(ax)
            ax.axhline(1, color="0.8", lw=0.5, zorder=0)
            ax.set_title(f"{DATASETS[ds]}, {WORKLOADS[wl]}")
            if j == 0:
                ax.set_ylabel("Median bounded q-error")
            ax.grid(axis="y", which="major", color="0.92", lw=0.5)
    h1, = axes[0, 1].plot([], [], "o", color="0.3")
    h2, = axes[0, 1].plot([], [], "o", color="0.3", mfc="white")
    top_legend(fig, [h1, h2], ["all queries (filled)", "non-empty queries (hollow)"], 2)
    return save(fig, "acc_main.pdf", TEXT_W, 110 if len(datasets) == 2 else 160)


# ---------------------------------------------------------------- acc_buckets
def acc_buckets(datasets: list[str]) -> Path:
    b = rows(NATIVE / "task1_bucket_table__planner_all.csv")
    fig, axes = plt.subplots(len(datasets), 2, sharey="row", layout="constrained")
    axes = np.atleast_2d(axes)
    handles = {}
    for i, ds in enumerate(datasets):
        for j, wl in enumerate(WORKLOADS):
            ax = axes[i, j]
            ns = {}
            for est in ("native_bbox", "separable", "path_hist_sum", "trajectory_sampling"):
                pts = {r["bucket"]: r for r in b if (r["dataset"], r["workload"], r["estimator"]) == (ds, wl, est)}
                if not pts:
                    continue
                name, colour, marker, dash = STYLE[est]
                xs = [k for k, bk in enumerate(BUCKETS) if bk in pts]
                ys = [float(pts[BUCKETS[k]]["median"]) for k in xs]
                for k in xs:
                    ns[k] = int(pts[BUCKETS[k]]["n_queries"])
                # buckets with n < 20 are faded: their markers and every segment touching them
                for k in range(len(xs) - 1):
                    faded = min(ns[xs[k]], ns[xs[k + 1]]) < MIN_BUCKET_N
                    ax.plot(xs[k:k + 2], ys[k:k + 2], linestyle=dash, color=colour, alpha=FADE if faded else 1.0)
                for x, y in zip(xs, ys):
                    ax.plot([x], [y], marker=marker, color=colour, alpha=1.0 if ns[x] >= MIN_BUCKET_N else FADE, ls="none")
                handles[name] = Line2D([], [], linestyle=dash, color=colour, marker=marker, markersize=4.5)
            present = sorted(ns)
            ax.set_xticks(present)
            ax.set_xticklabels([f"{BUCKET_LABELS[k]}\nn={ns[k]}" for k in present])
            for k, lab in zip(present, ax.get_xticklabels()):
                if ns[k] < MIN_BUCKET_N:
                    lab.set_alpha(FADE)
            ax.set_xlim(-0.5, len(BUCKETS) - 0.5)
            ax.set_yscale("log")
            plain_log(ax)
            ax.set_title(f"{DATASETS[ds]}, {WORKLOADS[wl]}")
            if j == 0:
                ax.set_ylabel("Median bounded q-error")
            if i == len(datasets) - 1:
                ax.set_xlabel("True cardinality")
            ax.grid(axis="y", color="0.92", lw=0.5)
    top_legend(fig, list(handles.values()), list(handles), 4)
    return save(fig, "acc_buckets.pdf", TEXT_W, 100 if len(datasets) == 2 else 150)


# ---------------------------------------------------------------- native_decomp
def decomp_records(ds: str, workload: str = "fixed_extent") -> list[tuple[int, float, float]]:
    """(truth, histogram error, semantic gap) per query, as in native_decomposition.csv."""
    if ds == "ais":
        native = rows(ROOT / "run/vldbj_rev/ais/native/explain_estimates.csv")
        boxes = rows(ROOT / "run/vldbj_rev/ais/native/box_counts.csv")
    else:
        native = rows(NATIVE / "explain_estimates_primary.csv")
        boxes = rows(NATIVE / "box_counts.csv")
    box = {(r["dataset"], r["workload"], int(r["query_index"])): int(r["box_count"]) for r in boxes}
    out = []
    for r in native:
        if r["dataset"] != ds or r["workload"] != workload:
            continue
        t, e = int(r["true_cardinality"]), float(r["native_bbox"])
        bc = box[(ds, workload, int(r["query_index"]))]
        out.append((t, np.log10((e + 1) / (bc + 1)), np.log10((bc + 1) / (t + 1))))
    return out


def bucket_of(t: int) -> int:
    return 0 if t == 0 else 1 if t <= 10 else 2 if t <= 100 else 3 if t <= 1000 else 4


def native_decomp(datasets: list[str]) -> list[Path]:
    """3x2 panels (rows datasets, columns fixed/mixed extent), one shared y range per row."""
    fig, axes = plt.subplots(len(datasets), 2, sharey="row", layout="constrained", squeeze=False)
    for i, ds in enumerate(datasets):
        for j, wl in enumerate(WORKLOADS):
            ax = axes[i, j]
            rec = decomp_records(ds, wl)
            present = [k for k in range(len(BUCKETS)) if any(bucket_of(r[0]) == k for r in rec)]
            for comp, off, colour in ((1, -0.18, "0.35"), (2, 0.18, "#56B4E9")):
                data = [[r[comp] for r in rec if bucket_of(r[0]) == k] for k in present]
                bp = ax.boxplot(data, positions=[slot + off for slot in range(len(present))], widths=0.32,
                                whis=(5, 95), showfliers=False, patch_artist=True,
                                medianprops={"color": "black", "lw": 0.9}, boxprops={"lw": 0.6},
                                whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
                for patch in bp["boxes"]:
                    patch.set_facecolor(colour); patch.set_alpha(0.85)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xticks(range(len(present)))
            ax.set_xticklabels([f"{BUCKET_LABELS[k]}\nn={sum(bucket_of(r[0]) == k for r in rec)}" for k in present])
            ax.set_xlim(-0.6, len(present) - 0.4)
            ax.set_title(f"{DATASETS[ds]}, {WORKLOADS[wl]}")
            ax.grid(axis="y", color="0.92", lw=0.5)
            if j == 0:
                ax.set_ylabel("log$_{10}$ ratio")
            if i == len(datasets) - 1:
                ax.set_xlabel("True cardinality")
    hs = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.5) for c in ("0.35", "#56B4E9")]
    top_legend(fig, hs, ["histogram error  log$_{10}$((e+1)/(b+1))  (positive = over)",
                         "semantic gap  log$_{10}$((b+1)/(p+1))"], 2)
    return [save(fig, "native_decomp.pdf", TEXT_W, 90)]


# ---------------------------------------------------------------- plan_regret_cdf
def plan_regret_cdf() -> Path:
    panels = (("T-Drive", ROOT / "run/vldbj_rev/plan_rerun/analysis/per_query_plan_metrics.csv",
               ("native", "true_cardinality_input", "trajectory_sampling", "joint_dm", "separable_dm")),
              ("Porto", ROOT / "run/vldbj_rev/plan_porto_nolearned/analysis/per_query_plan_metrics.csv",
               ("native", "true_cardinality_input", "trajectory_sampling")))
    fig, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
    for ax, (title, path, conds) in zip(axes, panels):
        data = rows(path)
        for c in conds:
            v = np.sort([float(r["regret"]) for r in data if r["condition"] == c])
            name, colour, _, dash = STYLE[c]
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colour, ls=dash, label=name)
        ax.axvline(1.25, color="0.4", ls="--", lw=0.7)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(REGRET_TICKS))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(1, max(REGRET_TICKS[-1], ax.get_xlim()[1]))
        ax.set_title(title)
        ax.set_xlabel("Regret (plan runtime / oracle runtime)")
        ax.grid(color="0.92", lw=0.5)
    axes[0].set_ylabel("Fraction of queries")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="lower right")
    return save(fig, "plan_regret_cdf.pdf", TEXT_W, 60)


# ---------------------------------------------------------------- learning_curve
def learning_curve() -> Path:
    budgets = (800, 1600, 3200, 6400)
    fig, ax = plt.subplots(layout="constrained")
    handles = []
    for est in ("joint", "separable", "no_coupling", "spatial_only"):
        name, colour, marker, _ = STYLE[est]
        med_mean, p95_mean, med_lo, med_hi = [], [], [], []
        for b in budgets:
            meds, p95s = [], []
            for f in sorted(glob.glob(str(ROOT / f"run/real_experiments/task7b/scalability/runs/learned/"
                                           f"tdrive__n1000000__mixed_extent__b{b}__{est}__s*.json"))):
                preds = json.loads(Path(f).read_text())["predictions"]
                t = np.array([p["true_cardinality"] for p in preds], float) + 1
                e = np.clip([p["predicted_cardinality"] for p in preds], 1, 1_000_000) + 1
                q = np.maximum(t / e, e / t)
                meds.append(np.median(q)); p95s.append(np.percentile(q, 95))
            med_mean.append(np.mean(meds)); p95_mean.append(np.mean(p95s))
            med_lo.append(min(meds)); med_hi.append(max(meds))
        h, = ax.plot(budgets, med_mean, color=colour, marker=marker, ls="-", label=name)
        handles.append(h)
        ax.fill_between(budgets, med_lo, med_hi, color=colour, alpha=0.15, lw=0)
        ax.plot(budgets, p95_mean, color=colour, marker=marker, ls="--", mfc="white")
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(["800", "1,600", "3,200", "6,400"])
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xlabel("Training labels")
    ax.set_ylabel("Bounded q-error")
    m, = ax.plot([], [], color="0.3", ls="-")
    q, = ax.plot([], [], color="0.3", ls="--")
    top_legend(fig, handles + [m, q], [STYLE[e][0] for e in ("joint", "separable", "no_coupling", "spatial_only")]
               + ["median", "p95"], 3)
    ax.grid(color="0.92", lw=0.5)
    return save(fig, "learning_curve.pdf", COL_W, 60)


# ---------------------------------------------------------------- tails_ccdf (optional)
def per_query_errors(ds: str, wl: str) -> dict[str, np.ndarray]:
    src, _ = SOURCES[ds]
    grouped = load_runs(src)
    out = {}
    first = next(e for e in ("trajectory_sampling", "joint") if (ds, wl, e) in grouped)
    truth = np.array([p["true_cardinality"] for p in grouped[(ds, wl, first)][0]["predictions"]])
    n_rows = 100_000.0
    for est in ("trajectory_sampling", "joint", "separable", "no_coupling", "spatial_only"):
        if (ds, wl, est) in grouped:
            out[est] = qerrors(truth, prediction_matrix(grouped[(ds, wl, est)]), "planner", n_rows).ravel()
    native = rows(ROOT / "run/vldbj_rev/ais/native/explain_estimates.csv") if ds == "ais" else rows(NATIVE / "explain_estimates_primary.csv")
    nat = sorted((r for r in native if r["dataset"] == ds and r["workload"] == wl), key=lambda r: int(r["query_index"]))
    out["native_bbox"] = qerrors(truth, np.array([[float(r["native_bbox"]) for r in nat]]), "planner", n_rows).ravel()
    hist_path = ROOT / ("run/vldbj_rev/ais/path_histogram/estimates.csv" if ds == "ais" else "run/vldbj_rev/path_histogram/estimates.csv")
    hist = sorted((r for r in rows(hist_path) if r["dataset"] == ds and r["workload"] == wl), key=lambda r: int(r["query_index"]))
    out["path_hist_sum"] = qerrors(truth, np.array([[float(r["path_hist_sum"]) for r in hist]]), "planner", n_rows).ravel()
    return out


def tails_ccdf(datasets: list[str]) -> Path:
    fig, axes = plt.subplots(len(datasets), 2, sharey=True, layout="constrained")
    axes = np.atleast_2d(axes)
    handles = {}
    for i, ds in enumerate(datasets):
        for j, wl in enumerate(WORKLOADS):
            ax = axes[i, j]
            for est, q in per_query_errors(ds, wl).items():
                name, colour, _, dash = STYLE[est]
                v = np.sort(q)
                handles[name], = ax.step(v, 1 - np.arange(len(v)) / len(v), where="post", color=colour, ls=dash)
            ax.set_xscale("log"); ax.set_yscale("log")
            plain_log(ax, "x", decades_only=max(q.max() for q in per_query_errors(ds, wl).values()) > 1000)
            ax.set_xlim(left=1)
            ax.set_title(f"{DATASETS[ds]}, {WORKLOADS[wl]}")
            if j == 0:
                ax.set_ylabel("P(q-error > x)")
            if i == len(datasets) - 1:
                ax.set_xlabel("Bounded q-error x")
            ax.grid(color="0.92", lw=0.5)
    top_legend(fig, list(handles.values()), list(handles), 4)
    return save(fig, "tails_ccdf.pdf", TEXT_W, 100 if len(datasets) == 2 else 140)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-ais", action="store_true", help="omit the AIS rows")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    taxi = ["tdrive", "porto"]
    acc_sets = taxi + ([] if args.no_ais else ["ais"])
    made = [acc_main(acc_sets), acc_buckets(acc_sets), *native_decomp(acc_sets),
            plan_regret_cdf(), learning_curve(), tails_ccdf(taxi)]  # tails: taxi datasets only (2x2)
    for p in made:
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
