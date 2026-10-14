"""Does `JointSpaceTimeGrid` earn its keep by *coupling*, or just by capacity?

The coupling diagnostic (`scripts/diagnose_coupling_advantage.py`) removed the two
confounds that hide the coupling term -- query-box-size decomposability and
counting noise at small cardinalities -- and found the joint grid beating its
own `use_coupling=False` ablation by 4-14%. But it won by that margin at
*every* `correlation_strength`, including 0.0, where the generator injects
essentially no space-time dependence. `use_coupling=False` removes the grid's
capacity and its interaction simultaneously, so it cannot say which one was
being paid for.

This script adds the arm that can: a **separable** grid with exactly the same
parameter count (4673 whole-net, either way) whose readout is provably
`f(lon, lat) + g(time)` for any parameters. Same workload, same confound
controls, same scale and repeat discipline as the coupling diagnostic.

Read the result like this:

* Joint beats separable, and the margin **grows with `correlation_strength`**,
  converging near zero at 0.0 -> the paper's explicit-coupling claim is real.
* Joint and separable are indistinguishable at every strength -> the benefit
  measured by the diagnostic is generic capacity and spatial-locality lookup, and the
  coupling claim has to be reframed around what is supported.

Arms (all at identical whole-net parameter count except `separable_lean`,
which is the deliberately parameter-lean resolution control):

    joint            shipped: JointSpaceTimeGrid + interaction features
    no_coupling      the ablation: grid readout zeroed, features off
    separable        PRIMARY CONTROL: capacity-matched separable grid,
                     interaction features off, so the arm is interaction-free
                     on both the input and the architecture side
    separable_ifeat  same grid, interaction features left ON -- isolates the
                     grid's joint index from the featurizer's product dims
    separable_lean   separable grid at the joint grid's own 8x8x8 resolution
                     (288 grid params, not 2048) -- guards the reading that
                     the matched control lost because 22x22 was too fine

Also reported per strength, and requiring no training at all: the held-out R²
of the best *separable* readout and the best *joint* readout fit in closed
form to the same targets. Their gap is the interaction headroom that exists in
this exact functional family, so it predicts whether the trained joint arm has
anything to win.

Run:
    uv run python scripts/spike_separable_control.py \
        --out run/revalidation/separable_grid_control.json
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.diagnose_coupling_advantage import fixed_extent_workload  # noqa: E402
from src.data_module.mobilitydb_loader import load_trajectories  # noqa: E402
from src.data_module.synthetic_generator import (  # noqa: E402
    SyntheticConfig,
    generate_synthetic_trajectories,
)
from src.estimator_module.coupled_learned.coupling import axis_coverage_weights  # noqa: E402
from src.estimator_module.coupled_learned.model import (  # noqa: E402
    GRID_EMBED_DIM,
    GRID_SHAPE,
    CoupledCardinalityNet,
    CoupledLearnedEstimator,
)
from src.estimator_module.coupled_learned.separable_coupling import (  # noqa: E402
    matched_separable_shape,
)
from src.eval_module.accuracy import evaluate_accuracy  # noqa: E402
from src.eval_module.correlation_analysis import measure_correlation_strength  # noqa: E402
from src.results_io import write_results_json  # noqa: E402
from src.utils.db import get_connection  # noqa: E402

logger = logging.getLogger(__name__)

DSN = "postgresql://postgres@localhost:5434/mobilitydb"
TABLE = "spike_separable_traj"

# Every scale constant below is the coupling diagnostic's, unchanged, so the
# joint-vs-no_coupling arm here is directly comparable with its published
# numbers and doubles as a check that this harness reproduces them.
N_TRAJECTORIES = 24000
N_POINTS_PER_TRAJ = 10
BOX_EXTENT = 2.0
WINDOW_HOURS = 1.0
N_TRAIN = 1500
N_EVAL = 500
EPOCHS = 1500
MODEL_SEEDS = (1, 2, 3)
DATA_SEED = 42
STRENGTHS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]

# name -> CoupledLearnedEstimator kwargs. "joint" is first because every ratio
# below is reported against it.
ARMS: dict[str, dict] = {
    "joint": dict(use_coupling=True, coupling_kind="joint", include_interactions=True),
    "no_coupling": dict(use_coupling=False, coupling_kind="joint", include_interactions=False),
    "separable": dict(use_coupling=True, coupling_kind="separable", include_interactions=False),
    "separable_ifeat": dict(use_coupling=True, coupling_kind="separable", include_interactions=True),
    "separable_lean": dict(
        use_coupling=True, coupling_kind="separable_lean", include_interactions=False
    ),
}
REFERENCE_ARM = "joint"


def arm_parameter_counts() -> dict[str, int]:
    """Build every arm's net and count its parameters, by direct construction.

    The comparison only means anything if the arms match, so this is asserted
    at run time and recorded in the output rather than argued for in a
    docstring -- the same check that was earlier run by hand on
    `use_coupling`.
    """
    counts: dict[str, int] = {}
    for name, kwargs in ARMS.items():
        net = CoupledCardinalityNet(
            use_coupling=kwargs["use_coupling"], coupling_kind=kwargs["coupling_kind"]
        )
        counts[name] = sum(p.numel() for p in net.parameters())
    matched = [n for n in ARMS if n != "separable_lean"]
    distinct = {counts[n] for n in matched}
    if len(distinct) != 1:
        raise AssertionError(f"capacity-matched arms disagree on parameter count: {counts}")
    return counts


def _raw_boxes(queries) -> np.ndarray:
    return np.array(
        [
            [q.predicate.min_lon, q.predicate.max_lon,
             q.predicate.min_lat, q.predicate.max_lat,
             q.predicate.start_time.timestamp(), q.predicate.end_time.timestamp()]
            for q in queries
        ],
        dtype=np.float64,
    )


def _axis_bounds(box: np.ndarray) -> list[tuple[float, float]]:
    """Per-axis (lo, hi) normalization bounds, mirroring how `Featurizer`
    derives its bounds from the *training* workload alone."""
    bounds = []
    for lo_col, hi_col in ((0, 1), (2, 3), (4, 5)):
        lo = float(box[:, [lo_col, hi_col]].min())
        hi = float(box[:, [lo_col, hi_col]].max())
        bounds.append((lo, hi if hi > lo else lo + 1.0))
    return bounds


def _coverage_design(queries, kind: str, bounds: list[tuple[float, float]]) -> np.ndarray:
    """Design matrix of coverage weights over the normalized unit cube.

    `kind="separable"` stacks the 22x22 spatial coverage weights beside the 28
    temporal ones: the exact function class `SeparableSpaceTimeGrid` can
    represent (a spatial table plus a temporal table, read out by coverage).
    `kind="joint"` uses the 8x8x8 outer product: the class
    `JointSpaceTimeGrid` can represent. Least squares on each therefore gives
    the best achievable *linear* readout of each architecture, with no
    training and no initialization luck involved.

    `bounds` must come from the training workload for both the train and the
    eval design, or the fitted coefficients index different cells in the two
    and the held-out R² is meaningless.
    """
    box = _raw_boxes(queries)
    for (lo_col, hi_col), (lo, hi) in zip(((0, 1), (2, 3), (4, 5)), bounds):
        box[:, [lo_col, hi_col]] = np.clip((box[:, [lo_col, hi_col]] - lo) / (hi - lo), 0.0, 1.0)

    t = torch.from_numpy(box)
    if kind == "joint":
        n_lon, n_lat, n_time = GRID_SHAPE
    else:
        n_lon, n_lat, n_time = matched_separable_shape(*GRID_SHAPE)
    w_lon = axis_coverage_weights(t[:, 0], t[:, 1], n_lon).numpy()
    w_lat = axis_coverage_weights(t[:, 2], t[:, 3], n_lat).numpy()
    w_time = axis_coverage_weights(t[:, 4], t[:, 5], n_time).numpy()

    spatial = np.einsum("ni,nj->nij", w_lon, w_lat).reshape(len(box), -1)
    if kind == "separable":
        blocks = [spatial, w_time]
    else:
        blocks = [np.einsum("ns,nk->nsk", spatial, w_time).reshape(len(box), -1)]
    return np.concatenate(blocks + [np.ones((len(box), 1))], axis=1)


RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = ridge * np.eye(x.shape[1])
    penalty[-1, -1] = 0.0  # never penalize the intercept column
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - pred) ** 2).sum() / sst) if sst > 0 else float("nan")


def _held_out_r2(train, evaluation, kind: str) -> tuple[float, float]:
    """Held-out R² of the best closed-form readout of `kind`'s function class.

    Both designs carry ~513 columns, so an unregularized fit is unstable
    whenever the workload is small relative to that -- badly enough to return
    a large negative held-out R² and make the two classes look different for
    a numerical reason rather than a structural one. The ridge strength is
    therefore chosen on a validation split of the *training* workload, by the
    same grid for both classes, and only then scored on `evaluation`. That
    keeps the comparison a statement about function classes.

    Returns:
        ``(held_out_r2, chosen_ridge)``.
    """
    bounds = _axis_bounds(_raw_boxes(train))
    x_tr = _coverage_design(train, kind, bounds)
    x_ev = _coverage_design(evaluation, kind, bounds)
    y_tr = np.log1p(np.array([q.true_cardinality for q in train], dtype=np.float64))
    y_ev = np.log1p(np.array([q.true_cardinality for q in evaluation], dtype=np.float64))

    cut = int(0.8 * len(x_tr))
    best_ridge, best_score = RIDGE_GRID[0], -np.inf
    for ridge in RIDGE_GRID:
        beta = _fit_ridge(x_tr[:cut], y_tr[:cut], ridge)
        score = _r2(y_tr[cut:], x_tr[cut:] @ beta)
        if np.isfinite(score) and score > best_score:
            best_ridge, best_score = ridge, score
    beta = _fit_ridge(x_tr, y_tr, best_ridge)
    return _r2(y_ev, x_ev @ beta), best_ridge


def _build_table(conn: psycopg.Connection, strength: float, args: argparse.Namespace) -> None:
    from psycopg import sql

    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(TABLE)))
        cur.execute(
            sql.SQL("CREATE TABLE {} (trajectory_id int, traj tgeompoint);").format(
                sql.Identifier(TABLE)
            )
        )
    df = generate_synthetic_trajectories(
        SyntheticConfig(
            n_trajectories=args.n_trajectories,
            n_points_per_traj=N_POINTS_PER_TRAJ,
            correlation_strength=strength,
            seed=DATA_SEED,
        )
    )
    load_trajectories(conn, TABLE, df)


def run_strength(conn: psycopg.Connection, strength: float, args: argparse.Namespace) -> dict:
    """Train every arm over `MODEL_SEEDS` on one strength's data."""
    _build_table(conn, strength, args)
    train = fixed_extent_workload(
        conn, TABLE, args.n_train, DATA_SEED, args.extent, args.window_hours
    )
    evaluation = fixed_extent_workload(
        conn, TABLE, args.n_eval, DATA_SEED + 1, args.extent, args.window_hours
    )

    scores: dict[str, list[float]] = {name: [] for name in ARMS}
    for model_seed in MODEL_SEEDS:
        for name, kwargs in ARMS.items():
            estimator = CoupledLearnedEstimator(seed=model_seed, epochs=args.epochs, **kwargs)
            estimator.fit(conn, TABLE, train_queries=train)
            scores[name].append(evaluate_accuracy(estimator, evaluation)["median"])

    cards = np.array([q.true_cardinality for q in evaluation], dtype=float)
    ref = np.array(scores[REFERENCE_ARM], dtype=float)
    sep_r2, sep_ridge = _held_out_r2(train, evaluation, "separable")
    joint_r2, joint_ridge = _held_out_r2(train, evaluation, "joint")
    row: dict = {
        "correlation_strength": strength,
        "measured_mi": measure_correlation_strength(conn, TABLE),
        "eval_p50_true_cardinality": float(np.median(cards)),
        "eval_n_nonzero": int((cards > 0).sum()),
        "separable_r2_heldout": sep_r2,
        "joint_r2_heldout": joint_r2,
        "r2_ridge_chosen": {"separable": sep_ridge, "joint": joint_ridge},
        "per_seed": {name: scores[name] for name in ARMS},
        "mean": {name: float(np.mean(scores[name])) for name in ARMS},
        "std": {name: float(np.std(scores[name], ddof=1)) for name in ARMS},
    }
    row["interaction_headroom_r2"] = row["joint_r2_heldout"] - row["separable_r2_heldout"]
    # Two ratio conventions, both >1 meaning "joint is better": the ratio of
    # means (comparable with the diagnostic's table) and the mean of the per-seed
    # paired ratios (which carries a spread, so a margin can be called against
    # the seed noise rather than eyeballed).
    row["ratio_over_joint"] = {
        name: float(np.mean(scores[name]) / np.mean(ref)) for name in ARMS
    }
    row["paired_ratio_over_joint_mean"] = {
        name: float(np.mean(np.array(scores[name]) / ref)) for name in ARMS
    }
    row["paired_ratio_over_joint_std"] = {
        name: float(np.std(np.array(scores[name]) / ref, ddof=1)) for name in ARMS
    }
    logger.info(
        "strength=%.2f mi=%.3f headroom_r2=%.3f separable/joint=%.3f no_coupling/joint=%.3f",
        strength, row["measured_mi"], row["interaction_headroom_r2"],
        row["ratio_over_joint"]["separable"], row["ratio_over_joint"]["no_coupling"],
    )
    # Printed in full, per_seed included: this run takes long enough that the
    # log is the recovery path if the process dies mid-sweep.
    print(json.dumps(row), flush=True)
    return row


def summarize(rows: list[dict]) -> dict:
    """Trend of each arm's advantage over the strength knob.

    The verdict this whole script exists to reach is not "is the ratio > 1"
    but "does it GROW with `correlation_strength`, from ~1.0 at 0.0". A flat
    ratio -- however large -- is the generic-capacity signature the diagnostic already
    found. So report the slope and the endpoint gap, not just the mean.
    """
    strengths = np.array([r["correlation_strength"] for r in rows], dtype=float)
    summary: dict = {}
    for name in ARMS:
        ratios = np.array([r["ratio_over_joint"][name] for r in rows], dtype=float)
        slope = float(np.polyfit(strengths, ratios, 1)[0]) if len(rows) > 1 else float("nan")
        summary[name] = {
            "ratio_at_min_strength": float(ratios[0]),
            "ratio_at_max_strength": float(ratios[-1]),
            "ratio_mean_over_strengths": float(ratios.mean()),
            "ratio_slope_per_unit_strength": slope,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dsn", default=DSN)
    parser.add_argument("--n-trajectories", type=int, default=N_TRAJECTORIES)
    parser.add_argument("--extent", type=float, default=BOX_EXTENT)
    parser.add_argument("--window-hours", type=float, default=WINDOW_HOURS)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--strengths", type=float, nargs="+", default=STRENGTHS)
    parser.add_argument(
        "--threads", type=int, default=4,
        help="torch intra-op threads. The default is deliberately low: these "
             "are tiny full-batch tensors, and letting torch use every core "
             "makes a fit ~2x slower and a single-query estimate() far worse.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(args.threads)
    counts = arm_parameter_counts()
    logger.info("arm parameter counts: %s", counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    conn = get_connection(args.dsn)
    rows: list[dict] = []
    try:
        for strength in args.strengths:
            rows.append(run_strength(conn, strength, args))
            # Checkpoint after every strength. The full sweep is 75 trainings;
            # losing all of it to an interrupted run is avoidable.
            _write(args, counts, rows, started, partial=len(rows) < len(args.strengths))
    finally:
        from psycopg import sql

        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(TABLE)))
        conn.close()

    print(f"\nwrote {args.out}")


def _write(
    args: argparse.Namespace, counts: dict[str, int], rows: list[dict],
    started: float, partial: bool,
) -> None:
    write_results_json(
        {
            "partial": partial,
            "run_config": {
                "n_trajectories": args.n_trajectories,
                "n_points_per_traj": N_POINTS_PER_TRAJ,
                "box_extent": args.extent,
                "window_hours": args.window_hours,
                "n_train": args.n_train,
                "n_eval": args.n_eval,
                "epochs": args.epochs,
                "model_seeds": list(MODEL_SEEDS),
                "data_seed": DATA_SEED,
                "strengths": list(args.strengths),
                "grid_shape": list(GRID_SHAPE),
                "grid_embed_dim": GRID_EMBED_DIM,
                "separable_shape": list(matched_separable_shape(*GRID_SHAPE)),
                "arms": {k: dict(v) for k, v in ARMS.items()},
                "arm_parameter_counts": counts,
            },
            "rows": rows,
            "summary": summarize(rows),
            "wall_seconds": round(time.time() - started, 1),
        },
        args.out,
    )
    logger.info("checkpointed %d/%d strengths to %s", len(rows), len(args.strengths), args.out)


if __name__ == "__main__":
    main()
