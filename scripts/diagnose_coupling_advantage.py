"""Why the same-architecture ablation does not show a coupling advantage.

The re-validation check expects `CoupledLearnedEstimator`'s
explicit `JointSpaceTimeGrid` term to beat its own `use_coupling=False`
ablation at high `correlation_strength`, and to tie at 0.0. It does not, on
the workload `run_correlation_sweep` actually uses. This script isolates the
two confounds that stand between the sweep and that measurement, so the
finding rests on evidence rather than on one sweep's numbers.

Confound 1 -- **box-size variance**. `generate_workload` draws each spatial
bound independently and each time window log-uniformly, so a query's
cardinality is driven mostly by how big the box is, which is decomposable
(`log volume = log dlon + log dlat + log dt`). An earlier architecture comparison measured the
size term explaining 76% of the log-target's variance and "swamping every
architectural difference"; its own benchmark
(`_make_interaction_heavy_queries`) holds extent fixed for exactly this
reason. This script's workload does the same.

Confound 2 -- **counting noise at small cardinalities**. Concentrating trips
into joint space-time modes empties most of the domain, so at high strength a
fixed-extent box catches 1-3 trips and median q-error measures Poisson
discreteness rather than model error -- both arms sit on the same floor.
Raising `n_trajectories` restores counts large enough for q-error to resolve
the models.

Run:
    uv run python scripts/diagnose_coupling_advantage.py \
        --out run/revalidation/coupling_advantage_diagnostic.json
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_module.mobilitydb_loader import load_trajectories  # noqa: E402
from src.data_module.synthetic_generator import (  # noqa: E402
    SyntheticConfig,
    generate_synthetic_trajectories,
)
from src.data_module.workload_generator import (  # noqa: E402
    STPredicate,
    WorkloadQuery,
    compute_ground_truth,
)
from src.estimator_module.coupled_learned.model import CoupledLearnedEstimator  # noqa: E402
from src.eval_module.accuracy import evaluate_accuracy  # noqa: E402
from src.eval_module.correlation_analysis import (  # noqa: E402
    SYNTHETIC_TIME_RANGE,
    measure_correlation_strength,
)
from src.results_io import write_results_json  # noqa: E402
from src.utils.db import get_connection  # noqa: E402

logger = logging.getLogger(__name__)

DSN = "postgresql://postgres@localhost:5434/mobilitydb"
TABLE = "diagnose_coupling_traj"

# Enough trips that a fixed-extent box still catches a double-digit count at
# correlation_strength=1.0, where the mode field has emptied most of the
# domain. At `SWEEP_N_TRAJECTORIES` (6000) the same workload has median true
# cardinality 3 and both arms sit on the counting-noise floor.
N_TRAJECTORIES = 24000
N_POINTS_PER_TRAJ = 10
# Fixed box extent, in domain units, and fixed query window, in hours.
BOX_EXTENT = 2.0
WINDOW_HOURS = 1.0
N_TRAIN = 1500
N_EVAL = 400
EPOCHS = 1500
MODEL_SEEDS = (1, 2, 3)
DATA_SEED = 42
STRENGTHS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


def fixed_extent_workload(
    conn: psycopg.Connection, table: str, n_queries: int, seed: int,
    extent: float, window_hours: float,
    bbox: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
    time_range: tuple[datetime, datetime] = SYNTHETIC_TIME_RANGE,
) -> list[WorkloadQuery]:
    """A workload whose queries differ only in WHERE and WHEN they sit.

    Every box has side `extent` and every window lasts `window_hours`, so the
    decomposable size term is constant across the workload and cannot explain
    any of the target's variance -- leaving the joint space-time part, which
    is what the ablation is trying to measure.

    `bbox`/`time_range` default to the synthetic module's own `[0, 10] x
    [0, 10]` domain and `SYNTHETIC_TIME_RANGE` respectively, so every
    existing caller (`diagnose_coupling_advantage.run_strength`,
    `spike_separable_control.run_strength`) is unaffected. A real-table
    caller passes that table's own derived bounds instead (see
    `real_experiment_steps.fetch_table_workload_bounds`) -- this is the only
    change needed to make the helper dataset-agnostic: the box-size-fixed / high-
    cardinality property this workload exists for does not depend on which
    domain the box sits in, only on the caller picking an `extent`/
    `window_hours` that is a sane fraction of that domain's own span.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.
        n_queries: Number of queries to generate.
        seed: RNG seed for reproducibility.
        extent: Fixed side length of every query box, in `bbox`'s own units
            (must be smaller than both `bbox` axis spans or query boxes
            would not fit inside the domain).
        window_hours: Fixed duration of every query's time window.
        bbox: ``(min_lon, max_lon, min_lat, max_lat)`` domain to draw query
            box origins from.
        time_range: ``(t0, t1)`` domain to draw query window starts from.
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    t0, t1 = time_range
    total_seconds = (t1 - t0).total_seconds() - window_hours * 3600.0
    rng = np.random.default_rng(seed)
    queries: list[WorkloadQuery] = []
    for _ in range(n_queries):
        lon0 = float(rng.uniform(min_lon, max_lon - extent))
        lat0 = float(rng.uniform(min_lat, max_lat - extent))
        start = t0 + timedelta(seconds=float(rng.uniform(0.0, total_seconds)))
        predicate = STPredicate(
            lon0, lon0 + extent, lat0, lat0 + extent,
            start, start + timedelta(hours=window_hours),
        )
        queries.append(WorkloadQuery(
            predicate=predicate,
            true_cardinality=compute_ground_truth(conn, table, predicate),
        ))
    return queries


def _build_table(conn: psycopg.Connection, strength: float, args: argparse.Namespace) -> None:
    from psycopg import sql
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(TABLE)))
        cur.execute(
            sql.SQL("CREATE TABLE {} (trajectory_id int, traj tgeompoint);").format(
                sql.Identifier(TABLE)
            )
        )
    df = generate_synthetic_trajectories(SyntheticConfig(
        n_trajectories=args.n_trajectories, n_points_per_traj=N_POINTS_PER_TRAJ,
        correlation_strength=strength, seed=DATA_SEED,
    ))
    load_trajectories(conn, TABLE, df)


def run_strength(conn: psycopg.Connection, strength: float, args: argparse.Namespace) -> dict:
    """Train both arms over `MODEL_SEEDS` on one strength's data."""
    _build_table(conn, strength, args)
    train = fixed_extent_workload(
        conn, TABLE, args.n_train, DATA_SEED, args.extent, args.window_hours
    )
    evaluation = fixed_extent_workload(
        conn, TABLE, args.n_eval, DATA_SEED + 1, args.extent, args.window_hours
    )
    scores: dict[bool, list[float]] = {True: [], False: []}
    for model_seed in MODEL_SEEDS:
        for use_coupling in (True, False):
            estimator = CoupledLearnedEstimator(
                seed=model_seed, epochs=args.epochs, use_coupling=use_coupling
            )
            estimator.fit(conn, TABLE, train_queries=train)
            scores[use_coupling].append(evaluate_accuracy(estimator, evaluation)["median"])
    cards = np.array([q.true_cardinality for q in evaluation], dtype=float)
    coupled, ablated = scores[True], scores[False]
    row = {
        "correlation_strength": strength,
        "measured_mi": measure_correlation_strength(conn, TABLE),
        "eval_p50_true_cardinality": float(np.median(cards)),
        "eval_n_nonzero": int((cards > 0).sum()),
        "coupled_median_qerror_per_seed": coupled,
        "coupled_no_coupling_median_qerror_per_seed": ablated,
        "coupled_mean": float(np.mean(coupled)),
        "coupled_no_coupling_mean": float(np.mean(ablated)),
        "ablation_ratio_no_coupling_over_coupled": float(np.mean(ablated) / np.mean(coupled)),
    }
    logger.info("strength=%s ratio=%.3f", strength, row["ablation_ratio_no_coupling_over_coupled"])
    print(json.dumps(row))
    return row


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    started = time.time()
    conn = get_connection(args.dsn)
    try:
        rows = [run_strength(conn, s, args) for s in args.strengths]
    finally:
        from psycopg import sql
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(TABLE)))
        conn.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_results_json({
        "run_config": {
            "n_trajectories": args.n_trajectories, "n_points_per_traj": N_POINTS_PER_TRAJ,
            "box_extent": args.extent, "window_hours": args.window_hours,
            "n_train": args.n_train, "n_eval": args.n_eval, "epochs": args.epochs,
            "model_seeds": list(MODEL_SEEDS), "data_seed": DATA_SEED,
            "strengths": list(args.strengths),
        },
        "rows": rows,
        "wall_seconds": round(time.time() - started, 1),
    }, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
