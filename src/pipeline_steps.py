"""Individual evaluation-step logic for the smoke pipeline (split out of
`run_pipeline.py`, which had grown too long once the cross-city transfer and
robustness wiring and the multi-estimator fitting loop were added).

`run_pipeline.run_smoke_pipeline` owns Hydra wiring, top-level orchestration,
and JSON serialization; the functions here do the actual per-step DB-backed
work (creating/loading a synthetic table, fitting the full estimator
registry, running cross-city transfer, running density/temporal-drift
robustness checks) and return plain result dicts for `run_smoke_pipeline` to
compose and stamp into `results.json`.

Every registered estimator's constructor is dispatched explicitly by name
in `build_estimator` rather than called uniformly with no arguments:
registered estimator constructors are NOT uniform (`CoupledLearnedEstimator`
takes `seed`/`epochs`/`use_coupling`; `SpatialOnlyMLEstimator` and
`SamplingEstimator` take `seed`; the rest take only their own tuned
hyperparameter defaults), and a naive ``EstimatorFactory(name)()`` call
would silently ignore both `seed` (breaking this run's reproducibility
contract) and `epochs` (leaving `coupled_learned` at its tuned production
default of 1500 epochs instead of the smoke-scale budget the whole point of
a "smoke" pipeline run depends on).
"""
import logging
from datetime import datetime
from typing import Callable

import psycopg
from psycopg import sql

from src.data_module.mobilitydb_loader import load_trajectories
from src.data_module.synthetic_generator import SyntheticConfig, generate_synthetic_trajectories
from src.data_module.workload_generator import WorkloadQuery, generate_workload
from src.estimator_module.base import ESTIMATOR_FACTORY, Estimator, EstimatorFactory
from src.estimator_module.coupled_learned.model import CoupledLearnedEstimator
from src.eval_module.correlation_analysis import measure_correlation_strength, sweep_table_name
from src.eval_module.generalization import evaluate_cross_city_transfer
from src.eval_module.robustness import evaluate_density_stratified, evaluate_temporal_drift

logger = logging.getLogger(__name__)

# Smoke-run constants. These are deliberately NOT configurable: they define
# what makes a smoke run a wiring smoke test rather than a real experiment.
# `run_pipeline.run_smoke_pipeline` stamps the values actually used into
# `results.json`'s own `run_config` block, so a saved result stays
# self-describing without needing this module as provenance.
SMOKE_TABLE = "smoke_pipeline_traj"
# Raised from 200 x 5 alongside the synthetic generator rewrite
# that made a trip stay in its own district rather than scatter its points
# across the whole domain. A local trip occupies a much smaller slice of the
# space-time cube, so a fixture this size stopped producing a workload with
# anything in it: measured 5/20 eval queries non-empty at 400 x 5, versus
# 17/20 at 4000 x 10. A smoke run whose queries are nearly all empty
# exercises the wiring but reports numbers that mean nothing.
SMOKE_N_TRAJECTORIES = 4000
SMOKE_N_POINTS_PER_TRAJ = 10
SMOKE_EPOCHS = 20
SMOKE_CORRELATION_STRENGTH = 0.7
SMOKE_SWEEP_STRENGTHS: list[float] = [0.0, 1.0]
# Number of eval queries fed to `evaluate_systems_impact`; each one costs a
# real EXPLAIN (ANALYZE) round-trip per estimator, so keep it small.
SMOKE_N_PLAN_QUERIES = 5
SMOKE_BBOX: tuple[float, float, float, float] = (0, 10, 0, 10)

# `evaluate_density_stratified`'s bucket count for the smoke run.
SMOKE_DENSITY_N_BINS = 3
# `evaluate_temporal_drift`'s train/eval split, picked well inside the
# smoke table's real ~24-hour data span (SYNTHETIC_TIME_RANGE) with margin
# on both sides -- see `run_robustness_step`'s docstring.
SMOKE_TEMPORAL_DRIFT_TRAIN_BEFORE = datetime(2026, 1, 1, 8, 0, 0)
SMOKE_TEMPORAL_DRIFT_EVAL_AFTER = datetime(2026, 1, 1, 16, 0, 0)

# The second synthetic "city" that cross-city transfer needs: a different
# correlation_strength than the primary smoke table's SMOKE_CORRELATION_
# STRENGTH (0.7), so transfer is exercised against genuinely different
# space-time structure, not an identical distribution under a new name.
# Its own seed is derived from the run's `seed` (see `run_cross_city_step`)
# rather than a fixed constant, so a smoke run stays fully reproducible
# from its single `seed` argument, matching this codebase's existing
# `seed + 1`-for-eval-workload convention.
SMOKE_CITY_B_TABLE = "smoke_pipeline_city_b_traj"
SMOKE_CITY_B_CORRELATION_STRENGTH = 0.2
SMOKE_CITY_B_SEED_OFFSET = 1000


def smoke_tables() -> list[str]:
    """Every table a smoke pipeline run creates: the primary table, the
    second "city" table, plus the one table per correlation strength
    that the nested `run_correlation_sweep` call creates internally.

    Derived from `sweep_table_name` rather than hardcoded, so this cannot
    drift from `run_correlation_sweep`'s actual naming scheme.
    """
    return [SMOKE_TABLE, SMOKE_CITY_B_TABLE] + [sweep_table_name(s) for s in SMOKE_SWEEP_STRENGTHS]


def drop_smoke_tables(conn: psycopg.Connection) -> None:
    """Drop every table a smoke pipeline run creates, so repeated runs do
    not accumulate junk in the shared MobilityDB instance."""
    with conn.cursor() as cur:
        for table in smoke_tables():
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {table};").format(table=sql.Identifier(table)))


def create_trajectory_table(conn: psycopg.Connection, table: str) -> None:
    """Drop (if present) and (re)create a ``(trajectory_id int, traj
    tgeompoint)`` table, the standard shape `load_trajectories` expects."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {table};").format(table=sql.Identifier(table)))
        cur.execute(
            sql.SQL("CREATE TABLE {table} (trajectory_id int, traj tgeompoint);").format(
                table=sql.Identifier(table)
            )
        )


def build_estimator(name: str, seed: int, epochs: int) -> Estimator:
    """Construct one registered estimator by name, threading this run's
    `seed` (and, for the trained-from-scratch coupled net, its smoke-scale
    `epochs` budget) through explicitly -- see module docstring for why a
    uniform no-arg `EstimatorFactory(name)()` call is not used here.

    Args:
        name: A key in `ESTIMATOR_FACTORY` (e.g. ``"coupled_learned"``).
        seed: Random seed forwarded to every estimator whose constructor
            accepts one (``coupled_learned``, ``spatial_only_ml``,
            ``sampling``, ``trajectory_sampling``); ignored by the rest, whose constructors take no
            such parameter.
        epochs: Training epoch budget forwarded only to
            ``coupled_learned`` (the only estimator here trained via
            gradient descent).

    Returns:
        A fresh, unfitted `Estimator` instance.
    """
    cls = EstimatorFactory(name)
    if name == "coupled_learned":
        return cls(seed=seed, epochs=epochs)
    if name in ("spatial_only_ml", "sampling", "trajectory_sampling"):
        return cls(seed=seed)
    return cls()


def fit_all_estimators(
    conn: psycopg.Connection,
    table: str,
    train_queries: list[WorkloadQuery],
    seed: int,
    epochs: int,
) -> dict[str, Estimator]:
    """Build and fit every registered estimator (`ESTIMATOR_FACTORY`)
    against ``table``, for the smoke pipeline's accuracy/systems-impact
    comparison (the orchestrator previously only ever exercised 2 of the (now 7)
    registered estimators).

    ``train_queries`` is passed to every estimator's `fit()` uniformly
    (including the estimators that ignore it, e.g. the histogram/sampling/
    native baselines, whose `fit()` signatures accept and discard
    ``train_queries`` -- see each module in `src/estimator_module/`), so
    this loop needs no per-estimator branching beyond `build_estimator`'s
    constructor dispatch.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.
        train_queries: Training workload shared across every estimator.
        seed: Forwarded to `build_estimator`.
        epochs: Forwarded to `build_estimator`.

    Returns:
        ``{name: fitted_estimator}``, one entry per key currently in
        `ESTIMATOR_FACTORY`, iterated in sorted name order for a
        deterministic fit sequence.
    """
    fitted: dict[str, Estimator] = {}
    for name in sorted(ESTIMATOR_FACTORY):
        estimator = build_estimator(name, seed=seed, epochs=epochs)
        estimator.fit(conn, table, train_queries=train_queries)
        fitted[name] = estimator
    logger.info("fit_all_estimators: fitted %s against table=%s", sorted(fitted), table)
    return fitted


def run_cross_city_step(
    conn: psycopg.Connection,
    source_table: str,
    source_train_queries: list[WorkloadQuery],
    seed: int,
    epochs: int,
    n_trajectories: int,
    n_points_per_traj: int,
    n_eval: int,
    bbox: tuple[float, float, float, float],
    time_range: tuple[datetime, datetime],
) -> dict:
    """Build a second synthetic "city" (`SMOKE_CITY_B_TABLE`, a different
    correlation_strength than ``source_table``'s and its own seed derived
    from ``seed``), generate its own eval workload, and run
    `evaluate_cross_city_transfer` from ``source_table`` to it.

    Uses `CoupledLearnedEstimator` as the transferred estimator rather
    than a simpler baseline, so this exercises the
    `Estimator.refit_normalization` hook meaningfully: the target city's
    coordinates are drawn independently of the source's, so without that
    hook every target feature would clip to a normalization-artifact
    constant.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        source_table: The already-loaded source table (the smoke
            pipeline's primary table).
        source_train_queries: Training workload already generated against
            ``source_table`` -- reused here rather than regenerated, since
            `evaluate_cross_city_transfer` fits a fresh estimator on
            exactly this workload.
        seed: This run's base seed. Also forwarded to the
            `CoupledLearnedEstimator` factory (``seed=seed``), matching
            every other estimator instantiation in this smoke run.
        epochs: Training epoch budget for the transferred estimator
            (`SMOKE_EPOCHS` from the caller).
        n_trajectories: Trajectory count for the second city's synthetic
            dataset (matches the primary table's own scale).
        n_points_per_traj: Points-per-trajectory for the second city's
            synthetic dataset.
        n_eval: Number of target-city evaluation queries to generate.
        bbox: Sampling bbox forwarded to `generate_workload`.
        time_range: Sampling time range forwarded to `generate_workload`
            (the synthetic generator's real 24-hour span; see
            `src.eval_module.correlation_analysis.SYNTHETIC_TIME_RANGE`).

    Returns:
        ``{"target_table": str, "transfer": dict, "target_measured_mi":
        float}``, where ``transfer`` is `evaluate_cross_city_transfer`'s
        own return dict and ``target_measured_mi`` is
        `measure_correlation_strength` on the target table --
        computed here (one extra DB round trip) so the caller can feed
        both source and target MI into `analyze_transfer_failure` without
        threading a second connection round trip through it separately.
    """
    table = SMOKE_CITY_B_TABLE
    city_b_seed = seed + SMOKE_CITY_B_SEED_OFFSET

    create_trajectory_table(conn, table)
    cfg = SyntheticConfig(
        n_trajectories=n_trajectories, n_points_per_traj=n_points_per_traj,
        correlation_strength=SMOKE_CITY_B_CORRELATION_STRENGTH, seed=city_b_seed,
    )
    df = generate_synthetic_trajectories(cfg)
    load_trajectories(conn, table, df)

    target_eval_queries = generate_workload(
        conn, table, n_queries=n_eval, seed=city_b_seed + 1,
        bbox=bbox, knn_fraction=0.0, time_range=time_range,
    )
    transfer = evaluate_cross_city_transfer(
        lambda: CoupledLearnedEstimator(seed=seed, epochs=epochs),
        conn, source_table, source_train_queries, table, target_eval_queries,
    )
    target_measured_mi = measure_correlation_strength(conn, table)
    logger.info(
        "run_cross_city_step: target_table=%s target_measured_mi=%.4f transfer_median=%.3f",
        table, target_measured_mi, transfer["median"],
    )
    return {"target_table": table, "transfer": transfer, "target_measured_mi": target_measured_mi}


def run_robustness_step(
    conn: psycopg.Connection,
    table: str,
    density_estimator: Estimator,
    eval_queries: list[WorkloadQuery],
    n_density_bins: int,
    temporal_drift_estimator_factory: Callable[[], Estimator],
    bbox: tuple[float, float, float, float],
    train_before: datetime,
    eval_after: datetime,
    n_train: int,
    n_eval: int,
    seed: int,
) -> dict:
    """Run the two DB-backed robustness checks against ``table``.

    `evaluate_density_stratified` reuses an already-fitted estimator and
    the smoke pipeline's own ``eval_queries`` (no extra DB round trip
    beyond the density fetch it does internally). `evaluate_temporal_drift`
    instead trains its own estimator via ``temporal_drift_estimator_factory``
    on a fresh train/eval split scoped to ``[table_start, train_before)`` /
    ``[eval_after, table_end)`` -- ``train_before``/``eval_after`` must fall
    strictly inside the table's real data span (see `evaluate_temporal_
    drift`'s own bound validation); the module-level `SMOKE_TEMPORAL_DRIFT_
    TRAIN_BEFORE`/`SMOKE_TEMPORAL_DRIFT_EVAL_AFTER` defaults are picked
    well inside the smoke table's real ~24-hour span (roughly hour 8 and
    hour 16) with margin on both sides.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.
        density_estimator: An already-fitted estimator to score for the
            density-stratified breakdown.
        eval_queries: Already-generated range-only eval workload (must
            have been generated with ``knn_fraction=0.0`` -- see
            `evaluate_density_stratified`'s own caller caveat).
        n_density_bins: Forwarded to `evaluate_density_stratified`.
        temporal_drift_estimator_factory: Zero-arg factory forwarded to
            `evaluate_temporal_drift`, which trains it fresh on its own
            internally-generated train split.
        bbox: Forwarded to `evaluate_temporal_drift`.
        train_before: Forwarded to `evaluate_temporal_drift`.
        eval_after: Forwarded to `evaluate_temporal_drift`.
        n_train: Forwarded to `evaluate_temporal_drift`.
        n_eval: Forwarded to `evaluate_temporal_drift`.
        seed: Forwarded to `evaluate_temporal_drift`.

    Returns:
        ``{"density_stratified": dict, "temporal_drift": dict}``.
    """
    density_stratified = evaluate_density_stratified(
        conn, table, density_estimator, eval_queries, n_density_bins=n_density_bins,
    )
    temporal_drift = evaluate_temporal_drift(
        conn, table, temporal_drift_estimator_factory, bbox=bbox,
        train_before=train_before, eval_after=eval_after,
        n_train=n_train, n_eval=n_eval, seed=seed,
    )
    return {"density_stratified": density_stratified, "temporal_drift": temporal_drift}
