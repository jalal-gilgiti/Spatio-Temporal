"""Real-dataset (T-Drive/Porto/NYC TLC) evaluation-battery logic.

This is the real-data counterpart of `src.pipeline_steps` (the smoke
pipeline's per-step DB-backed logic, kept apart from the Hydra wiring). It is
a SEPARATE module rather than an extension of `pipeline_steps.py` because
the two serve genuinely different runs: `pipeline_steps.py`'s `SMOKE_*`
constants and helpers assume a synthetic table on the known `[0, 10] x
[0, 10]` / 24-hour domain (`SyntheticConfig`'s own sampling domain), while
every function here takes an arbitrary already-loaded real table and
derives its bbox/time_range from the live data instead (see
`_fetch_table_spatial_bounds`/`_fetch_table_time_bounds` in
`src.eval_module.robustness`).

Two entry points:

  - `run_real_experiment_for_dataset`: the full single-dataset battery
    (accuracy for all registered estimators, systems impact,
    real-data MI measurement, density-stratified/temporal-drift
    robustness, design-guidance synthesis) against one already-loaded
    real table.
  - `run_cross_city_transfer_matrix`: cross-city transfer, run over
    every directed pair among a set of real tables (6 pairs for 3 cities).

The *synthetic* controlled correlation sweep (`run_correlation_sweep`) is
deliberately NOT run here: it generates its own synthetic data at each
`correlation_strength` and has no notion of an already-loaded real table
to sweep over. The real-data half that this module exercises instead is
`measure_correlation_strength` directly (the real MI number), which is
threaded into `build_design_guidance`'s `accuracy_by_dataset` argument the
same way `src.run_pipeline`/`src.pipeline_steps` already do for the
synthetic case -- see `run_real_experiment_for_dataset`'s
``design_guidance`` construction below.
"""
import logging
from datetime import datetime, timedelta
from typing import Callable

import psycopg

from src.data_module.workload_generator import WorkloadQuery, generate_workload
from src.estimator_module.coupled_learned.model import CoupledLearnedEstimator
from src.estimator_module.spatial_only_ml import SpatialOnlyMLEstimator
from src.eval_module.accuracy import evaluate_accuracy
from src.eval_module.correlation_analysis import measure_correlation_strength
from src.eval_module.generalization import evaluate_cross_city_transfer
from src.eval_module.robustness import (
    _fetch_table_spatial_bounds,
    _fetch_table_time_bounds,
    analyze_transfer_failure,
    evaluate_density_stratified,
    evaluate_selectivity_extremes,
    evaluate_temporal_drift,
)
from src.eval_module.synthesis import build_design_guidance
from src.eval_module.systems_impact import evaluate_systems_impact
from src.pipeline_steps import fit_all_estimators

logger = logging.getLogger(__name__)

# Real-experiment scale. Trimmed from the synthetic single-sweep scale
# (n_train=1500, n_eval=500, epochs=1500) because this task runs that
# pattern 3 times (once per real dataset) PLUS a 6-directed-pair cross-city
# transfer matrix, each pair training its own fresh CoupledLearnedEstimator
# -- roughly 3x the single-run cost plus 6 more coupled-net trainings, a
# much larger total workload than one synthetic sweep. Still large enough
# to be genuinely representative, not toy-scale: n_train=800 comfortably
# exceeds the smallest n_train (500) any convergence test in this repo
# tunes against, and epochs=1000 is the same order of magnitude as the
# tuned production default (1500), not a 10x-smaller smoke-scale budget.
REAL_N_TRAIN = 800
REAL_N_EVAL = 300
REAL_EPOCHS = 1000
REAL_SEED = 42

# Number of eval queries fed to `evaluate_systems_impact`; each one costs a
# real EXPLAIN (ANALYZE) round-trip per estimator (7 estimators here), so
# keep it small -- matches `pipeline_steps.SMOKE_N_PLAN_QUERIES`.
REAL_N_PLAN_QUERIES = 5

# `evaluate_density_stratified`'s bucket count.
REAL_DENSITY_N_BINS = 3

# `evaluate_temporal_drift`'s train/eval split, expressed as FRACTIONS of a
# table's own real [table_start, table_end] span rather than fixed
# calendar dates (unlike `pipeline_steps.SMOKE_TEMPORAL_DRIFT_*`, which can
# hardcode absolute datetimes because every smoke table shares the same
# synthetic 24-hour span). The 3 real tables span wildly different real
# calendar ranges (T-Drive ~6 days, Porto ~1 year, NYC TLC ~1 month), so a
# single hardcoded literal cannot be inside every table's span at once --
# fractions of each table's own derived span always are, by construction.
# 60% train / 15% gap / 25% eval: both windows comfortably clear
# `evaluate_temporal_drift`'s own 10%-of-span "may be too thin" warning
# threshold, with a real gap between them so this measures actual drift,
# not a near-adjacent split.
_TEMPORAL_DRIFT_TRAIN_FRACTION = 0.6
_TEMPORAL_DRIFT_GAP_FRACTION = 0.15


def derive_temporal_drift_split(
    table_start: datetime, table_end: datetime,
) -> tuple[datetime, datetime]:
    """Derive `evaluate_temporal_drift`'s `train_before`/`eval_after` as
    fixed fractions of a table's own real data span -- see module docstring
    for why a fixed calendar literal (`pipeline_steps.SMOKE_TEMPORAL_DRIFT_*`)
    does not work across 3 real tables with very different real spans.

    Args:
        table_start: Real start of the table's data (see
            `_fetch_table_time_bounds`).
        table_end: Real end of the table's data.

    Returns:
        (train_before, eval_after), both strictly inside
        ``[table_start, table_end]``.
    """
    span = table_end - table_start
    train_before = table_start + _TEMPORAL_DRIFT_TRAIN_FRACTION * span
    eval_after = table_start + (_TEMPORAL_DRIFT_TRAIN_FRACTION + _TEMPORAL_DRIFT_GAP_FRACTION) * span
    return train_before, eval_after


def fetch_table_workload_bounds(
    conn: psycopg.Connection, table: str,
) -> tuple[tuple[float, float, float, float], tuple[datetime, datetime]]:
    """Fetch the ``(bbox, time_range)`` pair `generate_workload` needs for an
    arbitrary already-loaded real table, deriving both from the live data
    rather than from a synthetic-domain literal (
    `generate_workload`'s `bbox`/`time_range` are required precisely so a
    real-dataset caller cannot silently fall back to the synthetic
    `(0, 10, 0, 10)` default).

    Returns:
        ``((min_lon, max_lon, min_lat, max_lat), (table_start, table_end))``.
    """
    bbox = _fetch_table_spatial_bounds(conn, table)
    time_range = _fetch_table_time_bounds(conn, table)
    return bbox, time_range


def run_real_experiment_for_dataset(
    conn: psycopg.Connection,
    table: str,
    dataset_name: str,
    n_train: int = REAL_N_TRAIN,
    n_eval: int = REAL_N_EVAL,
    epochs: int = REAL_EPOCHS,
    seed: int = REAL_SEED,
    n_plan_queries: int = REAL_N_PLAN_QUERIES,
    n_density_bins: int = REAL_DENSITY_N_BINS,
) -> dict:
    """Run the full evaluation battery against one
    already-loaded real table.

    Mirrors `src.run_pipeline.run_smoke_pipeline`'s composition (fit every
    registered estimator -> accuracy -> systems impact -> MI
    measurement -> robustness -> synthesis), but against a
    caller-supplied real ``table`` whose bbox/time_range are derived from
    the live data (`fetch_table_workload_bounds`) instead of a synthetic
    module constant, and with NO controlled sweep or cross-city
    step (`run_cross_city_transfer_matrix` below covers transfer separately,
    since a real transfer matrix compares tables pairwise rather than one
    table against a freshly-generated second "city").

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the already-loaded real trajectory table (e.g.
            ``"tdrive_traj"``).
        dataset_name: Short label for this dataset (e.g. ``"tdrive"``),
            used as the key into `build_design_guidance`'s
            ``accuracy_by_dataset``/``robustness_by_dataset`` dicts.
        n_train: Number of training queries.
        n_eval: Number of held-out evaluation queries.
        epochs: Training epoch budget forwarded to ``coupled_learned``.
        seed: Base random seed (data/workload/estimator reproducibility).
        n_plan_queries: Eval queries fed to `evaluate_systems_impact`.
        n_density_bins: Forwarded to `evaluate_density_stratified`.

    Returns:
        A dict with ``run_config``, ``accuracy`` (per estimator),
        ``n_eval_nonzero``, ``systems_impact``, ``measured_mi``,
        ``density_stratified``, ``temporal_drift``, ``selectivity_extremes``,
        and ``design_guidance`` -- the real-data analogue of
        `run_smoke_pipeline`'s return shape, minus the synthetic-only
        ``correlation_sweep``/``cross_city_transfer`` keys.
    """
    bbox, time_range = fetch_table_workload_bounds(conn, table)
    logger.info(
        "run_real_experiment_for_dataset(%s): table=%s bbox=%s time_range=%s..%s",
        dataset_name, table, bbox, time_range[0], time_range[1],
    )

    train_queries = generate_workload(
        conn, table, n_queries=n_train, seed=seed,
        bbox=bbox, knn_fraction=0.0, time_range=time_range,
    )
    eval_queries = generate_workload(
        conn, table, n_queries=n_eval, seed=seed + 1,
        bbox=bbox, knn_fraction=0.0, time_range=time_range,
    )
    n_eval_nonzero = sum(1 for q in eval_queries if q.true_cardinality > 0)
    if n_eval_nonzero < n_eval * 0.05:
        logger.warning(
            "run_real_experiment_for_dataset(%s): only %d/%d eval queries are non-empty -- "
            "accuracy numbers below may be dominated by trivial all-zero predictions",
            dataset_name, n_eval_nonzero, n_eval,
        )

    # Fit every registered estimator (7 at present).
    fitted = fit_all_estimators(conn, table, train_queries, seed=seed, epochs=epochs)
    accuracy = {name: evaluate_accuracy(est, eval_queries) for name, est in fitted.items()}

    # Systems impact: same full estimator set, on a small shared query slice.
    systems_impact = evaluate_systems_impact(
        conn, table, estimators=fitted, queries=eval_queries[:n_plan_queries],
    )

    # Real-data MI measurement (the synthetic sweep does not apply to
    # an already-loaded real table -- see module docstring).
    measured_mi = measure_correlation_strength(conn, table)

    # Robustness: density-stratified accuracy and temporal drift.
    train_before, eval_after = derive_temporal_drift_split(time_range[0], time_range[1])
    density_stratified = evaluate_density_stratified(
        conn, table, fitted["spatial_only_ml"], eval_queries, n_density_bins=n_density_bins,
    )
    temporal_drift = evaluate_temporal_drift(
        conn, table, lambda: SpatialOnlyMLEstimator(seed=seed),
        bbox=bbox, train_before=train_before, eval_after=eval_after,
        n_train=n_train, n_eval=n_eval, seed=seed,
    )
    selectivity_extremes = evaluate_selectivity_extremes(accuracy["coupled_learned"])

    # Design guidance, fed this one dataset's real MI/accuracy/
    # robustness -- correlation_sweep_results=[] since no synthetic sweep
    # was run here (see module docstring), so `correlation_threshold_mi`
    # will read None; `per_dataset_verdict[dataset_name]` is still computed
    # from this dataset's own coupled-vs-spatial-only accuracy comparison.
    design_guidance = build_design_guidance(
        [],
        accuracy_by_dataset={dataset_name: {
            "measured_mi": measured_mi,
            "coupled_median_qerror": accuracy["coupled_learned"]["median"],
            "spatial_only_median_qerror": accuracy["spatial_only_ml"]["median"],
        }},
        robustness_by_dataset={dataset_name: {
            "density_stratified": density_stratified,
            "temporal_drift": temporal_drift,
        }},
    )

    logger.info(
        "run_real_experiment_for_dataset(%s): measured_mi=%.4f coupled_median=%.3f "
        "spatial_only_median=%.3f verdict=%s",
        dataset_name, measured_mi, accuracy["coupled_learned"]["median"],
        accuracy["spatial_only_ml"]["median"], design_guidance["per_dataset_verdict"][dataset_name],
    )

    return {
        "run_config": {
            "dataset_name": dataset_name,
            "table": table,
            "seed": seed,
            "n_train": n_train,
            "n_eval": n_eval,
            "coupled_epochs": epochs,
            "n_plan_queries": n_plan_queries,
            "density_n_bins": n_density_bins,
            "bbox": list(bbox),
            "time_range": [t.isoformat() for t in time_range],
            "knn_fraction": 0.0,
            "estimator_names": sorted(fitted),
            "temporal_drift_train_before": train_before.isoformat(),
            "temporal_drift_eval_after": eval_after.isoformat(),
        },
        "accuracy": accuracy,
        "n_eval_nonzero": n_eval_nonzero,
        "systems_impact": systems_impact,
        "measured_mi": measured_mi,
        "density_stratified": density_stratified,
        "temporal_drift": temporal_drift,
        "selectivity_extremes": selectivity_extremes,
        "design_guidance": design_guidance,
    }


def run_cross_city_transfer_matrix(
    conn: psycopg.Connection,
    tables: dict[str, str],
    n_train: int = REAL_N_TRAIN,
    n_eval: int = REAL_N_EVAL,
    epochs: int = REAL_EPOCHS,
    seed: int = REAL_SEED,
) -> dict:
    """Run cross-city transfer over every directed pair among
    ``tables`` (6 pairs for 3 real cities), using `CoupledLearnedEstimator`
    as the transferred estimator -- the same choice `pipeline_steps.
    run_cross_city_step` makes for the synthetic case, so this exercises
    `Estimator.refit_normalization` the same way.

    Each table's own train/eval workload is generated ONCE (using that
    table's own derived bbox/time_range) and reused across every pair it
    participates in: as the SOURCE table's ``source_train_queries`` for
    every pair where it is the source, and as the TARGET table's
    ``target_eval_queries`` for every pair where it is the target. This
    means a real deployment's own eval workload is what "does transfer
    into city X work" is measured against, regardless of which other city
    supplied the source model -- and avoids regenerating the same
    workload (each query costs a DB round-trip) once per pair instead of
    once per table.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        tables: ``{dataset_name: table_name}``, e.g. ``{"tdrive":
            "tdrive_traj", "porto": "porto_traj", "nyc_tlc": "nyc_tlc_traj"}``.
            Iterated in sorted name order for a deterministic pair sequence.
        n_train: Training queries per source table.
        n_eval: Evaluation queries per target table.
        epochs: Training epoch budget for the transferred estimator.
        seed: Base random seed; every table's eval workload uses ``seed +
            1``, matching this codebase's existing convention.

    Returns:
        ``{"run_config": {...}, "per_table_measured_mi": {name: float},
        "pairs": {"source->target": {"transfer": {...},
        "transfer_failure_analysis": {...}}}}``, one ``pairs`` entry per
        ordered ``(source, target)`` pair with ``source != target``.
    """
    names = sorted(tables)
    per_table: dict[str, dict] = {}
    for name in names:
        table = tables[name]
        bbox, time_range = fetch_table_workload_bounds(conn, table)
        train_queries = generate_workload(
            conn, table, n_queries=n_train, seed=seed,
            bbox=bbox, knn_fraction=0.0, time_range=time_range,
        )
        eval_queries = generate_workload(
            conn, table, n_queries=n_eval, seed=seed + 1,
            bbox=bbox, knn_fraction=0.0, time_range=time_range,
        )
        measured_mi = measure_correlation_strength(conn, table)
        per_table[name] = {
            "table": table,
            "train_queries": train_queries,
            "eval_queries": eval_queries,
            "measured_mi": measured_mi,
        }
        logger.info(
            "run_cross_city_transfer_matrix: prepared %s (table=%s) measured_mi=%.4f",
            name, table, measured_mi,
        )

    pairs: dict[str, dict] = {}
    for source_name in names:
        for target_name in names:
            if source_name == target_name:
                continue
            source = per_table[source_name]
            target = per_table[target_name]
            factory: Callable[[], CoupledLearnedEstimator] = (
                lambda: CoupledLearnedEstimator(seed=seed, epochs=epochs)
            )
            transfer = evaluate_cross_city_transfer(
                factory, conn, source["table"], source["train_queries"],
                target["table"], target["eval_queries"],
            )
            failure = analyze_transfer_failure(
                source["measured_mi"], target["measured_mi"], transfer,
            )
            key = f"{source_name}->{target_name}"
            pairs[key] = {"transfer": transfer, "transfer_failure_analysis": failure}
            logger.info(
                "run_cross_city_transfer_matrix: %s median_qerror=%.3f flagged_shift_driven=%s",
                key, transfer["median"], failure["flagged_as_likely_shift_driven"],
            )

    return {
        "run_config": {
            "datasets": tables,
            "n_train": n_train,
            "n_eval": n_eval,
            "coupled_epochs": epochs,
            "seed": seed,
            "knn_fraction": 0.0,
        },
        "per_table_measured_mi": {name: per_table[name]["measured_mi"] for name in names},
        "pairs": pairs,
    }
