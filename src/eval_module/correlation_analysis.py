"""Correlation-strength analysis: quantify real space-time
correlation and correlate it against learned-estimator gain; sweep the
tunable synthetic generator to isolate the effect.

This is the paper's venue-fit addition: it answers "when is the coupled
learned estimator's explicit space-time coupling actually worth it?" via
(a) real-data mutual-information measurement (`measure_correlation_strength`)
and (b) a synthetic controlled sweep (`run_correlation_sweep`) that varies
`SyntheticConfig.correlation_strength` and compares the coupled estimator
against the spatial-only baseline at each level.
"""
import logging
from datetime import datetime
from typing import Callable

import numpy as np
import psycopg
from psycopg import sql
from sklearn.metrics import adjusted_mutual_info_score

from src.data_module.mobilitydb_loader import load_trajectories
from src.data_module.synthetic_generator import SyntheticConfig, generate_synthetic_trajectories
from src.data_module.workload_generator import generate_workload
from src.estimator_module.base import Estimator
from src.eval_module.accuracy import evaluate_accuracy

logger = logging.getLogger(__name__)

# `generate_synthetic_trajectories` stamps every instant inside a single
# 24-hour day (2026-01-01 00:00 -> 23:59; see `SyntheticConfig` usage in
# `run_correlation_sweep`). `generate_workload`'s own default time_range is
# a 7-day window, which would place ~6/7 of every sampled query window
# outside the data entirely and make most queries trivially empty, so the
# sweep must pass this span explicitly instead of relying on that default.
SYNTHETIC_TIME_RANGE: tuple[datetime, datetime] = (datetime(2026, 1, 1), datetime(2026, 1, 2))

# Size of the dataset `run_correlation_sweep` builds at each strength.
# Raised from 500 x 10 when `generate_synthetic_trajectories` was
# rewritten so that a trip stays in its own district instead of scattering its
# points across the whole domain. A local trip occupies far less of the
# space-time cube, so at 500 trips the median eval query matched nothing at
# all (measured: p50 true cardinality 0, over half the queries empty). A
# workload that empty pins every arm's median q-error near 1.0 and the sweep
# can no longer distinguish any of them. Measured at 6000 x 10: p50 true
# cardinality 10 at correlation_strength=0 and 4 at 1.0, with 83%/75% of eval
# queries non-empty.
SWEEP_N_TRAJECTORIES = 6000
SWEEP_N_POINTS_PER_TRAJ = 10


def sweep_table_name(strength: float) -> str:
    """Return the table name `run_correlation_sweep` uses for one nominal
    correlation strength.

    Public so that callers and tests can drop the sweep's tables afterwards
    without re-deriving (and drifting from) its private naming scheme. The
    float is sanitized into a bare identifier: ``.`` becomes ``_`` and a
    leading ``-`` becomes ``neg`` (e.g. ``0.0 -> "synth_corr_0_0"``).
    """
    return f"synth_corr_{str(strength).replace('.', '_').replace('-', 'neg')}"


def _fetch_lon_lat_hour(conn: psycopg.Connection, table: str) -> np.ndarray:
    """Fetch ONE representative instant per trajectory of ``table`` -- the
    trip's origin -- as (lon, lat, hour_of_day) rows.

    One row per *trip*, not per instant. Both forms
    measure "is a trip's place related to its time of day", but the
    per-instant form answers it on a sample whose rows are not independent:
    a trip contributes `n_points_per_traj` near-identical rows, while
    `adjusted_mutual_info_score`'s chance correction assumes independent
    rows. That mismatch leaves a large positive AMI on genuinely
    independent data, and its size depends on the sampling rate -- measured
    on `SyntheticConfig(correlation_strength=0.0)` data, per-instant AMI is
    0.170 at 500 trips x 10 points, 0.032 at 6000 x 10, and 0.007 at
    5000 x 1. Per-trip AMI on the same datasets is 0.007, 0.007 and 0.006:
    flat, and ~0 as an independence measure must be.

    That flatness is the whole point. The study compares measured
    correlation strength *across* datasets, and the real ones are sampled
    at wildly different rates (T-Drive ~177 s between fixes, Porto 15 s).
    Under the per-instant form the denser dataset would score higher purely
    from having more rows per trip, which is the exact failure mode the
    choice of AMI over raw mutual information was made to avoid -- see
    `measure_correlation_strength`. Reducing to one row per trip removes
    the sampling rate from the measurement entirely.

    The origin (`startValue` / `startTimestamp`) is the representative
    rather than a random instant so the value is deterministic for a given
    table, and because "which district a trip starts in, at what hour" is
    the quantity origin-destination analysis of real taxi data works in.

    Hour-of-day is extracted server-side with
    ``extract(hour from startTimestamp(traj))`` so it respects the
    connection's session ``TimeZone`` (the deployment's session
    timezone is ``Asia/Seoul``). Deriving it in Python from epoch seconds
    would instead yield the UTC hour, which is a 9-hour cyclic relabeling
    here -- harmless for mutual information on synthetic data, but wrong
    semantics for real datasets where "morning rush hour" coupling is
    defined in local time.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.

    Returns:
        An (n, 3) float array of (lon, lat, hour_of_day) rows, one per
        non-NULL trajectory, where ``hour_of_day`` is an integer-valued
        0..23 local hour. Always 2-D, even when the table is empty
        (``n == 0``).
    """
    with conn.cursor() as cur:
        query = sql.SQL(
            "SELECT ST_X(startValue({table}.traj)), ST_Y(startValue({table}.traj)), "
            "extract(hour from startTimestamp({table}.traj)) "
            "FROM {table} WHERE {table}.traj IS NOT NULL"
        ).format(table=sql.Identifier(table))
        cur.execute(query)
        rows = cur.fetchall()
    return np.array(rows, dtype=float).reshape(-1, 3)


def measure_correlation_strength(
    conn: psycopg.Connection,
    table: str,
    n_spatial_bins: int = 10,
    n_time_bins: int = 24,
) -> float:
    """Measure the adjusted mutual information between spatial location and
    local time of day across the trajectories in ``table``.

    The unit of observation is one *trip*, represented by its origin
    instant, not one trajectory instant -- see `_fetch_lon_lat_hour` for
    why (per-instant rows are not independent, which leaves AMI a large
    positive floor on independent data that scales with the sampling rate
    and so is not comparable across datasets).

    Location is discretized into an ``n_spatial_bins x n_spatial_bins``
    grid cell id (via independent per-axis `numpy.digitize` on lon/lat),
    and time is discretized into ``n_time_bins`` hour-of-day buckets.
    `sklearn.metrics.adjusted_mutual_info_score` is computed on the two
    resulting label arrays.

    AMI rather than raw `mutual_info_score` is deliberate. Raw MI carries a
    finite-sample bias that grows with both row count and bin count, and at
    the scales used here that bias dominates the signal: raw MI is ~0.20 for
    genuinely independent data at n=5000, and a 2000-row independent table
    can outscore a 5000-row maximally-coupled one. Since the study compares
    measured correlation strength *across* datasets of very different sizes
    (T-Drive, Porto, NYC TLC), that bias would invert the comparison. AMI is
    ~0 for independent data regardless of n, so it is comparable across
    datasets.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.
        n_spatial_bins: Number of bins per spatial axis (lon and lat).
        n_time_bins: Number of hour-of-day bins.

    Returns:
        Adjusted mutual information, ~0.0 for independent location/time and
        rising toward 1.0 as they become more dependent. Measured on the
        synthetic generator, ~0.007 at ``correlation_strength=0.0`` and
        ~0.25 at 1.0, at every dataset size tried. Unlike raw MI, AMI can
        be slightly negative (worse-than-chance agreement). An empty table
        (no trajectories) returns 0.0 -- there is no correlation signal to
        measure, and computing `.min()` / `adjusted_mutual_info_score` on
        empty arrays would otherwise raise.
    """
    points = _fetch_lon_lat_hour(conn, table)  # columns: lon, lat, hour_of_day
    if points.shape[0] == 0:
        logger.warning("measure_correlation_strength: table %r has no trajectories; returning 0.0", table)
        return 0.0

    # `np.linspace(a, b, k)` yields k *edges*, i.e. k-1 intervals, so ask for
    # n_bins + 1 edges to actually get n_bins bins as documented.
    lon_bins = np.digitize(points[:, 0], np.linspace(points[:, 0].min(), points[:, 0].max(), n_spatial_bins + 1))
    lat_bins = np.digitize(points[:, 1], np.linspace(points[:, 1].min(), points[:, 1].max(), n_spatial_bins + 1))
    # `np.digitize` returns 1..n_spatial_bins+1 (the max value lands in its own
    # top bin), so the row stride must be n_spatial_bins + 1 for the (lon, lat)
    # -> cell id map to be injective; a stride of n_spatial_bins collides,
    # e.g. (1, n+1) and (2, 1) both map to 2n+1.
    spatial_cell = lon_bins * (n_spatial_bins + 1) + lat_bins

    time_bin = np.digitize(points[:, 2], np.linspace(0, 24, n_time_bins + 1))

    return float(adjusted_mutual_info_score(spatial_cell, time_bin))


def run_correlation_sweep(
    coupled_factory: Callable[[], Estimator],
    spatial_only_factory: Callable[[], Estimator],
    correlation_strengths: list[float],
    n_train: int,
    n_eval: int,
    seed: int,
    conn: psycopg.Connection | None,
    coupled_no_coupling_factory: Callable[[], Estimator],
) -> list[dict]:
    """Sweep `SyntheticConfig.correlation_strength`, training a fresh
    coupled, spatial-only, and no-coupling-ablation estimator at each
    level, to isolate how much the coupled estimator's accuracy gain
    depends on real space-time correlation strength.

    For each strength, this generates a fresh synthetic dataset, loads it
    into a dedicated table, measures its actual (binned) adjusted mutual
    information (which need not equal the nominal ``strength`` exactly,
    since the generator's coupling is stochastic), trains all three
    estimators on the same range-only query workload, and evaluates all
    three on a held-out workload drawn from the same distribution.

    Both workloads are generated over the synthetic data's real 24-hour
    span (`SYNTHETIC_TIME_RANGE`) and with ``knn_fraction=0.0``; see the
    inline comment at the `generate_workload` calls for why both matter for
    the validity of the coupled-vs-spatial-only comparison.

    ``coupled_factory`` vs. ``spatial_only_factory`` compares two different
    model families (a torch MLP with an explicit `JointSpaceTimeGrid`
    coupling term vs. a `GradientBoostingRegressor`) -- a real comparison,
    but one that confounds "does explicit coupling help" with "which model
    family is better." ``coupled_no_coupling_factory`` is
    the real same-architecture ablation: a `CoupledLearnedEstimator`
    (or compatible `Estimator`) built with its explicit coupling term
    zeroed (`use_coupling=False`), so its architecture and
    parameter count are IDENTICAL to ``coupled_factory``'s. Comparing
    ``coupled_median_qerror`` against
    ``coupled_no_coupling_median_qerror`` isolates the coupling term's
    combined contribution -- both its lookup capacity (extra parameters) and
    any space-time interaction it captures -- holding model family fixed;
    per the capacity-matched `separable` control (see
    `coupled_learned/model.py`'s `COUPLING_KINDS` docstring), it does NOT by
    itself isolate interaction specifically, since `use_coupling=False`
    removes capacity and interaction together and cannot distinguish them.
    This is nonetheless what the re-validation check looks for:
    coupling-on beating coupling-off at high correlation_strength, and the
    two converging (no reliable coupling advantage) at
    correlation_strength=0.0.

    Args:
        coupled_factory: Zero-arg factory returning a fresh, untrained
            `CoupledLearnedEstimator` (or compatible `Estimator`) with its
            explicit coupling term enabled.
        spatial_only_factory: Zero-arg factory returning a fresh,
            untrained `SpatialOnlyMLEstimator` (or compatible `Estimator`).
        correlation_strengths: Nominal correlation strengths to sweep,
            each forwarded to `SyntheticConfig.correlation_strength`.
        n_train: Number of training queries to generate per strength.
        n_eval: Number of held-out evaluation queries to generate per
            strength.
        seed: Base random seed; the eval workload uses ``seed + 1`` so it
            does not overlap the train workload's query sample.
        conn: Open psycopg connection to the MobilityDB instance. May be
            ``None`` when ``correlation_strengths`` is empty, since it is
            never dereferenced in that case.
        coupled_no_coupling_factory: Zero-arg factory returning a fresh,
            untrained estimator with the SAME architecture as
            ``coupled_factory`` produces but its explicit coupling term
            disabled (e.g. ``lambda: CoupledLearnedEstimator(seed=seed,
            epochs=epochs, use_coupling=False)``). A required parameter,
            not defaulted with e.g. ``coupled_factory`` re-called with a
            flag flipped -- this function only knows ``coupled_factory``
            as an opaque zero-arg `Callable[[], Estimator]` and must not
            assume its exact constructor signature (`Estimator` makes no
            promise every implementation exposes a `use_coupling` kwarg at
            all).

    Returns:
        One dict per strength, in input order, where ``measured_mi`` is the
        adjusted mutual information reported by
        `measure_correlation_strength`:
        ``{"correlation_strength", "measured_mi", "coupled_median_qerror",
        "spatial_only_median_qerror", "coupled_no_coupling_median_qerror"}``.
        Returns ``[]`` cleanly if ``correlation_strengths`` is empty (in
        which case ``coupled_no_coupling_factory`` is never called either).

    Note:
        The per-strength tables this creates are left in place on return
        (they are useful to inspect after a sweep). Callers responsible for
        a shared database should drop them afterwards, deriving each name
        from `sweep_table_name` rather than re-implementing it.
    """
    results: list[dict] = []
    for strength in correlation_strengths:
        table = sweep_table_name(strength)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {table};").format(table=sql.Identifier(table)))
            cur.execute(
                sql.SQL("CREATE TABLE {table} (trajectory_id int, traj tgeompoint);").format(
                    table=sql.Identifier(table)
                )
            )
        cfg = SyntheticConfig(
            n_trajectories=SWEEP_N_TRAJECTORIES,
            n_points_per_traj=SWEEP_N_POINTS_PER_TRAJ,
            correlation_strength=strength,
            seed=seed,
        )
        df = generate_synthetic_trajectories(cfg)
        load_trajectories(conn, table, df)

        measured_mi = measure_correlation_strength(conn, table)
        # time_range: pin to the generator's actual 24-hour span (see
        # SYNTHETIC_TIME_RANGE) rather than generate_workload's 7-day default.
        # knn_fraction=0.0: SpatialOnlyMLEstimator drops kNN queries inside its
        # own fit(), so any kNN share would silently train the two estimators on
        # different-sized sets (at the 0.3 default: ~200 vs ~139 queries). A
        # range-only workload keeps the comparison apples-to-apples and matches
        # how the spatial-only ablation baseline is scoped.
        # bbox=(0, 10, 0, 10): matches `SyntheticConfig`'s actual [0, 10] x
        # [0, 10] sampling domain -- `generate_workload`'s `bbox` is now a
        # required parameter, so this choice is
        # explicit here instead of relying on an implicit default.
        train_queries = generate_workload(
            conn, table, n_queries=n_train, seed=seed,
            bbox=(0, 10, 0, 10), knn_fraction=0.0, time_range=SYNTHETIC_TIME_RANGE,
        )
        eval_queries = generate_workload(
            conn, table, n_queries=n_eval, seed=seed + 1,
            bbox=(0, 10, 0, 10), knn_fraction=0.0, time_range=SYNTHETIC_TIME_RANGE,
        )

        coupled = coupled_factory()
        coupled.fit(conn, table, train_queries=train_queries)
        coupled_summary = evaluate_accuracy(coupled, eval_queries)

        spatial_only = spatial_only_factory()
        spatial_only.fit(conn, table, train_queries=train_queries)
        spatial_only_summary = evaluate_accuracy(spatial_only, eval_queries)

        # The real same-architecture ablation arm -- see this
        # function's docstring. Trained/evaluated identically to the two
        # arms above, on the exact same train/eval workload, so the only
        # thing that differs across coupled_summary vs.
        # coupled_no_coupling_summary is the coupling term itself.
        coupled_no_coupling = coupled_no_coupling_factory()
        coupled_no_coupling.fit(conn, table, train_queries=train_queries)
        coupled_no_coupling_summary = evaluate_accuracy(coupled_no_coupling, eval_queries)

        logger.info(
            "run_correlation_sweep: strength=%s measured_mi=%.4f coupled_median=%.3f "
            "spatial_only_median=%.3f coupled_no_coupling_median=%.3f",
            strength, measured_mi, coupled_summary["median"], spatial_only_summary["median"],
            coupled_no_coupling_summary["median"],
        )
        results.append({
            "correlation_strength": strength,
            "measured_mi": measured_mi,
            "coupled_median_qerror": coupled_summary["median"],
            "spatial_only_median_qerror": spatial_only_summary["median"],
            "coupled_no_coupling_median_qerror": coupled_no_coupling_summary["median"],
        })
    return results
