"""ST range/kNN query workload generation and exact ground-truth cardinality."""
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import psycopg
from psycopg import sql


@dataclass(frozen=True)
class STPredicate:
    """Canonical spatio-temporal predicate consumed by every estimator/eval module.

    Represents either a spatio-temporal range predicate (default) or a
    spatio-temporal kNN predicate (when ``k`` and ``knn_center`` are set).
    """

    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    start_time: datetime
    end_time: datetime
    k: int | None = None
    knn_center: tuple[float, float] | None = None


@dataclass(frozen=True)
class WorkloadQuery:
    """A single query in a generated workload, paired with its ground-truth cardinality."""

    predicate: STPredicate
    true_cardinality: int


def stbox_literal(p: STPredicate) -> str:
    """Render an STPredicate's spatio-temporal range as a MobilityDB STBOX literal."""
    return (
        f"STBOX XT((({p.min_lon},{p.min_lat}),({p.max_lon},{p.max_lat})),"
        f"[{p.start_time.isoformat()},{p.end_time.isoformat()}))"
    )


def st_predicate_sql(p: STPredicate) -> str:
    """Real spatio-temporal intersection, not bbox overlap. `&&` on a
    tgeompoint tests whether the trajectory's OVERALL bounding box overlaps
    the query stbox -- true even when the trajectory's actual path never
    occupies that space during that time window (confirmed empirically). atStbox restricts the
    trajectory to the query's space-time region and returns non-null only
    if the restriction is non-empty, i.e. the path genuinely passes through.

    Returns a raw SQL fragment referencing an unqualified ``traj`` column
    (resolves fine whether or not the caller's query aliases the table,
    since no other table in these queries also has a ``traj`` column). Has
    no bound parameters: `stbox_literal` is already a safe,
    internally-controlled literal (built only from `STPredicate`'s own
    floats/datetimes, never from external input), consistent with how
    predicate_sql fragments are treated as trusted internal SQL everywhere
    else in this codebase.
    """
    return f"atStbox(traj, stbox '{stbox_literal(p)}') IS NOT NULL"


def compute_ground_truth(conn: psycopg.Connection, table: str, predicate: STPredicate) -> int:
    """Compute the exact cardinality of a predicate against a trajectory table.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.
        predicate: The range or kNN predicate to evaluate.

    Returns:
        Exact number of rows matching the predicate.
    """
    with conn.cursor() as cur:
        if predicate.k is not None:
            lon, lat = predicate.knn_center
            query = sql.SQL(
                "SELECT count(*) FROM (SELECT * FROM {table} "
                "ORDER BY traj <-> ST_Point(%s, %s) LIMIT %s) t"
            ).format(table=sql.Identifier(table))
            cur.execute(query, (lon, lat, predicate.k))
        else:
            query = sql.SQL(
                "SELECT count(*) FROM {table} WHERE " + st_predicate_sql(predicate)
            ).format(table=sql.Identifier(table))
            cur.execute(query)
        return cur.fetchone()[0]


# Real trips (after `segment_into_trips`) span roughly 0-45 minutes
# -- see `synthetic_generator.py`'s `max_trip_span`. Sampling a query
# window's start/end as two independent uniform draws across the full
# `time_range` (the original implementation) gives an expected window
# WIDTH of total_seconds / 3 (order statistics of 2 uniform draws): for a
# 24-hour time_range that is ~8 hours -- tens of times wider than any real
# trip. At that scale, whether a query "hits" a trajectory is dominated by
# where the multi-hour window happens to land relative to the data's span,
# not by genuine space-time structure. Confirmed as the reason
# `run_correlation_sweep`'s no-advantage-on-uncorrelated-data assertion
# failed (measured 1.51, threshold < 1.1).
#
# Sample the window WIDTH from a log-uniform distribution at trip scale
# (1 minute to 2 hours) instead, independent of `time_range`'s span, and
# place its START uniformly across `time_range`. Log-uniform (rather than
# plain uniform) avoids concentrating mass at the wide end of the range,
# so both a handful-of-minutes query and a ~2-hour query are plausible.
_MIN_QUERY_WINDOW_SECONDS = 60.0  # 1 minute
_MAX_QUERY_WINDOW_SECONDS = 7200.0  # 2 hours


def _sample_query_window(
    rng: np.random.Generator, t0: datetime, total_seconds: float
) -> tuple[datetime, datetime]:
    """Sample one query time window: START uniform across
    ``[t0, t0 + total_seconds)``, WIDTH log-uniform at trip scale (1 minute
    to 2 hours), clamped so END never exceeds ``t0 + total_seconds``.

    When the available span from the sampled start to the end of
    `time_range` is already at or below `_MIN_QUERY_WINDOW_SECONDS` (a
    narrow `time_range`, or a start landing near its end), the window
    simply fills the remaining span rather than sampling a width that
    could exceed it.
    """
    start_offset = float(rng.uniform(0, total_seconds))
    max_width = max(total_seconds - start_offset, 0.0)
    if max_width <= _MIN_QUERY_WINDOW_SECONDS:
        width = max_width
    else:
        hi = min(_MAX_QUERY_WINDOW_SECONDS, max_width)
        width = float(np.exp(rng.uniform(np.log(_MIN_QUERY_WINDOW_SECONDS), np.log(hi))))
    start = t0 + timedelta(seconds=start_offset)
    end = start + timedelta(seconds=width)
    return start, end


def generate_workload(
    conn: psycopg.Connection,
    table: str,
    n_queries: int,
    seed: int,
    bbox: tuple[float, float, float, float],
    knn_fraction: float = 0.0,
    time_range: tuple[datetime, datetime] = (datetime(2026, 1, 1), datetime(2026, 1, 8)),
) -> list[WorkloadQuery]:
    """Generate a random workload of ST range/kNN queries with exact ground truth.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.
        n_queries: Number of queries to generate.
        seed: Random seed for reproducibility.
        bbox: (min_lon, max_lon, min_lat, max_lat) bounds for sampling.
            Required, not defaulted. A defaulted bbox is the identical
            confound class already fixed for `time_range` below: this
            codebase's synthetic data lives on the [0, 10] x [0, 10]
            domain (see `synthetic_generator.py`), so a silent
            ``bbox=(0, 10, 0, 10)`` default matched every caller today --
            but a future real-dataset caller (T-Drive/Porto/NYC TLC
            coordinates, all far from (0, 10)) that forgot to pass this
            would silently draw an all-empty workload instead of erroring. Making it required forces
            every caller to make that choice visibly; every current call
            site in this codebase generates synthetic or synthetic-scale
            test data, so ``(0, 10, 0, 10)`` remains the correct value to
            pass explicitly at each of them.
        knn_fraction: Fraction of queries that should be kNN predicates.
            Defaults to ``0.0``: every
            current caller in this codebase (`run_correlation_sweep`,
            `evaluate_temporal_drift`, `run_smoke_pipeline`,
            `evaluate_systems_impact`'s callers) already explicitly passes
            ``knn_fraction=0.0``, since several downstream consumers
            (`SpatialOnlyMLEstimator.fit`, density/temporal-drift
            robustness analysis, systems-impact evaluation) either
            drop kNN queries silently or cannot yet handle them at all.
            Defaulting to 0.0 makes that the safe out-of-the-box choice
            instead of relying on every caller remembering to override it;
            pass a positive fraction explicitly where a kNN-mixed workload
            is actually wanted (e.g. `test_generate_workload_produces_
            requested_count_and_knn_fraction`).
        time_range: (start, end) datetime bounds for sampling. The window
            START is sampled uniformly across this range; the window WIDTH
            is sampled independently, at a scale comparable to a real trip
            (minutes to low hours -- see `_sample_query_window`), not as a
            fraction of `time_range`'s own span.

    Returns:
        List of WorkloadQuery, each paired with its exact true_cardinality.
    """
    rng = np.random.default_rng(seed)
    min_lon, max_lon, min_lat, max_lat = bbox
    t0, t1 = time_range
    total_seconds = (t1 - t0).total_seconds()

    queries: list[WorkloadQuery] = []
    for _ in range(n_queries):
        is_knn = rng.random() < knn_fraction
        lon0, lon1 = sorted(rng.uniform(min_lon, max_lon, 2))
        lat0, lat1 = sorted(rng.uniform(min_lat, max_lat, 2))
        start, end = _sample_query_window(rng, t0, total_seconds)
        if is_knn:
            center = (float(rng.uniform(min_lon, max_lon)), float(rng.uniform(min_lat, max_lat)))
            k = int(rng.choice([1, 5, 10, 50, 100]))
            p = STPredicate(lon0, lon1, lat0, lat1, start, end, k=k, knn_center=center)
        else:
            p = STPredicate(lon0, lon1, lat0, lat1, start, end)
        true_card = compute_ground_truth(conn, table, p)
        queries.append(WorkloadQuery(predicate=p, true_cardinality=true_card))
    return queries
