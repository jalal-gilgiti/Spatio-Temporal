"""Robustness and failure analysis: density-stratified accuracy,
temporal drift, selectivity extremes, and cross-city transfer failure
analysis.

Three deviations from a naive implementation are load-bearing and documented
inline at their call sites:

  - `evaluate_density_stratified` buckets queries by a RANK-based partition
    (sort by density, assign bucket by position), not a value-range mask.
    A value-range mask (`edges[i] <= x <= edges[i+1]`)
    double/triple/all-bucket-counts tied values; empirically, a workload
    with many zero-point-bbox queries (common for a selective ST workload)
    hits this hard -- a 200-query test workload inflated to 572 total
    bucketed queries under the naive mask.
  - `evaluate_temporal_drift` derives the table's real `table_start`/
    `table_end` from `startTimestamp`/`endTimestamp` aggregates over the
    data, instead of hardcoding a `[2026-01-01, 2026-12-31]` literal span.
    An assumed-vs-actual data-span mismatch manufactures
    a large fraction of trivially-empty queries; empirically, hardcoding
    the eval window's end to 2026-12-31 against a table that actually ends
    ~90 days in produced 65/100 empty eval queries, vs. 20/100 using the
    table's real end timestamp.
  - `evaluate_temporal_drift` also takes `bbox` as a required parameter (no
    default), mirroring `generate_workload`'s own required `bbox`.
    Fixing `generate_workload`'s `bbox` default alone was not sufficient: `evaluate_temporal_drift` was
    still hardcoding `bbox=(0, 10, 0, 10)` internally with no way for its
    own caller to override it -- reintroducing the identical confound one
    frame up, for the exact same reason the TIME half of this function
    already derives real table bounds instead of hardcoding them (previous
    bullet). Every current call site still passes `(0, 10, 0, 10)`
    explicitly, since every current table is synthetic data on that
    domain; this is a required-parameter fix, not a value change.
"""
import logging
from datetime import datetime
from typing import Callable

import numpy as np
import psycopg
from psycopg import sql

from src.data_module.workload_generator import WorkloadQuery, generate_workload
from src.estimator_module._point_utils import fetch_points
from src.estimator_module.base import Estimator
from src.eval_module.accuracy import evaluate_accuracy, q_error

logger = logging.getLogger(__name__)


def _fetch_table_time_bounds(conn: psycopg.Connection, table: str) -> tuple[datetime, datetime]:
    """Fetch the real ``[min(startTimestamp), max(endTimestamp)]`` span of a
    trajectory table.

    Used instead of a hardcoded calendar-year literal so that
    `evaluate_temporal_drift`'s train/eval query windows are always scoped
    to data that actually exists (see module docstring for the confound
    this avoids).

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.

    Returns:
        (table_start, table_end) naive datetimes (any session-timezone
        offset is stripped -- see note below).

    Raises:
        ValueError: If the table has no rows (bounds are undefined).
    """
    with conn.cursor() as cur:
        query = sql.SQL(
            "SELECT min(startTimestamp(traj)), max(endTimestamp(traj)) FROM {table}"
        ).format(table=sql.Identifier(table))
        cur.execute(query)
        table_start, table_end = cur.fetchone()
    if table_start is None or table_end is None:
        raise ValueError(f"table {table!r} has no trajectories; cannot determine time bounds")
    # psycopg returns tz-aware datetimes here (the deployment's session
    # TimeZone is Asia/Seoul), but every other datetime in this codebase
    # (STPredicate bounds, generate_workload's time_range, this function's
    # own train_before/eval_after parameters) is naive. Stripping tzinfo
    # keeps them comparable/subtractable in generate_workload, which does
    # `(t1 - t0).total_seconds()` and raises TypeError on a naive/aware mix.
    return table_start.replace(tzinfo=None), table_end.replace(tzinfo=None)


def _fetch_table_spatial_bounds(conn: psycopg.Connection, table: str) -> tuple[float, float, float, float]:
    """Fetch the real ``(min_lon, max_lon, min_lat, max_lat)`` bounding box of
    a trajectory table's ENTIRE path, not just each trajectory's start point.

    `generate_workload`'s `bbox` and `evaluate_temporal_drift`'s `bbox` are
    both required, non-defaulted parameters precisely because this codebase's synthetic data lives on
    a known `[0, 10] x [0, 10]` domain but a real table's coordinate extent
    (T-Drive/Porto/NYC TLC, none of them anywhere near `(0, 10)`) is not
    known in advance -- exactly the class of caller that fix was written to
    protect. This is the spatial counterpart of `_fetch_table_time_bounds`.

    Uses each trajectory's full ``stbox`` (via ``xmin``/``xmax``/``ymin``/
    ``ymax`` on ``traj::stbox``), not `startValue`, so the returned box
    covers the whole path a trajectory travels, not only where it began --
    a table whose trips end far from where they start (any real taxi trip)
    would otherwise report a bbox too small to contain most of the data.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.

    Returns:
        (min_lon, max_lon, min_lat, max_lat), as plain Python floats.

    Raises:
        ValueError: If the table has no rows (bounds are undefined).
    """
    with conn.cursor() as cur:
        query = sql.SQL(
            "SELECT min(xmin(traj::stbox)), max(xmax(traj::stbox)), "
            "min(ymin(traj::stbox)), max(ymax(traj::stbox)) FROM {table}"
        ).format(table=sql.Identifier(table))
        cur.execute(query)
        min_lon, max_lon, min_lat, max_lat = cur.fetchone()
    if min_lon is None or max_lon is None or min_lat is None or max_lat is None:
        raise ValueError(f"table {table!r} has no trajectories; cannot determine spatial bounds")
    return float(min_lon), float(max_lon), float(min_lat), float(max_lat)


def evaluate_density_stratified(
    conn: psycopg.Connection,
    table: str,
    estimator: Estimator,
    queries: list[WorkloadQuery],
    n_density_bins: int = 3,
) -> dict:
    """Bucket queries by the local point density inside their spatial bbox
    (points per unit area, computed from the fitted table) and report a
    per-bucket q-error summary.

    Bucket membership is a RANK-based partition (sort by density, assign
    bucket by position via ``np.argsort(..., kind="stable")``), not a
    value-range mask over quantile edges. A value-range mask
    (``edges[i] <= density <= edges[i+1]``) double/triple/all-bucket-counts
    queries that share the exact same density -- a common case here, since
    many range queries over a wide sampling bbox have zero points inside
    them and therefore identical (zero) density. This mirrors the
    `evaluate_accuracy` selectivity-bucket fix: every query lands
    in exactly one bucket, and ``sum(b["n"] for b in by_bin.values()) ==
    len(queries)`` always holds.

    Caller caveat -- filter kNN queries out of ``queries`` first. A kNN
    predicate's bbox is sampled independently of its ``knn_center``, and its
    true cardinality is just ``k`` regardless of what the bbox contains, so
    a kNN query's "local density in the bbox" bears no relation to its
    cardinality. Including kNN queries therefore injects pure noise into
    every density bucket. This function takes ``queries`` as a parameter
    and cannot enforce that itself; callers should generate their workload
    with ``knn_fraction=0.0`` (or drop kNN queries) before calling.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.
        estimator: Fitted estimator to evaluate.
        queries: Workload queries to bucket and score.
        n_density_bins: Number of equal-count density bins.

    Returns:
        ``{"by_density_bin": {bucket_idx: {"median_qerror": ..., "n":
        ...}}}``.
    """
    points = fetch_points(conn, table)
    query_densities = []
    for q in queries:
        p = q.predicate
        mask = (
            (points[:, 0] >= p.min_lon) & (points[:, 0] <= p.max_lon)
            & (points[:, 1] >= p.min_lat) & (points[:, 1] <= p.max_lat)
        )
        area = max((p.max_lon - p.min_lon) * (p.max_lat - p.min_lat), 1e-9)
        query_densities.append(mask.sum() / area)
    query_densities = np.array(query_densities)

    n = len(query_densities)
    sorted_indices = np.argsort(query_densities, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[sorted_indices] = np.arange(n)
    bucket_of_query = np.minimum(n_density_bins - 1, ranks * n_density_bins // n)

    by_bin: dict[int, dict] = {}
    for i in range(n_density_bins):
        mask = bucket_of_query == i
        if mask.sum() == 0:
            continue
        bucket_queries = [q for q, m in zip(queries, mask) if m]
        errors = [q_error(q.true_cardinality, estimator.estimate(q.predicate)) for q in bucket_queries]
        by_bin[i] = {"median_qerror": float(np.median(errors)), "n": len(bucket_queries)}
    return {"by_density_bin": by_bin}


def evaluate_temporal_drift(
    conn: psycopg.Connection,
    table: str,
    estimator_factory: Callable[[], Estimator],
    bbox: tuple[float, float, float, float],
    train_before: datetime,
    eval_after: datetime,
    n_train: int,
    n_eval: int,
    seed: int,
) -> dict:
    """Train on queries confined to ``[table_start, train_before)`` and
    evaluate (no retraining) on queries confined to ``[eval_after,
    table_end)``, where ``table_start``/``table_end`` are the table's real
    data bounds (see `_fetch_table_time_bounds`), not a hardcoded calendar
    literal -- avoiding manufacturing trivially-empty eval queries against
    a data-free tail (see module docstring).

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.
        estimator_factory: Zero-arg callable returning a fresh, unfitted
            ``Estimator``.
        bbox: (min_lon, max_lon, min_lat, max_lat) bounds passed straight
            through to both internal `generate_workload` calls. Required,
            not defaulted -- for the identical reason `generate_workload`'s
            own `bbox` is required: this codebase's
            synthetic/test data lives on the `[0, 10] x [0, 10]` domain, so
            a silently-hardcoded `(0, 10, 0, 10)` default matched every
            caller today, but a future real-dataset caller (T-Drive/Porto/
            NYC TLC coordinates, all far from `(0, 10)`) that forgot to
            override it would silently draw an all-empty workload instead
            of erroring. This function takes an arbitrary externally-
            supplied `table: str`, so it is exactly the kind of call site
            the required parameter protects -- hardcoding `bbox` internally
            here (as an earlier version did) reintroduced the same confound
            one frame up. Every current call site passes `(0, 10, 0, 10)`
            explicitly, since every current table is synthetic data on
            that domain.
        train_before: Exclusive upper bound of the training window.
        eval_after: Inclusive lower bound of the evaluation window.
        n_train: Number of training queries to generate.
        n_eval: Number of evaluation queries to generate.
        seed: Base random seed; the eval workload uses ``seed + 1``.

    Returns:
        The `evaluate_accuracy` summary dict for the eval split,
        plus provenance of the split actually used: ``"table_start"``,
        ``"table_end"`` (the derived real data bounds), ``"n_eval_nonzero"``
        (count of eval queries with ``true_cardinality > 0``), and
        ``"n_eval_knn"`` (count of eval queries with ``predicate.k is not
        None`` -- always 0 since this function forces
        ``knn_fraction=0.0`` internally, made assertable so a future
        regression of that internal default is directly testable rather
        than silently passing every other check). The provenance fields
        make the derived-vs-hardcoded bound distinction observable to
        callers and testable directly, rather than only inferable by
        re-deriving the workload externally.

    Raises:
        ValueError: If the table has no rows, or if either requested window
            falls outside the table's real data span (which would otherwise
            surface as an opaque ``low - high < 0`` error from inside
            ``numpy.random.Generator.uniform``).
    """
    table_start, table_end = _fetch_table_time_bounds(conn, table)

    # Validate the requested windows against the REAL data span before
    # generating anything. An inverted window (eval_after >= table_end)
    # otherwise crashes deep inside generate_workload's rng.uniform with
    # "ValueError: high - low < 0", pointing nowhere near the real problem.
    if not table_start < train_before:
        raise ValueError(
            f"train_before ({train_before}) must be after the table's actual "
            f"start ({table_start}); the training window is empty as specified"
        )
    if not eval_after < table_end:
        raise ValueError(
            f"eval_after ({eval_after}) must be before the table's actual end "
            f"({table_end}); the evaluation window is empty as specified"
        )

    # A merely-narrow (not inverted) window is legal but yields a
    # near-degenerate split with no real drift signal -- the same confound
    # class the derived bounds exist to prevent, just silent. Warn on it.
    #
    # Coverage is the MINIMUM of two ratios, each catching a blind spot the other one has on its own:
    #
    #   - overlap / requested_window_length: catches
    #     an OVER-WIDE window. Overlap with the table's real data can
    #     never exceed total_span, so requesting a window many times
    #     wider than the table (e.g. eval_after set years before
    #     table_start, or train_before set years past table_end) drives
    #     this ratio toward 0 even though the table's ENTIRE span is
    #     covered. An earlier version clamped overlap to total_span and
    #     divided by that SAME total_span, always reading exactly 100%
    #     for an over-wide window and never warning -- exactly backwards.
    #   - overlap / total_table_span: catches a NARROW window that is
    #     fully WITHIN the table's bounds. For any such window, overlap
    #     equals the window's own requested length exactly (they are the
    #     same quantity -- nothing gets clamped), so the first ratio
    #     alone is permanently 1.0 regardless of how narrow the window
    #     actually is relative to the table (e.g. `train_before =
    #     table_start + 1 minute` on a 24-hour table, 0.07% of its span).
    #     This second ratio is what the original formula measured.
    #
    # Taking the min of both catches an over-wide window OR a narrow
    # in-bounds window, without reintroducing either one's blind spot.
    # total_span_seconds > 0 is the practical case (a table spanning any
    # positive duration), but guarded rather than assumed, since a
    # single-instant table (table_start == table_end) would otherwise
    # divide by zero here; in that degenerate case only the over-wide
    # check still applies (there is no span to be "narrow relative to").
    min_span_fraction = 0.1
    requested_train_seconds = (train_before - table_start).total_seconds()
    requested_eval_seconds = (table_end - eval_after).total_seconds()
    overlap_train_seconds = (min(train_before, table_end) - table_start).total_seconds()
    overlap_eval_seconds = (table_end - max(eval_after, table_start)).total_seconds()
    total_span_seconds = (table_end - table_start).total_seconds()

    train_of_requested = overlap_train_seconds / requested_train_seconds
    eval_of_requested = overlap_eval_seconds / requested_eval_seconds
    train_of_span = overlap_train_seconds / total_span_seconds if total_span_seconds > 0 else 1.0
    eval_of_span = overlap_eval_seconds / total_span_seconds if total_span_seconds > 0 else 1.0
    train_fraction = min(train_of_requested, train_of_span)
    eval_fraction = min(eval_of_requested, eval_of_span)
    if train_fraction < min_span_fraction:
        logger.warning(
            "temporal drift train window [%s, %s) covers %.1f%% of its own "
            "requested length and %.1f%% of the table's %s..%s span; the split "
            "may be too thin to carry drift signal",
            table_start, train_before, 100 * train_of_requested, 100 * train_of_span,
            table_start, table_end,
        )
    if eval_fraction < min_span_fraction:
        logger.warning(
            "temporal drift eval window [%s, %s) covers %.1f%% of its own "
            "requested length and %.1f%% of the table's %s..%s span; the split "
            "may be too thin to carry drift signal",
            eval_after, table_end, 100 * eval_of_requested, 100 * eval_of_span,
            table_start, table_end,
        )

    # knn_fraction=0.0 (same reasoning as run_correlation_sweep):
    # kNN ground truth is `k` whenever the table holds at least k rows -- it
    # applies no temporal (or spatial) filter at all, so a kNN query is
    # invariant to whichever time window its predicate nominally claims.
    # At the 0.3 default, ~30% of both splits would carry zero temporal-drift
    # signal and simply dilute the measurement this function exists to make.
    # bbox=bbox: forwarded from this function's own required `bbox`
    # parameter (see Args and the module docstring's third bullet) instead
    # of a hardcoded literal -- a hardcoded value here would silently
    # reintroduce the same confound one frame up, since `table` (and therefore
    # the real coordinate domain) is caller-supplied.
    train_queries = generate_workload(
        conn, table, n_queries=n_train, seed=seed,
        bbox=bbox, knn_fraction=0.0, time_range=(table_start, train_before),
    )
    eval_queries = generate_workload(
        conn, table, n_queries=n_eval, seed=seed + 1,
        bbox=bbox, knn_fraction=0.0, time_range=(eval_after, table_end),
    )
    estimator = estimator_factory()
    estimator.fit(conn, table, train_queries=train_queries)
    summary = evaluate_accuracy(estimator, eval_queries)
    return {
        **summary,
        "table_start": table_start,
        "table_end": table_end,
        "n_eval_nonzero": sum(1 for q in eval_queries if q.true_cardinality > 0),
        # Provenance for the knn_fraction=0.0 choice above. Without
        # this, deleting that fix would not fail any existing test -- kNN
        # ground truth is always `k` (never zero), so a kNN-contaminated
        # eval split would silently pass n_eval_nonzero's check too.
        "n_eval_knn": sum(1 for q in eval_queries if q.predicate.k is not None),
    }


def evaluate_selectivity_extremes(accuracy_result: dict, extreme_fraction: float = 0.1) -> dict:
    """Post-process an `evaluate_accuracy` result's ``by_selectivity_bucket``
    into the smallest- and largest-true-cardinality buckets.

    Args:
        accuracy_result: An `evaluate_accuracy` return dict.
        extreme_fraction: Unused placeholder for a future finer-grained
            extreme-fraction cut; the current implementation reports the
            single smallest/largest decile bucket already produced by
            `evaluate_accuracy`.

    Returns:
        ``{"smallest_bucket": {...} | None, "largest_bucket": {...} | None}``.
    """
    buckets = accuracy_result["by_selectivity_bucket"]
    if not buckets:
        return {"smallest_bucket": None, "largest_bucket": None}
    keys = sorted(buckets.keys())
    return {"smallest_bucket": buckets[keys[0]], "largest_bucket": buckets[keys[-1]]}


def analyze_transfer_failure(source_mi: float, target_mi: float, transfer_summary: dict) -> dict:
    """Correlate the source/target mutual-information gap with cross-city
    transfer accuracy (`evaluate_cross_city_transfer` output).

    Decision rule (documented, not hidden): flag a transfer as likely
    driven by space-time-correlation shift if the target city's measured MI
    differs from the source city's by more than 20% of the source MI. This
    fixed threshold is a modeling choice to defend explicitly in the
    paper's discussion, not a tuned/validated cutoff.

    Args:
        source_mi: Measured correlation strength on the source
            table.
        target_mi: Measured correlation strength on the target
            table.
        transfer_summary: An `evaluate_cross_city_transfer` result
            dict, must contain a ``"median"`` q-error.

    Returns:
        ``{"mi_gap": ..., "transfer_median_qerror": ...,
        "flagged_as_likely_shift_driven": bool}``.
    """
    mi_gap = abs(source_mi - target_mi)
    threshold = 0.2 * max(source_mi, 1e-6)
    return {
        "mi_gap": mi_gap,
        "transfer_median_qerror": transfer_summary["median"],
        "flagged_as_likely_shift_driven": mi_gap > threshold,
    }
