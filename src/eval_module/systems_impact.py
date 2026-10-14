"""Downstream/systems-impact evaluation: real plan/latency comparison across
estimators on the same query set. Builds on the
`capture_plan` harness so latency numbers reflect an actual PostgreSQL
EXPLAIN (ANALYZE, FORMAT JSON) run under each estimator's hinted
cardinality, not a synthetic proxy."""
import logging

import numpy as np
import psycopg

from src.data_module.workload_generator import WorkloadQuery, st_predicate_sql
from src.estimator_module.base import Estimator
from src.integration_module.plan_capture import PlanResult, capture_plan

logger = logging.getLogger(__name__)

__all__ = ["evaluate_systems_impact"]


def evaluate_systems_impact(
    conn: psycopg.Connection,
    table: str,
    estimators: dict[str, Estimator],
    queries: list[WorkloadQuery],
) -> dict:
    """Run the same query set through `capture_plan` under each estimator's
    hinted cardinality estimate, and summarize per-estimator latency so
    estimators can be compared against each other on identical queries.

    For each `(name, estimator)` pair in `estimators`, every query in
    `queries` is run through `capture_plan`, which injects that
    estimator's `estimate()` output via pg_hint_plan and captures the
    resulting EXPLAIN (ANALYZE, FORMAT JSON) plan. Returns::

        {name: {"mean_latency_ms": ..., "median_latency_ms": ...,
                "plan_results": [PlanResult, ...]}}

    where `plan_results` is positionally aligned with `queries`.

    `mean_latency_ms`/`median_latency_ms` are computed over each query's
    `PlanResult.actual_latency_ms`, which -- per `capture_plan`'s and
    `PlanResult`'s docstrings -- is EXPLAIN ANALYZE's post-planning
    execution time only. It excludes query planning time, the Python-side
    wall-clock cost of `estimator.estimate()` itself, and the
    `LOAD 'pg_hint_plan'` round-trip. A slow estimator's own overhead is
    therefore invisible to these numbers: this function reports the
    estimator's *downstream plan/execution-latency impact*, not the
    end-to-end wall-clock cost of using that estimator.

    Args:
        conn: An open psycopg connection to the target database.
        table: Target table name, forwarded to `capture_plan` (which quotes
            it safely via `build_hinted_query`).
        estimators: Mapping from a display name to a fitted `Estimator`.
        queries: The shared query set every estimator is evaluated against.
            Must be non-empty: mean/median latency over zero queries is
            undefined, and numpy would otherwise silently return NaN. Every
            query must be a range predicate (``predicate.k is None`` -- see
            Raises below); kNN predicates are not yet supported.

    Returns:
        Per-estimator summary dict, keyed by the same names as `estimators`.

    Raises:
        ValueError: If `queries` is empty, or if any query is a kNN
            predicate (``predicate.k is not None``). This function is not
            yet kNN-aware: it builds every
            served query via `st_predicate_sql`, a plain space-time
            intersection WHERE-clause fragment with no relation to a kNN
            query's actual ``ORDER BY traj <-> ST_Point(...) LIMIT k``
            shape (see `workload_generator.compute_ground_truth`'s kNN
            branch for that shape). Silently running the range-shaped
            query while hinting `estimator.estimate()`'s (possibly
            kNN-aware) cardinality would mismatch the served query against
            the hint without any error -- rejecting loudly here is
            preferred over that silent mismatch. Building a real
            kNN-appropriate hinted/served query would need
            `build_hinted_query`'s join-wrapper mechanism reworked for an
            ORDER BY/LIMIT top-N plan shape (pg_hint_plan's ``Rows()``
            hint corrects join cardinality estimates, which is not the
            mechanism a LIMIT-bounded kNN plan is shaped by) -- substantial
            new plumbing beyond this function's current scope, so this
            raises instead of building support for it. Filter kNN queries
            out of `queries` before calling (e.g. generate the workload
            with ``knn_fraction=0.0``, the default).
        Whatever `capture_plan`/`estimator.estimate` raise for a given
        estimator (e.g. a `RuntimeError` from a broken estimator, or a
        `ValueError` for a non-finite estimate) propagates uncaught. This
        aborts the whole call rather than silently omitting that
        estimator's key from the returned dict or filling it with partial
        results -- a broken estimator must fail the comparison loudly, not
        corrupt it.
    """
    if not queries:
        raise ValueError(
            "queries must be non-empty: mean/median latency over zero queries is undefined"
        )
    n_knn = sum(1 for wq in queries if wq.predicate.k is not None)
    if n_knn:
        raise ValueError(
            f"evaluate_systems_impact does not support kNN predicates yet ({n_knn} of "
            f"{len(queries)} queries have predicate.k is not None); filter kNN queries out "
            "of `queries` before calling (e.g. generate the workload with knn_fraction=0.0)"
        )

    results: dict[str, dict] = {}
    for name, estimator in estimators.items():
        plan_results: list[PlanResult] = []
        for wq in queries:
            predicate_sql = st_predicate_sql(wq.predicate)
            plan_results.append(capture_plan(conn, table, predicate_sql, estimator, wq.predicate))
        latencies = [pr.actual_latency_ms for pr in plan_results]
        results[name] = {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "plan_results": plan_results,
        }
        logger.debug(
            "evaluated systems impact for estimator=%s over %d queries", name, len(queries)
        )
    return results
