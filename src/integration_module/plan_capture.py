"""Capture real query plans + latency under a given estimator's hinted
cardinality. Builds on the `build_hinted_query` join-wrapper
mechanism, so the top-level EXPLAIN node for a hinted query is a join node
(e.g. `Nested Loop`), not a bare `Seq Scan` — `PlanResult.actual_rows` still
reads the top-level node's `Actual Rows`, which is correct regardless of node
type."""
import logging
from dataclasses import dataclass

import psycopg

from src.data_module.workload_generator import STPredicate
from src.estimator_module.base import Estimator
from src.integration_module.cardinality_hint import build_hinted_query

logger = logging.getLogger(__name__)

__all__ = ["PlanResult", "capture_plan"]


@dataclass(frozen=True)
class PlanResult:
    """The captured plan + latency + row-count evidence for one hinted query run.

    `actual_latency_ms` is PostgreSQL's post-planning EXPLAIN ANALYZE
    execution time only (the JSON plan's `Execution Time`, in milliseconds).
    It excludes query *planning* time (`Planning Time` in the same JSON
    plan, which can be comparable in magnitude to execution time for cheap
    queries), the Python-side wall-clock cost of `estimator.estimate()`
    (called before the timed EXPLAIN), and the `LOAD 'pg_hint_plan'`
    round-trip. Callers comparing latency across estimators (e.g. a later
    systems-impact evaluation) must account for this scope explicitly, since
    a slow estimator's own overhead is invisible to this field.
    """

    plan_json: dict
    actual_latency_ms: float
    estimated_rows: float
    actual_rows: int


def capture_plan(
    conn: psycopg.Connection,
    table: str,
    predicate_sql: str,
    estimator: Estimator,
    predicate: STPredicate,
) -> PlanResult:
    """Run a predicate through EXPLAIN (ANALYZE, FORMAT JSON) with the
    estimator's cardinality estimate injected via pg_hint_plan, and capture
    the resulting plan, latency, and row counts.

    Args:
        conn: An open psycopg connection to the target database.
        table: Target table name (safely quoted inside `build_hinted_query`).
        predicate_sql: The raw WHERE-clause predicate fragment (see
            `build_hinted_query`).
        estimator: A fitted `Estimator` used to produce the row-count estimate.
        predicate: The `STPredicate` passed to `estimator.estimate`.

    Returns:
        A `PlanResult` with the raw plan JSON, measured execution time,
        the estimator's row estimate, and the plan's actual row count.
        Note `PlanResult.actual_latency_ms` is EXPLAIN ANALYZE's
        post-planning execution time only — it does NOT include Planning
        Time, `estimator.estimate()`'s own wall-clock cost, or the
        `LOAD 'pg_hint_plan'` round-trip. See `PlanResult`'s docstring.

    Raises:
        Whatever `estimator.estimate` or `build_hinted_query` raise (e.g. a
        `ValueError` for a non-finite estimate) propagates uncaught — a
        broken estimator or invalid estimate should fail loudly here rather
        than be swallowed or produce a misleading `PlanResult`.
    """
    estimated_rows = estimator.estimate(predicate)
    hinted_query = build_hinted_query(table, predicate_sql, estimated_rows)
    with conn.cursor() as cur:
        cur.execute("LOAD 'pg_hint_plan';")
        cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {hinted_query}")
        plan_json = cur.fetchone()[0][0]
    plan_node = plan_json["Plan"]
    logger.debug("captured plan for table=%s: %s", table, hinted_query)
    return PlanResult(
        plan_json=plan_json,
        actual_latency_ms=plan_json["Execution Time"],
        estimated_rows=estimated_rows,
        actual_rows=plan_node["Actual Rows"],
    )
