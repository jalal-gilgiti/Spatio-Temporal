"""Inject an estimator's cardinality estimate into a live query via
pg_hint_plan's Rows() hint (chosen over an in-process selectivity hook).
Rows() only corrects JOIN row counts, so single-table predicates are wrapped
in a join against a 1-row dummy subquery; the OFFSET 0 fence is load-bearing
— without it Postgres optimizes the trivial join away and the hint silently
does nothing (confirmed empirically)."""
import logging
import math

import psycopg
from psycopg import sql

from src.data_module.workload_generator import STPredicate
from src.estimator_module.base import Estimator

logger = logging.getLogger(__name__)

__all__ = ["build_hinted_query", "run_with_estimator"]


def build_hinted_query(
    table: str,
    predicate_sql: str,
    estimated_rows: float | None,
    select_clause: str = "t.*",
) -> str:
    """Build a join-wrapped query, optionally carrying a pg_hint_plan Rows() hint.

    Args:
        table: Target table name, safely quoted via ``sql.Identifier`` (never
            interpolated as a raw f-string). Quoting is resolved eagerly via
            ``sql.Identifier(table).as_string(None)`` and the result spliced
            into the query text, rather than deferring interpolation to a
            ``sql.SQL(...).format(...)`` Composed object as in other tasks —
            functionally equivalent, safe against injection, just a
            different call pattern.
        predicate_sql: The raw WHERE-clause predicate fragment to evaluate
            against the table's ``t`` alias, e.g.
            ``"atStbox(t.traj, stbox '...') IS NOT NULL"`` (see
            `workload_generator.st_predicate_sql`) -- real spatio-temporal
            intersection, not the bbox-overlap ``&&`` operator.
        estimated_rows: The estimator's row-count estimate. When ``None``,
            the SAME join-wrapped structure is returned without a Rows()
            hint, so unhinted/baseline latency comparisons
            aren't confounded by the extra join node.
        select_clause: The SELECT-list expression; defaults to ``t.*``.

    Returns:
        The full query string, ready to execute (or to EXPLAIN).
    """
    table_sql = sql.Identifier(table).as_string(None)
    base = (
        f"SELECT {select_clause} FROM {table_sql} t, (SELECT 1 AS x OFFSET 0) AS dummy "
        f"WHERE ({predicate_sql}) AND dummy.x = 1"
    )
    if estimated_rows is None:
        return base
    if math.isnan(estimated_rows) or math.isinf(estimated_rows):
        # Fail loud rather than let round() raise its own less-informative
        # ValueError (NaN) or OverflowError (+/-inf) several frames removed
        # from the actual problem. A caller with a broken estimator should
        # see a clear, attributable error, not a confusing exception deep
        # inside query-building code (mirrors the fail-loud pattern used in
        # CoupledLearnedEstimator for non-finite outputs).
        raise ValueError(f"estimated_rows must be finite, got {estimated_rows}")
    # Clamp to at least 1: pg_hint_plan's Rows() hint expects a positive row
    # count. A zero or negative estimate (possible from some estimators on
    # edge-case predicates) would otherwise be emitted verbatim into the
    # hint, which is not a meaningful cardinality and risks the hint being
    # rejected or misapplied by the planner.
    rows = max(1, round(estimated_rows))
    return f"/*+ Rows(t dummy #{rows}) */ {base}"


def run_with_estimator(
    conn: psycopg.Connection,
    table: str,
    predicate_sql: str,
    estimator: Estimator,
    predicate: STPredicate,
) -> str:
    """Estimate cardinality, build the hinted query, execute it, and return the SQL run.

    Args:
        conn: An open psycopg connection to the target database.
        table: Target table name (see ``build_hinted_query``).
        predicate_sql: The raw WHERE-clause predicate fragment (see ``build_hinted_query``).
        estimator: A fitted ``Estimator`` used to produce the row-count estimate.
        predicate: The ``STPredicate`` passed to ``estimator.estimate``.

    Returns:
        The hinted query string that was actually executed, for logging/reproducibility.
    """
    estimated_rows = estimator.estimate(predicate)
    hinted_query = build_hinted_query(table, predicate_sql, estimated_rows)
    with conn.cursor() as cur:
        cur.execute("LOAD 'pg_hint_plan';")
        cur.execute(hinted_query)
    logger.debug("executed hinted query: %s", hinted_query)
    return hinted_query
