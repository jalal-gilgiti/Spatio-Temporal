"""Baseline: read Postgres/MobilityDB's own planner row-count estimate via EXPLAIN."""
import re

import psycopg
from psycopg import sql

from src.data_module.workload_generator import STPredicate, st_predicate_sql
from src.estimator_module.base import Estimator, register_estimator

_ROWS_RE = re.compile(r"rows=(\d+)")


@register_estimator("native")
class NativeMobilityDBEstimator(Estimator):
    def __init__(self) -> None:
        self._table: str | None = None
        self._conn: psycopg.Connection | None = None

    def fit(self, conn: psycopg.Connection, table: str, train_queries: "list | None" = None) -> None:
        self._table = table
        self._conn = conn

    def estimate(self, predicate: STPredicate) -> float:
        with self._conn.cursor() as cur:
            query = sql.SQL(
                "EXPLAIN SELECT * FROM {table} WHERE " + st_predicate_sql(predicate)
            ).format(table=sql.Identifier(self._table))
            cur.execute(query)
            plan_line = cur.fetchone()[0]
        match = _ROWS_RE.search(plan_line)
        if not match:
            raise ValueError(f"could not parse row estimate from EXPLAIN output: {plan_line!r}")
        return float(match.group(1))
