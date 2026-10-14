"""Uniform sampling baseline with full trajectory-intersection semantics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import psycopg
from psycopg import sql

from src.data_module.workload_generator import STPredicate, st_predicate_sql
from src.estimator_module.base import Estimator, register_estimator


@dataclass(frozen=True)
class TrajectorySampleEstimate:
    """One exact-predicate evaluation over the fixed trajectory sample."""

    estimated_cardinality: float
    sample_hits: int
    latency_ms: float


@register_estimator("trajectory_sampling")
class TrajectorySamplingEstimator(Estimator):
    """Estimate full-path intersection counts from a fixed uniform row sample.

    After drawing ``n`` complete trajectory rows uniformly without replacement from a
    table of ``N`` rows, a query with ``h`` hits in the sample is estimated as
    ``N * h / n``. Hits are evaluated in MobilityDB with the same ``atStbox`` predicate
    used by ``compute_ground_truth``; no origin or representative point is used.
    ``sample_count`` supplies the scalability study's fixed-count policy; when present it takes
    precedence over ``sample_fraction`` while preserving the original fraction API.
    """

    _instance_counter = 0

    def __init__(
        self,
        sample_fraction: float = 0.01,
        seed: int = 0,
        sample_count: int | None = None,
    ):
        if not 0.0 < sample_fraction <= 1.0:
            raise ValueError("sample_fraction must be in (0, 1]")
        if sample_count is not None and sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self.sample_fraction = float(sample_fraction)
        self.sample_count = None if sample_count is None else int(sample_count)
        self.seed = int(seed)
        self.n_total = 0
        self.n_sample = 0
        self.sample_ids: list[int] = []
        self.storage_bytes = 0
        self.last_estimate: TrajectorySampleEstimate | None = None
        self._conn: psycopg.Connection | None = None
        self._sample_table: str | None = None

    @staticmethod
    def _find_id_column(conn: psycopg.Connection, table: str) -> str:
        """Return the conventional integer row identifier used by ``table``."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = ANY(current_schemas(false)) AND table_name = %s "
                "AND column_name IN ('trajectory_id', 'id') "
                "ORDER BY CASE column_name WHEN 'trajectory_id' THEN 0 ELSE 1 END LIMIT 1",
                (table,),
            )
            row = cur.fetchone()
        if row is None:
            raise ValueError(
                f"{table!r} needs a trajectory_id or id column for reproducible sampling"
            )
        return str(row[0])

    def fit(
        self,
        conn: psycopg.Connection,
        table: str,
        train_queries: "list | None" = None,
    ) -> None:
        """Draw the fixed sample and materialize complete trajectories in a temp table."""
        del train_queries
        id_column = self._find_id_column(conn, table)
        with conn.cursor() as cur:
            query = sql.SQL("SELECT {id_col} FROM {table} ORDER BY {id_col}").format(
                id_col=sql.Identifier(id_column), table=sql.Identifier(table)
            )
            cur.execute(query)
            all_ids = [int(row[0]) for row in cur.fetchall()]

        self.n_total = len(all_ids)
        requested = (
            self.sample_count
            if self.sample_count is not None
            else max(1, int(round(self.n_total * self.sample_fraction)))
        )
        self.n_sample = 0 if self.n_total == 0 else min(self.n_total, requested)
        rng = np.random.default_rng(self.seed)
        if self.n_sample:
            positions = rng.choice(self.n_total, size=self.n_sample, replace=False)
            self.sample_ids = sorted(all_ids[int(i)] for i in positions)
        else:
            self.sample_ids = []

        type(self)._instance_counter += 1
        sample_tag = (
            f"n{self.sample_count}"
            if self.sample_count is not None
            else f"f{int(round(self.sample_fraction * 1_000_000))}"
        )
        self._sample_table = (
            f"trajectory_sample_{self.seed}_{sample_tag}_{type(self)._instance_counter}"
        )
        with conn.cursor() as cur:
            create = sql.SQL(
                "CREATE TEMP TABLE {sample} ON COMMIT PRESERVE ROWS AS "
                "SELECT {id_col} AS sample_id, traj FROM {source} "
                "WHERE {id_col} = ANY(%s)"
            ).format(
                sample=sql.Identifier(self._sample_table),
                id_col=sql.Identifier(id_column),
                source=sql.Identifier(table),
            )
            cur.execute(create, (self.sample_ids,))
            cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(self._sample_table)))
            cur.execute(
                sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(self._sample_table))
            )
            materialized_count = int(cur.fetchone()[0])
            if materialized_count != self.n_sample:
                raise RuntimeError(
                    f"expected {self.n_sample} sampled rows, materialized {materialized_count}"
                )
            cur.execute("SELECT pg_total_relation_size(%s::regclass)", (self._sample_table,))
            self.storage_bytes = int(cur.fetchone()[0])
        self._conn = conn

    def estimate_with_details(self, predicate: STPredicate) -> TrajectorySampleEstimate:
        """Evaluate the exact range predicate and return estimate, hits, and latency."""
        if predicate.k is not None:
            raise ValueError("TrajectorySamplingEstimator supports range predicates only")
        if self._conn is None or self._sample_table is None:
            raise RuntimeError("fit() must be called before estimate()")
        if self.n_sample == 0:
            result = TrajectorySampleEstimate(0.0, 0, 0.0)
            self.last_estimate = result
            return result

        query = sql.SQL("SELECT count(*) FROM {sample} WHERE " + st_predicate_sql(predicate)).format(
            sample=sql.Identifier(self._sample_table)
        )
        started = perf_counter()
        with self._conn.cursor() as cur:
            cur.execute(query)
            hits = int(cur.fetchone()[0])
        latency_ms = (perf_counter() - started) * 1000.0
        estimate = self.n_total * hits / self.n_sample
        result = TrajectorySampleEstimate(float(estimate), hits, float(latency_ms))
        self.last_estimate = result
        return result

    def estimate(self, predicate: STPredicate) -> float:
        return self.estimate_with_details(predicate).estimated_cardinality

    def close(self) -> None:
        """Drop the materialized temporary sample without changing source data."""
        if self._conn is not None and self._sample_table is not None:
            with self._conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(self._sample_table)
                    )
                )
        self._sample_table = None
        self._conn = None
