"""Sampling-based selectivity estimator baseline (spec §5)."""
import numpy as np
import psycopg

from src.data_module.workload_generator import STPredicate
from src.estimator_module._point_utils import fetch_trajectory_representatives
from src.estimator_module.base import Estimator, register_estimator


@register_estimator("sampling")
class SamplingEstimator(Estimator):
    """Estimates selectivity by extrapolating from a uniform random sample of points."""

    def __init__(self, sample_rate: float = 0.1, seed: int = 0):
        self.sample_rate = sample_rate
        self.seed = seed

    def fit(self, conn: psycopg.Connection, table: str, train_queries: "list | None" = None) -> None:
        points = fetch_trajectory_representatives(conn, table)
        self.n_total = len(points)
        rng = np.random.default_rng(self.seed)
        # Guard against the empty-table edge case: `max(1, ...)` would ask
        # rng.choice for 1 sample from a population of 0, which raises
        # ValueError ("Cannot take a larger sample than population when
        # 'replace=False'"). Cap n_sample at n_total so an empty table
        # yields an empty (but valid) sample instead of crashing.
        n_sample = min(self.n_total, max(1, int(self.n_total * self.sample_rate)))
        idx = rng.choice(self.n_total, size=n_sample, replace=False)
        self.sample = points[idx]

    def estimate(self, predicate: STPredicate) -> float:
        if len(self.sample) == 0:
            return 0.0
        mask = (
            (self.sample[:, 0] >= predicate.min_lon) & (self.sample[:, 0] <= predicate.max_lon)
            & (self.sample[:, 1] >= predicate.min_lat) & (self.sample[:, 1] <= predicate.max_lat)
            & (self.sample[:, 2] >= predicate.start_time.timestamp())
            & (self.sample[:, 2] <= predicate.end_time.timestamp())
        )
        return (mask.sum() / len(self.sample)) * self.n_total
