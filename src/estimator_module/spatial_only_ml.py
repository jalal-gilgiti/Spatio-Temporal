"""Spatial-only learned CE baseline, grounded in the WISE-2024/LIAS-Laboratory
"Selectivity Estimation for Spatial Filters Using Optimizer Feedback"
methodology: regression over spatial predicate features only, no temporal
features — isolates the value the coupled estimator adds by
including the temporal dimension and explicit coupling."""
from typing import TYPE_CHECKING

import numpy as np
import psycopg
from sklearn.ensemble import GradientBoostingRegressor

from src.data_module.workload_generator import STPredicate
from src.estimator_module.base import Estimator, register_estimator

if TYPE_CHECKING:
    from src.data_module.workload_generator import WorkloadQuery


def _spatial_features(p: STPredicate) -> np.ndarray:
    return np.array([p.min_lon, p.max_lon, p.min_lat, p.max_lat, p.max_lon - p.min_lon, p.max_lat - p.min_lat])


@register_estimator("spatial_only_ml")
class SpatialOnlyMLEstimator(Estimator):
    def __init__(self, seed: int = 0):
        self.model = GradientBoostingRegressor(random_state=seed)

    def fit(self, conn: psycopg.Connection, table: str, train_queries: "list[WorkloadQuery] | None" = None) -> None:
        if not train_queries:
            raise ValueError("SpatialOnlyMLEstimator requires train_queries")
        train_queries = [q for q in train_queries if q.predicate.k is None]
        if not train_queries:
            raise ValueError("SpatialOnlyMLEstimator requires train_queries")
        X = np.stack([_spatial_features(q.predicate) for q in train_queries])
        y = np.log1p(np.array([q.true_cardinality for q in train_queries], dtype=float))
        self.model.fit(X, y)

    def estimate(self, predicate: STPredicate) -> float:
        x = _spatial_features(predicate).reshape(1, -1)
        return float(np.expm1(self.model.predict(x)[0]))
