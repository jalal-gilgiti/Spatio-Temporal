"""Independent-marginals and joint 3D histogram baselines (spec §5)."""
import numpy as np
import psycopg

from src.data_module.workload_generator import STPredicate
from src.estimator_module._point_utils import fetch_trajectory_representatives
from src.estimator_module.base import Estimator, register_estimator


@register_estimator("independent_hist")
class IndependentHistogramEstimator(Estimator):
    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def fit(self, conn: psycopg.Connection, table: str, train_queries: "list | None" = None) -> None:
        points = fetch_trajectory_representatives(conn, table)
        self.n_total = len(points)
        self.lon_hist, self.lon_edges = np.histogram(points[:, 0], bins=self.n_bins)
        self.lat_hist, self.lat_edges = np.histogram(points[:, 1], bins=self.n_bins)
        self.time_hist, self.time_edges = np.histogram(points[:, 2], bins=self.n_bins)

    def estimate(self, predicate: STPredicate) -> float:
        p_lon = self._range_fraction(self.lon_hist, self.lon_edges, predicate.min_lon, predicate.max_lon)
        p_lat = self._range_fraction(self.lat_hist, self.lat_edges, predicate.min_lat, predicate.max_lat)
        p_time = self._range_fraction(
            self.time_hist, self.time_edges,
            predicate.start_time.timestamp(), predicate.end_time.timestamp(),
        )
        return self.n_total * p_lon * p_lat * p_time  # independence assumption

    @staticmethod
    def _range_fraction(hist: np.ndarray, edges: np.ndarray, lo: float, hi: float) -> float:
        total = hist.sum()
        if total == 0:
            return 0.0
        mask = (edges[:-1] < hi) & (edges[1:] > lo)
        return hist[mask].sum() / total


@register_estimator("joint_hist")
class JointHistogramEstimator(Estimator):
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def fit(self, conn: psycopg.Connection, table: str, train_queries: "list | None" = None) -> None:
        points = fetch_trajectory_representatives(conn, table)
        self.n_total = len(points)
        self.hist, self.edges = np.histogramdd(points, bins=self.n_bins)

    def estimate(self, predicate: STPredicate) -> float:
        lon_edges, lat_edges, time_edges = self.edges
        lon_mask = (lon_edges[:-1] < predicate.max_lon) & (lon_edges[1:] > predicate.min_lon)
        lat_mask = (lat_edges[:-1] < predicate.max_lat) & (lat_edges[1:] > predicate.min_lat)
        time_mask = (
            (time_edges[:-1] < predicate.end_time.timestamp())
            & (time_edges[1:] > predicate.start_time.timestamp())
        )
        submatrix = self.hist[np.ix_(lon_mask, lat_mask, time_mask)]
        return float(submatrix.sum())
