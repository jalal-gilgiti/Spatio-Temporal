"""MinSkew-inspired histogram over trajectory origins.

This implementation recursively splits longitude/latitude/time buckets. It is an
origin proxy, not a predictive moving-object or velocity-distribution model. The
unambiguous public and registry name is ``MinSkew-Origin``.
"""
from dataclasses import dataclass, field

import numpy as np
import psycopg

from src.data_module.workload_generator import STPredicate
from src.estimator_module._point_utils import fetch_trajectory_representatives
from src.estimator_module.base import Estimator, register_estimator


@dataclass
class _Bucket:
    """A single MinSkew bucket: an axis-aligned space-time box plus the
    points it currently owns."""

    lo: np.ndarray  # [lon, lat, time]
    hi: np.ndarray
    points: np.ndarray
    scale: np.ndarray  # root bucket's per-dimension extent, for scale-invariant split-dim choice
    count: int = field(init=False)

    def __post_init__(self) -> None:
        self.count = len(self.points)

    def skew(self) -> float:
        """Variance of point counts across the bucket's 8 space-time octants.

        Buckets with too few points (< 4) can't meaningfully be split
        further, so they report zero skew and are left alone by the
        recursive splitter.
        """
        if self.count < 4:
            return 0.0
        mid = (self.lo + self.hi) / 2
        octant_counts = []
        for signs in np.ndindex(2, 2, 2):
            mask = np.ones(self.count, dtype=bool)
            for dim, s in enumerate(signs):
                mask &= (self.points[:, dim] >= mid[dim]) if s else (self.points[:, dim] < mid[dim])
            octant_counts.append(mask.sum())
        return float(np.var(octant_counts))

    def split(self) -> list["_Bucket"]:
        """Split this bucket in half along its longest dimension.

        "Longest" is measured relative to the root bucket's own per-dimension
        extent (``self.scale``), not raw units: longitude/latitude degrees
        (~O(10)) and epoch seconds (~O(1e9)) are not comparable magnitudes,
        so comparing raw ``hi - lo`` would always pick the time dimension
        and the tree would never learn the spatial density.
        """
        dim = int(np.argmax((self.hi - self.lo) / self.scale))
        mid_val = (self.lo[dim] + self.hi[dim]) / 2
        lower_mask = self.points[:, dim] < mid_val
        children = []
        for mask, is_lower in [(lower_mask, True), (~lower_mask, False)]:
            lo, hi = self.lo.copy(), self.hi.copy()
            if is_lower:
                hi[dim] = mid_val
            else:
                lo[dim] = mid_val
            children.append(_Bucket(lo=lo, hi=hi, points=self.points[mask], scale=self.scale))
        return children


@register_estimator("minskew_origin")
class MinSkewOriginHistogramEstimator(Estimator):
    """MinSkew-inspired origin histogram.

    It recursively splits the highest-skew bucket and assumes uniform density within
    each leaf. It represents a trajectory by its first instant, so it does not share
    the full-path intersection semantics of the evaluation labels.
    """

    def __init__(self, n_buckets: int = 64):
        self.n_buckets = n_buckets
        self.buckets: list[_Bucket] = []

    def fit(self, conn: psycopg.Connection, table: str, train_queries: "list | None" = None) -> None:
        points = fetch_trajectory_representatives(conn, table)
        if points.shape[0] == 0:
            # No rows: leave buckets empty so estimate() sums to 0 instead of
            # crashing on points.min(axis=0) over a zero-size array.
            self.buckets = []
            return
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        scale = np.clip(hi - lo, 1e-9, None)
        buckets = [_Bucket(lo=lo, hi=hi, points=points, scale=scale)]
        while len(buckets) < self.n_buckets:
            skews = [b.skew() for b in buckets]
            if max(skews) <= 0:
                break
            worst = int(np.argmax(skews))
            buckets[worst:worst + 1] = buckets[worst].split()
        # The fitted histogram only needs each bucket's box (lo/hi) and
        # count to answer queries (see estimate()) — the raw point arrays
        # are dead weight once splitting is done, and keeping them would
        # make this "histogram" retain the entire dataset in memory,
        # misrepresenting its footprint against learned estimators later.
        for b in buckets:
            b.points = np.empty((0, 3), dtype=float)
        self.buckets = buckets

    def estimate(self, predicate: STPredicate) -> float:
        q_lo = np.array([predicate.min_lon, predicate.min_lat, predicate.start_time.timestamp()])
        q_hi = np.array([predicate.max_lon, predicate.max_lat, predicate.end_time.timestamp()])
        total = 0.0
        for b in self.buckets:
            frac = 1.0
            for d in range(3):
                extent = b.hi[d] - b.lo[d]
                if extent == 0:
                    # Degenerate dimension (e.g. every point shares one
                    # timestamp): there's no extent to take a ratio of, so
                    # treat it as fully inside the query iff the bucket's
                    # single value on this dimension lies within the query
                    # range, instead of dividing by a clipped-up extent
                    # (which silently zeroed out every such bucket).
                    frac *= 1.0 if (q_lo[d] <= b.lo[d] <= q_hi[d]) else 0.0
                else:
                    overlap = min(b.hi[d], q_hi[d]) - max(b.lo[d], q_lo[d])
                    frac *= max(overlap, 0.0) / extent
                if frac == 0.0:
                    break
            total += b.count * frac
        return total
