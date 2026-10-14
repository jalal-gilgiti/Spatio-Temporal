"""Predicate featurization for the coupled learned estimator.

Beyond the marginal predicate bounds, `transform()` emits explicit space-time
*product* features. They exist so the model's own input carries a coupling
signal that can be switched off by name, so the coupling is concretely
ablatable.

Not every product is a coupling term, and the layout below keeps the two apart:
the space-time volume is a product that is decomposable on the log target, so
it sits in the always-on block, while only the position-by-position terms are
ablatable. `space_time_volume` carries the full argument.
"""
from datetime import datetime

import numpy as np

from src.data_module.workload_generator import STPredicate

# Layout of transform()'s output. Kept as module constants (not inferred at
# runtime) because the model's input layer must be sized before fit() runs,
# and because `JointSpaceTimeGrid` slices the box dims out of the feature
# vector by position.
BOX_SLICE = slice(0, 6)
"""Dims 0-5: [min_lon, max_lon, min_lat, max_lat, start_t, end_t]."""

VOLUME_INDEX = 7
"""Dim 7: the query's space-time volume. A product, but *not* a coupling term,
and deliberately never ablated -- see `space_time_volume` for why."""

MARGINAL_DIM = 8
"""Dims 0-7: the box dims, k, and space-time volume. None of these carries a
space-time *interaction*, so all of them survive the coupling ablation."""

INTERACTION_SLICE = slice(8, 10)
"""Dims 8-9: the genuine space-time coupling products (the ablatable ones)."""

FEATURE_DIM = 10


def space_time_volume(box: np.ndarray) -> np.ndarray:
    """The query box's space-time volume, ``Δlon * Δlat * Δt``.

    This is a product of spatial and temporal quantities, but it is *not* a
    space-time coupling term, and it is deliberately kept out of the ablatable
    `INTERACTION_SLICE`:

    * On the model's ``log1p(cardinality)`` target it is exactly decomposable:
      ``log volume = log Δlon + log Δlat + log Δt`` is a sum of three per-axis
      terms, so a model with no interaction capability whatsoever can represent
      it perfectly. A genuine interaction is precisely what *cannot* be written
      that way (`interaction_features`, and the same argument applied to the
      grid in `coupling.py`).
    * It is what an independence-assumption estimator already computes
      (``E[count] = density * area * Δt``). It is the baseline capability the
      coupling term is supposed to improve on, not part of the improvement.

    Bundling it into the ablated-away set would handicap the
    ``use_coupling=False`` arm on a capability unrelated to space-time
    correlation, biasing the same-architecture comparison toward
    "coupling helps" for a reason that has nothing to do with coupling. The
    architecture comparison measured this same point: query size is
    decomposable and explains 76% of the log-target's variance on a
    log-uniform-extent workload.

    Args:
        box: Shape ``(..., 6)``, normalized to [0, 1]:
            ``[min_lon, max_lon, min_lat, max_lat, start_t, end_t]``.

    Returns:
        Shape ``(..., 1)``, in [0, 1] (every factor is).
    """
    lon_extent = box[..., 1] - box[..., 0]
    lat_extent = box[..., 3] - box[..., 2]
    t_extent = box[..., 5] - box[..., 4]
    return np.stack([lon_extent * lat_extent * t_extent], axis=-1).astype(np.float32)


def interaction_features(box: np.ndarray) -> np.ndarray:
    """The explicit space-time *coupling* terms for one or more normalized boxes.

    These are the only features `include_interactions=False` removes, so this
    function defines what "explicit coupling" means on the input side. Both
    terms multiply a spatial *position* by a temporal *position*, which is what
    makes each an interaction rather than a restatement of the marginals:
    neither is recoverable from the marginal dims by a linear model, and each
    separates two query sets that agree on every marginal but differ in how
    space and time are paired up -- the four-way contrast that any additive
    ``f(space) + g(time)`` model necessarily scores at zero (see
    `test_interaction_dims_separate_queries_with_identical_marginals`).

    * ``lon_center * t_center`` and ``lat_center * t_center`` -- "where" times
      "when", the terms that answer whether a location's cardinality depends on
      the time it is asked about.

    The query's space-time volume is a product too, but it is decomposable on
    the log target and belongs to the always-on block; `space_time_volume`
    documents why.

    Args:
        box: Shape ``(..., 6)``, normalized to [0, 1]:
            ``[min_lon, max_lon, min_lat, max_lat, start_t, end_t]``.

    Returns:
        Shape ``(..., 2)``, also in [0, 1] (every factor is), so these dims
        need no separate scaling to sit alongside the marginal ones.
    """
    lon_center = 0.5 * (box[..., 0] + box[..., 1])
    lat_center = 0.5 * (box[..., 2] + box[..., 3])
    t_center = 0.5 * (box[..., 4] + box[..., 5])
    return np.stack(
        [
            lon_center * t_center,
            lat_center * t_center,
        ],
        axis=-1,
    ).astype(np.float32)


class Featurizer:
    """Maps an `STPredicate` to a fixed-length, min-max normalized feature vector.

    Output layout (shape ``(10,)``):

    * dims 0-5 (`BOX_SLICE`): ``[min_lon, max_lon, min_lat, max_lat,
      start_epoch, end_epoch]``, min-max normalized by fitted bounds.
    * dim 6: ``k / max_k`` for kNN predicates, else 0.
    * dim 7 (`VOLUME_INDEX`): the space-time volume from `space_time_volume`.
      Decomposable, therefore always on.
    * dims 8-9 (`INTERACTION_SLICE`): the genuine space-time coupling products
      from `interaction_features`.

    Dims 0-7 (`MARGINAL_DIM`) are the block that carries no space-time
    interaction and so survives ablation intact.

    Normalizing by fitted bounds puts range-query extents and kNN's ``k`` on
    comparable scales for the downstream model.

    Args:
        include_interactions: When False, dims 8-9 are emitted as zeros while
            the vector keeps its shape. This is the feature half of the
            same-architecture ablation: a "no explicit coupling"
            estimator must differ from the real one *only* in the coupling
            term, which means the tensor shapes -- and therefore the layer
            shapes and parameter count -- have to stay identical rather than
            shrinking to a narrower input. It must also differ *only* in
            coupling, which is why the volume dim is excluded from this switch:
            zeroing it too would take away a decomposable capability that an
            independence-assumption model has as well, and make the ablated arm
            look worse for a reason unrelated to coupling.
    """

    def __init__(
        self,
        lon_range: tuple[float, float],
        lat_range: tuple[float, float],
        time_range: tuple[datetime, datetime],
        max_k: int,
        include_interactions: bool = True,
    ):
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.time_range = (time_range[0].timestamp(), time_range[1].timestamp())
        self.max_k = max_k
        self.include_interactions = include_interactions

    def _norm(self, val: float, lo: float, hi: float) -> float:
        raw = (val - lo) / (hi - lo) if hi > lo else 0.0
        return float(np.clip(raw, 0.0, 1.0))

    def transform(self, p: STPredicate) -> np.ndarray:
        if p.k is not None:
            # kNN predicates: compute_ground_truth derives true_cardinality
            # from knn_center alone, ignoring the (independently sampled) bbox. Use
            # knn_center as a degenerate point for the spatial dims so the featurized
            # spatial signal actually correlates with the ground-truth cardinality.
            if p.knn_center is None:
                raise ValueError("STPredicate has k set but knn_center is None")
            min_lon = max_lon = p.knn_center[0]
            min_lat = max_lat = p.knn_center[1]
        else:
            min_lon, max_lon = p.min_lon, p.max_lon
            min_lat, max_lat = p.min_lat, p.max_lat

        box = np.array([
            self._norm(min_lon, *self.lon_range),
            self._norm(max_lon, *self.lon_range),
            self._norm(min_lat, *self.lat_range),
            self._norm(max_lat, *self.lat_range),
            self._norm(p.start_time.timestamp(), *self.time_range),
            self._norm(p.end_time.timestamp(), *self.time_range),
        ], dtype=np.float32)

        # Note: k=0 (an explicit 0-neighbor kNN query) and k=None (a range query)
        # both encode to 0.0 in dim 6 — an inherent collision of this scheme.
        k_feature = np.float32(
            float(np.clip(p.k / self.max_k, 0.0, 1.0)) if p.k is not None else 0.0
        )
        interactions = (
            interaction_features(box)
            if self.include_interactions
            else np.zeros(2, dtype=np.float32)
        )
        return np.concatenate(
            [box, [k_feature], space_time_volume(box), interactions]
        ).astype(np.float32)
