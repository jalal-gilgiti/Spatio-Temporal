"""Query-driven coupled cardinality estimator, the paper's core model.

The architecture was chosen by a second architecture comparison. The first
one compared cross-attention against a plain concatenation MLP that had no
grid and no embedding in it; the MLP won, and that model therefore had no
identifiable space-time coupling mechanism at all. The second comparison
added a joint-grid-embedding arm, plus an additive control showing that the
benchmark task really does require coupling.

The model here is `marginal MLP + explicit joint space-time grid`: the
coupling lives in `JointSpaceTimeGrid` (see coupling.py), is switchable with
one constructor flag, and is the term the paper's coupling ablation removes.
"""
import logging
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import psycopg
import torch
import torch.nn as nn

from src.data_module.workload_generator import STPredicate
from src.estimator_module.base import Estimator, register_estimator
from src.estimator_module.coupled_learned.coupling import JointSpaceTimeGrid
from src.estimator_module.coupled_learned.featurize import BOX_SLICE, FEATURE_DIM, Featurizer
from src.estimator_module.coupled_learned.separable_coupling import SeparableSpaceTimeGrid
from src.estimator_module.coupled_learned.train import train_coupled_model

if TYPE_CHECKING:
    from src.data_module.workload_generator import WorkloadQuery

logger = logging.getLogger(__name__)

# Grid geometry, from a one-factor-at-a-time sweep over
# resolution, embedding width, and init scale. 8x8x8 beat
# 4^3 and 6^3 outright and matched 12^3 at a third of the parameters;
# embed_dim=4 tied embed_dim=8 (1.040 vs 1.042 median q-error, well inside the
# 5% tie threshold) with 32% fewer parameters, so the tie broke on
# parameter count -- the same rule the first comparison used.
GRID_SHAPE = (8, 8, 8)
GRID_EMBED_DIM = 4
# Cell embeddings start at exactly zero, so an untrained model's coupling
# contribution is identically zero and training departs from the
# marginal-only solution. Unlike a hidden layer, the grid needs no random
# init to break symmetry: each cell receives a different gradient because
# each query covers a different set of cells. Zero init also means cells no
# training query ever covered contribute nothing at inference instead of
# injecting their random init into held-out predictions.
GRID_INIT_SCALE = 0.0

COUPLING_KINDS = ("joint", "separable", "separable_lean")
"""Which coupling term `CoupledCardinalityNet` installs.

``"joint"`` is the shipped `JointSpaceTimeGrid`. ``"separable"`` is the
capacity-matched control: identical parameter count, but structurally unable to
represent a space-time interaction (see `separable_coupling.py`). It exists so
the paper can separate "the coupling term buys accuracy" from "the extra
parameters buy accuracy" -- `use_coupling=False` removes both at once and
cannot distinguish them.

``"separable_lean"`` is the same control at the joint grid's *own* per-axis
resolution (8x8 spatial + 8 temporal) instead of the parameter-matched one
(22x22 + 28), so it has far fewer parameters. It is a diagnostic companion,
not a headline arm: it separates "the joint index matters" from "the matched
control's finer marginal resolution hurt it", which a single separable arm
cannot.
"""

__all__ = [
    "COUPLING_KINDS",
    "CoupledCardinalityNet",
    "CoupledLearnedEstimator",
    "FEATURE_DIM",
]


def _make_coupling(
    kind: str,
    grid_shape: tuple[int, int, int],
    embed_dim: int,
    init_scale: float,
) -> nn.Module:
    """Build the coupling term named by `kind`.

    Lives here rather than in either coupling module because it is the only
    place both are needed, and importing one from the other would be circular.
    """
    if kind == "joint":
        return JointSpaceTimeGrid(
            n_lon=grid_shape[0], n_lat=grid_shape[1], n_time=grid_shape[2],
            embed_dim=embed_dim, init_scale=init_scale,
        )
    if kind == "separable":
        return SeparableSpaceTimeGrid.matched_to(
            grid_shape, embed_dim=embed_dim, init_scale=init_scale
        )
    if kind == "separable_lean":
        return SeparableSpaceTimeGrid(
            n_lon=grid_shape[0], n_lat=grid_shape[1], n_time=grid_shape[2],
            embed_dim=embed_dim, init_scale=init_scale,
        )
    raise ValueError(f"unknown coupling kind {kind!r}, expected one of {COUPLING_KINDS}")


def _featurizer_bounds_from_queries(
    queries: "list[WorkloadQuery]",
) -> tuple[tuple[float, float], tuple[float, float], tuple[datetime, datetime], int]:
    """Derive `Featurizer`'s ``(lon_range, lat_range, time_range, max_k)``
    bounds from a set of queries' predicates.

    Shared by `CoupledLearnedEstimator.fit()` (bounds from the source
    training queries) and `refit_normalization()` (bounds from a
    different table's queries, for cross-city transfer -- see
    `refit_normalization`'s docstring for why that refit exists).

    Returns:
        ``(lon_range, lat_range, time_range, max_k)``. ``max_k`` is ``0``
        if no query in ``queries`` is a kNN predicate; callers pass
        ``max_k or 1`` to `Featurizer` (``max_k=0`` would divide by zero
        in `Featurizer.transform`'s ``k / max_k``).
    """
    lons: list[float] = []
    lats: list[float] = []
    times: list[datetime] = []
    max_k = 0
    for q in queries:
        p = q.predicate
        lons.extend((p.min_lon, p.max_lon))
        lats.extend((p.min_lat, p.max_lat))
        times.extend((p.start_time, p.end_time))
        if p.k is not None:
            max_k = max(max_k, p.k)
            # Featurizer.transform() substitutes knn_center for the spatial
            # dims of kNN predicates, so the fitted envelope must cover
            # those centers too -- otherwise a center outside the range
            # bboxes gets clipped to 0.0/1.0 and loses its spatial signal.
            if p.knn_center is not None:
                lons.append(p.knn_center[0])
                lats.append(p.knn_center[1])
    return (min(lons), max(lons)), (min(lats), max(lats)), (min(times), max(times)), max_k


class CoupledCardinalityNet(nn.Module):
    """Marginal MLP plus an explicit, ablatable joint space-time coupling term.

    Two branches consume the same feature vector:

    * `marginal`: an MLP over all `FEATURE_DIM` features — the whole of the
      previously shipped model.
    * `coupling`: a `JointSpaceTimeGrid` read out from the box dims. This is
      the space-time interaction term proper: a learned joint ST histogram
      whose cell embeddings are indexed by the joint ``(lon, lat, time)``
      triple and therefore cannot be decomposed into a spatial effect plus a
      temporal effect.

    Their outputs are concatenated and passed through a two-layer head, so
    the coupling term can modulate the marginal prediction rather than only
    offsetting it.

    Args:
        use_coupling: When False the coupling readout is replaced by zeros.
            Every layer shape, the parameter count, and the featurizer output
            shape are unchanged, so `use_coupling=False` is a true
            same-architecture ablation of the coupling term and nothing else
            -- this is what the coupling ablation compares against. Pair it with
            `Featurizer(include_interactions=False)` (which
            `CoupledLearnedEstimator` does automatically) to remove the
            explicit interaction *features* as well; leaving those on would
            let the marginal branch reconstruct part of the coupling and
            understate the ablation.
        coupling_kind: Which coupling term to install, see `COUPLING_KINDS`.
            ``"separable"`` swaps in the capacity-matched, interaction-free
            control; the parameter count is identical (4673 either way), so it
            is a same-capacity ablation of the *interaction* alone, where
            `use_coupling=False` is an ablation of capacity and interaction
            together.
    """

    def __init__(
        self,
        dim: int = 32,
        use_coupling: bool = True,
        grid_shape: tuple[int, int, int] = GRID_SHAPE,
        embed_dim: int = GRID_EMBED_DIM,
        init_scale: float = GRID_INIT_SCALE,
        coupling_kind: str = "joint",
    ):
        super().__init__()
        self.use_coupling = use_coupling
        self.coupling_kind = coupling_kind
        self.marginal = nn.Sequential(
            nn.Linear(FEATURE_DIM, dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.ReLU(),
        )
        # grid_shape/embed_dim/init_scale are constructor args rather than
        # hard-coded so the architecture sweep could vary the grid
        # geometry on the class that actually ships; the defaults above are
        # what that sweep selected.
        self.coupling = _make_coupling(coupling_kind, grid_shape, embed_dim, init_scale)
        self.head = nn.Sequential(
            nn.Linear(dim + embed_dim, dim), nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        marginal = self.marginal(x)
        if self.use_coupling:
            coupled = self.coupling(x[:, BOX_SLICE])
        else:
            # Zeros rather than a narrower head: see `use_coupling` above.
            coupled = torch.zeros(
                x.shape[0], self.coupling.embed_dim, dtype=x.dtype, device=x.device
            )
        return self.head(torch.cat([marginal, coupled], dim=-1)).squeeze(-1)


@register_estimator("coupled_learned")
class CoupledLearnedEstimator(Estimator):
    """Query-driven estimator: like `SpatialOnlyMLEstimator`,
    this requires `train_queries` (raises `ValueError` without them)
    since the model has no way to learn cardinalities from `conn`/`table`
    alone.

    Args:
        use_coupling: Propagated to both `CoupledCardinalityNet` (zeroes the
            grid readout) and `Featurizer` (zeroes the interaction dims).
            False gives the ablation arm: identical architecture,
            identical parameter count, no explicit space-time coupling.
        coupling_kind: ``"joint"`` (shipped) or ``"separable"`` (the
            capacity-matched control). See `COUPLING_KINDS`.
        include_interactions: Overrides the `Featurizer` interaction dims
            independently of `use_coupling`, which otherwise drives them.
            Needed because the separable arm has to be interaction-free on
            *both* sides -- an interaction-free grid paired with the
            lon-center-times-t-center input features would leave the marginal
            MLP a coupling pathway and stop the arm being a clean control.
            ``None`` keeps the coupled behaviour of following `use_coupling`,
            so every existing call site is unaffected.
    """

    # Defaults tuned end-to-end against the synthetic space-time-correlated
    # convergence test: full-batch Adam at the initial
    # lr=1e-3/epochs=200 was still *worse* than a
    # constant predictor on that benchmark. lr=1e-2 matches the architecture sweep;
    # epochs=1500 is the point where median q-error is comfortably (>2x)
    # below the constant-predictor baseline. Re-checked against the grid
    # architecture: the grid converges faster than the old plain MLP, so
    # these remain valid (with more margin than before), and the sweep used
    # the same budget for every arm.
    def __init__(
        self,
        seed: int = 0,
        epochs: int = 1500,
        lr: float = 1e-2,
        use_coupling: bool = True,
        coupling_kind: str = "joint",
        include_interactions: bool | None = None,
    ):
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.use_coupling = use_coupling
        if coupling_kind not in COUPLING_KINDS:
            raise ValueError(
                f"unknown coupling kind {coupling_kind!r}, expected one of {COUPLING_KINDS}"
            )
        self.coupling_kind = coupling_kind
        self.include_interactions = (
            use_coupling if include_interactions is None else include_interactions
        )
        # The net is deliberately NOT constructed here: weight init consumes
        # the global torch RNG, so building it before fit()'s
        # torch.manual_seed(self.seed) would make `seed` inert (init would
        # depend on the ambient RNG position instead). It is built inside
        # fit(), immediately after seeding -- which also makes fit()
        # idempotent, since every call starts from a fresh net.
        self.model: CoupledCardinalityNet | None = None
        self.featurizer: Featurizer | None = None

    def fit(
        self,
        conn: psycopg.Connection,
        table: str,
        train_queries: "list[WorkloadQuery] | None" = None,
    ) -> None:
        if not train_queries:
            raise ValueError("CoupledLearnedEstimator requires train_queries")
        torch.manual_seed(self.seed)
        self.model = CoupledCardinalityNet(
            use_coupling=self.use_coupling, coupling_kind=self.coupling_kind
        )

        lon_range, lat_range, time_range, max_k = _featurizer_bounds_from_queries(train_queries)
        self.featurizer = Featurizer(
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=time_range,
            max_k=max_k or 1,
            include_interactions=self.include_interactions,
        )

        X = torch.from_numpy(np.stack([self.featurizer.transform(q.predicate) for q in train_queries]))
        y = torch.from_numpy(
            np.log1p(np.array([q.true_cardinality for q in train_queries], dtype=np.float32))
        )

        losses = train_coupled_model(self.model, X, y, epochs=self.epochs, lr=self.lr)
        # Convergence signal: without this the per-epoch losses are discarded
        # and a real experiment run has no way to tell whether training
        # converged. (epochs=0 is legal but trains nothing, hence the guard.)
        if losses:
            logger.info(
                "coupled_learned fit: n=%d epochs=%d lr=%g coupling=%s "
                "first_loss=%.6f final_loss=%.6f",
                len(train_queries), self.epochs, self.lr, self.use_coupling,
                losses[0], losses[-1],
            )
            self._assert_finite(losses[-1])

    def refit_normalization(
        self,
        conn: psycopg.Connection,
        table: str,
        target_queries: "list[WorkloadQuery] | None" = None,
    ) -> None:
        """Refit ONLY `self.featurizer`'s coordinate-normalization bounds
        against a different table's query workload, leaving the trained
        model weights (`self.model`) untouched. Overrides `Estimator`'s
        default no-op -- see its docstring for the full rationale.

        `fit()` derives `self.featurizer`'s min-max bounds purely from the
        SOURCE city's `train_queries`. For cross-city transfer
        (`evaluate_cross_city_transfer`), when the TARGET
        city's coordinates fall entirely outside that fitted range, every
        target feature clips to 0.0/1.0 (`Featurizer._norm`'s `np.clip`),
        so every target query looks identical to the model regardless of
        its actual predicate -- "transfer failure" would then be a
        normalization artifact of the featurizer, not a real measurement
        of whether the LEARNED WEIGHTS generalize. Calling this after `fit()` and before `estimate()`-based
        evaluation against a target table fixes that: bounds are rebuilt
        from `target_queries` with the exact same logic `fit()` uses for
        `train_queries` (`_featurizer_bounds_from_queries`), including
        `include_interactions=self.include_interactions` so the rebuilt
        `Featurizer` stays consistent with the trained architecture's
        ablation setting (`self.include_interactions` falls back to
        `self.use_coupling` when not overridden, so this is behaviorally
        identical to following `use_coupling` directly for every current
        caller).

        `conn`/`table` are accepted for interface symmetry with `fit()`
        (see `Estimator.refit_normalization`) but unused here, the same as
        `fit()` itself never touches them -- this is a query-driven
        estimator; only the queries' own predicate ranges determine
        normalization bounds.

        Args:
            conn: Unused (see above).
            table: Unused (see above).
            target_queries: The workload whose predicates define the new
                normalization bounds (`evaluate_cross_city_transfer`
                passes `target_eval_queries`). A falsy value (``None`` or
                ``[]``) is a no-op, leaving the existing (source-fitted)
                featurizer in place, so an empty target workload still
                surfaces as an error from downstream `evaluate_accuracy`
                rather than from this hook.

        Raises:
            ValueError: If `fit()` has not been called yet.
        """
        if not target_queries:
            return
        if self.featurizer is None:
            raise ValueError("call fit() before refit_normalization()")
        lon_range, lat_range, time_range, max_k = _featurizer_bounds_from_queries(target_queries)
        self.featurizer = Featurizer(
            lon_range=lon_range,
            lat_range=lat_range,
            time_range=time_range,
            max_k=max_k or 1,
            include_interactions=self.include_interactions,
        )

    def _assert_finite(self, final_loss: float) -> None:
        """Fail loudly on a diverged run. Without this, NaN parameters would
        propagate to estimate(), where `max(0.0, nan)` returns 0.0 (NaN
        comparisons are always False) -- silently producing clean-looking
        zero estimates everywhere instead of an error."""
        assert self.model is not None
        if not np.isfinite(final_loss) or not all(
            torch.isfinite(p).all().item() for p in self.model.parameters()
        ):
            raise RuntimeError(
                "training diverged: model produced non-finite values "
                f"(final loss={final_loss}); try a smaller lr than {self.lr}"
            )

    def estimate(self, predicate: STPredicate) -> float:
        if self.featurizer is None or self.model is None:
            raise ValueError("call fit() before estimate()")
        self.model.eval()
        x = torch.from_numpy(self.featurizer.transform(predicate)).unsqueeze(0)
        with torch.no_grad():
            log_est = self.model(x).item()
        est = float(np.expm1(log_est))
        # Guard before the clip below: max(0.0, nan) is 0.0 in Python, so an
        # unchecked NaN would masquerade as a legitimate zero estimate.
        if not np.isfinite(est):
            raise RuntimeError(
                "training diverged: model produced non-finite values "
                f"(raw log-estimate={log_est})"
            )
        # Clip to 0: cardinality can never be negative, but an undertrained
        # net's raw output near 0 can expm1() to a small negative value
        # (e.g. with very few epochs) -- nothing else in the forward pass
        # enforces non-negativity.
        return max(0.0, est)
