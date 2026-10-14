"""A parameter-matched additive-readout control for `JointSpaceTimeGrid`.

This module exists because of an open finding, not a feature request. The synthetic
re-validation measured `JointSpaceTimeGrid` beating its own `use_coupling=False`
ablation by 4-14% on a fixed-extent, high-cardinality workload -- but it won by
that margin at *every* `correlation_strength`, including 0.0, where the
generator injects essentially no space-time dependence. A coupling term that
helps equally when there is no coupling to exploit is not being paid for its
coupling. The leading explanation was that the grid contributes **generic
capacity and spatial locality** -- a learned lookup table indexed by where the
query box sits, which the marginal MLP would otherwise have to reconstruct from
raw normalized coordinates.

`use_coupling=False` cannot separate those two explanations, because it removes
the active grid capacity and joint indexing at the same time.
`SeparableSpaceTimeGrid` removes joint space-time indexing from the grid readout:

* It owns exactly as many learnable parameters as the joint grid it replaces,
  and emits a vector of the same width, so every downstream layer shape and the
  total network parameter count are unchanged (`matched_separable_shape`).
* Its readout is, for **any** setting of its parameters,
  ``f(lon, lat) + g(time)``. A spatial table and a temporal table are looked up
  independently and *summed*; nothing is ever indexed by a joint (space, time)
  pair. This is the same algebraic argument the earlier
  `AdditiveNoCoupling` control used, applied to the grid architecture
  specifically instead of to the whole network. The surrounding MLP and nonlinear
  head remain capable of representing space-time interactions.

Note which interactions are and are not removed. The spatial table is indexed
by the joint ``(lon, lat)`` pair, so space-space interaction is fully retained
-- deliberately. The paper's claim is about space-*time* coupling; taking away
lon-lat interaction as well would hand the joint arm an advantage on a
capability nobody is claiming, exactly the error `space_time_volume` documents
for the volume feature.

The experiment this enables: if the joint grid beats the separable grid *and*
the margin grows with `correlation_strength`, converging near zero at 0.0, the
paper's "explicit space-time coupling improves accuracy" claim is supported in
a sharp, defensible form. If the two are indistinguishable at every strength,
the explicit grid mechanism has not earned an accuracy claim; the experiment alone
does not identify which remaining network component explains the result. The
measurement is in `run/revalidation/separable_grid_control.json`.
"""
import torch
import torch.nn as nn

from src.estimator_module.coupled_learned.coupling import axis_coverage_weights

__all__ = ["SeparableSpaceTimeGrid", "matched_separable_shape"]


def matched_separable_shape(n_lon: int, n_lat: int, n_time: int) -> tuple[int, int, int]:
    """Pick a separable geometry with *exactly* as many cells as a joint grid.

    Both modules store one free embedding vector of width ``embed_dim`` per
    cell, so equal cell counts give equal parameter counts exactly -- not
    approximately. The joint grid spends ``n_lon * n_lat * n_time`` cells on a
    3-D product grid; the separable grid spends the same budget on a 2-D
    spatial table plus a 1-D temporal table.

    The allocation rule is: **the largest square spatial grid that still leaves
    the joint grid's own temporal resolution for time**, with every remaining
    cell given to time. Concretely, for the shipped ``(8, 8, 8)`` geometry (512
    cells) this returns ``(22, 22, 28)`` -- 484 spatial cells plus 28 temporal
    cells, again 512.

    Two things about that rule are load-bearing:

    * It is deterministic and derived, so the parameter match is a property of
      the code rather than a hand-tuned constant that could drift.
    * It makes the control **conservative in the right direction**. The
      separable arm ends up with strictly *finer* marginal resolution than the
      joint arm (22 spatial cells per axis versus 8, 28 temporal versus 8),
      because it is not paying the cubic cost of a joint index. The hypothesis
      under test is that the joint grid's measured advantage comes from generic
      spatial-locality lookup; the separable arm is handed more lookup
      resolution than the joint arm has, so if the joint arm still wins, lookup
      resolution is not the explanation.

    Args:
        n_lon: Joint grid's longitude resolution.
        n_lat: Joint grid's latitude resolution.
        n_time: Joint grid's time resolution.

    Returns:
        ``(n_lon_s, n_lat_s, n_time_s)`` with
        ``n_lon_s * n_lat_s + n_time_s == n_lon * n_lat * n_time``.

    Raises:
        ValueError: If the joint grid is too small to split this way (fewer
            than ``n_time + 1`` cells, i.e. no room for even a 1x1 spatial
            table alongside the required temporal resolution).
    """
    total = n_lon * n_lat * n_time
    if total < n_time + 1:
        raise ValueError(
            f"joint grid of {total} cells is too small to split into a spatial "
            f"table plus at least {n_time} temporal cells"
        )
    # Largest s with s*s <= total - n_time. isqrt would be exact, but the loop
    # form keeps the invariant visible and the sizes here are tiny.
    side = 1
    while (side + 1) * (side + 1) <= total - n_time:
        side += 1
    return side, side, total - side * side


class SeparableSpaceTimeGrid(nn.Module):
    """A learned spatial table plus a learned temporal table, **summed**.

    Drop-in replacement for `JointSpaceTimeGrid`: same ``(N, 6)`` box input,
    same ``(N, embed_dim)`` output, same coverage-weighted readout, same
    `axis_coverage_weights` treatment of degenerate point predicates. The one
    difference is the one under test -- the embeddings are indexed per axis
    group and added, never indexed jointly across space and time.

    The decomposability property, stated precisely: for any parameters and any
    box,

        readout(space, time) = S(lon_range, lat_range) + T(time_range)

    where ``S`` depends only on the spatial bounds (dims 0-3) and ``T`` only on
    the temporal bounds (dims 4-5). It follows immediately that the four-way
    interaction contrast

        readout(s1, t1) + readout(s2, t2) - readout(s1, t2) - readout(s2, t1)

    is identically zero, for every parameter setting -- the exact converse of
    `JointSpaceTimeGrid`'s defining property, and the reason this module can
    serve as an interaction-free *grid-readout* control. The complete estimator is
    not interaction-free. `test_separable_coupling.py` asserts both the contrast and
    the stronger additive identity for this module.

    Args:
        n_lon: Spatial table's longitude resolution.
        n_lat: Spatial table's latitude resolution.
        n_time: Temporal table's resolution.
        embed_dim: Width of every cell embedding, and of the readout. Must
            match the joint grid's so the downstream head keeps its shape.
        init_scale: Standard deviation of the cell-embedding init. Defaults to
            0.0 for the same reason `GRID_INIT_SCALE` does: an untrained
            coupling contribution is then exactly zero, and cells no training
            query ever covered contribute nothing to a held-out prediction
            instead of injecting their random init into it.

    The defaults are the geometry `matched_separable_shape` returns for the
    shipped ``(8, 8, 8)`` joint grid; `matched_to` derives them rather than
    trusting the literals, and is what `model.py` calls.
    """

    def __init__(
        self,
        n_lon: int = 22,
        n_lat: int = 22,
        n_time: int = 28,
        embed_dim: int = 4,
        init_scale: float = 0.0,
    ):
        super().__init__()
        if min(n_lon, n_lat, n_time) < 1:
            raise ValueError(
                f"grid resolution must be >= 1 per axis, got ({n_lon}, {n_lat}, {n_time})"
            )
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        self.n_lon = n_lon
        self.n_lat = n_lat
        self.n_time = n_time
        self.embed_dim = embed_dim
        # Two independent tables. `spatial` is indexed by the joint (lon, lat)
        # pair -- space-space interaction is retained on purpose (see the
        # module docstring); only the space-TIME index is broken.
        self.spatial = nn.Parameter(torch.randn(n_lon, n_lat, embed_dim) * init_scale)
        self.temporal = nn.Parameter(torch.randn(n_time, embed_dim) * init_scale)

    @classmethod
    def matched_to(
        cls,
        joint_shape: tuple[int, int, int],
        embed_dim: int,
        init_scale: float = 0.0,
    ) -> "SeparableSpaceTimeGrid":
        """Build the separable grid that parameter-matches a joint geometry.

        Args:
            joint_shape: The joint grid's ``(n_lon, n_lat, n_time)``.
            embed_dim: The joint grid's embedding width, reused unchanged.
            init_scale: Passed through to ``__init__``.

        Returns:
            An instance whose `numel` over parameters equals a
            `JointSpaceTimeGrid` of ``joint_shape`` and ``embed_dim``.
        """
        n_lon, n_lat, n_time = matched_separable_shape(*joint_shape)
        return cls(
            n_lon=n_lon, n_lat=n_lat, n_time=n_time,
            embed_dim=embed_dim, init_scale=init_scale,
        )

    def forward(self, box: torch.Tensor) -> torch.Tensor:
        """Read out the (decomposable) coupling vector for a batch of boxes.

        Args:
            box: Shape ``(N, 6)``:
                ``[min_lon, max_lon, min_lat, max_lat, start_t, end_t]``,
                each normalized to [0, 1].

        Returns:
            Shape ``(N, embed_dim)``.
        """
        if box.ndim != 2 or box.shape[1] != 6:
            raise ValueError(f"expected box of shape (N, 6), got {tuple(box.shape)}")

        w_lon = axis_coverage_weights(box[:, 0], box[:, 1], self.n_lon)
        w_lat = axis_coverage_weights(box[:, 2], box[:, 3], self.n_lat)
        w_time = axis_coverage_weights(box[:, 4], box[:, 5], self.n_time)

        # Contract one axis at a time, mirroring JointSpaceTimeGrid, so the
        # (N, n_lon, n_lat) weight tensor is never materialized.
        partial = torch.einsum("ijd,nj->nid", self.spatial, w_lat)
        spatial = torch.einsum("nid,ni->nd", partial, w_lon)
        temporal = torch.einsum("kd,nk->nd", self.temporal, w_time)
        # THE line under test: a sum, not a joint index.
        return spatial + temporal

    def extra_repr(self) -> str:
        return (
            f"n_lon={self.n_lon}, n_lat={self.n_lat}, n_time={self.n_time}, "
            f"embed_dim={self.embed_dim}"
        )
