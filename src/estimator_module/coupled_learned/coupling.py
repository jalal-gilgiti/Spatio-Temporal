"""The explicit space-time coupling term for the coupled learned estimator.

This module exists because an earlier review found
that the shipped `CoupledCardinalityNet` had *no* identifiable coupling
mechanism at all -- it was a plain MLP over concatenated features whose own
docstring conceded "the coupling is implicit in the joint MLP's weights."
An implicit term cannot be ablated, so the paper's central claim had no
ablation to point at.

`JointSpaceTimeGrid` is the explicit replacement: a learned joint space-time
histogram. Space-time is discretized into a joint ``(n_lon, n_lat, n_time)``
grid, each cell owns a free embedding vector, and a query's coupling feature
is the *coverage-weighted average* of the embeddings of the cells its
space-time box overlaps.

Why this is a genuine coupling term, not a relabelled MLP:

* The cell embedding is indexed by the *joint* triple ``(i, j, k)``. It is a
  free rank-3 tensor, so the readout ``sum_ijk w_i w_j w_k E[i,j,k]`` is a
  general trilinear form in the marginal coverage weights. Nothing about it
  factors into a spatial effect plus a temporal effect -- unlike an
  independence-assumption histogram, whose joint density is by construction
  the product of its marginals.
* It is removable in one line: zero the readout and the surrounding network
  is exactly a marginal-feature MLP, with identical layer shapes and
  parameter count. That is the same-architecture ablation
  (`CoupledCardinalityNet(use_coupling=False)`).
* It is inspectable: `grid` is a real ST histogram, so a trained model's
  coupling term can be visualized per cell rather than argued for.

An earlier architecture comparison selected this design over the shipped concat-MLP and over
cross-attention, with numbers.
"""
import torch
import torch.nn as nn

# Guards the per-axis weight normalization. The minimum-width rule below
# makes the un-normalized weights strictly positive, so this only ever
# defends against float underflow, never against a real zero.
_WEIGHT_EPS = 1e-12


def axis_coverage_weights(lo: torch.Tensor, hi: torch.Tensor, n_cells: int) -> torch.Tensor:
    """Distribute a 1-D query interval over `n_cells` equal cells of [0, 1].

    Returns a ``(N, n_cells)`` tensor whose rows sum to 1: the fraction of the
    query interval's (widened) extent that falls in each cell.

    Two behaviours fall out of one rule -- widen any interval narrower than
    one cell, symmetrically about its center, to exactly one cell width:

    * A wide box gets true coverage weights, so the readout is an average over
      the cells the query actually touches.
    * A degenerate point (``lo == hi``, which is how `Featurizer` encodes a kNN
      predicate's center) gets the triangular kernel ``max(0, 1 - n|x - c_i|)``
      -- i.e. linear interpolation between the two nearest cell centers. This
      is exactly the box-coverage formula for a one-cell-wide interval, so
      points and boxes need no separate code path, and a point query's
      gradient w.r.t. its position stays non-zero (a hard one-hot lookup would
      be piecewise constant and give the grid no positional gradient at all).

    Args:
        lo: Lower bound of each query interval, shape ``(N,)``, nominally in
            [0, 1] (clamped defensively -- `Featurizer` already clips, but this
            module is also used directly by the separable-control script).
        hi: Upper bound, shape ``(N,)``. Inverted inputs are tolerated.
        n_cells: Number of cells along this axis.

    Returns:
        Coverage weights, shape ``(N, n_cells)``, each row summing to 1.
    """
    cell_width = 1.0 / n_cells
    lo = lo.clamp(0.0, 1.0)
    hi = hi.clamp(0.0, 1.0)
    low = torch.minimum(lo, hi)
    high = torch.maximum(lo, hi)

    center = 0.5 * (low + high)
    half_width = torch.clamp(0.5 * (high - low), min=0.5 * cell_width)
    span_lo = (center - half_width).unsqueeze(-1)
    span_hi = (center + half_width).unsqueeze(-1)

    edges = torch.arange(n_cells, dtype=lo.dtype, device=lo.device) * cell_width
    cell_lo = edges
    cell_hi = edges + cell_width

    overlap = torch.clamp(
        torch.minimum(span_hi, cell_hi) - torch.maximum(span_lo, cell_lo), min=0.0
    )
    # The widened span always has length >= cell_width and its center lies in
    # [0, 1], so at least half a cell is always covered: the sum is positive.
    return overlap / overlap.sum(dim=-1, keepdim=True).clamp(min=_WEIGHT_EPS)


class JointSpaceTimeGrid(nn.Module):
    """A learned joint space-time histogram, read out by box coverage.

    Forward maps a batch of normalized space-time boxes
    ``(min_lon, max_lon, min_lat, max_lat, start_t, end_t)`` to an
    ``embed_dim`` coupling vector per query.

    Design notes:

    * Weights are computed per axis and combined by outer product, so the
      *weights* are separable but the *embeddings they index* are not. That is
      the whole point: separable weights keep the cost at
      ``O(n_lon + n_lat + n_time)`` per query to build, while the free joint
      embedding tensor supplies the non-decomposable interaction.
    * The per-axis weights are normalized, so the readout is a location
      signal ("which region of space-time does this query sit in") rather than
      a volume signal. Volume is already available to the network as explicit
      extent and space-time-volume features from `Featurizer`, and leaving it
      out here keeps the readout's scale independent of query size, which
      trains far more stably than an unnormalized integral.
    * The defaults here are deliberately generic; the geometry the estimator
      actually ships with lives in `model.py` (`GRID_SHAPE`, `GRID_EMBED_DIM`,
      `GRID_INIT_SCALE`) next to the sweep that chose it.
    """

    def __init__(
        self,
        n_lon: int = 8,
        n_lat: int = 8,
        n_time: int = 8,
        embed_dim: int = 8,
        init_scale: float = 0.1,
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
        # One free embedding per JOINT (lon, lat, time) cell -- this tensor is
        # the coupling term. Small init keeps the coupling contribution near
        # zero at step 0, so training starts from the marginal-only solution
        # and has to earn the coupling.
        self.grid = nn.Parameter(
            torch.randn(n_lon, n_lat, n_time, embed_dim) * init_scale
        )

    def forward(self, box: torch.Tensor) -> torch.Tensor:
        """Read out the coupling vector for a batch of normalized boxes.

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

        # Contract one axis at a time rather than in a single 4-operand einsum:
        # the fused form can materialize the full (N, n_lon, n_lat, n_time)
        # weight tensor, which is 512 floats per query at the default
        # resolution and grows cubically with it.
        partial = torch.einsum("ijkd,nk->nijd", self.grid, w_time)
        partial = torch.einsum("nijd,nj->nid", partial, w_lat)
        return torch.einsum("nid,ni->nd", partial, w_lon)

    def extra_repr(self) -> str:
        return (
            f"n_lon={self.n_lon}, n_lat={self.n_lat}, n_time={self.n_time}, "
            f"embed_dim={self.embed_dim}"
        )
