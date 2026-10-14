"""Training loop for `CoupledCardinalityNet` (spec §4), factored out of
`CoupledLearnedEstimator.fit()` so it can be reused standalone (e.g. by a
future hyperparameter-sweep or evaluation script) without instantiating a
full `Estimator`."""
import torch
import torch.nn as nn


def train_coupled_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
) -> list[float]:
    """Fit `model` to predict log1p(cardinality) via full-batch Adam/MSE.

    Args:
        model: A `CoupledCardinalityNet` (or any `nn.Module` with the same
            ``(N, FEATURE_DIM) -> (N,)`` signature).
        X: Featurized training predicates, shape ``(N, FEATURE_DIM)``.
        y: Training targets (log1p-transformed cardinalities), shape ``(N,)``.
        epochs: Number of full-batch gradient steps.
        lr: Adam learning rate.

    Returns:
        Per-epoch training loss. `CoupledLearnedEstimator.fit()` logs the
        first/final values as a convergence signal and uses the final value
        to detect divergence; other callers can inspect the full curve.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
