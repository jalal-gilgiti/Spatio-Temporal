"""q-error accuracy evaluation, broken out by selectivity bucket."""
import numpy as np

from src.data_module.workload_generator import WorkloadQuery
from src.estimator_module.base import Estimator


def q_error(true_card: int, est_card: float) -> float:
    """Compute the (symmetric) q-error between a true and estimated cardinality.

    ``q_error = max(true_c / est_c, est_c / true_c)`` where both operands use
    additive ``+1`` smoothing (``true_c = max(true_card, 0) + 1``, likewise
    for ``est_c``), the standard convention in the learned cardinality
    estimation literature. The result is always >= 1.0, with 1.0 meaning a
    perfect estimate.

    ``+1`` smoothing (rather than an epsilon floor like ``max(x, 1e-3)``)
    keeps the penalty for near-zero true cardinalities proportionate: a
    true=0, est=1 off-by-one miss scores q_error=2 (a small, sensible
    penalty), instead of the epsilon-floor scheme's q_error=1000, which put
    a single off-by-one miss on par with a million-fold miss at scale.

    A defensive ``max(x, 0)`` floor precedes the ``+1`` in case a caller
    passes a negative estimate; the estimators already clip
    their outputs to non-negative, so this should never trigger in practice.
    """
    true_c = max(true_card, 0) + 1
    est_c = max(est_card, 0) + 1
    return max(true_c / est_c, est_c / true_c)


def evaluate_accuracy(estimator: Estimator, queries: list[WorkloadQuery]) -> dict:
    """Evaluate an estimator's q-error accuracy over a workload.

    Returns a dict with overall summary statistics (``median``, ``p90``,
    ``p95``, ``max``), the per-query q-error list (``per_query``, aligned
    positionally with ``queries``), and a breakdown by selectivity decile
    of ``true_cardinality`` (``by_selectivity_bucket``: ``{bucket_idx:
    {"median": ..., "n": ...}}``).

    Bucket membership is a RANK-based partition, not a value-range mask:
    queries are sorted by ``true_cardinality`` (ties broken stably by
    original array order, via ``np.argsort(..., kind="stable")`` for
    reproducibility), and each query's bucket is determined by its
    position/rank in that sorted order (``bucket = min(9, rank * 10 //
    n)``). This guarantees every query lands in exactly one bucket and the
    ``n`` values across all 10 buckets always sum to ``len(queries)``,
    regardless of how many queries share the same true_cardinality (e.g.
    several empty-result queries). A value-range mask over decile
    boundaries would instead double/triple/all-bucket-count tied queries,
    corrupting the per-decile breakdown.
    """
    per_query = [q_error(q.true_cardinality, estimator.estimate(q.predicate)) for q in queries]
    errors = np.array(per_query)

    true_cards = np.array([q.true_cardinality for q in queries])
    n = len(true_cards)
    sorted_indices = np.argsort(true_cards, kind="stable")
    ranks = np.empty(n, dtype=int)
    ranks[sorted_indices] = np.arange(n)
    bucket_of_query = np.minimum(9, ranks * 10 // n)

    by_bucket: dict[int, dict] = {}
    for i in range(10):
        mask = bucket_of_query == i
        if mask.sum() == 0:
            continue
        by_bucket[i] = {
            "median": float(np.median(errors[mask])),
            "n": int(mask.sum()),
        }

    return {
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
        "per_query": per_query,
        "by_selectivity_bucket": by_bucket,
    }
