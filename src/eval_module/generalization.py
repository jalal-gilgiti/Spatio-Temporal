"""Cross-city generalization harness: train on one city's data,
evaluate on another without retraining."""
from typing import Callable

import psycopg

from src.data_module.workload_generator import WorkloadQuery
from src.estimator_module.base import Estimator
from src.eval_module.accuracy import evaluate_accuracy


def evaluate_cross_city_transfer(
    estimator_factory: Callable[[], Estimator],
    conn: psycopg.Connection,
    source_table: str,
    source_train_queries: list[WorkloadQuery],
    target_table: str,
    target_eval_queries: list[WorkloadQuery],
) -> dict:
    """Train a fresh estimator on ``source_table`` and evaluate it (no
    retraining) on ``target_table``.

    Builds a fresh estimator via ``estimator_factory`` (so the same factory
    can be reused across multiple source/target pairs without state
    leaking between runs), fits it on ``source_train_queries`` against
    ``source_table``, refits ONLY its coordinate-normalization state
    against ``target_eval_queries`` (``Estimator.refit_normalization`` --
    a no-op for most registered estimators; see below), then scores it
    against ``target_eval_queries`` from ``target_table`` using
    ``evaluate_accuracy``.

    The ``refit_normalization`` call exists to fix a normalization
    artifact in cross-city transfer:
    ``CoupledLearnedEstimator.fit()`` derives its ``Featurizer``'s min-max
    bounds purely from ``source_train_queries``. When the target city's
    coordinates fall entirely outside that fitted range, every target
    feature clips to 0.0/1.0 and every target query looks identical to
    the model -- "transfer failure" would then be an artifact of the
    featurizer's normalization, not a real measurement of whether the
    LEARNED WEIGHTS generalize. Refitting only the normalization bounds
    (not retraining) is deliberate: a real deployment always knows its
    own table's coordinate extent, so this does not violate the
    "no retraining" contract the way refitting model weights would.
    Estimators with no such coordinate-scale-dependent internal state
    (the histogram-based ones, sampling, native, ``SpatialOnlyMLEstimator``)
    inherit ``Estimator``'s no-op default and are unaffected.

    Returns the same accuracy-summary dict shape as ``evaluate_accuracy``
    (``median``, ``p90``, ``p95``, ``max``, ``per_query``,
    ``by_selectivity_bucket``), with two extra labels merged in:
    ``source_table`` and ``target_table``. ``source_table == target_table``
    is accepted (a degenerate no-transfer case); ``target_eval_queries``
    must be non-empty, since ``evaluate_accuracy`` requires at least one
    query to compute summary statistics.

    Args:
        estimator_factory: Zero-arg callable returning a fresh, unfitted
            ``Estimator`` instance.
        conn: Open psycopg connection to the MobilityDB instance.
        source_table: Table to fit the estimator on.
        source_train_queries: Training workload against ``source_table``.
        target_table: Table to evaluate the fitted estimator against.
        target_eval_queries: Evaluation workload against ``target_table``.

    Returns:
        Accuracy-summary dict (see ``evaluate_accuracy``) plus
        ``"source_table"`` and ``"target_table"`` labels.
    """
    estimator = estimator_factory()
    estimator.fit(conn, source_table, train_queries=source_train_queries)
    estimator.refit_normalization(conn, target_table, target_queries=target_eval_queries)
    summary = evaluate_accuracy(estimator, target_eval_queries)
    summary["source_table"] = source_table
    summary["target_table"] = target_table
    return summary
