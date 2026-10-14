"""Common interface + Factory/Registry for all cardinality estimators."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import psycopg

from src.data_module.workload_generator import STPredicate

if TYPE_CHECKING:
    from src.data_module.workload_generator import WorkloadQuery

ESTIMATOR_FACTORY: dict[str, type["Estimator"]] = {}


class Estimator(ABC):
    @abstractmethod
    def fit(
        self,
        conn: psycopg.Connection,
        table: str,
        train_queries: "list[WorkloadQuery] | None" = None,
    ) -> None: ...

    @abstractmethod
    def estimate(self, predicate: STPredicate) -> float: ...

    def refit_normalization(
        self,
        conn: psycopg.Connection,
        table: str,
        target_queries: "list[WorkloadQuery] | None" = None,
    ) -> None:
        """Optional hook: refit ONLY an estimator's internal coordinate-
        normalization state against a different table's query workload,
        leaving any trained/learned parameters untouched. Default: a no-op.

        Used by `evaluate_cross_city_transfer`, after `fit()`
        and before scoring against the target table, so an estimator whose
        `estimate()` depends on bounds fitted from the SOURCE table's
        coordinates (e.g. `CoupledLearnedEstimator`'s min-max-normalized
        `Featurizer`) can adapt its normalization to the TARGET table's
        real coordinate range without retraining any learned weights.
        Without this, a target city whose
        coordinates fall entirely outside the source's fitted range clips
        to a constant feature vector, making transfer "failure" a
        normalization artifact rather than a real measurement of whether
        the learned weights generalize.

        This is intentionally allowed and distinct from retraining: a real
        deployment would always know its own table's coordinate extent (it
        is metadata about the data, not something learned from labels), so
        refitting normalization bounds for a new table does not violate
        the "no retraining" contract the way refitting model weights would.

        Most registered estimators here (the histogram-based ones,
        sampling, native, `SpatialOnlyMLEstimator`) have no such
        coordinate-scale-dependent internal state -- they either query the
        live table directly at estimate time or use tree-based models with
        no min-max feature scaling -- so this default no-op is correct for
        them unchanged; only `CoupledLearnedEstimator` overrides it.

        Args:
            conn: Open psycopg connection to the MobilityDB instance.
            table: Name of the table whose coordinate extent normalization
                should be refit against.
            target_queries: The workload whose predicates define the new
                normalization bounds, if an estimator's implementation
                derives bounds from queries rather than a live `conn`/
                `table` round-trip.
        """
        return None


def register_estimator(name: str):
    def decorator(cls: type[Estimator]) -> type[Estimator]:
        ESTIMATOR_FACTORY[name] = cls
        return cls
    return decorator


def EstimatorFactory(name: str) -> type[Estimator]:
    try:
        return ESTIMATOR_FACTORY[name]
    except KeyError:
        available = sorted(ESTIMATOR_FACTORY)
        raise KeyError(
            f"no estimator registered as {name!r}; available: {available}"
        ) from None
