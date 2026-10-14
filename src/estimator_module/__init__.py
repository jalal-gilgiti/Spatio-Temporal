"""Common interface + Factory/Registry for all cardinality estimators.

Importing this package must register every concrete estimator with
`ESTIMATOR_FACTORY` via each module's `@register_estimator(...)` decorator --
a decorator only runs when its defining module is actually imported, and
nothing else in this codebase was guaranteed to import all concrete estimator
modules before the first `EstimatorFactory(...)` call. Before this fix, this
file imported only `.base`, so `ESTIMATOR_FACTORY` stayed empty until/unless
some unrelated import happened to pull in a concrete estimator module first
-- a fragile, import-order-dependent registry (a regression test checks
that a plain `import src.estimator_module`, with nothing else imported
first, populates the registry).

The imports below exist purely for their `@register_estimator` side effect;
nothing in this file uses their names directly.
"""
from .base import ESTIMATOR_FACTORY, Estimator, EstimatorFactory, register_estimator
from . import native_mobilitydb  # noqa: F401  (registers "native")
from . import histograms  # noqa: F401  (registers "independent_hist", "joint_hist")
from . import minskew_origin  # noqa: F401  (registers "minskew_origin")
from . import sampling  # noqa: F401  (registers "sampling")
from . import trajectory_sampling  # noqa: F401  (registers "trajectory_sampling")
from . import spatial_only_ml  # noqa: F401  (registers "spatial_only_ml")
from .coupled_learned import model as _coupled_learned_model  # noqa: F401  (registers "coupled_learned")

__all__ = ["Estimator", "register_estimator", "EstimatorFactory", "ESTIMATOR_FACTORY"]
