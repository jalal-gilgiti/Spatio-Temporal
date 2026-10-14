"""Tunable-correlation synthetic trajectory generator.

`correlation_strength` is the correlation knob of the synthetic study: 0.0 means a trip's location
is statistically independent of the time of day it happens, 1.0 means the two
are maximally dependent. What the knob must produce, for the
same-architecture ablation to be able to measure anything, is a *genuine
space-time interaction* -- a field where cardinality depends on WHERE and
WHEN jointly and not on WHERE plus WHEN separately.

The construction (which replaced an earlier one):

* A fixed set of `n_modes` joint space-time modes ("this district, at this
  hour"). Each mode owns a spatial center, an hour-of-day center, and a
  share of the trips.
* Every trip draws a *temporal* mode and a *spatial* mode. With probability
  `correlation_strength` the two are the SAME mode; otherwise they are drawn
  independently. That single Bernoulli is the whole knob.
* Trips are local: a trip starts at its spatial mode's center (plus noise)
  and travels a short distance at constant heading and speed, instead of
  scattering its points across the whole domain.

Two properties this buys, both of which the original mean-shift construction
(`lon = base_lon + correlation_strength * 4 * sin(2*pi*hour/24)`) lacked:

* **Marginal invariance.** Both mode indices are marginally
  ``Categorical(weights)`` for every value of `correlation_strength`, because
  mixing a draw with an independent draw from the same distribution leaves
  that distribution unchanged. So the spatial point distribution and the
  hour-of-day distribution are the same at every strength, and the knob moves
  *only* the dependence between them. A sweep over it therefore isolates
  coupling rather than confounding it with "the data got clumpier".
* **Non-decomposability.** At strength 1 the joint density is
  ``sum_k w_k f_k(lon, lat) g_k(hour)`` with different modes sitting in
  different places, which is not any ``f(lon, lat) + g(hour)``: knowing the
  hour tells you where the mass is, and no space-only-plus-time-only model
  can express that. At strength 0 the mode-based part of the density
  factorizes into ``(sum_k w_k f_k)(lon, lat) * (sum_k w_k g_k)(hour)``, so
  there is (almost -- see below) nothing for a coupling term to find. The
  original construction instead shifted the expected longitude by a smooth
  function of the hour alone, which a plain MLP over the query box captures
  about as well as an explicit coupling term -- measured, and the reason for
  this rewrite.

**The strength-0 factorization is exact only at ``background_fraction=0``.**
It is tempting to say the joint density factorizes exactly at strength 0, and
earlier versions of this docstring did. That is false at the default
``background_fraction=0.1``, for a reason that has nothing to do with the
coupling knob: a *mixture* of two independent-pair distributions is not
itself an independent pair. Writing ``Z ~ Bernoulli(b)`` for the background
selector, ``F``/``G`` for the mode-based spatial and temporal densities and
``U``/``V`` for the uniform ones, the strength-0 joint is
``(1-b) F(x) G(h) + b U(x) V(h)``, whose residual against its own marginals is

    p(x, h) - p(x) p(h) = b (1 - b) (F - U)(x) (G - V)(h)

-- zero only at ``b = 0`` or ``b = 1``. Measured at the default ``b = 0.1``
(4000 trips, one point per trip, 20 seeds): adjusted mutual information at
strength 0 is +0.0062 +/- 0.0029, positive on 20 seeds out of 20, i.e.
systematic rather than noise. At ``b = 0`` the same measurement gives +0.0001
+/- 0.0012, positive on 9 of 20 -- centered on zero, as an exact
factorization requires.

The residual is small enough not to disturb what the knob is for: it is ~2%
of the strength-1 AMI (+0.280 +/- 0.019 over the same 20 seeds). But a
strength-0 measurement should be compared against ~0.006, not against 0, and
any experiment that needs a genuinely exact factorization at strength 0 must
set ``background_fraction=0`` and accept the empty-query problem that knob
exists to solve.
"""
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from src.data_module.trip_segmenter import segment_into_trips

# The [0, 10] x [0, 10] domain every caller's `bbox` already assumes (see
# `generate_workload`'s `bbox` docstring and every call site's (0, 10, 0, 10)).
DOMAIN_MIN = 0.0
DOMAIN_MAX = 10.0

# Hour-of-day centers are confined well inside the day for the same reason,
# with extra room at the top for a trip's own forward extent (up to ~54
# minutes at n_points_per_traj=10; see `_time_offsets_hours`).
_MODE_HOUR_LO = 3.0
_MODE_HOUR_HI = 20.0
_MAX_START_HOUR = 22.5
_MAX_HOUR = 23.999


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for one synthetic trajectory dataset.

    Args:
        n_trajectories: Number of trips to generate (before
            `segment_into_trips` possibly splits a long one).
        n_points_per_traj: Number of sampled points per trip.
        correlation_strength: 0.0 = a trip's place is independent of its
            time of day; 1.0 = a trip's place and time are drawn from the
            same joint space-time mode. Values in between are the
            probability of that coupling, per trip.
        seed: Seeds both the mode field and the per-trip draws, so a
            (config -> dataset) mapping is fully reproducible. Note that
            two configs differing only in `correlation_strength` share the
            same mode field, which is what makes a correlation sweep a controlled
            comparison.
        n_modes: Number of joint space-time modes.
        spatial_sigma: Standard deviation, in domain units, of a trip's
            origin around its spatial mode center.
        temporal_sigma_hours: Standard deviation, in hours, of a trip's
            start time around its temporal mode center.
        speed_range: (min, max) trip speed in domain units per hour, drawn
            per trip. With trips lasting under an hour this keeps a trip's
            path comparable in length to `spatial_sigma`, i.e. local to its
            district rather than sweeping the map.
        background_fraction: Share of trips drawn uniformly in space and in
            time instead of from a mode. Independent of
            `correlation_strength`, so it does not blunt the knob: it adds a
            floor of traffic everywhere, which keeps a query that misses
            every mode from returning exactly zero. Without it, well over 40%
            of a generated eval workload is empty and median q-error
            collapses toward 1.0 for every estimator, discriminating between
            none of them. The background component is separable on its own,
            but *mixing* it with the mode-based trips leaves a small residual
            space-time dependence even at strength 0 (AMI ~0.006 at the
            default 0.1, ~2% of the strength-1 signal) -- see the module
            docstring for the derivation and the measurement.
        gps_jitter: Standard deviation of per-point positional noise.
    """

    n_trajectories: int
    n_points_per_traj: int
    correlation_strength: float
    seed: int
    n_modes: int = 5
    spatial_sigma: float = 0.9
    temporal_sigma_hours: float = 1.2
    speed_range: tuple[float, float] = (0.6, 1.8)
    background_fraction: float = 0.1
    gps_jitter: float = 0.03


@dataclass(frozen=True)
class SpaceTimeModes:
    """The joint space-time mode field: `lon`, `lat`, `hour` centers and
    mixture `weights`, each of shape ``(n_modes,)``."""

    lon: np.ndarray
    lat: np.ndarray
    hour: np.ndarray
    weights: np.ndarray


def _stratified_positions(
    rng: np.random.Generator, n_modes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Place `n_modes` spatial centers on distinct cells of a coarse grid over
    the domain, jittered inside their cell.

    Stratified rather than i.i.d. uniform because a generated query box is
    large: the workload samples each spatial bound independently across the
    whole domain, so the median box is about a third of the domain wide. Two
    mode centers a short distance apart therefore fall inside the *same* query
    box most of the time, and "which district is busy right now" stops being
    answerable from the box -- which is precisely the joint signal an explicit
    coupling term exists to read. Spreading the centers keeps them separable
    at the resolution the workload actually queries at.
    """
    side = int(np.ceil(np.sqrt(n_modes)))
    cell = (DOMAIN_MAX - DOMAIN_MIN) / side
    chosen = rng.choice(side * side, size=n_modes, replace=False)
    row, col = np.divmod(chosen, side)
    # Jitter is capped so a center cannot leave its own cell, which is what
    # keeps the minimum separation guarantee the stratification buys.
    jitter = rng.uniform(-0.25, 0.25, size=(2, n_modes)) * cell
    lon = DOMAIN_MIN + (col + 0.5) * cell + jitter[0]
    lat = DOMAIN_MIN + (row + 0.5) * cell + jitter[1]
    return lon, lat


def sample_space_time_modes(rng: np.random.Generator, cfg: SyntheticConfig) -> SpaceTimeModes:
    """Draw the joint space-time mode field for one config.

    Hour centers are stratified (one per equal slice of
    ``[_MODE_HOUR_LO, _MODE_HOUR_HI]``, jittered inside its slice) rather
    than drawn i.i.d. uniform: two modes landing on the same hour would be
    temporally indistinguishable, and a query's time window would then not
    identify which region of space is busy -- exactly the coupling signal
    this field exists to create. Spatial centers are stratified too, for the
    analogous reason at query-box scale (`_stratified_positions`). Which
    district is paired with which hour is still random -- the hour slices are
    filled in order but the spatial cells are drawn in a random permutation --
    so the hour -> place map carries no fixed structure a model could exploit
    without learning this particular field.

    Public (rather than a private helper) because the mode field is the
    object a diagnostic or a figure wants to talk about -- "which district is
    busy at which hour" -- independently of the trips drawn from it.
    """
    if cfg.n_modes < 1:
        raise ValueError(f"n_modes must be >= 1, got {cfg.n_modes}")
    edges = np.linspace(_MODE_HOUR_LO, _MODE_HOUR_HI, cfg.n_modes + 1)
    lon, lat = _stratified_positions(rng, cfg.n_modes)
    return SpaceTimeModes(
        lon=lon,
        lat=lat,
        hour=rng.uniform(edges[:-1], edges[1:]),
        # Dirichlet(5) rather than exactly-equal weights: real cities have
        # busier and quieter districts, and a concentration this high keeps
        # every mode populated enough to be learnable at the dataset sizes
        # used here.
        weights=rng.dirichlet(np.full(cfg.n_modes, 5.0)),
    )


def assign_modes(
    rng: np.random.Generator, cfg: SyntheticConfig, modes: SpaceTimeModes
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each trip a temporal mode and a spatial mode.

    ``correlation_strength`` is the per-trip probability that the two are the
    same mode. The spatial assignment is ``where(coupled, time_mode,
    independent_mode)`` where both candidates are i.i.d.
    ``Categorical(weights)``, so the spatial mode's marginal law is
    ``Categorical(weights)`` at every strength -- see the module docstring on
    marginal invariance, which is what makes a strength sweep a controlled
    experiment rather than a confounded one.

    Returns:
        ``(time_mode, space_mode)``, each of shape ``(n_trajectories,)``.
    """
    time_mode = rng.choice(cfg.n_modes, size=cfg.n_trajectories, p=modes.weights)
    independent_mode = rng.choice(cfg.n_modes, size=cfg.n_trajectories, p=modes.weights)
    coupled = rng.random(cfg.n_trajectories) < cfg.correlation_strength
    return time_mode, np.where(coupled, time_mode, independent_mode)


def _time_offsets_hours(rng: np.random.Generator, cfg: SyntheticConfig) -> np.ndarray:
    """Per-point elapsed time within a trip, shape ``(n_traj, n_points)``.

    Each point is 1-6 minutes after the previous one, with the first at
    offset 0. Bounding every step strictly below the 20-minute ``max_gap``
    passed to `segment_into_trips` guarantees segmentation only ever splits
    on ``max_trip_span`` (producing coherent multi-point sub-trips) and never
    gap-shatters a trip into singleton, single-instant rows -- the
    dynamic-range degeneracy D1 exists to prevent, at instant granularity.
    """
    step_minutes = rng.uniform(1.0, 6.0, size=(cfg.n_trajectories, cfg.n_points_per_traj))
    step_minutes[:, 0] = 0.0
    return np.cumsum(step_minutes, axis=1) / 60.0


def generate_synthetic_trajectories(cfg: SyntheticConfig) -> pd.DataFrame:
    """Generate a trip-scale synthetic dataset with tunable space-time coupling.

    Returns:
        A DataFrame with columns ``trajectory_id``, ``lon``, ``lat``,
        ``timestamp``, holding ``n_trajectories * n_points_per_traj`` rows,
        every instant inside a single 24-hour day (2026-01-01; see
        `src.eval_module.correlation_analysis.SYNTHETIC_TIME_RANGE`).
        ``trajectory_id`` is re-assigned by `segment_into_trips`, so the
        number of distinct ids can exceed `cfg.n_trajectories`.
    """
    rng = np.random.default_rng(cfg.seed)
    modes = sample_space_time_modes(rng, cfg)
    time_mode, space_mode = assign_modes(rng, cfg, modes)

    offset_hours = _time_offsets_hours(rng, cfg)
    start_hours = np.clip(
        rng.normal(modes.hour[time_mode], cfg.temporal_sigma_hours), 0.0, _MAX_START_HOUR
    )
    origin_lon = rng.normal(modes.lon[space_mode], cfg.spatial_sigma)
    origin_lat = rng.normal(modes.lat[space_mode], cfg.spatial_sigma)

    # Background traffic: uniform in space AND in time, so this component
    # carries no coupling of its own -- though mixing it with the mode-based
    # trips does leave a small residual dependence at strength 0 (module
    # docstring). Applied after the mode draws (rather than by shrinking the
    # mode population) so the mode field and its assignment stream are
    # untouched by this knob.
    is_background = rng.random(cfg.n_trajectories) < cfg.background_fraction
    start_hours = np.where(
        is_background, rng.uniform(0.0, _MAX_START_HOUR, size=cfg.n_trajectories), start_hours
    )
    origin_lon = np.where(
        is_background, rng.uniform(DOMAIN_MIN, DOMAIN_MAX, size=cfg.n_trajectories), origin_lon
    )
    origin_lat = np.where(
        is_background, rng.uniform(DOMAIN_MIN, DOMAIN_MAX, size=cfg.n_trajectories), origin_lat
    )
    hours = np.clip(start_hours[:, None] + offset_hours, 0.0, _MAX_HOUR)
    # A trip travels in one direction at a constant speed, so its points stay
    # near its origin district. Drawing every point's position independently
    # over the whole domain (the previous construction) made each trip's
    # interpolated path sweep the entire map, which erases the spatial
    # localization the mode field exists to create.
    heading = rng.uniform(0.0, 2.0 * np.pi, size=cfg.n_trajectories)
    speed = rng.uniform(cfg.speed_range[0], cfg.speed_range[1], size=cfg.n_trajectories)
    jitter_shape = (cfg.n_trajectories, cfg.n_points_per_traj)
    lon = np.clip(
        origin_lon[:, None] + (speed * np.cos(heading))[:, None] * offset_hours
        + rng.normal(0.0, cfg.gps_jitter, size=jitter_shape),
        DOMAIN_MIN, DOMAIN_MAX,
    )
    lat = np.clip(
        origin_lat[:, None] + (speed * np.sin(heading))[:, None] * offset_hours
        + rng.normal(0.0, cfg.gps_jitter, size=jitter_shape),
        DOMAIN_MIN, DOMAIN_MAX,
    )

    df = pd.DataFrame({
        "trajectory_id": np.repeat(np.arange(cfg.n_trajectories), cfg.n_points_per_traj),
        "lon": lon.ravel(),
        "lat": lat.ravel(),
        "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(hours.ravel(), unit="h"),
    })

    # Re-segment into trip-scale trajectories: a row that is
    # a multi-hour per-entity history rather than a bounded trip makes
    # spatio-temporal correlation unmeasurable.
    return segment_into_trips(
        df,
        max_trip_span=timedelta(minutes=45),
        max_gap=timedelta(minutes=20),
    )
