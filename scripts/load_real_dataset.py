"""Driver: reads one real dataset's `conf/dataset/<name>.yaml`,
normalizes it to the standard `(trajectory_id, lon, lat, timestamp)` point
shape via the right reader, applies dataset-specific cleaning
(`segment_into_trips` for T-Drive; a real-city bbox filter for T-Drive and
Porto, both of which carry a real ~1% GPS-fix-failure outlier rate --
see `filter_bbox`'s docstring), loads the result into MobilityDB, and runs
a cheap sanity check (a handful of ground-truth range queries plus one
baseline estimator's `.fit()`/`.estimate()`) so a load failure or a silent
all-empty/all-total table is caught immediately rather than discovered
downstream by the experiments.

Used for BOTH the small-sample verification load (e.g. ``--limit 500``)
and the full/documented-subset load (a larger ``--limit``) -- same code
path, different scale, so a small load can be verified before
committing to a full load.

Usage (from repo root):
    uv run python scripts/load_real_dataset.py tdrive --limit 500 --table tdrive_traj_sample
    uv run python scripts/load_real_dataset.py tdrive --limit 100000 --table tdrive_traj
    uv run python scripts/load_real_dataset.py porto --limit 500 --table porto_traj_sample
    uv run python scripts/load_real_dataset.py nyc_tlc --limit 500 --table nyc_tlc_traj_sample
"""
import argparse
import glob
import hashlib
import json
import logging
import math
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_module.mobilitydb_loader import filter_bbox, load_trajectories, read_raw_csv
from src.data_module.parquet_reader import load_zone_centroids, read_raw_parquet
from src.data_module.porto_polyline_reader import read_porto_polyline
from src.data_module.trip_segmenter import segment_into_trips
from src.data_module.tdrive_sampling import (
    TDRIVE_DATASET_SEED,
    select_tdrive_sources,
)
from src.data_module.workload_generator import STPredicate, compute_ground_truth, generate_workload
from src.estimator_module.histograms import IndependentHistogramEstimator
from src.pipeline_steps import create_trajectory_table
from src.utils.db import get_connection

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("load_real_dataset")

DSN = "postgresql://postgres@localhost:5434/mobilitydb"

# Real-city bboxes (min_lon, max_lon, min_lat, max_lat) for `filter_bbox`,
# generous enough to keep every genuine in-city point while dropping the
# GPS-fix-failure outliers confirmed empirically during data acquisition (see
# `filter_bbox`'s docstring). NYC TLC needs no bbox filter: its points are
# taxi-zone centroids computed from the official shapefile, inherently
# NYC-only.
TDRIVE_BBOX = (115.4, 117.5, 39.4, 41.6)  # Beijing
PORTO_BBOX = (-8.8, -8.4, 40.9, 41.3)  # Porto metro area

# `segment_into_trips` parameters, matching `synthetic_generator.py`'s
# real-trip-scale convention (see its own `max_trip_span`/`max_gap`).
TRIP_MAX_SPAN = timedelta(minutes=45)
TRIP_MAX_GAP = timedelta(minutes=20)

# Observed trips-per-taxi-file ratio from a 199-file T-Drive timing probe
# (23769 trip segments / 199 taxi files, from a loader timing
# probe) used only to size how many raw taxi files to read before
# segmenting, so a T-Drive load doesn't have to segment all 10357 taxis'
# full-week logs (~40 min, measured) just to reach a much smaller `--limit`.
_TDRIVE_TRIPS_PER_FILE = 23769 / 199
_TDRIVE_FILE_COUNT_SAFETY_MARGIN = 1.2


def _load_dataset_config(name: str) -> DictConfig:
    return OmegaConf.load(f"conf/dataset/{name}.yaml")


def _subsample_trajectories(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Randomly keep exactly `limit` whole trajectories (all their points),
    if `df` currently holds more than that -- never truncates a
    trajectory's own points, only drops whole trajectories, so every kept
    trajectory stays a complete, undistorted trip."""
    ids = df["trajectory_id"].unique()
    if len(ids) <= limit:
        return df
    rng = np.random.default_rng(seed)
    keep = rng.choice(ids, size=limit, replace=False)
    return df[df["trajectory_id"].isin(keep)]


def load_tdrive(
    cfg: DictConfig,
    limit: int | None,
    seed: int = TDRIVE_DATASET_SEED,
    *,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Load T-Drive after seeded source-taxi selection.

    An earlier implementation used ``sorted(glob(...))[:n_files]`` and
    thereby selected a lexicographic filename prefix.  The source taxi is now
    the sampling unit: numeric ids establish a stable population and a seeded
    permutation chooses the files before cleaning and segmentation.  The
    source id is retained on every point so downstream manifests can disclose
    that multiple trip segments may come from one taxi.
    """
    all_paths = glob.glob(cfg.raw_path)
    if limit is not None:
        n_files = min(
            len(all_paths),
            math.ceil(limit / _TDRIVE_TRIPS_PER_FILE * _TDRIVE_FILE_COUNT_SAFETY_MARGIN),
        )
    else:
        n_files = len(all_paths)
    selection = select_tdrive_sources(cfg.raw_path, seed=seed, n_files=n_files)
    paths = list(selection.selected_paths)
    logger.info(
        "tdrive: reading %d randomized source taxis from %d valid files (seed=%d)",
        len(paths), selection.valid_file_count, seed,
    )

    names = list(cfg.raw_column_names)
    frames = []
    for path, source_taxi_id in zip(paths, selection.selected_taxi_ids, strict=True):
        frame = read_raw_csv(path, dict(cfg.column_map), names=names)
        frame["source_taxi_id"] = source_taxi_id
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    logger.info("tdrive: %d raw points from %d taxis", len(df), df["trajectory_id"].nunique())

    df = filter_bbox(df, TDRIVE_BBOX)
    df = segment_into_trips(df, max_trip_span=TRIP_MAX_SPAN, max_gap=TRIP_MAX_GAP)
    logger.info("tdrive: %d trip-segments after segment_into_trips", df["trajectory_id"].nunique())

    if limit is not None:
        df = _subsample_trajectories(df, limit, seed)
        if df["trajectory_id"].nunique() < limit:
            raise RuntimeError(
                f"randomized T-Drive source sample yielded only "
                f"{df['trajectory_id'].nunique()} segments for requested limit {limit}; "
                "increase the source-file safety margin"
            )
    if not return_metadata:
        return df
    result_ids = sorted(int(value) for value in df["trajectory_id"].unique())
    result_hash = hashlib.sha256(
        json.dumps(result_ids, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = selection.to_dict()
    metadata.pop("selected_paths", None)
    metadata.update({
        "segmentation": {
            "max_trip_span_minutes": TRIP_MAX_SPAN.total_seconds() / 60.0,
            "max_gap_minutes": TRIP_MAX_GAP.total_seconds() / 60.0,
            "point_bbox": list(TDRIVE_BBOX),
        },
        "resulting_trajectory_count": len(result_ids),
        "resulting_trajectory_ids_sha256": result_hash,
        "contributing_source_taxi_count": int(df["source_taxi_id"].nunique()),
        "multiple_segments_per_taxi": bool(
            df[["trajectory_id", "source_taxi_id"]].drop_duplicates()
            .groupby("source_taxi_id").size().max() > 1
        ),
        "independence_statement": (
            "Rows are coherent trip segments, not independent vehicles; multiple segments "
            "from the same source taxi are present and source_taxi_id is retained."
        ),
    })
    return df, metadata


def load_porto(cfg: DictConfig, limit: int | None, seed: int) -> pd.DataFrame:
    # `read_porto_polyline` Bernoulli-samples across the WHOLE file (see
    # its own docstring) rather than stopping at the
    # first `limit` surviving rows, so the loaded sample is spread across
    # Porto's full July 2013 - June 2014 collection window instead of
    # always being the temporally-earliest slice. It can slightly
    # overshoot `limit` (sampling margin + drop-rate variance), so trim
    # to exactly `limit` here, same as `load_tdrive`.
    df = read_porto_polyline(cfg.raw_path, max_trips=limit, seed=seed)
    df = filter_bbox(df, PORTO_BBOX)
    if limit is not None:
        df = _subsample_trajectories(df, limit, seed)
    return df


def load_nyc_tlc(cfg: DictConfig, limit: int | None, seed: int) -> pd.DataFrame:
    # `read_raw_parquet` reads+filters every matched file in full (cheap,
    # vectorized) and takes one uniform random sample across ALL of them
    # combined, rather than "the first `limit` trips
    # in file order" -- for a single month that used to mean only the
    # first ~3.4 days ever got loaded. Its internal `.sample(n=limit,
    # random_state=seed)` already returns an exact count, but the trim
    # below is applied anyway for defensive consistency with the other
    # two loaders (a no-op when the count is already <= limit).
    zone_centroids = load_zone_centroids(cfg.zone_shapefile)
    df = read_raw_parquet(cfg.raw_path, zone_centroids, max_trips=limit, seed=seed)
    if limit is not None:
        df = _subsample_trajectories(df, limit, seed)
    return df


_ANCHOR_LON_PAD = 0.01
_ANCHOR_LAT_PAD = 0.01
_ANCHOR_TIME_PAD = timedelta(minutes=30)


def _anchor_predicate(sub: pd.DataFrame) -> STPredicate:
    """Build an STPredicate whose box generously contains one trajectory's
    ENTIRE (lon, lat, timestamp) extent -- with this much padding,
    `compute_ground_truth` is structurally guaranteed to count that
    trajectory (`atStbox` restricted to its whole non-empty life), making
    "did this anchor's own query find itself" a real bug signal rather
    than a density-dependent coin flip -- see `verify_load`'s docstring
    for why a purely-random workload check was replaced
    with this."""
    return STPredicate(
        min_lon=float(sub["lon"].min()) - _ANCHOR_LON_PAD,
        max_lon=float(sub["lon"].max()) + _ANCHOR_LON_PAD,
        min_lat=float(sub["lat"].min()) - _ANCHOR_LAT_PAD,
        max_lat=float(sub["lat"].max()) + _ANCHOR_LAT_PAD,
        start_time=sub["timestamp"].min().to_pydatetime() - _ANCHOR_TIME_PAD,
        end_time=sub["timestamp"].max().to_pydatetime() + _ANCHOR_TIME_PAD,
    )


def verify_load(conn: psycopg.Connection, table: str, df: pd.DataFrame, seed: int = 42) -> None:
    """Cheap post-load sanity check against the freshly loaded real table.

    Two checks, deliberately not the same signal:

    1. ANCHOR check (primary correctness signal, density-independent):
       for up to 20 randomly chosen already-loaded trajectories, build a
       query box padded around that ONE trajectory's own full extent and
       confirm `compute_ground_truth` finds it (`>= 1`) -- structurally
       guaranteed by `atStbox` semantics (see `_anchor_predicate`), so
       any anchor returning 0 is a genuine loading/ground-truth bug, not
       a density artifact. This replaced a purely-random `generate_
       workload`-based zero-count check: once Porto/
       NYC TLC's sampling was fixed to spread trajectories across their
       FULL collection window instead of the temporally-earliest slice,
       a small (e.g. 500-trajectory) sample became genuinely sparse
       relative to a full year/month, and uniform-random query windows
       at trip scale legitimately miss it almost every time -- that isn't
       a bug, but the old check (all-zero => warn) couldn't tell the
       difference.
    2. `generate_workload`-based random queries (secondary, informational
       only): realistic-shaped queries, logged for context, but a low or
       zero non-zero-count here is EXPECTED for a small/sparse sample and
       is not treated as a failure signal.

    Plus one baseline estimator's `.fit()`/`.estimate()` against the
    table, to catch a crash in that interface against real data.
    """
    rng = np.random.default_rng(seed)
    traj_ids = df["trajectory_id"].unique()
    n_anchors = min(20, len(traj_ids))
    anchor_ids = rng.choice(traj_ids, size=n_anchors, replace=False)

    anchor_counts = []
    for tid in anchor_ids:
        sub = df[df["trajectory_id"] == tid]
        p = _anchor_predicate(sub)
        anchor_counts.append(compute_ground_truth(conn, table, p))
    n_anchor_found = sum(1 for c in anchor_counts if c >= 1)
    logger.info(
        "verify_load(%s): anchor check %d/%d self-found (should always be %d/%d); counts=%s",
        table, n_anchor_found, n_anchors, n_anchors, n_anchors, anchor_counts,
    )
    if n_anchor_found < n_anchors:
        logger.error(
            "verify_load(%s): %d anchor trajectories did NOT find themselves via their own "
            "padded query box -- this is a real loading/ground-truth bug, not a density artifact",
            table, n_anchors - n_anchor_found,
        )

    bbox = (float(df["lon"].min()), float(df["lon"].max()), float(df["lat"].min()), float(df["lat"].max()))
    time_range = (df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime())
    n_total = df["trajectory_id"].nunique()
    workload = generate_workload(
        conn, table, n_queries=20, seed=seed, bbox=bbox, knn_fraction=0.0, time_range=time_range
    )
    counts = [q.true_cardinality for q in workload]
    n_nonzero = sum(1 for c in counts if c > 0)
    n_nontotal = sum(1 for c in counts if c < n_total)
    logger.info(
        "verify_load(%s): [informational] %d/20 random queries non-zero, %d/20 non-total "
        "(table has %d trajectories, span=%s..%s); counts=%s -- low/zero non-zero here is "
        "EXPECTED and not a bug signal for a small or temporally-sparse sample",
        table, n_nonzero, n_nontotal, n_total, time_range[0], time_range[1], counts,
    )

    est = IndependentHistogramEstimator()
    est.fit(conn, table)
    sample_estimate = est.estimate(_anchor_predicate(df[df["trajectory_id"] == anchor_ids[0]]))
    logger.info(
        "verify_load(%s): independent_hist.fit() + .estimate() succeeded, estimate=%.2f "
        "(true=%d) for first anchor query",
        table, sample_estimate, anchor_counts[0],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["tdrive", "porto", "nyc_tlc"])
    parser.add_argument("--limit", type=int, default=None, help="Target number of trajectories to load.")
    parser.add_argument("--table", type=str, default=None, help="Override the target table name.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Sampling seed (default: 70707 for T-Drive, legacy 42 otherwise).",
    )
    parser.add_argument("--dsn", type=str, default=DSN)
    parser.add_argument("--no-verify", action="store_true", help="Skip the post-load sanity check.")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else (
        TDRIVE_DATASET_SEED if args.dataset == "tdrive" else 42
    )

    cfg = _load_dataset_config(args.dataset)
    table = args.table or cfg.table

    t0 = time.time()
    if args.dataset == "tdrive":
        df = load_tdrive(cfg, args.limit, seed)
    elif args.dataset == "porto":
        df = load_porto(cfg, args.limit, seed)
    else:
        df = load_nyc_tlc(cfg, args.limit, seed)
    logger.info("%s: %d points / %d trajectories prepared in %.1fs", args.dataset, len(df), df["trajectory_id"].nunique(), time.time() - t0)

    if len(df) == 0:
        raise RuntimeError(f"{args.dataset}: reader produced 0 points -- aborting before touching the DB")

    # `load_trajectories` inserts by `int(trajectory_id)` into a plain
    # `int` column; groupby/subsampling above can leave a non-contiguous
    # or non-zero-based id space (e.g. tdrive's raw taxi_id, or a random
    # subsample) -- remap to a dense 0-based range so ids stay small and
    # collision-free across repeated runs against the same table.
    remap = {old: new for new, old in enumerate(sorted(df["trajectory_id"].unique()))}
    df = df.assign(trajectory_id=df["trajectory_id"].map(remap))

    conn = get_connection(args.dsn)
    create_trajectory_table(conn, table)
    t0 = time.time()
    n = load_trajectories(conn, table, df)
    logger.info("%s: loaded %d trajectories into %s in %.1fs", args.dataset, n, table, time.time() - t0)

    if not args.no_verify:
        verify_load(conn, table, df, seed=seed)
    conn.close()


if __name__ == "__main__":
    main()
