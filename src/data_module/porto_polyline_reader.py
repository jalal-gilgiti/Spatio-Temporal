"""Parser for the Porto taxi (ECML/PKDD 2015) dataset's ``POLYLINE`` column.

Porto's raw ``train.csv`` (see
`scripts/download_real_datasets.sh` for the exact UCI mirror URL used) does
NOT have flat ``lon``/``lat`` columns -- each row is one already-complete
trip, and its path is packed into a single ``POLYLINE`` cell: a JSON array
of ``[lon, lat]`` pairs sampled every 15 seconds, starting at the trip's
``TIMESTAMP`` (Unix seconds, trip start). This module expands that packed
shape into the standard per-point ``(trajectory_id, lon, lat, timestamp)``
stream `load_trajectories` expects.

Each row is already one bounded trip by construction (Porto is naturally
trip-scale, unlike T-Drive's multi-day per-taxi logs), so callers do not
need to run `segment_into_trips` on this reader's output (checked
empirically on the loaded sample).

Sampling note: an earlier version of this reader stopped as soon as
`max_trips` surviving rows were found, i.e. "the first N trips in file
order." `train.csv` is essentially chronological (July 2013 start), so
that made every loaded sample the temporally-earliest slice of a
year-long collection -- for `max_trips=100000` out of 1,710,670 rows,
the earliest ~6%, with no coverage of the rest of the year at all. This
directly undermines any real cross-dataset temporal/correlation analysis
downstream. `read_porto_polyline` now Bernoulli-samples every row
(seeded, before the expensive per-row POLYLINE parse) while scanning the
WHOLE file, so the loaded sample is spread across the full collection
window -- see `max_trips`/`seed` below.
"""
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_POINT_INTERVAL_SECONDS = 15

# Total data rows in the official UCI-mirrored `train.csv` (`wc -l` minus
# the header) -- used only to size
# the Bernoulli keep-probability below, not assumed authoritative for
# anything else (a differently-sized `train.csv` still works correctly,
# just with a less precisely-aimed `max_trips`).
_PORTO_TOTAL_ROWS = 1_710_670
# Sampled rows are further reduced by MISSING_DATA/empty-POLYLINE drops
# (empirically a small fraction, ~0.4%) and Bernoulli variance, so aim
# somewhat over `max_trips` -- any overshoot is trimmed by the caller
# (`scripts/load_real_dataset.py`'s final random subsample), matching the
# same margin-then-trim pattern already used for T-Drive's raw-file count.
_SAMPLE_MARGIN = 1.15


def read_porto_polyline(path: str, max_trips: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Normalize Porto's raw ``train.csv`` to the standard trajectory point shape.

    Trips with ``MISSING_DATA == True`` have an unreliable/incomplete
    ``POLYLINE`` (per the dataset's own documentation) and are dropped, as
    are trips whose ``POLYLINE`` is an empty array (no GPS fix at all).

    Raw ``TRIP_ID`` values (e.g. ``1372636858620000589``) exceed
    PostgreSQL's 32-bit ``int`` range used for `trajectory_id` throughout
    this codebase (see `create_trajectory_table`), so a fresh 0-based
    sequential integer id is assigned per surviving trip instead of
    reusing ``TRIP_ID``.

    Args:
        path: Path to Porto's raw ``train.csv``.
        max_trips: If given, independently Bernoulli-sample rows (see
            module docstring) at a keep probability sized so the WHOLE
            file is scanned while landing near this many surviving
            trips -- NOT an exact count; the caller trims any overshoot.
            ``None`` reads every trip in the file (no sampling needed).
        seed: Seed for the sampling RNG, for reproducibility.

    Returns:
        DataFrame with columns ``trajectory_id, lon, lat, timestamp``, one
        row per GPS fix (~every 15 simulated seconds along each trip).
    """
    if max_trips is not None:
        keep_prob = min(1.0, (max_trips / _PORTO_TOTAL_ROWS) * _SAMPLE_MARGIN)
    else:
        keep_prob = 1.0
    rng = np.random.default_rng(seed)

    usecols = ["TIMESTAMP", "MISSING_DATA", "POLYLINE"]
    reader = pd.read_csv(path, usecols=usecols, chunksize=50_000)

    out_frames: list[pd.DataFrame] = []
    next_id = 0
    n_dropped_empty = 0

    # Deliberately no early `break` on `next_id >= max_trips`: every chunk
    # across the WHOLE file must be visited for `keep_prob` sampling to
    # actually cover the full temporal span (see module docstring).
    for chunk in reader:
        if keep_prob < 1.0:
            chunk = chunk[rng.random(len(chunk)) < keep_prob]
        # pandas infers MISSING_DATA's "True"/"False" strings as bool.
        chunk = chunk[~chunk["MISSING_DATA"]]
        for row in chunk.itertuples(index=False):
            coords = json.loads(row.POLYLINE)
            if not coords:
                n_dropped_empty += 1
                continue
            start_ts = pd.Timestamp(row.TIMESTAMP, unit="s")
            timestamps = [start_ts + pd.Timedelta(seconds=_POINT_INTERVAL_SECONDS * i) for i in range(len(coords))]
            out_frames.append(
                pd.DataFrame({
                    "trajectory_id": next_id,
                    "lon": [c[0] for c in coords],
                    "lat": [c[1] for c in coords],
                    "timestamp": timestamps,
                })
            )
            next_id += 1

    logger.info(
        "read_porto_polyline: %d trips loaded, %d empty-polyline trips dropped (from %s)",
        next_id, n_dropped_empty, path,
    )
    if not out_frames:
        return pd.DataFrame(columns=["trajectory_id", "lon", "lat", "timestamp"])
    return pd.concat(out_frames, ignore_index=True)
