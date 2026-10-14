"""Reader for NYC TLC yellow-cab monthly trip parquet files.

Verified empirically against downloaded files (2024-01 AND the older
2015-01 vintage, both re-published by TLC as parquet): every currently-served vintage records ``PULocationID`` /
``DOLocationID`` TAXI ZONE ids for pickup/dropoff, not raw lat/lon --
TLC's July-2016 switch to zone ids was applied retroactively to the
parquet re-publication, so there is no "older vintage still has lat/lon"
escape hatch via this endpoint (`docs`: the plan's original ledger
anticipated this might be findable; it is not, as of this check).

NYC TLC trip records are also never a continuous GPS trace to begin
with (unlike T-Drive/Porto) -- they report exactly two space-time
endpoints per trip: pickup (``PULocationID``, ``tpep_pickup_datetime``)
and dropoff (``DOLocationID``, ``tpep_dropoff_datetime``). This module
therefore builds a 2-point-per-trip stream, using each zone's polygon
CENTROID (computed from the official TLC taxi-zone shapefile) as a
documented approximation of the pickup/dropoff coordinate -- trips
loaded via this reader are ZONE-resolution, not GPS-resolution. Each
row is already one complete, bounded trip by construction, so callers
do not need `segment_into_trips` on this reader's output either.
"""
import glob
import logging
import re

import pandas as pd
import pyproj
import shapefile

logger = logging.getLogger(__name__)

# NYC TLC's own monthly parquet files carry a real, if rare, data-quality
# defect: a handful of rows have a pickup/dropoff timestamp wildly outside
# the file's nominal month (confirmed empirically on 2024-01: 4 rows dated
# 2002-12-31, evidently a meter-clock fault at the source, not a parsing
# bug). A single such row blows up any downstream
# min/max-derived time range by decades, which silently starves
# `generate_workload`'s query-window sampling (every sampled window lands
# nowhere near the real data, so `compute_ground_truth` returns 0 for
# essentially every query). `_FILENAME_MONTH_RE` recovers each file's
# nominal ``YYYY-MM`` from its name (TLC's own naming convention, e.g.
# ``yellow_tripdata_2024-01.parquet``) so `read_raw_parquet` can drop rows
# outside a generous buffer around that month, while still keeping
# legitimate trips that start just before/after the calendar boundary
# (e.g. a trip beginning at 23:58 on the last day of the prior month).
_FILENAME_MONTH_RE = re.compile(r"(\d{4})-(\d{2})")
_DATE_SANITY_BUFFER_DAYS = 3

# NAD83 / New York Long Island (US feet) -- the TLC taxi-zone shapefile's
# native CRS (read from its own .prj; see `scripts/download_real_datasets.sh`).
_ZONE_SHAPEFILE_CRS = "EPSG:2263"
_WGS84_CRS = "EPSG:4326"


def _ring_signed_area_and_centroid(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Shoelace-formula signed area and centroid of one closed polygon ring."""
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(points)
    for i in range(n - 1):  # shapefile rings repeat the first point as the last
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area /= 2.0
    if area == 0.0:
        # Degenerate ring; fall back to the plain vertex average.
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return 0.0, sum(xs) / len(xs), sum(ys) / len(ys)
    cx /= 6.0 * area
    cy /= 6.0 * area
    return area, cx, cy


def load_zone_centroids(shapefile_path: str) -> pd.DataFrame:
    """Compute one representative (lon, lat) centroid per TLC taxi zone.

    Reads the official TLC taxi-zone shapefile (EPSG:2263, NY State Plane
    Long Island, US feet) and, for each ``LocationID``, uses the
    LARGEST-area ring among that zone's polygon parts (some zones are
    multi-part -- islands, disjoint pieces) as its representative shape,
    to avoid a small sliver or hole pulling the centroid off zone. That
    ring's shoelace centroid is then reprojected to WGS84 lon/lat via
    `pyproj` (bundles its own PROJ data, no system GIS deps required).

    Args:
        shapefile_path: Path to the ``.shp`` file (its sibling ``.dbf``/
            ``.shx`` must sit alongside it, standard shapefile layout).

    Returns:
        DataFrame with columns ``LocationID, lon, lat`` -- one row per
        zone in the shapefile (263 for the standard TLC release).
    """
    sf = shapefile.Reader(shapefile_path)
    transformer = pyproj.Transformer.from_crs(_ZONE_SHAPEFILE_CRS, _WGS84_CRS, always_xy=True)

    rows = []
    for shape_rec in sf.iterShapeRecords():
        location_id = shape_rec.record["LocationID"]
        shp = shape_rec.shape
        parts = list(shp.parts) + [len(shp.points)]
        best_area, best_cx, best_cy = 0.0, None, None
        for i in range(len(parts) - 1):
            ring = shp.points[parts[i]:parts[i + 1]]
            area, cx, cy = _ring_signed_area_and_centroid(ring)
            if abs(area) > abs(best_area) or best_cx is None:
                best_area, best_cx, best_cy = area, cx, cy
        lon, lat = transformer.transform(best_cx, best_cy)
        rows.append({"LocationID": location_id, "lon": lon, "lat": lat})

    return pd.DataFrame(rows)


def _expected_month_bounds(path: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Parse a ``YYYY-MM`` out of `path`'s filename and return a buffered
    ``(start, end)`` sanity window, or ``None`` if no such pattern is found
    (in which case the caller skips the date-sanity filter entirely rather
    than guessing)."""
    match = _FILENAME_MONTH_RE.search(path)
    if match is None:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.DateOffset(months=1)
    buffer = pd.Timedelta(days=_DATE_SANITY_BUFFER_DAYS)
    return start - buffer, end + buffer


def read_raw_parquet(
    path_glob: str, zone_centroids: pd.DataFrame, max_trips: int | None = None, seed: int = 42
) -> pd.DataFrame:
    """Normalize NYC TLC yellow-cab monthly parquet file(s) to the
    standard trajectory point shape.

    Args:
        path_glob: Glob matching one or more monthly parquet files (e.g.
            ``data/raw/nyc_tlc/*.parquet``).
        zone_centroids: Output of `load_zone_centroids` -- maps
            ``PULocationID``/``DOLocationID`` to an approximate lon/lat.
        max_trips: If given, take a uniform random sample of this many
            surviving trips across ALL matched files combined (an
            earlier version instead took "the first N
            trips in file order", which for a single 2024-01 file meant
            the temporally-earliest ~3.4 days only -- not even a full
            week, no day-of-week signal at all, which would have biased the
            cross-dataset comparisons).
            Every file is read+filtered in full regardless (parquet
            reads are already vectorized/fast -- ~17s measured for the
            full 2.96M-row 2024-01 file -- so there
            is no need to stop early the way `read_porto_polyline`'s
            much more expensive per-row CSV parse does); ``None`` keeps
            every surviving trip, no sampling.
        seed: Seed for the sampling RNG, for reproducibility.

    Returns:
        DataFrame with columns ``trajectory_id, lon, lat, timestamp``,
        exactly 2 rows (pickup, dropoff) per surviving trip. Trips are
        dropped when: dropoff is not strictly after pickup (bad/zero-
        duration records), either endpoint's zone id has no centroid
        (zone 264 "Unknown"/265 "N/A" are not real shapefile polygons), or
        either endpoint's timestamp falls outside a generous buffer around
        the file's own nominal month (see `_expected_month_bounds` --
        catches rare meter-clock-fault rows, e.g. a handful of real
        2024-01 rows dated 2002-12-31).
    """
    paths = sorted(glob.glob(path_glob))
    if not paths:
        raise FileNotFoundError(f"no files matched {path_glob!r}")

    pu_centroids = zone_centroids.rename(
        columns={"LocationID": "PULocationID", "lon": "pu_lon", "lat": "pu_lat"}
    )
    do_centroids = zone_centroids.rename(
        columns={"LocationID": "DOLocationID", "lon": "do_lon", "lat": "do_lat"}
    )

    survivor_frames: list[pd.DataFrame] = []
    n_dropped_bad_duration = 0
    n_dropped_bad_date = 0
    n_dropped_unknown_zone = 0

    # Every matched file is processed fully -- no early stopping on
    # `max_trips` -- so the pool sampled below spans every downloaded
    # month evenly, not just whichever file(s) happened to be read first.
    for path in paths:
        df = pd.read_parquet(
            path,
            columns=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID"],
        )

        bounds = _expected_month_bounds(path)
        if bounds is not None:
            lo, hi = bounds
            valid_date = df["tpep_pickup_datetime"].between(lo, hi) & df["tpep_dropoff_datetime"].between(lo, hi)
            n_dropped_bad_date += int((~valid_date).sum())
            df = df[valid_date]

        valid_duration = df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]
        n_dropped_bad_duration += int((~valid_duration).sum())
        df = df[valid_duration]

        n_before_zone_filter = len(df)
        df = df.merge(pu_centroids, on="PULocationID", how="inner")
        df = df.merge(do_centroids, on="DOLocationID", how="inner")
        n_dropped_unknown_zone += n_before_zone_filter - len(df)

        if not df.empty:
            survivor_frames.append(df)

    if not survivor_frames:
        logger.info(
            "read_raw_parquet: 0 trips loaded, %d dropped (bad date), %d dropped (bad duration), "
            "%d dropped (unknown zone) from %d file(s)",
            n_dropped_bad_date, n_dropped_bad_duration, n_dropped_unknown_zone, len(paths),
        )
        return pd.DataFrame(columns=["trajectory_id", "lon", "lat", "timestamp"])

    survivors = pd.concat(survivor_frames, ignore_index=True)
    if max_trips is not None and len(survivors) > max_trips:
        survivors = survivors.sample(n=max_trips, random_state=seed).reset_index(drop=True)

    logger.info(
        "read_raw_parquet: %d trips loaded, %d dropped (bad date), %d dropped (bad duration), "
        "%d dropped (unknown zone) from %d file(s)",
        len(survivors), n_dropped_bad_date, n_dropped_bad_duration, n_dropped_unknown_zone, len(paths),
    )

    ids = range(len(survivors))
    return pd.DataFrame({
        "trajectory_id": list(ids) + list(ids),
        "lon": list(survivors["pu_lon"]) + list(survivors["do_lon"]),
        "lat": list(survivors["pu_lat"]) + list(survivors["do_lat"]),
        "timestamp": list(survivors["tpep_pickup_datetime"]) + list(survivors["tpep_dropoff_datetime"]),
    })
