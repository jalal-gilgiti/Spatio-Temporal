"""Load trajectory point data (any source) into MobilityDB as tgeompoint rows."""
import glob
import logging

import pandas as pd
import psycopg
from psycopg import sql

logger = logging.getLogger(__name__)


def read_raw_csv(
    path: str, column_map: dict[str, str], names: list[str] | None = None
) -> pd.DataFrame:
    """Normalize a raw dataset CSV to the standard trajectory point shape.

    Args:
        path: Path to the raw CSV file.
        column_map: Mapping from raw column names to the standard names
            ``trajectory_id, lon, lat, timestamp``.
        names: Explicit raw column names, for a CSV with no header row
            (T-Drive's per-taxi release: ``taxi_id,timestamp,lon,lat`` with
            no header line -- verified against the actual downloaded files,
            see `scripts/download_real_datasets.sh`). When given, the file
            is read with ``header=None`` and these names applied first,
            then `column_map` renames as usual. ``None`` (default)
            preserves the original header-CSV behavior.

    Returns:
        DataFrame with columns ``trajectory_id, lon, lat, timestamp``.
    """
    df = pd.read_csv(path, header=None if names else "infer", names=names)
    df = df.rename(columns=column_map)[["trajectory_id", "lon", "lat", "timestamp"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # An empty raw file has no data to infer numeric dtype from, so pandas
    # leaves lon/lat/trajectory_id as `object` for that one file; when many
    # per-file frames are later concatenated (`read_raw_csv_glob`), even a
    # single such object-dtype frame upcasts the whole merged column to
    # `object` (confirmed against real T-Drive data: 21 of
    # 10357 per-taxi files are 0-row). Force numeric dtypes explicitly here
    # so every caller gets clean float64/int64 regardless of what any one
    # source file looked like in isolation.
    df["lon"] = pd.to_numeric(df["lon"])
    df["lat"] = pd.to_numeric(df["lat"])
    df["trajectory_id"] = pd.to_numeric(df["trajectory_id"])
    return df


def read_raw_csv_glob(
    path_glob: str, column_map: dict[str, str], names: list[str] | None = None
) -> pd.DataFrame:
    """Read and concatenate every file matching `path_glob` via `read_raw_csv`.

    T-Drive's raw release is one headerless per-taxi text file per taxi
    (thousands of files), not the single-CSV shape `read_raw_csv` alone
    handles -- this is the multi-file counterpart used for that dataset.

    Args:
        path_glob: Glob pattern matching one file per source unit (e.g.
            ``data/raw/tdrive/extracted/*.txt``).
        column_map: Forwarded to `read_raw_csv` for each matched file.
        names: Forwarded to `read_raw_csv` for each matched file.

    Returns:
        DataFrame with columns ``trajectory_id, lon, lat, timestamp``,
        concatenated across every matched file.

    Raises:
        FileNotFoundError: If `path_glob` matches no files.
    """
    paths = sorted(glob.glob(path_glob))
    if not paths:
        raise FileNotFoundError(f"no files matched {path_glob!r}")
    frames = [read_raw_csv(p, column_map, names=names) for p in paths]
    n_empty = sum(1 for f in frames if len(f) == 0)
    if n_empty:
        logger.warning(
            "read_raw_csv_glob: %d of %d matched files were empty (0 rows)", n_empty, len(paths)
        )
    return pd.concat(frames, ignore_index=True)


def filter_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Drop points whose (lon, lat) falls outside a sane geographic bbox.

    Both real datasets loaded by this project carry
    a small fraction of GPS-fix-failure points (a classic ``(0, 0)``
    sentinel, or a wild reading thousands of km from the city in
    question): ~1% of T-Drive's ~17.7M points and ~1% of a 20k-trip Porto
    sample (confirmed empirically). Segmentation
    (`segment_into_trips`) is purely time-based and would not catch these
    -- a single garbage point sits silently inside an otherwise normal
    trip's `tgeompoint`, wrecking that trajectory's bounding box and, via
    `atStbox`, corrupting ground-truth cardinality for any query whose
    space-time window happens to intersect the point's (bogus)
    interpolated position. Filtering by each city's real bbox before
    loading removes this class of corruption. `bbox` is
    ``(min_lon, max_lon, min_lat, max_lat)``, matching `generate_workload`'s
    convention.

    Args:
        df: DataFrame with a ``lon``/``lat`` column (any other columns,
            e.g. ``trajectory_id``/``timestamp``, pass through untouched).
        bbox: ``(min_lon, max_lon, min_lat, max_lat)`` bounds to keep.

    Returns:
        A copy of `df` containing only in-bbox rows.
    """
    min_lon, max_lon, min_lat, max_lat = bbox
    in_bbox = df["lon"].between(min_lon, max_lon) & df["lat"].between(min_lat, max_lat)
    n_dropped = int((~in_bbox).sum())
    if n_dropped:
        logger.warning(
            "filter_bbox: dropped %d of %d point(s) (%.2f%%) outside bbox %s",
            n_dropped, len(df), 100.0 * n_dropped / len(df), bbox,
        )
    return df[in_bbox].copy()


def load_trajectories(conn: psycopg.Connection, table: str, df: pd.DataFrame) -> int:
    """Group points by trajectory_id, build one tgeompoint sequence per
    trajectory, and insert into ``table``.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Target table name, expected to have columns
            ``(trajectory_id int, traj tgeompoint)``.
        df: DataFrame with columns ``trajectory_id, lon, lat, timestamp``.

    Returns:
        Number of trajectories (i.e. rows) inserted.
    """
    insert_stmt = sql.SQL(
        "INSERT INTO {table} (trajectory_id, traj) VALUES (%s, %s::tgeompoint)"
    ).format(table=sql.Identifier(table))

    inserted = 0
    with conn.cursor() as cur:
        for traj_id, group in df.sort_values("timestamp").groupby("trajectory_id"):
            n_before = len(group)
            group = group.drop_duplicates(subset="timestamp", keep="first")
            n_dropped = n_before - len(group)
            if n_dropped:
                logger.warning(
                    "trajectory_id=%s: dropped %d row(s) with duplicate timestamps",
                    traj_id,
                    n_dropped,
                )
            instants = ", ".join(
                f"POINT({row.lon} {row.lat})@{row.timestamp.isoformat()}"
                for row in group.itertuples()
            )
            cur.execute(insert_stmt, (int(traj_id), f"[{instants}]"))
            inserted += 1
    logger.info("loaded %d trajectories into %s", inserted, table)
    return inserted
