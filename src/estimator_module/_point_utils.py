"""Shared point-fetching helper for histogram-style estimators."""
import numpy as np
import psycopg
from psycopg import sql


def fetch_points(conn: psycopg.Connection, table: str) -> np.ndarray:
    """Fetch every trajectory instant of ``table`` as (lon, lat, epoch_seconds) rows.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.

    Returns:
        An (n, 3) float array of (lon, lat, epoch_seconds) rows. Always
        2-D, even when the table is empty (``n == 0``), so downstream
        indexing/unpacking never sees a shapeless ``(0,)`` array.
    """
    with conn.cursor() as cur:
        query = sql.SQL(
            "SELECT ST_X(g), ST_Y(g), extract(epoch from t) "
            "FROM {table}, LATERAL unnest(instants({table}.traj)) i, "
            "LATERAL (SELECT getValue(i) AS g, getTimestamp(i) AS t) sub"
        ).format(table=sql.Identifier(table))
        cur.execute(query)
        rows = cur.fetchall()
    # columns: lon, lat, epoch_seconds; reshape defensively so an empty
    # result set still yields a properly-shaped (0, 3) array instead of
    # a shapeless (0,) array that breaks downstream indexing/unpacking.
    return np.array(rows, dtype=float).reshape(-1, 3)


def fetch_trajectory_representatives(conn: psycopg.Connection, table: str) -> np.ndarray:
    """Fetch one representative point per TRAJECTORY row of ``table``, as
    (lon, lat, epoch_seconds) rows -- not one row per instant.

    The one-row-per-trajectory representation prevents instant-count scaling, but it
    does *not* match the labels' full-path-intersection semantics: a path may enter a
    query box even when its first instant does not. Callers must disclose this origin
    approximation when comparing against trajectory-level ground truth.

    Args:
        conn: Open psycopg connection to the MobilityDB instance.
        table: Name of the table holding a ``traj tgeompoint`` column.

    Returns:
        An (n, 3) float array of (lon, lat, epoch_seconds) rows, one per
        trajectory row. Always 2-D, even when the table is empty (``n ==
        0``), so downstream indexing/unpacking never sees a shapeless
        ``(0,)`` array.
    """
    with conn.cursor() as cur:
        query = sql.SQL(
            "SELECT ST_X(g), ST_Y(g), extract(epoch from t) "
            "FROM {table}, LATERAL (SELECT startInstant({table}.traj) AS i) sub, "
            "LATERAL (SELECT getValue(i) AS g, getTimestamp(i) AS t) vals"
        ).format(table=sql.Identifier(table))
        cur.execute(query)
        rows = cur.fetchall()
    # columns: lon, lat, epoch_seconds; reshape defensively so an empty
    # result set still yields a properly-shaped (0, 3) array instead of
    # a shapeless (0,) array that breaks downstream indexing/unpacking.
    return np.array(rows, dtype=float).reshape(-1, 3)
