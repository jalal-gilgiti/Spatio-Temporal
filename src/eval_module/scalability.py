"""Primitives for nested data-size and training-workload scaling."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from src.data_module.workload_generator import STPredicate, WorkloadQuery, stbox_literal
from src.eval_module.accuracy import q_error


def predicate_dict(predicate: STPredicate) -> dict[str, Any]:
    return {
        "min_lon": predicate.min_lon, "max_lon": predicate.max_lon,
        "min_lat": predicate.min_lat, "max_lat": predicate.max_lat,
        "start_time": predicate.start_time.isoformat(),
        "end_time": predicate.end_time.isoformat(),
        "k": predicate.k, "knn_center": predicate.knn_center,
    }


def predicate_from_dict(raw: dict[str, Any]) -> STPredicate:
    return STPredicate(
        float(raw["min_lon"]), float(raw["max_lon"]),
        float(raw["min_lat"]), float(raw["max_lat"]),
        datetime.fromisoformat(raw["start_time"]),
        datetime.fromisoformat(raw["end_time"]),
        raw.get("k"), tuple(raw["knn_center"]) if raw.get("knn_center") else None,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalized_bounds(df: pd.DataFrame) -> tuple[tuple[float, ...], tuple[datetime, datetime]]:
    bbox = (
        float(df["lon"].min()), float(df["lon"].max()),
        float(df["lat"].min()), float(df["lat"].max()),
    )
    time_range = (
        df["timestamp"].min().to_pydatetime(),
        df["timestamp"].max().to_pydatetime(),
    )
    return bbox, time_range


def make_nested_pool(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Randomly rank whole trajectories once so every prefix is nested."""
    ids = np.asarray(sorted(df["trajectory_id"].unique()))
    permutation = np.random.default_rng(seed).permutation(ids)
    rank = pd.Series(np.arange(len(permutation), dtype=np.int64), index=permutation)
    result = df.copy()
    result["trajectory_id"] = result["trajectory_id"].map(rank)
    if result["trajectory_id"].isna().any():
        raise RuntimeError("failed to assign a nested rank to every trajectory")
    result["trajectory_id"] = result["trajectory_id"].astype(np.int64)
    return result.sort_values(["trajectory_id", "timestamp"]).reset_index(drop=True)


def generate_predicates(
    workload: str,
    n_queries: int,
    seed: int,
    bbox: tuple[float, float, float, float],
    time_range: tuple[datetime, datetime],
    extent: float,
    window_hours: float,
) -> list[STPredicate]:
    """Generate labels-free predicates once for reuse at every nested scale."""
    if workload not in {"fixed_extent", "mixed_extent"}:
        raise ValueError(f"unknown workload {workload!r}")
    rng = np.random.default_rng(seed)
    min_lon, max_lon, min_lat, max_lat = bbox
    t0, t1 = time_range
    total_seconds = (t1 - t0).total_seconds()
    predicates = []
    for _ in range(n_queries):
        if workload == "fixed_extent":
            width = min(extent, max_lon - min_lon)
            height = min(extent, max_lat - min_lat)
            duration = min(window_hours * 3600.0, total_seconds)
            lon0 = float(rng.uniform(min_lon, max_lon - width)) if width < max_lon - min_lon else min_lon
            lat0 = float(rng.uniform(min_lat, max_lat - height)) if height < max_lat - min_lat else min_lat
            offset = float(rng.uniform(0, total_seconds - duration)) if duration < total_seconds else 0.0
        else:
            lon0, lon1 = sorted(rng.uniform(min_lon, max_lon, 2))
            lat0, lat1 = sorted(rng.uniform(min_lat, max_lat, 2))
            width, height = float(lon1 - lon0), float(lat1 - lat0)
            offset = float(rng.uniform(0, total_seconds))
            available = max(total_seconds - offset, 0.0)
            if available <= 60.0:
                duration = available
            else:
                duration = float(np.exp(rng.uniform(np.log(60.0), np.log(min(7200.0, available)))))
        start = t0 + timedelta(seconds=offset)
        predicates.append(STPredicate(
            lon0, lon0 + width, lat0, lat0 + height,
            start, start + timedelta(seconds=duration),
        ))
    return predicates


def exact_indexable_sql(predicate: STPredicate, alias: str = "t") -> str:
    box = stbox_literal(predicate)
    return (
        f"{alias}.traj && stbox '{box}' AND "
        f"atStbox({alias}.traj, stbox '{box}') IS NOT NULL"
    )


def label_predicates(
    conn: psycopg.Connection, table: str, predicates: Iterable[STPredicate]
) -> tuple[list[WorkloadQuery], list[float]]:
    queries, latencies = [], []
    table_name = sql.Identifier(table).as_string(None)
    with conn.cursor() as cur:
        for predicate in predicates:
            started = time.perf_counter()
            cur.execute(
                f"SELECT count(*) FROM {table_name} t WHERE "
                f"{exact_indexable_sql(predicate)}"
            )
            truth = int(cur.fetchone()[0])
            latencies.append((time.perf_counter() - started) * 1000.0)
            queries.append(WorkloadQuery(predicate, truth))
    return queries, latencies


def _trajectory_payloads(df: pd.DataFrame):
    for trajectory_id, group in df.groupby("trajectory_id", sort=True):
        group = group.drop_duplicates(subset="timestamp", keep="first")
        instants = ", ".join(
            f"POINT({row.lon} {row.lat})@{row.timestamp.isoformat()}"
            for row in group.itertuples()
        )
        yield int(trajectory_id), f"[{instants}]"


def bulk_load_trajectories(
    conn: psycopg.Connection, table: str, df: pd.DataFrame, batch_size: int = 2_000
) -> int:
    """Load trajectories in one transaction using bounded executemany batches."""
    statement = sql.SQL(
        "INSERT INTO {table} (trajectory_id, traj) VALUES (%s, %s::tgeompoint)"
    ).format(table=sql.Identifier(table))
    inserted = 0
    batch = []
    with conn.transaction():
        with conn.cursor() as cur:
            for payload in _trajectory_payloads(df):
                batch.append(payload)
                if len(batch) >= batch_size:
                    cur.executemany(statement, batch)
                    inserted += len(batch)
                    batch.clear()
            if batch:
                cur.executemany(statement, batch)
                inserted += len(batch)
    return inserted


@dataclass
class PeakMemory:
    start_rss_bytes: int
    peak_rss_bytes: int

    @property
    def delta_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)


class PeakRSSMonitor:
    """Sample process RSS while native NumPy/PyTorch training code runs."""

    def __init__(self, interval_seconds: float = 0.02):
        self.interval_seconds = interval_seconds
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.start_rss = _current_rss_bytes()
        self.peak_rss = self.start_rss

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak_rss = max(self.peak_rss, _current_rss_bytes())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_rss = max(self.peak_rss, _current_rss_bytes())

    @property
    def result(self) -> PeakMemory:
        return PeakMemory(self.start_rss, self.peak_rss)


def _current_rss_bytes() -> int:
    """Read Linux process RSS without adding a benchmark-only dependency."""
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmRSS is unavailable in /proc/self/status")


def prediction_summary(predictions: list[dict[str, Any]]) -> dict[str, float]:
    errors = np.asarray([row["q_error"] for row in predictions], dtype=float)
    latencies = np.asarray([row["latency_ms"] for row in predictions], dtype=float)
    return {
        "median_q_error": float(np.median(errors)),
        "p90_q_error": float(np.percentile(errors, 90)),
        "p95_q_error": float(np.percentile(errors, 95)),
        "max_q_error": float(np.max(errors)),
        "inference_p50_ms": float(np.median(latencies)),
        "inference_p95_ms": float(np.percentile(latencies, 95)),
    }


def evaluate_estimator(estimator, queries: list[WorkloadQuery]) -> tuple[list[dict], dict]:
    if queries:
        estimator.estimate(queries[0].predicate)
    predictions = []
    for index, query in enumerate(queries):
        started = time.perf_counter()
        predicted = float(estimator.estimate(query.predicate))
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not math.isfinite(predicted):
            raise RuntimeError(f"non-finite prediction at query {index}")
        predictions.append({
            "query_index": index,
            "true_cardinality": int(query.true_cardinality),
            "predicted_cardinality": predicted,
            "q_error": float(q_error(query.true_cardinality, predicted)),
            "latency_ms": latency_ms,
        })
    return predictions, prediction_summary(predictions)
