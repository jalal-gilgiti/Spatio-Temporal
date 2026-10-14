"""Splits raw (trajectory_id, lon, lat, timestamp) point streams into
trip-scale segments (spec §1/§2 D1: ground-truth semantics require rows
that are bounded trips, not multi-day per-entity histories, or a bbox
covering an entire city renders spatio-temporal correlation unmeasurable)."""
from datetime import timedelta

import numpy as np
import pandas as pd


def segment_into_trips(
    df: pd.DataFrame,
    max_trip_span: timedelta,
    max_gap: timedelta,
) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    # The original implementation performed two ``.loc`` scalar lookups for
    # every point.  That is equivalent but takes tens of minutes on T-Drive's
    # 17M points.  Work per source trajectory and split it first at mandatory
    # gap boundaries; within each gap-contiguous run, ``searchsorted`` finds
    # the first timestamp strictly beyond the greedy segment start + max span.
    # ``side='right'`` is important: a point exactly on the max-span boundary
    # remains in the segment, matching the original ``span > max_trip_span``.
    ordered = df.sort_values(["trajectory_id", "timestamp"])
    out_frames = []
    next_id = 0
    max_gap_ns = int(max_gap.total_seconds() * 1_000_000_000)
    max_span_ns = int(max_trip_span.total_seconds() * 1_000_000_000)
    for _, group in ordered.groupby("trajectory_id", sort=True):
        group = group.reset_index(drop=True)
        times = group["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        gap_starts = np.flatnonzero(np.diff(times) > max_gap_ns) + 1
        run_bounds = np.concatenate(([0], gap_starts, [len(group)]))
        ids = np.empty(len(group), dtype=np.int64)
        for run_start, run_end in zip(run_bounds[:-1], run_bounds[1:], strict=True):
            segment_start = int(run_start)
            while segment_start < run_end:
                segment_end = int(np.searchsorted(
                    times,
                    times[segment_start] + max_span_ns,
                    side="right",
                    sorter=None,
                ))
                segment_end = min(max(segment_end, segment_start + 1), int(run_end))
                ids[segment_start:segment_end] = next_id
                next_id += 1
                segment_start = segment_end
        group = group.copy()
        group["trajectory_id"] = ids
        out_frames.append(group)

    return pd.concat(out_frames, ignore_index=True)
