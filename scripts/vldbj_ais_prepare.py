"""Prepare Danish AIS (DMA) ship trajectories, one day at a time.

Pipeline, following the MobilityDB/PyMEOS AIS examples for cleaning:
1. Stream one daily zip from the DMA bucket (aisdata.ais.dk) to a scratch file.
2. Read the CSV inside the zip in chunks (no extraction to disk) and keep only:
   MMSI != 0; latitude in [40.18, 84.17]; longitude in [-16.1, 32.88];
   0 <= SOG <= 1022; deduplicated on (MMSI, timestamp).
3. Drop stationary points: SOG < 0.5 kn, or navigational status "Moored" /
   "At anchor". This removes port stops from the movement stream.
4. Downsample to at most one point per --resample-seconds per MMSI.
5. Write a per-day parquet (mmsi, timestamp, lon, lat) and delete the zip.

``--segment`` then pools all prepared days and cuts trips: a new trip starts at a
gap of more than 30 minutes, which includes every port stop of 30 minutes or more
because stationary points were removed in step 3. Trips with fewer than
--min-points points or shorter than --min-minutes are discarded. The earliest-
starting --n-trips trips become the pool (trajectory_id, timestamp, lon, lat) in
the format the loaders expect.

``--day-report`` prints trips/day and points/trip so the number of days can be
chosen before more downloads (download budget: 5 GB).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_module.trip_segmenter import segment_into_trips  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "http://aisdata.ais.dk.s3.eu-central-1.amazonaws.com"
COLS = {"# Timestamp": "ts", "MMSI": "mmsi", "Latitude": "lat", "Longitude": "lon",
        "Navigational status": "status", "SOG": "sog"}
STATIONARY = {"Moored", "At anchor"}


def day_url(day: date) -> str:
    return f"{BUCKET}/{day.year}/aisdk-{day.isoformat()}.zip"


def prepare_day(day: date, work: Path, out: Path, resample_s: int) -> dict:
    target = out / f"ais_{day.isoformat()}.parquet"
    if target.exists():
        return json.loads((out / f"ais_{day.isoformat()}.json").read_text())
    zpath = work / f"aisdk-{day.isoformat()}.zip"
    started = time.perf_counter()
    with urllib.request.urlopen(day_url(day), timeout=120) as resp, zpath.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=8 << 20)
    download_s = time.perf_counter() - started
    zip_bytes = zpath.stat().st_size
    sha = hashlib.sha256()
    with zpath.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            sha.update(block)
    started = time.perf_counter()
    kept, raw_rows = [], 0
    with zipfile.ZipFile(zpath) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as fh:
            reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
                                 usecols=list(COLS), chunksize=2_000_000, low_memory=False)
            for chunk in reader:
                raw_rows += len(chunk)
                c = chunk.rename(columns=COLS)
                c = c[(c.mmsi != 0) & c.lat.between(40.18, 84.17) & c.lon.between(-16.1, 32.88)
                      & c.sog.between(0, 1022)]
                c = c[(c.sog >= 0.5) & ~c.status.isin(STATIONARY)]
                c = c.assign(timestamp=pd.to_datetime(c.ts, format="%d/%m/%Y %H:%M:%S", errors="coerce"))
                kept.append(c.loc[c.timestamp.notna(), ["mmsi", "timestamp", "lon", "lat"]])
    zpath.unlink()
    df = pd.concat(kept, ignore_index=True).drop_duplicates(["mmsi", "timestamp"])
    df["bucket"] = df.timestamp.dt.floor(f"{resample_s}s")
    df = (df.sort_values(["mmsi", "timestamp"]).drop_duplicates(["mmsi", "bucket"])
          .drop(columns="bucket").reset_index(drop=True))
    df.to_parquet(target, index=False)
    info = {"day": day.isoformat(), "source_url": day_url(day), "zip_bytes": zip_bytes,
            "zip_sha256": sha.hexdigest(), "resample_seconds": resample_s, "raw_rows": raw_rows,
            "kept_points": len(df), "vessels": int(df.mmsi.nunique()),
            "download_s": download_s, "parse_s": time.perf_counter() - started,
            "parquet_bytes": target.stat().st_size}
    (out / f"ais_{day.isoformat()}.json").write_text(json.dumps(info, indent=1))
    return info


def pooled_trips(out: Path, min_points: int, min_minutes: float):
    days = sorted(out.glob("ais_20*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in days], ignore_index=True)
    df = df.rename(columns={"mmsi": "trajectory_id"}).drop_duplicates(["trajectory_id", "timestamp"])
    trips = segment_into_trips(df, max_trip_span=timedelta(days=3650), max_gap=timedelta(minutes=30))
    g = trips.groupby("trajectory_id").timestamp.agg(["count", "min", "max"])
    ok = g[(g["count"] >= min_points) & ((g["max"] - g["min"]).dt.total_seconds() >= min_minutes * 60)]
    return days, trips, g, ok


def segment(out: Path, n_trips: int, min_points: int, min_minutes: float) -> dict:
    days, trips, g, ok = pooled_trips(out, min_points, min_minutes)
    chosen = ok.sort_values("min").head(n_trips).index
    pool = trips[trips.trajectory_id.isin(chosen)].copy()
    remap = {old: new for new, old in enumerate(sorted(chosen, key=lambda i: (g.loc[i, "min"], i)))}
    pool["trajectory_id"] = pool.trajectory_id.map(remap).astype(np.int64)
    pool = pool.sort_values(["trajectory_id", "timestamp"]).reset_index(drop=True)
    path = out / "ais_pool.parquet"
    pool.to_parquet(path, index=False)
    spans = (ok.loc[chosen, "max"] - ok.loc[chosen, "min"]).dt.total_seconds() / 3600
    return {"days": [p.stem for p in days], "candidate_trips": int(len(ok)), "selected_trips": len(chosen),
            "points": int(len(pool)), "points_per_trip_median": float(ok.loc[chosen, "count"].median()),
            "span_hours_median": float(spans.median()), "span_hours_p95": float(spans.quantile(0.95)),
            "time_range": [str(pool.timestamp.min()), str(pool.timestamp.max())],
            "bbox": [float(pool.lon.min()), float(pool.lon.max()), float(pool.lat.min()), float(pool.lat.max())],
            "pool_bytes": path.stat().st_size}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat, default=date(2025, 1, 6))
    ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "data/ais_prepared")
    ap.add_argument("--work", type=Path, default=ROOT / "data/ais_tmp")
    ap.add_argument("--resample-seconds", type=int, default=60)
    ap.add_argument("--until-trips", type=int, default=None,
                    help="stop once the pooled days yield at least this many valid trips")
    ap.add_argument("--check-every", type=int, default=5)
    ap.add_argument("--segment", action="store_true")
    ap.add_argument("--n-trips", type=int, default=100_000)
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--min-minutes", type=float, default=5.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    if args.segment:
        print(json.dumps(segment(args.out, args.n_trips, args.min_points, args.min_minutes), indent=1))
        return
    for k in range(args.days):
        info = prepare_day(args.start + timedelta(days=k), args.work, args.out, args.resample_seconds)
        print(json.dumps(info))
        sys.stdout.flush()
        if args.until_trips and (k + 1) % args.check_every == 0:
            _days, _trips, _g, ok = pooled_trips(args.out, args.min_points, args.min_minutes)
            print(json.dumps({"after_days": k + 1, "valid_trips": int(len(ok))}))
            sys.stdout.flush()
            if len(ok) >= args.until_trips:
                break


if __name__ == "__main__":
    main()
