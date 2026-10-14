"""Dataset-validity audit and corrected T-Drive pool preparation.

This script deliberately stops before workload generation or estimator runs
when no locally available third genuine trajectory dataset qualifies.  That is
the explicit decision rule of the audit, not an experiment failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.load_real_dataset import (  # noqa: E402
    PORTO_BBOX,
    TDRIVE_BBOX,
    TRIP_MAX_GAP,
    TRIP_MAX_SPAN,
    _TDRIVE_FILE_COUNT_SAFETY_MARGIN,
    _TDRIVE_TRIPS_PER_FILE,
    _subsample_trajectories,
    load_tdrive,
)
from src.data_module.mobilitydb_loader import filter_bbox, read_raw_csv  # noqa: E402
from src.data_module.tdrive_sampling import (  # noqa: E402
    TDRIVE_DATASET_SEED,
    TDRIVE_SELECTION_POLICY,
    enumerate_tdrive_sources,
)
from src.data_module.trip_segmenter import segment_into_trips  # noqa: E402
from src.eval_module.scalability import make_nested_pool  # noqa: E402
from src.results_io import write_results_json  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "run" / "real_experiments" / "task7_dataset_validity"
SCALES = (100_000, 250_000, 500_000, 1_000_000)
LEGACY_DATASET_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_rows(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((",".join(str(value) for value in row) + "\n").encode())
    return digest.hexdigest()


def _quantiles(values: pd.Series | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {}
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _trip_lengths_km(frame: pd.DataFrame) -> np.ndarray:
    ordered = frame.sort_values(["trajectory_id", "timestamp"])
    same = ordered["trajectory_id"].eq(ordered["trajectory_id"].shift())
    lat1 = np.radians(ordered["lat"].shift().to_numpy(dtype=float))
    lat2 = np.radians(ordered["lat"].to_numpy(dtype=float))
    dlat = lat2 - lat1
    dlon = np.radians(
        ordered["lon"].to_numpy(dtype=float) - ordered["lon"].shift().to_numpy(dtype=float)
    )
    hav = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    distance = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
    distance[~same.to_numpy()] = 0.0
    return pd.Series(distance).groupby(ordered["trajectory_id"].to_numpy()).sum().to_numpy()


def _origin_ami(frame: pd.DataFrame, n_spatial_bins: int = 10) -> float:
    origins = (
        frame.sort_values(["trajectory_id", "timestamp"])
        .groupby("trajectory_id", sort=False).first()
    )
    lon = origins["lon"].to_numpy()
    lat = origins["lat"].to_numpy()
    hour = origins["timestamp"].dt.hour.to_numpy()
    lon_bin = np.digitize(lon, np.linspace(lon.min(), lon.max(), n_spatial_bins + 1))
    lat_bin = np.digitize(lat, np.linspace(lat.min(), lat.max(), n_spatial_bins + 1))
    cells = lon_bin * (n_spatial_bins + 1) + lat_bin
    time_bin = np.digitize(hour, np.linspace(0, 24, 25))
    return float(adjusted_mutual_info_score(cells, time_bin))


def describe_trajectories(frame: pd.DataFrame) -> dict[str, Any]:
    grouped = frame.groupby("trajectory_id", sort=False)
    starts = grouped["timestamp"].min()
    ends = grouped["timestamp"].max()
    point_counts = grouped.size()
    taxis = int(frame["source_taxi_id"].nunique()) if "source_taxi_id" in frame else None
    hours = (
        frame.sort_values(["trajectory_id", "timestamp"])
        .groupby("trajectory_id", sort=False).first()["timestamp"].dt.hour
        .value_counts().reindex(range(24), fill_value=0).sort_index()
    )
    bbox = [
        float(frame["lon"].min()), float(frame["lon"].max()),
        float(frame["lat"].min()), float(frame["lat"].max()),
    ]
    return {
        "source_taxi_count": taxis,
        "trajectory_count": int(frame["trajectory_id"].nunique()),
        "point_count": int(len(frame)),
        "duration_seconds": _quantiles((ends - starts).dt.total_seconds()),
        "point_count_per_trajectory": _quantiles(point_counts),
        "trip_length_km": _quantiles(_trip_lengths_km(frame)),
        "spatial_coverage": {
            "bbox_wgs84": bbox,
            "bbox_degree_area": float((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])),
        },
        "temporal_coverage": {
            "start": frame["timestamp"].min().isoformat(),
            "end": frame["timestamp"].max().isoformat(),
            "span_seconds": float(
                (frame["timestamp"].max() - frame["timestamp"].min()).total_seconds()
            ),
        },
        "trip_origin_hour_of_day_counts": {str(key): int(value) for key, value in hours.items()},
        "origin_space_time_adjusted_mutual_information": _origin_ami(frame),
    }


def load_legacy_lexicographic_tdrive(cfg, limit: int) -> tuple[pd.DataFrame, dict]:
    """Reproduce the removed loader policy for descriptive audit only."""
    import glob

    all_paths = sorted(glob.glob(cfg.raw_path))
    n_files = min(
        len(all_paths),
        math.ceil(limit / _TDRIVE_TRIPS_PER_FILE * _TDRIVE_FILE_COUNT_SAFETY_MARGIN),
    )
    paths = all_paths[:n_files]
    frames = []
    for path in paths:
        frame = read_raw_csv(path, dict(cfg.column_map), names=list(cfg.raw_column_names))
        if not frame.empty:
            frame["source_taxi_id"] = int(Path(path).stem)
            frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw = filter_bbox(raw, TDRIVE_BBOX)
    raw = segment_into_trips(raw, max_trip_span=TRIP_MAX_SPAN, max_gap=TRIP_MAX_GAP)
    raw = _subsample_trajectories(raw, limit, LEGACY_DATASET_SEED)
    return raw, {
        "policy": "lexicographically sorted path prefix (removed; descriptive audit only)",
        "seed_for_post_segmentation_trajectory_sample": LEGACY_DATASET_SEED,
        "selected_source_file_count": len(paths),
        "selected_source_taxi_ids": sorted(int(Path(path).stem) for path in paths),
    }


def _inventory_candidate_record() -> dict[str, Any]:
    raw = REPO_ROOT / "data" / "raw"
    nyc_path = raw / "nyc_tlc" / "yellow_tripdata_2024-01.parquet"
    porto_path = raw / "porto" / "train.csv"
    berlin = REPO_ROOT.parents[1] / ".build" / "MobilityDB" / "meos" / "examples" / "data" / "berlinmod_instants.csv"
    ais = REPO_ROOT.parents[1] / ".build" / "MobilityDB" / "meos" / "examples" / "data" / "ais_instants.csv"
    ais_longest = ais.with_name("ais_trips_longest.csv")

    nyc_schema = None
    nyc_rows = None
    if nyc_path.exists():
        parquet = pq.ParquetFile(nyc_path)
        nyc_rows = parquet.metadata.num_rows
        nyc_schema = parquet.schema.names

    candidates = [
        {
            "name": "NYC TLC yellow taxi trip records",
            "local_source_files": [str(nyc_path.relative_to(REPO_ROOT))] if nyc_path.exists() else [],
            "source_row_count": nyc_rows,
            "source_fields": nyc_schema,
            "temporal_coverage": "2024-01 (filename and existing reader sanity bounds)",
            "coordinate_system": "zone IDs; centroids derived from EPSG:2263 polygons and transformed to EPSG:4326",
            "path_resolution": "two endpoint zone centroids per trip; no measured intermediate path",
            "scalability_capacity": "millions of endpoint records, but invalid for path-intersection semantics",
            "decision": "reject",
            "reason": "No GPS/polyline path exists in the local/public representation; interpolation is artificial.",
        },
        {
            "name": "Porto taxi",
            "local_source_files": [str(porto_path.relative_to(REPO_ROOT))] if porto_path.exists() else [],
            "trajectory_count": 1_710_670,
            "temporal_coverage": "2013-07 through 2014-06",
            "coordinate_system": "EPSG:4326",
            "path_resolution": "ordered GPS polyline at reconstructable 15-second intervals",
            "scalability_capacity": "supports 1M trajectories",
            "decision": "retain as existing dataset, not a third distinct replacement",
            "reason": "Valid genuine trajectory dataset already in the final set.",
        },
        {
            "name": "MobilityDB BerlinMOD example",
            "local_source_files": [str(berlin)] if berlin.exists() else [],
            "trajectory_count": 154 if berlin.exists() else None,
            "temporal_coverage": "2020-06 example window",
            "coordinate_system": "EPSG:3857",
            "path_resolution": "dense ordered generated points",
            "scalability_capacity": "154 local trip sequences; below evaluation requirements",
            "decision": "reject",
            "reason": "Synthetic benchmark output, not a genuine measured trajectory dataset.",
        },
        {
            "name": "MobilityDB AIS example",
            "local_source_files": [str(value) for value in (ais, ais_longest) if value.exists()],
            "trajectory_count": 10 if ais_longest.exists() else None,
            "temporal_coverage": "2021-01-08 (one day)",
            "coordinate_system": "source latitude/longitude; assembled example uses projected EPSG:25832",
            "path_resolution": "ordered AIS points",
            "scalability_capacity": "10 assembled longest trajectories (5 MMSIs in instant file)",
            "decision": "reject",
            "reason": "Genuine ordered points but far too few independent trajectories for the main evaluation.",
        },
        {
            "name": "Geolife / another external GPS or AIS corpus",
            "local_source_files": [],
            "trajectory_count": None,
            "temporal_coverage": None,
            "coordinate_system": None,
            "path_resolution": "would be evaluated after acquisition",
            "scalability_capacity": "unknown until legally acquired and inventoried",
            "decision": "not available locally",
            "reason": "The audit does not invent or evaluate an unavailable replacement.",
        },
    ]
    return {
        "task": 7,
        "decision": "blocked_no_qualifying_local_third_genuine_trajectory_dataset",
        "nyc_tlc_audit": {
            "genuine_path_available": False,
            "only_pickup_dropoff_zone_ids_available": True,
            "interpolation_currently_used": (
                "read_raw_parquet maps PULocationID/DOLocationID to taxi-zone polygon centroids, "
                "emits exactly pickup and dropoff instants, and MobilityDB linearly interpolates "
                "the resulting two-instant tgeompoint"
            ),
            "validity_conclusion": (
                "This is an endpoint-derived mobility record and cannot be presented as a measured "
                "trajectory for STBOX path-intersection ground truth."
            ),
        },
        "candidates": candidates,
        "final_choice": None,
        "required_next_action": (
            "Acquire and document a legally accessible genuine ordered-path dataset, then resume "
            "the audit from workload regeneration."
        ),
        "downstream_actions_intentionally_not_run": [
            "replacement full experiments", "changed-dataset workload generation",
            "core accuracy battery", "scalability rerun", "systems sanity check",
            "manuscript consolidation and compilation",
        ],
    }


def write_decision_record(output_dir: Path) -> dict[str, Any]:
    decision = _inventory_candidate_record()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results_json(decision, output_dir / "dataset_selection_decision.json")
    lines = [
        "# Dataset-selection decision", "",
        "Status: **stopped — no qualifying third genuine trajectory dataset is available locally.**", "",
        "NYC TLC exposes pickup/dropoff timestamps and taxi-zone IDs, not a GPS/polyline path. "
        "The current reader maps the two zones to centroids and MobilityDB interpolates one line "
        "between them. That line is not a measured taxi path and is excluded from headline evidence.", "",
        "| Candidate | Local capacity | Decision | Reason |", "|---|---:|---|---|",
    ]
    for row in decision["candidates"]:
        capacity = row.get("scalability_capacity") or "unknown"
        lines.append(f"| {row['name']} | {capacity} | {row['decision']} | {row['reason']} |")
    lines += ["", "No replacement was selected. Experiments on a third dataset must wait until "
              "a legally accessible genuine ordered-path corpus is acquired and audited.", ""]
    (output_dir / "dataset_selection_decision.md").write_text("\n".join(lines))
    return decision


def prepare_corrected_tdrive(output_dir: Path, max_scale: int, audit_scale: int) -> dict[str, Any]:
    cfg = OmegaConf.load(REPO_ROOT / "conf" / "dataset" / "tdrive.yaml")
    started = time.perf_counter()
    LOGGER.info("preparing corrected randomized T-Drive pool at max scale %d", max_scale)
    frame, selection = load_tdrive(
        cfg, max_scale, TDRIVE_DATASET_SEED, return_metadata=True
    )
    frame = frame.assign(source_segment_id=frame["trajectory_id"].astype("int64"))
    pool = make_nested_pool(frame, TDRIVE_DATASET_SEED)
    prepared = output_dir / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    pool_path = prepared / f"tdrive_corrected__max_{max_scale}.parquet"
    pool.to_parquet(pool_path, index=False, compression="zstd")
    LOGGER.info("saved corrected pool: %s", pool_path)

    selected_ids_path = prepared / "selected_taxi_ids.csv"
    with selected_ids_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selection_rank", "source_taxi_id"])
        writer.writerows(enumerate(selection["selected_taxi_ids"]))

    trajectory_manifest = (
        pool.groupby("trajectory_id", sort=True)
        .agg(
            source_taxi_id=("source_taxi_id", "first"),
            source_segment_id=("source_segment_id", "first"),
            point_count=("timestamp", "size"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        ).reset_index()
    )
    if trajectory_manifest["trajectory_id"].duplicated().any():
        raise RuntimeError("duplicate trajectories in corrected T-Drive manifest")
    manifest_path = prepared / "trajectory_manifest.parquet"
    trajectory_manifest.to_parquet(manifest_path, index=False, compression="zstd")

    scale_rows = []
    previous_ids: set[int] | None = None
    for scale in SCALES:
        if scale > max_scale:
            continue
        subset = trajectory_manifest[trajectory_manifest["trajectory_id"] < scale]
        ids = set(int(value) for value in subset["trajectory_id"])
        expected = min(scale, len(trajectory_manifest))
        if len(ids) != expected or ids != set(range(expected)):
            raise RuntimeError(f"T-Drive scale {scale} is not a unique dense prefix")
        proper = None if previous_ids is None else previous_ids < ids
        if previous_ids is not None and not proper:
            raise RuntimeError(f"T-Drive scale nesting failed at {scale}")
        scale_rows.append({
            "scale": scale,
            "trajectory_count": len(ids),
            "trajectory_identity_sha256": _sha256_rows(
                subset[["source_taxi_id", "source_segment_id"]].itertuples(index=False, name=None)
            ),
            "rank_sha256": _sha256_rows((value,) for value in subset["trajectory_id"]),
            "proper_superset_of_previous": proper,
        })
        previous_ids = ids
        LOGGER.info("validated nested T-Drive scale %d", scale)

    new_audit_frame = pool[pool["trajectory_id"] < audit_scale].copy()
    legacy_frame, legacy_meta = load_legacy_lexicographic_tdrive(cfg, audit_scale)
    LOGGER.info("computing old-vs-new descriptive audit at scale %d", audit_scale)
    audit = {
        "comparison_scale": audit_scale,
        "selection_is_descriptive_only_not_performance_based": True,
        "old_lexicographic_prefix": {
            "selection": legacy_meta,
            "distribution": describe_trajectories(legacy_frame),
        },
        "corrected_randomized": {
            "selection": {
                "seed": TDRIVE_DATASET_SEED,
                "policy": TDRIVE_SELECTION_POLICY,
                "selected_source_file_count_for_max_pool": selection["selected_file_count"],
                "contributing_source_taxi_count_at_audit_scale": int(
                    new_audit_frame["source_taxi_id"].nunique()
                ),
            },
            "distribution": describe_trajectories(new_audit_frame),
        },
    }
    write_results_json(audit, output_dir / "tdrive_old_vs_randomized_distribution.json")

    selection.pop("selected_paths", None)
    selection["selected_taxi_ids_artifact"] = str(selected_ids_path.relative_to(REPO_ROOT))
    selection["selected_taxi_ids_file_sha256"] = _sha256_file(selected_ids_path)
    selection["trajectory_manifest_artifact"] = str(manifest_path.relative_to(REPO_ROOT))
    selection["trajectory_manifest_file_sha256"] = _sha256_file(manifest_path)
    selection["prepared_pool_artifact"] = str(pool_path.relative_to(REPO_ROOT))
    selection["prepared_pool_file_sha256"] = _sha256_file(pool_path)
    selection["scale_manifests"] = scale_rows
    selection["preparation_seconds"] = time.perf_counter() - started
    selection["nested_validation"] = (
        "PASS: unique dense whole-trajectory ranks; 100k < 250k < 500k < 1M"
    )
    write_results_json(selection, output_dir / "tdrive_corrected_sampling_manifest.json")
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-tdrive", action="store_true")
    parser.add_argument("--max-scale", type=int, default=1_000_000)
    parser.add_argument("--audit-scale", type=int, default=100_000)
    args = parser.parse_args()

    decision = write_decision_record(args.output_dir)
    sources, excluded, matched = enumerate_tdrive_sources(
        str(REPO_ROOT / "data" / "raw" / "tdrive" / "extracted" / "*.txt")
    )
    inventory = {
        "matched_source_files": matched,
        "valid_nonempty_numeric_id_files": len(sources),
        "excluded_files": excluded,
        "population_taxi_id_min": min(value.taxi_id for value in sources),
        "population_taxi_id_max": max(value.taxi_id for value in sources),
    }
    write_results_json(inventory, args.output_dir / "tdrive_source_inventory.json")
    if args.prepare_tdrive:
        prepare_corrected_tdrive(args.output_dir, args.max_scale, args.audit_scale)

    status = {
        "task": 7,
        "complete": False,
        "stop_reason": decision["decision"],
        "tdrive_prepared": bool(args.prepare_tdrive),
        "headline_experiments_run": False,
    }
    write_results_json(status, args.output_dir / "task7_status.json")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
