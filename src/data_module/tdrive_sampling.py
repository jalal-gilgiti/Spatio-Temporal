"""Reproducible source-taxi selection for the T-Drive release.

T-Drive stores one taxi per headerless text file.  Sampling a prefix of the
lexicographically sorted filenames is therefore a sample of file names, not a
defensible sample of vehicles.  This module centralizes the sampling policy:
enumerate valid source files, sort by numeric taxi id only to establish a
stable population, then select a prefix of one seeded random permutation.
"""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


TDRIVE_DATASET_SEED = 70707
TDRIVE_SELECTION_POLICY = "numeric-id population; numpy PCG64 permutation; seeded prefix"


@dataclass(frozen=True)
class TDriveSource:
    taxi_id: int
    path: str
    size_bytes: int


@dataclass(frozen=True)
class TDriveSourceSelection:
    seed: int
    policy: str
    matched_file_count: int
    valid_file_count: int
    selected_file_count: int
    population_sha256: str
    selected_ids_sha256: str
    selected_taxi_ids: tuple[int, ...]
    selected_paths: tuple[str, ...]
    excluded_files: tuple[dict[str, str], ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _fingerprint_ids(ids: list[int] | tuple[int, ...]) -> str:
    payload = json.dumps(list(ids), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def enumerate_tdrive_sources(path_glob: str) -> tuple[list[TDriveSource], list[dict[str, str]], int]:
    """Return the valid one-file-per-taxi population in numeric-id order.

    Empty files and non-numeric filenames cannot yield a trajectory and are
    retained in the exclusion audit.  Duplicate numeric taxi ids are rejected
    because they would make the sampling unit ambiguous.
    """
    matched = [Path(value) for value in glob.glob(path_glob)]
    if not matched:
        raise FileNotFoundError(f"no files matched {path_glob!r}")

    valid: list[TDriveSource] = []
    excluded: list[dict[str, str]] = []
    seen: set[int] = set()
    for path in matched:
        try:
            taxi_id = int(path.stem)
        except ValueError:
            excluded.append({"path": str(path), "reason": "non_numeric_filename"})
            continue
        size = path.stat().st_size
        if size == 0:
            excluded.append({"path": str(path), "reason": "empty_file"})
            continue
        if taxi_id in seen:
            raise ValueError(f"duplicate T-Drive taxi id {taxi_id}: {path}")
        seen.add(taxi_id)
        valid.append(TDriveSource(taxi_id, str(path), size))
    valid.sort(key=lambda value: value.taxi_id)
    return valid, sorted(excluded, key=lambda value: value["path"]), len(matched)


def select_tdrive_sources(
    path_glob: str,
    *,
    seed: int = TDRIVE_DATASET_SEED,
    n_files: int | None = None,
) -> TDriveSourceSelection:
    """Select source taxis without replacement from a deterministic population."""
    sources, excluded, matched_count = enumerate_tdrive_sources(path_glob)
    if not sources:
        raise ValueError("T-Drive source population contains no non-empty numeric-id files")
    count = len(sources) if n_files is None else min(int(n_files), len(sources))
    if count <= 0:
        raise ValueError("n_files must be positive")
    order = np.random.default_rng(seed).permutation(len(sources))[:count]
    selected = [sources[int(index)] for index in order]
    population_ids = [value.taxi_id for value in sources]
    selected_ids = [value.taxi_id for value in selected]
    return TDriveSourceSelection(
        seed=seed,
        policy=TDRIVE_SELECTION_POLICY,
        matched_file_count=matched_count,
        valid_file_count=len(sources),
        selected_file_count=len(selected),
        population_sha256=_fingerprint_ids(population_ids),
        selected_ids_sha256=_fingerprint_ids(selected_ids),
        selected_taxi_ids=tuple(selected_ids),
        selected_paths=tuple(value.path for value in selected),
        excluded_files=tuple(excluded),
    )
