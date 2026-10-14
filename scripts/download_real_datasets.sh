#!/usr/bin/env bash
# Downloads T-Drive, Porto, and NYC TLC raw data into data/raw/<dataset>/
# (gitignored -- see .gitignore). Idempotent: each step skips work whose
# output already exists on disk, so re-running after an interruption
# resumes rather than re-downloading everything.
#
# Usage (from repo root):
#   bash scripts/download_real_datasets.sh [tdrive|porto|nyc_tlc|zones|all]
# Default target is "all".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"

# Microsoft's CDN (Akamai NetStorage) 403s a bare `curl` with no User-Agent
# / Referer, so every T-Drive
# request below sends both.
UA="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
TDRIVE_REFERER="https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/"

# ---------------------------------------------------------------------------
# T-Drive: Beijing taxi GPS traces (Microsoft Research). Public release,
# split across 14 zip parts named 01..014 (note the inconsistent zero-
# padding -- 01-09 then 010-014 -- confirmed against the actual working
# URLs, not a typo). ~139MB compressed total, unzips to per-taxi text
# files (`<taxi_id>.txt`, one file per taxi, no header,
# `taxi_id,timestamp,lon,lat` per line -- verified against real downloaded
# files, see `src/data_module/mobilitydb_loader.py`'s
# `read_raw_csv_glob` / conf/dataset/tdrive.yaml's `raw_column_names`).
download_tdrive() {
  local dir="$RAW/tdrive"
  mkdir -p "$dir/extracted"
  local parts=(01 02 03 04 05 06 07 08 09 010 011 012 013 014)
  for p in "${parts[@]}"; do
    local zip="$dir/${p}.zip"
    if [[ -f "$zip" ]]; then
      echo "[tdrive] $zip already downloaded, skipping"
    else
      echo "[tdrive] downloading part $p"
      curl -sL -A "$UA" -e "$TDRIVE_REFERER" \
        "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/${p}.zip" \
        -o "$zip"
    fi
    unzip -o -q "$zip" -d "$dir/extracted"
  done
  echo "[tdrive] done: $(find "$dir/extracted" -name '*.txt' | wc -l) per-taxi files extracted"
}

# ---------------------------------------------------------------------------
# Porto taxi (ECML/PKDD 2015 competition data). Kaggle API credentials are
# not configured in this environment (no ~/.kaggle/kaggle.json, `kaggle`
# CLI absent), so this uses the UCI Machine
# Learning Repository's official mirror of the same competition data
# instead (dataset 339, "Taxi Service Trajectory - Prediction Challenge,
# ECML PKDD 2015"): https://archive.ics.uci.edu/dataset/339/ . The zip
# contains train.csv.zip nested one level in; both levels are extracted
# here so `data/raw/porto/train.csv` ends up directly usable.
download_porto() {
  local dir="$RAW/porto"
  mkdir -p "$dir"
  if [[ -f "$dir/train.csv" ]]; then
    echo "[porto] $dir/train.csv already present, skipping"
    return
  fi
  local outer="$dir/porto_uci.zip"
  echo "[porto] downloading UCI mirror (~530MB)"
  curl -sL \
    "https://archive.ics.uci.edu/static/public/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015.zip" \
    -o "$outer"
  unzip -o -q "$outer" train.csv.zip -d "$dir"
  unzip -o -q "$dir/train.csv.zip" -d "$dir"
  rm -f "$outer" "$dir/train.csv.zip"
  echo "[porto] done: $(wc -l < "$dir/train.csv") lines in train.csv"
}

# ---------------------------------------------------------------------------
# NYC TLC yellow-cab trip records, official public parquet files (no auth).
# Pass one or more YYYY-MM month strings as extra args to override the
# default (2024-01 only, ~50MB / ~2.96M trips -- kept to one month to keep
# load time reasonable).
download_nyc_tlc() {
  local dir="$RAW/nyc_tlc"
  mkdir -p "$dir"
  local months=("${@:-2024-01}")
  for m in "${months[@]}"; do
    local out="$dir/yellow_tripdata_${m}.parquet"
    if [[ -f "$out" ]]; then
      echo "[nyc_tlc] $out already downloaded, skipping"
      continue
    fi
    echo "[nyc_tlc] downloading $m"
    curl -sL "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_${m}.parquet" -o "$out"
  done
}

# NYC TLC taxi zone shapefile: every currently-served parquet vintage
# reports PULocationID/DOLocationID zone ids rather than raw lat/lon
# (verified against both 2024-01 and 2015-01 -- there is no
# older-vintage escape hatch via this endpoint), so
# `src/data_module/parquet_reader.py` needs each zone's polygon centroid
# as an approximate coordinate. Official shapefile, same CDN as the trip
# data, EPSG:2263 (NAD83 NY State Plane Long Island, US feet).
download_nyc_zones() {
  local dir="$RAW/nyc_tlc/taxi_zones"
  mkdir -p "$dir"
  if [[ -f "$dir/taxi_zones.shp" ]]; then
    echo "[nyc_zones] $dir/taxi_zones.shp already present, skipping"
    return
  fi
  local zip="$RAW/nyc_tlc/taxi_zones.zip"
  echo "[nyc_zones] downloading taxi zone shapefile"
  curl -sL "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip" -o "$zip"
  unzip -o -q "$zip" -d /tmp/taxi_zones_extract_$$
  # The zip may nest a taxi_zones/ subdir or place files at top level
  # depending on how it was produced; handle both.
  if [[ -f /tmp/taxi_zones_extract_$$/taxi_zones.shp ]]; then
    cp /tmp/taxi_zones_extract_$$/taxi_zones.* "$dir/"
  else
    cp /tmp/taxi_zones_extract_$$/*/taxi_zones.* "$dir/"
  fi
  rm -rf /tmp/taxi_zones_extract_$$ "$zip"
  echo "[nyc_zones] done: $(ls "$dir" | wc -l) shapefile component files"
}

target="${1:-all}"
case "$target" in
  tdrive) download_tdrive ;;
  porto) download_porto ;;
  nyc_tlc) shift || true; download_nyc_tlc "$@"; download_nyc_zones ;;
  zones) download_nyc_zones ;;
  all)
    download_tdrive
    download_porto
    download_nyc_tlc
    download_nyc_zones
    ;;
  *)
    echo "usage: $0 [tdrive|porto|nyc_tlc|zones|all]" >&2
    exit 1
    ;;
esac
