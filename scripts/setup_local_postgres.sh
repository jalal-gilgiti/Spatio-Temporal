#!/usr/bin/env bash
# scripts/setup_local_postgres.sh
#
# Idempotent setup of the user-space PostgreSQL/MobilityDB server used for all
# experiments: a conda-forge PostgreSQL 17 + PostGIS environment, with
# MobilityDB v1.3.0 and pg_hint_plan built from source. No Docker and no root
# access are needed.
#
# State lives under ROOT (default: the repository root):
#   - conda env:  $ROOT/.conda-envs/spatiotemp_pg
#   - data dir:   $ROOT/.pgdata
#   - build dir:  $ROOT/.build
#
# Safe to re-run at any point: every stage is guarded so a partially-completed
# run can be resumed without erroring.
#
# Usage: [ROOT=/path] [CONDA_SH=/path/to/conda.sh] bash scripts/setup_local_postgres.sh

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_DIR="${ROOT}/.conda-envs/spatiotemp_pg"
PGDATA="${ROOT}/.pgdata"
BUILD_DIR="${ROOT}/.build"
PG_PORT=5434
DB_NAME="mobilitydb"

MOBILITYDB_REPO="https://github.com/MobilityDB/MobilityDB.git"
MOBILITYDB_TAG="v1.3.0"
PG_HINT_PLAN_REPO="https://github.com/ossc-db/pg_hint_plan"
PG_HINT_PLAN_BRANCH="PG17"

echo "==> Step 1/6: conda environment"
source "${CONDA_SH}"

if conda env list | grep -q "${CONDA_ENV_DIR}"; then
    echo "    conda env already exists at ${CONDA_ENV_DIR}, skipping creation"
else
    echo "    creating conda env at ${CONDA_ENV_DIR}"
    conda create -y -p "${CONDA_ENV_DIR}" -c conda-forge \
        postgresql=17 postgis cmake json-c gsl flex bison
fi

conda activate "${CONDA_ENV_DIR}"

echo "==> Step 2/6: initdb + start PostgreSQL on port ${PG_PORT}"
if [ -s "${PGDATA}/PG_VERSION" ]; then
    echo "    data dir already initialized at ${PGDATA}, skipping initdb"
else
    echo "    running initdb into ${PGDATA}"
    mkdir -p "${PGDATA}"
    initdb -D "${PGDATA}" -U postgres
    # trust auth for local + loopback TCP: a private, single-user instance.
    # Use password authentication instead on a shared machine.
    {
        echo "local   all             all                                     trust"
        echo "host    all             all             127.0.0.1/32            trust"
        echo "host    all             all             ::1/128                 trust"
    } >> "${PGDATA}/pg_hba.conf"
fi

if pg_isready -p "${PG_PORT}" >/dev/null 2>&1; then
    echo "    server already running on port ${PG_PORT}"
else
    echo "    starting server on port ${PG_PORT}"
    pg_ctl -D "${PGDATA}" -l "${PGDATA}/logfile" -o "-p ${PG_PORT}" start
    for _ in $(seq 1 30); do
        pg_isready -p "${PG_PORT}" >/dev/null 2>&1 && break
        sleep 1
    done
fi

echo "==> Step 3/6: build MobilityDB from tag ${MOBILITYDB_TAG}"
mkdir -p "${BUILD_DIR}"
if [ -f "${CONDA_ENV_DIR}/lib/postgresql/mobilitydb.so" ]; then
    echo "    mobilitydb.so already installed, skipping MobilityDB build"
else
    if [ -d "${BUILD_DIR}/MobilityDB/.git" ]; then
        echo "    MobilityDB repo already cloned"
    else
        git clone "${MOBILITYDB_REPO}" "${BUILD_DIR}/MobilityDB"
    fi
    git -C "${BUILD_DIR}/MobilityDB" checkout "${MOBILITYDB_TAG}"

    # NOTE: MobilityDB HEAD fails to build under GCC 14's stricter default
    # error promotion for implicit-function-declaration in vendored pgtypes
    # code -- v1.3.0 builds clean. -D_GNU_SOURCE is required because the
    # vendored snprintf.c needs strchrnul, which requires this feature-test
    # macro; normally inherited from PostgreSQL's own build system, not from
    # MobilityDB's standalone CMake build.
    mkdir -p "${BUILD_DIR}/MobilityDB/build"
    (
        cd "${BUILD_DIR}/MobilityDB/build"
        cmake .. -DCMAKE_C_FLAGS="-D_GNU_SOURCE"
        make -j"$(nproc)"
        make install
    )
fi

echo "==> Step 4/6: build pg_hint_plan from branch ${PG_HINT_PLAN_BRANCH}"
if [ -f "${CONDA_ENV_DIR}/lib/postgresql/pg_hint_plan.so" ]; then
    echo "    pg_hint_plan.so already installed, skipping pg_hint_plan build"
else
    if [ -d "${BUILD_DIR}/pg_hint_plan/.git" ]; then
        echo "    pg_hint_plan repo already cloned"
    else
        git clone -b "${PG_HINT_PLAN_BRANCH}" "${PG_HINT_PLAN_REPO}" "${BUILD_DIR}/pg_hint_plan"
    fi

    # NOTE: PG_CONFIG records conda's *build-time* toolchain paths (compiler,
    # flex, bison), which are not installed as runtime dependencies of this
    # env. Override with the system gcc and conda's own flex/bison, both
    # confirmed present/installable.
    (
        cd "${BUILD_DIR}/pg_hint_plan"
        make USE_PGXS=1 \
            PG_CONFIG="${CONDA_ENV_DIR}/bin/pg_config" \
            CC=gcc \
            FLEX="${CONDA_ENV_DIR}/bin/flex" \
            BISON="${CONDA_ENV_DIR}/bin/bison"
        make USE_PGXS=1 \
            PG_CONFIG="${CONDA_ENV_DIR}/bin/pg_config" \
            install
    )
fi

echo "==> Step 5/6: create database and role"
if psql -p "${PG_PORT}" -U postgres -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}'" | grep -q 1; then
    echo "    database ${DB_NAME} already exists"
else
    createdb -p "${PG_PORT}" -U postgres "${DB_NAME}"
fi

echo "==> Step 6/6: create extensions"
psql -p "${PG_PORT}" -U postgres -d "${DB_NAME}" -v ON_ERROR_STOP=1 <<'SQL'
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS mobilitydb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_hint_plan;
SQL
# Workload timestamps are stored without a time zone and were interpreted in
# Asia/Seoul when the experiments ran; keep that so query windows match.
psql -p "${PG_PORT}" -U postgres -d "${DB_NAME}" -v ON_ERROR_STOP=1 \
    -c "ALTER DATABASE ${DB_NAME} SET timezone TO 'Asia/Seoul'"

echo "==> Done. DSN: postgresql://postgres@localhost:${PG_PORT}/${DB_NAME}"
