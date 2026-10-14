#!/usr/bin/env bash
# Rebuild everything the released files allow and check it against the paper.
#
#   bash reproduce.sh
#
# No database is needed. The script works in a temporary copy and modifies nothing here.
# It reruns the analysis on the released per-query data (Porto, synthetic), compares the
# rebuilt tables with the released results, and classifies every number of the paper
# (rebuilt/macro_status.csv). T-Drive and AIS per-query data are not distributed; the
# numbers that need them are listed, and the README explains how to rebuild them from the
# original downloads.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/check_release.py

cat <<'EOF'

Not rebuilt here, because they need T-Drive or AIS per-query data:
  - the numbers marked "needs withheld data" in rebuilt/macro_status.csv;
  - the six figures in paper/figures/ (each has T-Drive panels);
  - for Porto rows, the bootstrap intervals of the non-empty-query table and the p-values
    of the paired tests, which are computed jointly with the T-Drive and AIS rows.
To rebuild them: load T-Drive and AIS from the original downloads, recompute and verify the
T-Drive answers with scripts/rebuild_answers.py, run the experiment scripts (README,
"Re-running the experiments"), then scripts/vldbj_generate_numbers.py and
scripts/vldbj_figures.py.
EOF
