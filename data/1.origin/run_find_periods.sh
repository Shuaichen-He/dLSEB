#!/bin/bash
# ============================================================
# run_find_periods.sh
# Batch-run 1.find_valid_periods.py for all three stations,
# finding qualifying 10-day valid periods and saving CSVs
# to each station folder.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
SCRIPT="1.find_valid_periods.py"

echo "============================================"
echo "  Batch find valid periods — 3 stations total"
echo "============================================"
echo ""

# ---- Station 1: Ejin Desert station ----
echo ">>> [1/3] Ejin Desert station"
"$PYTHON" "$SCRIPT" \
    "Ejin Desert station" \
    --out "Ejin Desert station" \
    --name "Ejin"
echo ""

# ---- Station 2: Huazhaizi desert station ----
echo ">>> [2/3] Huazhaizi desert station"
"$PYTHON" "$SCRIPT" \
    "Huazhaizi desert station" \
    --out "Huazhaizi desert station" \
    --name "Huazhaizi"
echo ""

# ---- Station 3: Shenshawo sandy desert ----
# Lower data quality: additionally allow missing values for sensible heat (Hs) and water vapour (H2O)
echo ">>> [3/3] Shenshawo sandy desert"
"$PYTHON" "$SCRIPT" \
    "Shenshawo sandy desert" \
    --out "Shenshawo sandy desert" \
    --name "Shenshawo" \
    --ignore-nan-cols Hs H2O
echo ""

echo "============================================"
echo "  All done! Output files:"
echo "    Ejin Desert station/Ejin_valid_10d_periods.csv"
echo "    Ejin Desert station/Ejin_valid_10d_periods_summary.csv"
echo "    Huazhaizi desert station/Huazhaizi_valid_10d_periods.csv"
echo "    Huazhaizi desert station/Huazhaizi_valid_10d_periods_summary.csv"
echo "    Shenshawo sandy desert/Shenshawo_valid_10d_periods.csv"
echo "    Shenshawo sandy desert/Shenshawo_valid_10d_periods_summary.csv"
echo "============================================"
