#!/bin/bash
# ============================================================
# run_select_best.sh
# Batch-run 2.select_best_period.py for all three stations,
# selecting the best period from step-1 valid-period CSVs
# and outputting meteo_var.csv / soil.csv / ec_var.csv /
# air_constants.csv to each station subfolder.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
SCRIPT="2.select_best_period.py"
ORIGIN_DIR="../1.origin"

echo "============================================"
echo "  Batch select best periods — 3 stations total"
echo "============================================"
echo ""

# ---- Station 1: Ejin Desert station ----
echo ">>> [1/3] Ejin Desert station"
"$PYTHON" "$SCRIPT" \
    --summary "$ORIGIN_DIR/Ejin Desert station/Ejin_valid_10d_periods_summary.csv" \
    --data    "$ORIGIN_DIR/Ejin Desert station/Ejin_valid_10d_periods.csv" \
    --out     "Ejin"
echo ""

# ---- Station 2: Huazhaizi desert station ----
echo ">>> [2/3] Huazhaizi desert station"
"$PYTHON" "$SCRIPT" \
    --summary "$ORIGIN_DIR/Huazhaizi desert station/Huazhaizi_valid_10d_periods_summary.csv" \
    --data    "$ORIGIN_DIR/Huazhaizi desert station/Huazhaizi_valid_10d_periods.csv" \
    --out     "Huazhaizi"
echo ""

# ---- Station 3: Shenshawo sandy desert ----
# Note: This site has H2O/Hs NaNs; the script auto-applies the NaN-filtering & imputation branch.
echo ">>> [3/3] Shenshawo sandy desert"
"$PYTHON" "$SCRIPT" \
    --summary "$ORIGIN_DIR/Shenshawo sandy desert/Shenshawo_valid_10d_periods_summary.csv" \
    --data    "$ORIGIN_DIR/Shenshawo sandy desert/Shenshawo_valid_10d_periods.csv" \
    --out     "Shenshawo"
echo ""

echo "============================================"
echo "  All done! Output files:"
echo "    Ejin/meteo_var.csv"
echo "    Ejin/soil.csv"
echo "    Ejin/ec_var.csv"
echo "    Ejin/air_constants.csv"
echo "    Huazhaizi/meteo_var.csv"
echo "    Huazhaizi/soil.csv"
echo "    Huazhaizi/ec_var.csv"
echo "    Huazhaizi/air_constants.csv"
echo "    Shenshawo/meteo_var.csv"
echo "    Shenshawo/soil.csv"
echo "    Shenshawo/ec_var.csv"
echo "    Shenshawo/air_constants.csv"
echo "============================================"
