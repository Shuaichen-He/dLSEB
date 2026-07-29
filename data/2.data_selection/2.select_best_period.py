"""
select_best_period.py
=====================
Select the optimal 10-day period from valid periods (summary CSV) and
generate training data CSV files.

Selection strategy (auto-detects data quality):
  - Clean sites (Ejin, Huazhaizi): pick the period with the largest max(Hs)/max(LE) ratio
  - Noisy sites (Shenshawo): first take top-5 periods with fewest Hs+H2O NaNs,
    then pick the best by ratio, then impute missing values

Usage:
  python 2.select_best_period.py --summary <summary_csv> --data <data_csv> --out <output_dir>

Output (4 CSV files):
  - meteo_var.csv    : DR, DLR_Cor, Ta_5m, Press, UR, ULR_Cor, Rain
  - soil.csv         : Gs_1, Gs_2, Gs_3, Ts_0cm, Ts_2cm, Ts_4cm, Ts_10cm, Ts_20cm, Ms_2cm
  - ec_var.csv       : Wnd, H2O, Hs, LE, Ustar, L
  - air_constants.csv: rho (approximated via ideal-gas law from Press and Ta_5m)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Output column definitions (matching data_loader.py) ──
METEO_COLS = ["DR", "DLR_Cor", "Ta_5m", "Press", "UR", "ULR_Cor", "Rain"]
SOIL_COLS = [
    "Gs_1",
    "Gs_2",
    "Gs_3",
    "Ts_0cm",
    "Ts_2cm",
    "Ts_4cm",
    "Ts_10cm",
    "Ts_20cm",
    "Ts_40cm",
    "Ts_60cm",
    "Ts_100cm",
    "Ms_2cm",
]
EC_COLS = ["Wnd", "H2O", "Hs", "LE", "Ustar", "L"]

# ── NaN handling: max number of top periods when pre-filtering by NaN count ──
TOP_N_BY_NAN = 5


# ═══════════════════════════════════════════════════════════
#  Imputation utilities
# ═══════════════════════════════════════════════════════════

def interpolate_series(series: pd.Series) -> pd.Series:
    """
    Impute NaN values in a series:
      - Edge NaN (leading / trailing) → fill with 0.0
      - Interior NaN → linear interpolation

    Returns
    -------
    pd.Series
        Imputed copy of the input series.
    """
    s = series.copy()
    if s.notna().all():
        return s

    # 1) All NaN → fill 0
    if s.isna().all():
        return s.fillna(0.0)

    # 2) Leading NaN → 0.0
    first_valid = s.first_valid_index()
    if first_valid is not None and first_valid > s.index[0]:
        s.loc[: first_valid - 1] = 0.0

    # 3) Trailing NaN → 0.0
    last_valid = s.last_valid_index()
    if last_valid is not None and last_valid < s.index[-1]:
        s.loc[last_valid + 1:] = 0.0

    # 4) Interior NaN → linear interpolation
    s = s.interpolate(method="linear", limit_direction="both")

    # 5) Safety fallback
    s = s.fillna(0.0)
    return s


# ═══════════════════════════════════════════════════════════
#  Final-periods log
# ═══════════════════════════════════════════════════════════

_LOG_HEADER = (
    f"{'Site':<12} {'Year':>6}  {'Start Date':<12}  {'End Date':<12}  Strategy\n"
    f"{'-' * 75}\n"
)


def write_log_entry(log_path: Path, site: str, year: int,
                    start_date, end_date, strategy: str) -> None:
    """Append one selected-period record to *log_path*.

    If the file does not exist (or is empty), a header line is written first.
    """
    write_header = (not log_path.exists()) or (log_path.stat().st_size == 0)
    with open(log_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                f"Final selected periods | last updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'=' * 75}\n"
            )
            f.write(_LOG_HEADER)
        f.write(
            f"{site:<12} {year:>6}  "
            f"{str(start_date.date()):<12}  {str(end_date.date()):<12}  "
            f"{strategy}\n"
        )


# ═══════════════════════════════════════════════════════════
#  Period ranking logic
# ═══════════════════════════════════════════════════════════

def compute_ratio(period_data: pd.DataFrame) -> float:
    """Compute max(non-NaN Hs) / max(non-NaN LE) ratio for a period.

    Returns -inf if LE or Hs is entirely NaN.
    """
    hs = period_data["Hs"].dropna()
    le = period_data["LE"].dropna()
    if hs.empty or le.empty:
        return -float("inf")
    max_hs = hs.max()
    max_le = le.max()
    if max_le == 0:
        return float("inf") if max_hs > 0 else -float("inf")
    return max_hs / max_le


def rank_by_ratio(summary: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Rank all periods by max(Hs)/max(LE) ratio in descending order."""
    records = []
    for _, row in summary.iterrows():
        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"])
        mask = (data["time"] >= start) & (data["time"] < end + pd.Timedelta(days=1))
        period = data[mask]

        hs = period["Hs"].dropna()
        le = period["LE"].dropna()
        max_hs = float(hs.max()) if not hs.empty else np.nan
        max_le = float(le.max()) if not le.empty else np.nan
        ratio = max_hs / max_le if (pd.notna(max_le) and max_le != 0) else np.nan

        records.append({
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "year": row["year"],
            "max_Hs": max_hs,
            "max_LE": max_le,
            "ratio": ratio,
            "n_rows": len(period),
        })

    return pd.DataFrame(records).sort_values(
        "ratio", ascending=False, na_position="last"
    ).reset_index(drop=True)


def rank_by_nan_count(
    summary: pd.DataFrame, data: pd.DataFrame
) -> pd.DataFrame:
    """Rank all periods by Hs + H2O NaN count in ascending order (fewer NaNs rank higher)."""
    records = []
    for _, row in summary.iterrows():
        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"])
        mask = (data["time"] >= start) & (data["time"] < end + pd.Timedelta(days=1))
        period = data[mask]

        h2o_nan = int(period["H2O"].isna().sum())
        hs_nan = int(period["Hs"].isna().sum())

        records.append({
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "year": row["year"],
            "H2O_NaN": h2o_nan,
            "Hs_NaN": hs_nan,
            "total_NaN": h2o_nan + hs_nan,
            "n_rows": len(period),
        })

    return pd.DataFrame(records).sort_values(
        ["total_NaN", "Hs_NaN", "H2O_NaN"]
    ).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════
#  Output utilities
# ═══════════════════════════════════════════════════════════

def write_outputs(best_data: pd.DataFrame, out_dir: Path) -> None:
    """Write the best-period data to meteo_var.csv / soil.csv / ec_var.csv / air_constants.csv.

    Parameters
    ----------
    best_data : pd.DataFrame
        The selected best-period data.
    out_dir : Path
        Output directory for the CSV files.
    """
    # ── Check required columns ──
    for label, cols in [
        ("meteo_var.csv", METEO_COLS),
        ("soil.csv", SOIL_COLS),
        ("ec_var.csv", EC_COLS),
    ]:
        missing = [c for c in cols if c not in best_data.columns]
        if missing:
            print(f"[Error] Missing columns for {label}: {missing}")
    if any(
        [c for c in METEO_COLS if c not in best_data.columns]
        + [c for c in SOIL_COLS if c not in best_data.columns]
        + [c for c in EC_COLS if c not in best_data.columns]
    ):
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Write CSVs ──
    best_data[METEO_COLS].to_csv(out_dir / "meteo_var.csv", index=False)
    best_data[SOIL_COLS].to_csv(out_dir / "soil.csv", index=False)
    best_data[EC_COLS].to_csv(out_dir / "ec_var.csv", index=False)

    n = len(best_data)
    print(f"\n  meteo_var.csv   ({n} rows)  {METEO_COLS}")
    print(f"  soil.csv        ({n} rows)  {SOIL_COLS}")
    print(f"  ec_var.csv      ({n} rows)  {EC_COLS}")

    # ── air_constants.csv ──
    Rd = 287.058
    p = best_data["Press"]
    t = best_data["Ta_5m"]
    valid = p.notna() & t.notna()
    rho = pd.Series(np.nan, index=best_data.index)
    if valid.any():
        T_k = t + 273.15
        rho[valid] = (p[valid] * 100) / (T_k[valid] * Rd)
        rho = rho.fillna(rho[valid].mean()).round(4)
    else:
        rho = rho.fillna(1.15).round(4)

    pd.DataFrame({"rho": rho}).to_csv(out_dir / "air_constants.csv", index=False)
    print(f"  air_constants.csv ({n} rows)  rho_mean={float(rho.mean()):.4f}")

    print(f"\nAll files saved to {out_dir}/")


# ═══════════════════════════════════════════════════════════
#  Main workflow
# ═══════════════════════════════════════════════════════════

def main():
    """Main entry point: select the best period and write training CSVs."""
    parser = argparse.ArgumentParser(
        description="Select the best period from valid periods and output training data CSVs.\n"
        "Auto-detects H2O/Hs NaNs and applies the corresponding selection & imputation strategy."
    )
    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to valid_10d_periods_summary.csv",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to valid_10d_periods.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory (meteo_var.csv, soil.csv, etc. will be saved here)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to final_periods.log (default: <out_dir>/../final_periods.log)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    data_path = Path(args.data)
    out_dir = Path(args.out)
    log_path = Path(args.log) if args.log else out_dir.parent / "final_periods.log"
    site_name = out_dir.name  # e.g. "Ejin", "Huazhaizi", "Shenshawo"

    # ── 1. Read data ──
    if not summary_path.exists():
        print(f"[Error] Summary file not found: {summary_path}")
        sys.exit(1)
    if not data_path.exists():
        print(f"[Error] Data file not found: {data_path}")
        sys.exit(1)

    summary = pd.read_csv(summary_path)
    print(f"Read summary: {len(summary)} valid periods")

    data = pd.read_csv(data_path)
    data["time"] = pd.to_datetime(data["time"])
    print(f"Read data: {len(data):,} rows, {len(data.columns)} columns")

    # ── 2. Check for H2O / Hs NaNs ──
    h2o_nan_total = int(data["H2O"].isna().sum()) if "H2O" in data.columns else 0
    hs_nan_total = int(data["Hs"].isna().sum()) if "Hs" in data.columns else 0
    has_nan = h2o_nan_total > 0 or hs_nan_total > 0

    # ═══════════════════════════════════════════════════════
    #  Branch A: Clean data — no H2O / Hs NaNs
    # ═══════════════════════════════════════════════════════
    if not has_nan:
        print("\nData has no H2O/Hs NaNs — selecting directly by max(Hs)/max(LE) ratio.")

        ratio_df = rank_by_ratio(summary, data)

        print(f"\n{'='*80}")
        print("Period max(Hs)/max(LE) ratios (top 20):")
        print(
            f"{'Rank':<5} {'start_date':<12} {'end_date':<12} "
            f"{'max_Hs':>10} {'max_LE':>10} {'ratio':>10} {'rows':>8} {'year':>6}"
        )
        print("-" * 80)
        for i, r in ratio_df.head(20).iterrows():
            print(
                f"{i+1:<5} {r['start_date']:<12} {r['end_date']:<12} "
                f"{r['max_Hs']:>10.3f} {r['max_LE']:>10.3f} "
                f"{r['ratio']:>10.4f} {int(r['n_rows']):>8} {int(r['year']):>6}"
            )

        # Select best period
        best = ratio_df.iloc[0]
        if pd.isna(best["ratio"]):
            print("\n[Error] No period with a computable ratio.")
            sys.exit(1)

        start_date = pd.Timestamp(best["start_date"])
        end_date = pd.Timestamp(best["end_date"])

        print(f"\n{'='*80}")
        print(f"Best period: {start_date.date()} ~ {end_date.date()}")
        print(f"   max(Hs) = {best['max_Hs']:.4f}, max(LE) = {best['max_LE']:.4f}")
        print(f"   ratio = max(Hs)/max(LE) = {best['ratio']:.4f}")
        print(f"   {int(best['n_rows'])} rows @30min, year: {int(best['year'])}")

        # Clip best period
        mask = (data["time"] >= start_date) & (
            data["time"] < end_date + pd.Timedelta(days=1)
        )
        best_data = data[mask].sort_values("time").reset_index(drop=True)
        print(f"   After clipping: {len(best_data)} rows")

        # ── Log selected period ──
        write_log_entry(
            log_path, site_name, int(best["year"]),
            start_date, end_date, "max(Hs)/max(LE)",
        )

        write_outputs(best_data, out_dir)
        return

    # ═══════════════════════════════════════════════════════
    #  Branch B: Data with H2O / Hs NaNs — filter by NaN count first
    # ═══════════════════════════════════════════════════════
    print(f"\n{'!'*80}")
    print("Warning: H2O / Hs NaNs present (these were ignored during the initial filtering step):")
    print(f"    - H2O column: {h2o_nan_total} NaN(s)")
    print(f"    - Hs  column: {hs_nan_total} NaN(s)")
    print("\nStrategy:")
    print(f"  1. Sort periods by Hs + H2O NaN count ascending, take top {TOP_N_BY_NAN} (or fewer)")
    print(f"  2. Within those {TOP_N_BY_NAN} periods, pick the one with the largest max(Hs)/max(LE) ratio")
    print("  3. Impute NaNs in the selected period:")
    print("       - H2O: fill with column mean")
    print("       - Hs:  edge NaNs → 0, interior NaNs → linear interpolation")
    print(f"{'!'*80}")

    # ── Step 1: Sort by NaN count ──
    nan_df = rank_by_nan_count(summary, data)

    print(f"\n{'='*80}")
    print("Period H2O + Hs NaN counts (top 20):")
    print(
        f"{'Rank':<5} {'start_date':<12} {'end_date':<12} "
        f"{'H2O_NaN':>8} {'Hs_NaN':>8} {'total_NaN':>10} {'year':>6}"
    )
    print("-" * 80)
    for i, r in nan_df.head(20).iterrows():
        print(
            f"{i+1:<5} {r['start_date']:<12} {r['end_date']:<12} "
            f"{int(r['H2O_NaN']):>8} {int(r['Hs_NaN']):>8} "
            f"{int(r['total_NaN']):>10} {int(r['year']):>6}"
        )

    # ── Step 2: Take top-N periods with fewest NaNs ──
    n_top = min(TOP_N_BY_NAN, len(nan_df))
    top_nan = nan_df.head(n_top)
    print(f"\nTaking top {n_top} periods with fewest NaNs, then selecting by ratio.")

    # ── Step 3: Compute ratio within the top-N and select best ──
    top_summary = summary[
        summary["start_date"].isin(top_nan["start_date"])
        & summary["end_date"].isin(top_nan["end_date"])
    ]
    ratio_df = rank_by_ratio(top_summary, data)

    print(f"\n{'='*80}")
    print(f"max(Hs)/max(LE) ratios for top {n_top} low-NaN periods:")
    print(
        f"{'Rank':<5} {'start_date':<12} {'end_date':<12} "
        f"{'max_Hs':>10} {'max_LE':>10} {'ratio':>10} {'rows':>8} {'year':>6}"
    )
    print("-" * 80)
    for i, r in ratio_df.iterrows():
        print(
            f"{i+1:<5} {r['start_date']:<12} {r['end_date']:<12} "
            f"{r['max_Hs']:>10.3f} {r['max_LE']:>10.3f} "
            f"{r['ratio']:>10.4f} {int(r['n_rows']):>8} {int(r['year']):>6}"
        )

    best = ratio_df.iloc[0]
    if pd.isna(best["ratio"]):
        print("\n[Error] No computable ratio among these periods.")
        sys.exit(1)

    start_date = pd.Timestamp(best["start_date"])
    end_date = pd.Timestamp(best["end_date"])

    # Look up NaN details for the selected period
    best_nan_row = nan_df[
        (nan_df["start_date"] == best["start_date"])
        & (nan_df["end_date"] == best["end_date"])
    ].iloc[0]

    print(f"\n{'='*80}")
    print(f"Best period: {start_date.date()} ~ {end_date.date()}")
    print(
        f"   H2O NaN={int(best_nan_row['H2O_NaN'])}, "
        f"Hs NaN={int(best_nan_row['Hs_NaN'])}"
    )
    print(f"   max(Hs) = {best['max_Hs']:.4f}, max(LE) = {best['max_LE']:.4f}")
    print(f"   ratio = max(Hs)/max(LE) = {best['ratio']:.4f}")
    print(f"   {int(best['n_rows'])} rows @30min, year: {int(best['year'])}")

    # ── Step 4: Clip best period ──
    mask = (data["time"] >= start_date) & (
        data["time"] < end_date + pd.Timedelta(days=1)
    )
    best_data = data[mask].sort_values("time").reset_index(drop=True)
    print(f"   After clipping: {len(best_data)} rows")

    # ── Step 5: Impute H2O / Hs NaNs ──
    before_h2o = int(best_data["H2O"].isna().sum())
    before_hs = int(best_data["Hs"].isna().sum())

    if before_h2o > 0:
        h2o_mean = best_data["H2O"].mean()
        best_data["H2O"] = best_data["H2O"].fillna(h2o_mean)
        print(f"  H2O imputation: NaN {before_h2o} → 0, filled with mean {h2o_mean:.4f}")

    if before_hs > 0:
        best_data["Hs"] = interpolate_series(best_data["Hs"])
        after_hs = int(best_data["Hs"].isna().sum())
        print(f"  Hs  imputation: NaN {before_hs} → {after_hs} "
              f"(edges→0, interior→linear)")

    # ── Step 6: Write outputs ──
    print(f"\nOutput directory: {out_dir}")

    # ── Log selected period ──
    write_log_entry(
        log_path, site_name, int(best["year"]),
        start_date, end_date,
        "top-5 fewest-NaN → max(Hs)/max(LE) + impute",
    )

    write_outputs(best_data, out_dir)


if __name__ == "__main__":
    main()
