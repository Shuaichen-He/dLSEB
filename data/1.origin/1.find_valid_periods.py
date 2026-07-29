"""
find_valid_periods.py
=====================
Find continuous 10-day valid periods from raw AWS + EC flux/meteorological
data that satisfy the dLSEB model requirements.

Selection criteria (all four must be met simultaneously):
  1. No missing values — all variables except LE (latent heat) have no NaN
     within the 10-day period. Extra columns can be exempted via --ignore-nan-cols
     (e.g., Hs, H2O).
  2. No precipitation — Rain == 0.0 at every 30-min step within the period.
  3. Temperature floor  — minimum Ts_2cm and Ta_5m within the period > -0.5 degC.
  4. Dry soil — near-surface soil moisture Ms_2cm < 5% throughout the period.

Usage
-----
  python find_valid_periods.py <station_dir> [--out output/]

station_dir must contain AWS/ and EC/ subdirectories with multi-year .xlsx files:
  - AWS/*.xlsx  (10-min resolution; columns: TIMESTAMP, Rain, Ms_2cm, Ta_5m, Ts_2cm, DR, ...)
  - EC/*.xlsx   (30-min resolution; columns: Date/Time, Hs, LE, Wnd, H2O)

Output
------
  Two CSV files:
  - ***_valid_10d_periods_summary.csv  Period summary (start/end dates, etc.)
  - ***_valid_10d_periods.csv         30-min raw data for qualifying periods
  Each period summary is also printed to the terminal.

Design: All data is aligned to the EC 30-min time grid.
        AWS (10-min) is aggregated to 30-min via sum (Rain) / mean (others).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ======================== Constants ========================
FILL_VALUES = [-6999, -9999]  # fill/missing-value markers
WINDOW_DAYS = 10  # period length (days)
EC_STEP_MIN = 30  # EC time step (minutes)

TEMP_MIN_THRESH = 0  # minimum temperature threshold (degC)
MS_MAX_THRESH = 5.0  # near-surface soil moisture upper bound (%)

DEFAULT_EC_HEIGHT = 4.5  # default EC measurement height z (m), used for z/L → L

# EC key columns (after standardisation)
EC_COLS_NEEDED = ["Hs", "LE", "Wnd", "H2O", "Ustar", "L"]

# Extra AWS variables used by data_loader.py (besides Rain, Ms_2cm, Ta_5m, Ts_2cm).
# These columns may be absent; only existing columns are loaded.
AWS_EXTRA_COLS = [
    "DR",
    "DLR_Cor",
    "Press",
    "UR",
    "ULR_Cor",
    "Gs_1",
    "Gs_2",
    "Gs_3",
    "Ts_0cm",
    # Note: Ts_2cm is handled separately in temp_cols; do not add here to avoid duplicate columns.
    "Ts_4cm",
    "Ts_10cm",
    "Ts_20cm",
    "Ts_40cm",
    "Ts_60cm",
    "Ts_100cm",
]


# ======================== Data Loading ========================


def _collect_xlsx(path: str) -> list[str]:
    """Collect xlsx file paths from a file or directory."""
    p = Path(path)
    if p.is_file() and p.suffix in (".xlsx", ".xls"):
        return [str(p)]
    if p.is_dir():
        files = sorted(p.rglob("*.xlsx"))
        # Exclude LAS directory
        files = [f for f in files if "LAS" not in str(f)]
        if not files:
            raise FileNotFoundError(f"No .xlsx files found in directory: {path}")
        return [str(f) for f in files]
    raise FileNotFoundError(f"Invalid path: {path}")


def load_aws(path: str) -> pd.DataFrame:
    """
    Load AWS Excel data (single file or all .xlsx in a directory), return a DataFrame.
    Keeps required columns: TIMESTAMP, Rain, Ms_2cm, Ta_5m, Ts_2cm, + extra meteo/soil columns.
    """
    files = _collect_xlsx(path)
    print(f"  Reading AWS files ({len(files)}):")
    for f in files:
        print(f"    {Path(f).name}")

    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Time column
    if "TIMESTAMP" not in df.columns:
        raise KeyError(f"AWS: time column 'TIMESTAMP' not found. Columns: {df.columns.tolist()}")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df[df["TIMESTAMP"].notna()].copy()

    # ── Required columns ──
    # Precipitation
    rain_col = None
    for c in df.columns:
        if c.lower() == "rain":
            rain_col = c
            break
    if rain_col is None:
        raise KeyError(f"AWS: precipitation column (Rain) not found. Columns: {df.columns.tolist()}")

    # Soil moisture Ms_2cm
    ms_cols = []
    if "Ms_2cm" in df.columns:
        ms_cols.append("Ms_2cm")
    else:
        raise KeyError(f"AWS: Ms_2cm column not found. Columns: {list(df.columns)}")

    # Temperature columns (optional)
    temp_cols = []
    for c in ["Ta_5m", "Ts_2cm"]:
        if c in df.columns:
            temp_cols.append(c)
        else:
            print(f"  ! AWS temperature column not found: {c}")

    # ── Extra meteo/soil columns (optional) ──
    extra_cols = [c for c in AWS_EXTRA_COLS if c in df.columns]

    keep_raw = ["TIMESTAMP", rain_col] + ms_cols + temp_cols + extra_cols
    keep = list(dict.fromkeys(keep_raw))  # de-duplicate, preserving order
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={"TIMESTAMP": "time", rain_col: "Rain"})

    # Fill values → NaN + coerce to numeric (column-wise to avoid pandas 3.x replace error on datetime columns)
    for col in df.columns:
        if col == "time":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(FILL_VALUES, np.nan)

    df = df.sort_values("time").reset_index(drop=True)

    print(f"  AWS: {len(df):,} rows, time span {df['time'].min()} ~ {df['time'].max()}")
    print(f"     Rain=Rain, soil_moisture=Ms_2cm, temp={temp_cols}, extra={extra_cols}")
    return df


def load_ec(path: str, ec_height: float | None = None) -> pd.DataFrame:
    """Load EC Excel data (single file or all .xlsx in a directory), standardise column names, and return a DataFrame.

    - Case-insensitive matching for Ustar / ustar.
    - If L column is missing but z/L is present, compute L = z / (z/L) (requires ec_height).
    """
    files = _collect_xlsx(path)
    print(f"  Reading EC files ({len(files)}):")
    for f in files:
        print(f"    {Path(f).name}")

    dfs = []
    for f in files:
        df = pd.read_excel(f)
        df.columns = df.columns.str.strip()
        # Some files use 'Date' instead of 'Date/Time'
        if "Date" in df.columns and "Date/Time" not in df.columns:
            df = df.rename(columns={"Date": "Date/Time"})
        # Some files use 'H' instead of 'Hs'
        if "H" in df.columns and "Hs" not in df.columns:
            df = df.rename(columns={"H": "Hs"})
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # ── Case-insensitive column name standardisation ──
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        # ustar / Ustar / USTAR → Ustar
        if col_lower == "ustar" and "Ustar" not in df.columns:
            rename_map[col] = "Ustar"
        # z/L / Z/L → z/L (temporarily, converted to L later)
        if (
            col_lower in ("z/l", "z_l")
            and "L" not in df.columns
            and "z/L" not in df.columns
        ):
            rename_map[col] = "z/L"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Column name standardised (case): {rename_map}")

    # ── z/L → L conversion ──
    if "L" not in df.columns and "z/L" in df.columns:
        if ec_height is None:
            raise ValueError(
                "EC data has no L column but has z/L column. "
                "Provide measurement height z (m) via --ec-height "
                f"to compute L = z / (z/L). Columns: {df.columns.tolist()}"
            )
        print(f"  Computing L = {ec_height} / (z/L) ...")
        df["L"] = np.where(
            df["z/L"] != 0,
            ec_height / df["z/L"],
            np.nan,
        )
        df = df.drop(columns=["z/L"])

    # Time column standardisation (column names vary by year; try each in order)
    for tcol in ["Date/Time", "Date", "TIMESTAMP"]:
        if tcol in df.columns:
            time_col = tcol
            break
    else:
        raise KeyError(f"EC: no time column found. Columns: {df.columns.tolist()}")

    missing = [c for c in EC_COLS_NEEDED if c not in df.columns]
    if missing:
        raise KeyError(f"EC: missing required columns: {missing}. Columns: {df.columns.tolist()}")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()

    # Align timestamps to 30-min: round to nearest 30 minutes
    df[time_col] = df[time_col].dt.round(f"{EC_STEP_MIN}min")

    keep = [time_col] + EC_COLS_NEEDED
    df = df[keep].copy()
    df = df.rename(columns={time_col: "time"})
    # Fill values → NaN + coerce to numeric (column-wise to avoid pandas 3.x replace error on datetime columns)
    for col in df.columns:
        if col == "time":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(FILL_VALUES, np.nan)

    df = df.sort_values("time").reset_index(drop=True)
    print(f"  EC:  {len(df):,} rows, time span {df['time'].min()} ~ {df['time'].max()}")
    return df


# ======================== AWS → 30 min Aggregation ========================


def resample_aws_to_30min(aws: pd.DataFrame) -> pd.DataFrame:
    """
    Resample AWS (10 min) to the EC (30 min) grid.
    - Rain: sum (accumulate 10 min totals to 30 min totals)
    - Other columns: mean
    Resample window: left-closed right-open, label=right, aligned with EC timestamps.

    Note: skipna=False — if any raw value in a window is NaN, the result is NaN.
    """
    aws = aws.set_index("time")

    # Separate Rain from other columns
    agg = {}
    for col in aws.columns:
        if col == "Rain":
            agg[col] = lambda x: x.sum(skipna=False)
        else:
            agg[col] = lambda x: x.mean(skipna=False)

    resampled = aws.resample(f"{EC_STEP_MIN}min", label="right", closed="right").agg(
        agg
    )
    resampled = resampled.reset_index()
    resampled = resampled[resampled["time"].notna()]
    return resampled


# ======================== Merge & Period Search ========================


def merge_and_search(
    aws_30: pd.DataFrame,
    ec: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    temp_min: float = TEMP_MIN_THRESH,
    ms_max: float = MS_MAX_THRESH,
    ignore_nan_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge AWS and EC data (inner join on 30-min timestamps),
    then search for all consecutive valid periods along the time axis.

    Conditions:
      1. All variables except LE (and those in ignore_nan_cols) are non-NaN
      2. Rain == 0 everywhere
      3. Window minimum of Ts_2cm and Ta_5m > temp_min
      4. Window maximum of Ms_2cm < ms_max

    Parameters:
      ignore_nan_cols: Additional column names allowed to contain NaN
                       (LE is always ignored). Default None.

    Returns:
      - valid_data: 30-min data for qualifying periods (original columns only)
      - summary_df: Period summary DataFrame
    """
    merged = pd.merge(aws_30, ec, on="time", how="inner")
    merged = merged.sort_values("time").reset_index(drop=True)
    print(f"  After merge: {len(merged):,} rows")

    if merged.empty:
        print("  [Error] No data after merge.")
        return pd.DataFrame(), pd.DataFrame()

    # ── Collect strict variables for NaN checking ──
    # LE is always ignored; extra columns in ignore_nan_cols are also ignored
    if ignore_nan_cols is None:
        ignore_nan_cols = []
    skip_cols = {"time", "Rain", "LE"} | set(ignore_nan_cols)
    vars_strict = [col for col in merged.columns if col not in skip_cols]

    print(f"  Variables required to be NaN-free ({len(vars_strict)}): {vars_strict}")

    # ── Row-wise flagging ──
    # Condition 1: all variables except LE are non-NaN
    merged["all_valid"] = merged[vars_strict].notna().all(axis=1)

    # Condition 2: no precipitation
    merged["no_rain"] = merged["Rain"].eq(0.0)

    # Condition 4: soil moisture below upper bound
    merged["ms_ok"] = merged["Ms_2cm"].notna() & merged["Ms_2cm"].lt(ms_max)

    # Combined row condition (excluding temperature, which is checked at window level)
    merged["row_ok"] = merged["all_valid"] & merged["no_rain"] & merged["ms_ok"]

    # ── Group by date, build daily summary ──
    merged["date"] = merged["time"].dt.date

    agg_dict = {
        "n_rows": ("time", "count"),
        "n_ok_rows": ("row_ok", "sum"),
        "n_all_valid": ("all_valid", "sum"),
        "n_no_rain": ("no_rain", "sum"),
        "n_ms_ok": ("ms_ok", "sum"),
        "Rain_sum": ("Rain", "sum"),
    }
    # Daily mean temperature (for window-level min check)
    if "Ta_5m" in merged.columns:
        agg_dict["Ta_5m_mean"] = ("Ta_5m", "mean")
    if "Ts_2cm" in merged.columns:
        agg_dict["Ts_2cm_mean"] = ("Ts_2cm", "mean")
    # Daily max Ms_2cm (for window-level check)
    agg_dict["Ms_2cm_max"] = ("Ms_2cm", "max")

    daily = merged.groupby("date", sort=True).agg(**agg_dict).reset_index()

    dates = daily["date"].values
    n_dates = len(dates)

    # ── Search windows ──
    results = []
    for i in range(n_dates - window_days + 1):
        start_date = dates[i]
        end_date = dates[i + window_days - 1]

        # Check date continuity
        expected_end = start_date + pd.Timedelta(days=window_days - 1)
        if end_date != expected_end:
            continue

        window_mask = (daily["date"] >= start_date) & (daily["date"] <= end_date)
        window = daily[window_mask]

        # Conditions 1 & 2 & 4: every row of every day within the window must satisfy
        if (window["n_ok_rows"] != window["n_rows"]).any():
            continue

        # Condition 3: window minimum temperatures > threshold
        ts_min = (
            window["Ts_2cm_mean"].min()
            if "Ts_2cm_mean" in window.columns
            else float("inf")
        )
        ta_min = (
            window["Ta_5m_mean"].min()
            if "Ta_5m_mean" in window.columns
            else float("inf")
        )
        if ts_min <= temp_min or ta_min <= temp_min:
            continue

        ms_max_val = window["Ms_2cm_max"].max()
        total_rows = int(window["n_rows"].sum())
        valid_rows = int(window["n_all_valid"].sum())

        results.append(
            {
                "start_date": start_date,
                "end_date": end_date,
                "total_30min": total_rows,
                "valid_30min": valid_rows,
                "Ts_2cm_min": round(float(ts_min), 2)
                if ts_min != float("inf")
                else None,
                "Ta_5m_min": round(float(ta_min), 2)
                if ta_min != float("inf")
                else None,
                "Ms_2cm_max": round(float(ms_max_val), 3),
            }
        )

    if results:
        summary_df = pd.DataFrame(results)
        print(f"\n  Found {len(results)} qualifying {window_days}-day periods")
        for r in results:

            def _fmt(v):
                return f"{v}°C" if v is not None else "N/A"

            print(
                f"    {r['start_date']} ~ {r['end_date']}  |  "
                f"Ts_min={_fmt(r['Ts_2cm_min'])}  Ta_min={_fmt(r['Ta_5m_min'])}  "
                f"Ms_max={r['Ms_2cm_max']}% "
            )
    else:
        print(f"\n  ! No qualifying {window_days}-day period found.")
        return pd.DataFrame(), pd.DataFrame()

    # ── Clip 30-min data of qualifying periods ──
    period_mask = pd.Series(False, index=merged.index)
    for r in results:
        mask = (merged["time"] >= pd.Timestamp(r["start_date"])) & (
            merged["time"] < pd.Timestamp(r["end_date"]) + pd.Timedelta(days=1)
        )
        period_mask = period_mask | mask

    # Output only original columns (excluding intermediate flag columns)
    out_cols = [
        c
        for c in merged.columns
        if c not in ("all_valid", "no_rain", "ms_ok", "row_ok", "date")
    ]
    result = merged.loc[period_mask, out_cols].copy()
    result = result.sort_values("time").reset_index(drop=True)
    return result, summary_df


# ======================== Yearly File Pairing ========================


def _extract_year(filepath: str) -> int:
    """Extract a 4-digit year from a filename, e.g. '2015.Desert.AWS.xlsx' → 2015."""
    m = re.search(r"(\d{4})", Path(filepath).name)
    if m is None:
        raise ValueError(f"Cannot extract year from filename: {filepath}")
    return int(m.group(1))


def _pair_by_year(
    aws_files: list[str], ec_files: list[str]
) -> dict[int, tuple[str, str]]:
    """Pair AWS and EC files by year. Returns {year: (aws_path, ec_path)}."""
    aws_map = {}
    for f in aws_files:
        y = _extract_year(f)
        if y in aws_map:
            print(f"  ! Multiple AWS files for year {y}: {aws_map[y]} and {f} — using the latter.")
        aws_map[y] = f

    ec_map = {}
    for f in ec_files:
        y = _extract_year(f)
        if y in ec_map:
            print(f"  ! Multiple EC files for year {y}: {ec_map[y]} and {f} — using the latter.")
        ec_map[y] = f

    common_years = sorted(set(aws_map.keys()) & set(ec_map.keys()))
    missing_aws = set(ec_map.keys()) - set(aws_map.keys())
    missing_ec = set(aws_map.keys()) - set(ec_map.keys())
    if missing_aws:
        print(f"  ! Years missing AWS files: {sorted(missing_aws)} — skipping.")
    if missing_ec:
        print(f"  ! Years missing EC files: {sorted(missing_ec)} — skipping.")

    return {y: (aws_map[y], ec_map[y]) for y in common_years}


# ======================== Main ========================


def main():
    """Main entry point: parse arguments, load data year-by-year, find valid periods, and save results."""
    parser = argparse.ArgumentParser(
        description="Find 10-day valid periods satisfying dLSEB requirements from raw AWS + EC Excel data.\n"
        "Load and process data year-by-year to avoid silent data loss from cross-year column name discrepancies.\n"
        "The station directory (AWS/ + EC/) must contain multi-year .xlsx files."
    )
    parser.add_argument(
        "station_dir", type=str, help="Station directory (contains AWS/ and EC/ subdirectories)"
    )
    parser.add_argument("--out", type=str, default="./", help="Output directory (default: current directory)")
    parser.add_argument(
        "--name", type=str, default=None, help="Output file prefix (default: extracted from directory name)"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=WINDOW_DAYS,
        help=f"Period length in days (default: {WINDOW_DAYS})",
    )
    parser.add_argument(
        "--temp-min",
        type=float,
        default=TEMP_MIN_THRESH,
        help=f"Minimum temperature threshold degC (default: {TEMP_MIN_THRESH})",
    )
    parser.add_argument(
        "--ms-max",
        type=float,
        default=MS_MAX_THRESH,
        help=f"Maximum soil moisture percent (default: {MS_MAX_THRESH})",
    )
    parser.add_argument(
        "--ec-height",
        type=float,
        default=DEFAULT_EC_HEIGHT,
        help="EC measurement height z (m). If EC data has no L column but has a z/L column, "
        "L is computed via L = z / (z/L). No effect for sites that already have an L column. Default: 2.5 m.",
    )
    parser.add_argument(
        "--ignore-nan-cols",
        nargs="*",
        default=None,
        help="Additional column names allowed to contain NaN (space-separated). LE is always ignored. "
        "Example: --ignore-nan-cols Hs H2O allows missing values in sensible heat and water vapour. "
        "When not specified, all variables except LE must be NaN-free.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = Path(args.station_dir)
    aws_dir = base / "AWS" if (base / "AWS").is_dir() else base
    ec_dir = base / "EC" if (base / "EC").is_dir() else base

    station_name = args.name if args.name else base.name

    print("=" * 60)
    print(f"Find {args.window_days}-day valid periods — {station_name}")

    ignore_desc = "LE"
    if args.ignore_nan_cols:
        ignore_desc += ", " + ", ".join(args.ignore_nan_cols)
    print(
        f"  Criteria: no NaN except {ignore_desc}, Rain=0, Ts_2cm&Ta_5m > {args.temp_min}°C, "
        f"Ms_2cm < {args.ms_max}%"
    )
    print("  Strategy: load & process year-by-year to avoid cross-year column name differences")
    if args.ec_height is not None:
        print(f"  EC height: z = {args.ec_height} m (for z/L → L conversion)")
    print("=" * 60)

    # ── Collect files and pair by year ──
    aws_files = _collect_xlsx(str(aws_dir))
    ec_files = _collect_xlsx(str(ec_dir))

    year_pairs = _pair_by_year(aws_files, ec_files)
    if not year_pairs:
        print("\n[Error] No AWS+EC paired years found. Exiting.")
        return

    print(f"\nTotal {len(year_pairs)} years to process: {list(year_pairs.keys())}")

    # ── Process year-by-year ──
    all_summaries: list[pd.DataFrame] = []
    all_valid_data: list[pd.DataFrame] = []
    total_periods = 0

    for year in sorted(year_pairs.keys()):
        aws_path, ec_path = year_pairs[year]

        print(f"\n{'─' * 50}")
        print(f"Year {year}")
        print(f"    AWS: {Path(aws_path).name}")
        print(f"    EC:  {Path(ec_path).name}")

        try:
            aws = load_aws(str(aws_path))
            ec = load_ec(str(ec_path), ec_height=args.ec_height)
        except Exception as e:
            print(f"  ! Loading failed: {e}. Skipping.")
            continue

        aws_30 = resample_aws_to_30min(aws)

        valid_data, summary = merge_and_search(
            aws_30,
            ec,
            window_days=args.window_days,
            temp_min=args.temp_min,
            ms_max=args.ms_max,
            ignore_nan_cols=args.ignore_nan_cols,
        )

        if not summary.empty:
            summary["year"] = year
            all_summaries.append(summary)
            all_valid_data.append(valid_data)
            total_periods += len(summary)

    # ── Aggregate output ──
    if not all_summaries:
        print("\nNo qualifying periods found. No output files generated.")
        return

    final_summary = (
        pd.concat(all_summaries, ignore_index=True)
        .sort_values("start_date")
        .reset_index(drop=True)
    )
    final_data = (
        pd.concat(all_valid_data, ignore_index=True)
        .sort_values("time")
        .reset_index(drop=True)
    )

    print(f"\n{'=' * 60}")
    print(f"Summary: {total_periods} qualifying {args.window_days}-day periods in total")
    print(
        f"  Time range: {final_summary['start_date'].min()} ~ {final_summary['end_date'].max()}"
    )
    for _, r in final_summary.iterrows():
        ts_m = f"{r['Ts_2cm_min']}°C" if pd.notna(r.get("Ts_2cm_min")) else "N/A"
        ta_m = f"{r['Ta_5m_min']}°C" if pd.notna(r.get("Ta_5m_min")) else "N/A"
        print(
            f"    {r['start_date']} ~ {r['end_date']}  "
            f"Ts_min={ts_m}  Ta_min={ta_m}  Ms_max={r['Ms_2cm_max']}%  "
            f"({int(r['valid_30min'])}/{int(r['total_30min'])} rows)  [{r['year']}]"
        )

    # ── Output ──
    summary_path = out_dir / f"{station_name}_valid_10d_periods_summary.csv"
    final_summary.to_csv(summary_path, index=False)
    print(f"\nPeriod summary saved: {summary_path}")

    data_path = out_dir / f"{station_name}_valid_10d_periods.csv"
    final_data.to_csv(data_path, index=False)
    print(
        f"Qualifying {args.window_days}-day period data saved "
        f"({len(final_data):,} rows): {data_path}"
    )


if __name__ == "__main__":
    main()
