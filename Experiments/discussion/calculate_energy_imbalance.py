"""
Energy Balance Closure Analysis for Three Stations

Uses observed G at 6 cm directly as surface heat flux (standard EC approach).
Convention: Rn (down+), H (up+), G_obs (down+).  Balance: Rn = H + G.

Closure metrics:
  1. MAE = mean(|Rn - H - G_obs|)  (Mean Absolute Error)
  2. EBR (Energy Balance Ratio) = (H + G_obs) / Rn  (mean, filtered |Rn|>10)
  3. Linear regression: (H + G_obs) = a * Rn + b  (OLS)
  4. Residual / Rn (fraction, |Rn|>10)
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
LOG_FILE = os.path.join(SCRIPT_DIR, "energy_closure.log")


def log_print(*args, **kwargs):
    """Print to both stdout and log file."""
    print(*args, **kwargs)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        print(*args, **kwargs, file=f)


def load_site_data(site_name):
    site_dir = os.path.join(BASE_DIR, "data", "2.data_selection", site_name)
    meteo = pd.read_csv(os.path.join(site_dir, "meteo_var.csv"))
    soil = pd.read_csv(os.path.join(site_dir, "soil.csv"))
    ec = pd.read_csv(os.path.join(site_dir, "ec_var.csv"))
    return meteo, soil, ec


def compute_closure(meteo, soil, ec):
    """Compute standard EC energy balance closure metrics.

    Convention:
      Rn     = DR - UR + DLR - ULR          (downward positive)
      H      = Hs                            (upward positive)
      G_obs  = mean(Gs_1, Gs_2, Gs_3)       (downward positive, plate convention)

    Balance: Rn = H + G  →  Residual = Rn - H - G_obs
    """
    Rn = (
        meteo["DR"].values
        - meteo["UR"].values
        + meteo["DLR_Cor"].values
        - meteo["ULR_Cor"].values
    )
    H = ec["Hs"].values
    G_obs = (soil["Gs_1"].values + soil["Gs_2"].values + soil["Gs_3"].values) / 3.0

    residual = Rn - H - G_obs           # signed
    turbulent = H + G_obs               # total turbulent + ground flux

    return Rn, H, G_obs, residual, turbulent


def main():
    # Clear log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")

    results = {}

    for site_name in ["Ejin", "Huazhaizi", "Shenshawo"]:
        meteo, soil, ec = load_site_data(site_name)
        Rn, H, G_obs, residual, turbulent = compute_closure(meteo, soil, ec)

        # ── 1. MAE & residual statistics ──
        res_abs = {
            "max": np.max(np.abs(residual)),
            "min": np.min(np.abs(residual)),
            "mae": np.mean(np.abs(residual)),   # Mean Absolute Error
            "std": np.std(np.abs(residual)),
        }

        # ── 2. EBR: (H+G)/Rn, filtered by |Rn|>10 ──
        mask = np.abs(Rn) > 10
        ebr_vals = turbulent[mask] / Rn[mask]
        ebr = {"mean": np.mean(ebr_vals), "std": np.std(ebr_vals)}

        # ── 3. Linear regression: turbulent = a * Rn + b ──
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            Rn[mask], turbulent[mask]
        )

        # ── 4. Residual fraction: |residual| / |Rn| ──
        res_frac = np.abs(residual[mask]) / np.abs(Rn[mask])

        results[site_name] = {
            "res_abs": res_abs,
            "ebr": ebr,
            "regression": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
            },
            "res_frac_mean": np.mean(res_frac),
            "res_frac_std": np.std(res_frac),
            "n": len(Rn),
            "n_used": mask.sum(),
        }

    # ═════════════════════════════════════════════════════════════════════
    # Print results
    # ═════════════════════════════════════════════════════════════════════

    sep = "=" * 90

    log_print(f"\n{sep}")
    log_print("  Energy Balance Closure Analysis")
    log_print("  Convention: Rn (down+), H (up+), G_obs (down+).  Rn = H + G")
    log_print("  Using observed G at 6 cm as surface heat flux")
    log_print(f"{sep}")

    # ── Metric 1: MAE ──
    log_print(f"\n  {'─'*80}")
    log_print("  Metric 1: MAE = mean(|Rn − H − G_obs|)  (W m^{-2})")
    log_print(f"  {'─'*80}")
    log_print(f"  {'Station':<12} {'Max':>8} {'Min':>8} {'MAE':>8} {'Std':>8}")
    log_print(f"  {'-'*50}")
    for site in ["Ejin", "Huazhaizi", "Shenshawo"]:
        a = results[site]["res_abs"]
        log_print(f"  {site:<12} {a['max']:>8.1f} {a['min']:>8.1f} {a['mae']:>8.1f} {a['std']:>8.1f}")

    # ── Metric 2: EBR ──
    log_print(f"\n  {'─'*80}")
    log_print("  Metric 2: Energy Balance Ratio  EBR = (H + G_obs) / Rn  (|Rn| > 10 W m^{-2})")
    log_print(f"  {'─'*80}")
    log_print(f"  {'Station':<12} {'N':>6}  {'EBR Mean':>10} {'EBR Std':>10}")
    log_print(f"  {'-'*40}")
    for site in ["Ejin", "Huazhaizi", "Shenshawo"]:
        r = results[site]
        log_print(f"  {site:<12} {r['n_used']:>6}  {r['ebr']['mean']:>10.4f} {r['ebr']['std']:>10.4f}")

    # ── Metric 3: Linear regression ──
    log_print(f"\n  {'─'*80}")
    log_print("  Metric 3: OLS Regression  (H + G_obs) = a · Rn + b")
    log_print(f"  {'─'*80}")
    log_print(f"  {'Station':<12} {'a (slope)':>10} {'b (W m^{{-2}})':>14} {'R²':>10}")
    log_print(f"  {'-'*45}")
    for site in ["Ejin", "Huazhaizi", "Shenshawo"]:
        r = results[site]
        reg = r["regression"]
        log_print(f"  {site:<12} {reg['slope']:>10.4f} {reg['intercept']:>10.2f} {reg['r_squared']:>10.4f}")

    # ── Metric 4: Residual fraction ──
    log_print(f"\n  {'─'*80}")
    log_print("  Metric 4: Residual Fraction  |Rn − H − G| / |Rn|  (|Rn| > 10)")
    log_print(f"  {'─'*80}")
    log_print(f"  {'Station':<12} {'Mean':>10} {'Std':>10}")
    log_print(f"  {'-'*35}")
    for site in ["Ejin", "Huazhaizi", "Shenshawo"]:
        r = results[site]
        log_print(f"  {site:<12} {r['res_frac_mean']:>10.4f} {r['res_frac_std']:>10.4f}")

    # ── Flux component summary ──
    log_print(f"\n  {'─'*80}")
    log_print("  Flux Components Summary (mean ± std, W m^{-2})")
    log_print(f"  {'─'*80}")
    log_print(f"  {'Station':<12} {'Rn':>16} {'H':>16} {'G_obs':>16}")
    log_print(f"  {'-'*65}")
    for site_name in ["Ejin", "Huazhaizi", "Shenshawo"]:
        meteo, soil, ec = load_site_data(site_name)
        Rn, H, G_obs, _, _ = compute_closure(meteo, soil, ec)
        log_print(f"  {site_name:<12} {Rn.mean():>8.1f} ± {Rn.std():>6.1f}"
                  f"  {H.mean():>8.1f} ± {H.std():>6.1f}"
                  f"  {G_obs.mean():>8.1f} ± {G_obs.std():>6.1f}")

    log_print(f"\n  → Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
