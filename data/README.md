# Data Processing Pipeline for LSM Energy Balance Parameter Estimation

This directory contains a three-step data processing pipeline that prepares
observational data from three desert stations (Ejin, Huazhaizi, Shenshawo) for
land surface model (LSM) parameter estimation.

## Directory Overview

```
data/
├── 1.origin/                  # Step 1 — Raw data & valid-period search
├── 2.data_selection/          # Step 2 — Best-period selection & training CSV export
├── 3.estimate_parameters/     # Step 3 — Parameter estimation & figure generation
└── README.md
```

---

## 1.origin/ — Raw Data & Valid Period Search

**Purpose**: Search raw AWS (10-min) + EC (30-min) Excel data for consecutive
10-day periods that satisfy the dLSEB model requirements.

**Contents**:

| Path | Description |
|------|-------------|
| `1.find_valid_periods.py` | Core Python script for finding valid periods |
| `run_find_periods.sh` | Batch-run script for all 3 stations |
| `Ejin Desert station/AWS/`, `EC/` | Raw Excel data (2022–2024) + data description docx |
| `Huazhaizi desert station/AWS/`, `EC/` | Raw Excel data (2018–2020) + data description docx |
| `Shenshawo sandy desert/AWS/`, `EC/` | Raw Excel data (2012–2014) + data description docx |

**Output** (per station):
- `{Station}_valid_10d_periods.csv` — 30-min data for all qualifying periods
- `{Station}_valid_10d_periods_summary.csv` — Period-level summary (dates, temperature extremes, etc.)

**Selection criteria** (all four simultaneously):
1. No missing values — all variables except LE (and user-specified exemptions) are NaN-free
2. No precipitation — Rain == 0.0 at every 30-min step
3. Temperature floor — min(Ts_2cm, Ta_5m) > 0 °C
4. Dry soil — near-surface soil moisture Ms_2cm < 5 %

**Design**: Data from AWS (10-min) and EC (30-min) are aligned on a common
30-min grid. AWS precipitation is summed; other AWS variables are averaged.

**Usage**:

```bash
# Single station
python "1.find_valid_periods.py" "Ejin Desert station" \
    --out "Ejin Desert station" --name "Ejin"

# With NaN-tolerant columns (for poorer-quality data)
python "1.find_valid_periods.py" "Shenshawo sandy desert" \
    --out "Shenshawo sandy desert" --name "Shenshawo" \
    --ignore-nan-cols Hs H2O

# Batch (all 3 stations at once)
bash run_find_periods.sh
```

**Key CLI arguments**:
- `--out` — output directory (default: current directory)
- `--name` — output file prefix (default: station directory name)
- `--window-days` — period length in days (default: 10)
- `--temp-min` — minimum temperature threshold °C (default: 0)
- `--ms-max` — maximum soil moisture % (default: 5.0)
- `--ec-height` — EC measurement height for z/L → L conversion (default: 4.5 m)
- `--ignore-nan-cols` — extra column names allowed to contain NaN (space-separated)

---

## 2.data_selection/ — Best-Period Selection & Training CSVs

**Purpose**: From the valid periods found in Step 1, select the single best
10-day period and export 4 training-data CSV files.

**Contents**:

| Path | Description |
|------|-------------|
| `2.select_best_period.py` | Core Python script for best-period selection |
| `run_select_best.sh` | Batch-run script for all 3 stations |
| `final_periods.log` | Auto-generated summary of selected periods |
| `Ejin/`, `Huazhaizi/`, `Shenshawo/` | Per-station output directories |

**Output** (per station, 4 CSV files):
- `meteo_var.csv` — meteorological variables: DR, DLR_Cor, Ta_5m, Press, UR, ULR_Cor, Rain
- `soil.csv` — soil variables: Gs_1/2/3, Ts_0–100cm, Ms_2cm
- `ec_var.csv` — eddy-covariance variables: Wnd, H2O, Hs, LE, Ustar, L
- `air_constants.csv` — air density (ρ), computed via the ideal gas law from Press and Ta_5m

Additionally, a `final_periods.log` is written to the `2.data_selection/` directory
listing each site's chosen year, start/end dates, and selection strategy.

**Selection strategy** (auto-detected by data quality):
- **Clean data** (Ejin, Huazhaizi): directly pick the period with the largest
  max(Hs) / max(LE) ratio (sensible heat dominates latent heat → Bowen ratio
  maximisation).
- **Noisy data** (Shenshawo): first rank periods by fewest Hs + H2O NaNs (top 5),
  then pick the best ratio among those, and finally impute any NaNs (H2O → column
  mean; Hs → edge zeros, interior linear interpolation).

**Usage**:

```bash
# Single station
python "2.select_best_period.py" \
    --summary "../1.origin/Ejin Desert station/Ejin_valid_10d_periods_summary.csv" \
    --data    "../1.origin/Ejin Desert station/Ejin_valid_10d_periods.csv" \
    --out     "Ejin"

# Batch (all 3 stations at once)
bash run_select_best.sh
```

**CLI arguments**:
- `--summary` — path to `*_valid_10d_periods_summary.csv` (from Step 1)
- `--data` — path to `*_valid_10d_periods.csv` (from Step 1)
- `--out` — output directory for the 4 CSV files (also used to infer site name for logging)
- `--log` — path to `final_periods.log` (default: `<out>/../final_periods.log`, i.e. `2.data_selection/final_periods.log`)

---

## 3.estimate_parameters/ — LSM Parameter Estimation

**Purpose**: Estimate six land surface parameters from the training data
produced in Step 2, and generate diagnostic figures.

**Contents**:

| Path | Description |
|------|-------------|
| `3.parameter_estimate.py` | Core Python script for parameter estimation |
| `parameter_est.log` | Log file: per-site summaries + cross-site comparison table |
| `parameters_summary_combined.svg` | Combined 4-row × 3-column summary figure (all 3 sites) |
| `Ejin/`, `Huazhaizi/`, `Shenshawo/` | Per-station output directories |

**Output** (per station):
- `{site_name}.npy` — NumPy archive of estimated parameter values (site-named, in this folder)
- `{site_name}/parameters_summary.svg` — 2×2 summary figure (albedo, emissivity, roughness length, soil conductivity)
- `{site_name}/soil_T_fit_0-10cm.jpg` — observed vs sinusoidally fitted soil temperature (0–10 cm)
- `{site_name}/soil_T_fit_20-100cm.jpg` — observed vs sinusoidally fitted soil temperature (20–100 cm)
- `{site_name}/thermal_diffusivity.jpg` — thermal diffusivity from amplitude-damping and phase-lag methods

**Output** (combined, generated by `--all`):
- `parameter_est.log` — plain-text log with per-site parameter summaries and cross-site comparison table
- `parameters_summary_combined.svg` — 4-row × 3-column combined summary figure:
  - Rows: albedo, emissivity, roughness length, soil conductivity
  - Columns: Huazhaizi, Ejin, Shenshawo

The `.npy` file is a dict with keys: `site`, `alpha`, `sigma`, `sigma_r2`, `z0m`, `k`, `k_r2`, `C`, `lam_A_mean`, `lam_phi_mean`.

**Estimated parameters**:
| Parameter | Symbol | Method |
|-----------|--------|--------|
| Surface shortwave albedo | α | ΣR_su / ΣR_sd over daytime (08:00–18:00) |
| Surface longwave emissivity | σ | Linear regression through origin: R_lu↑ − R_ld↓ = σ(σ₀T_s⁴ − R_ld↓) |
| Momentum roughness length | z₀ₘ | MOST inversion + KDE mode (ln-space) |
| Soil thermal conductivity | k | Linear regression through origin: −G₆cm = k · ΔT/Δz |
| Soil thermal diffusivity | λ | Amplitude-damping + phase-lag methods (sinusoidal fit of 8 soil layers) |
| Volumetric heat capacity | C | C = k / λ̄ (thickness-weighted harmonic mean) |

**Usage**:

```bash
# Single site
python "3.parameter_estimate.py" --site Huazhaizi

# All 3 sites at once
python "3.parameter_estimate.py" --all

# With custom data/output roots
python "3.parameter_estimate.py" --site Ejin \
    --data /path/to/2.data_selection \
    --output /path/to/output
```

**CLI arguments**:
- `--site` — single site name: Huazhaizi, Ejin, or Shenshawo
- `--all` — run all three sites in batch
- `--data` — override data base directory (default: `../2.data_selection`)
- `--output` — override output base directory (default: current directory)

---

## Full Pipeline Workflow

```bash
# Step 1 — Find valid 10-day periods (per station)
cd "1.origin"
bash run_find_periods.sh

# Step 2 — Select best period & export training CSVs
cd "../2.data_selection"
bash run_select_best.sh

# Step 3 — Estimate parameters & generate figures
cd "../3.estimate_parameters"
python "3.parameter_estimate.py" --all
```

**Prerequisites**: The scripts require Python 3 with the following packages:
`numpy`, `pandas`, `matplotlib`, `scipy`, `scikit-learn`, `seaborn`.

---

## Station Summary

| Station | Location | AWS Years | EC Years | Data Notes |
|---------|----------|-----------|----------|------------|
| **Ejin** | 100.99°E, 42.11°N | 2022–2024 | 2022–2024 | Clean data |
| **Huazhaizi** | 100.32°E, 38.77°N | 2018–2020 | 2018–2020 | Clean data |
| **Shenshawo** | 100.49°E, 37.46°N | 2012–2014 | 2012–2014 | Some Hs/H2O NaNs; auto-imputation applied |
