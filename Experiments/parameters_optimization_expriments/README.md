# Parameter Optimization Experiments

This directory contains scripts and results for the dLSEB parameter optimisation experiments, which test four loss-function configurations that control which observation variables contribute to the optimisation objective.

## Experiments

| Experiment | Loss terms | Description |
|:-----------|:-----------|:------------|
| **RSL** | Rsu + Rlu | Radiation only |
| **RHS** | Rsu + Rlu + H | Radiation + sensible heat |
| **RHT** | Rsu + Rlu + H + T | Radiation + H + temperature |
| **ALL** | T + Rsu + Rlu + H + G(6cm) | All terms |

Initial parameters are loaded from pre-estimated `.npy` files under `/data/3.estimate_parameters/`. Optimisation runs 500 epochs (with early stopping) using Adam with three independent learning-rate groups (α/σ/z₀ₘ, k, C) and optional T₀ soil-temperature data assimilation.

## Directory Structure

```
parameters_optimization_expriments/
├── 1.run_experiments.py       # Main optimisation runner
├── 2.batch_visualization.py   # Batch visualisation & metrics
├── 3.summary.py               # Grouped bar / line charts (RMSE / Cor / Std / Bias)
├── summary.log                # Latest optimisation summary
├── RMSE_Cor_Param.log          # RMSE & correlation summary across all sites/experiments
├── rmse_bars.svg              # RMSE grouped bar chart
├── cor_bars.svg               # Correlation grouped bar chart
├── std_bars.svg               # Std Ratio grouped bar chart
├── bias_bars.svg              # Bias grouped bar chart
├── rmse_lines.svg             # RMSE grouped line chart
├── cor_lines.svg              # Correlation grouped line chart
├── std_lines.svg              # Std Ratio grouped line chart
├── bias_lines.svg             # Bias grouped line chart
├── 0.EST/                     # EST (pre-estimated, no-optimisation) forward runs
│   ├── run_estimation.py       # Script to generate EST results
│   ├── EB_result_*.npy         # Energy-balance outputs per site
│   └── T_result_*.npy          # Soil-temperature profiles per site
├── 1.ALL/                     # ALL experiment results
├── 2.RSL/                     # RSL experiment results
├── 3.RHS/                     # RHS experiment results
└── 4.RHT/                     # RHT experiment results
```

Each experiment directory (`1.ALL/` … `4.RHT/`) contains one subdirectory per site (`Huazhaizi/`, `Ejin/`, `Shenshawo/`) with:

| File | Description |
|:-----|:------------|
| `{EXP}.log` | Console output log of the optimisation run |
| `T_loss_values.npy` | Loss value at each epoch |
| `params_hat_values.npy` | Parameter vectors [α, σ, z₀ₘ, k, log(C)] at each epoch |
| `EB_result.npy` | Modelled Rsu, Rlu, H, G (4 × 480) |
| `T_result.npy` | Modelled 10-layer soil temperature (480 × 10) |
| `losses.svg` | 6-panel figure: loss + 5 parameter evolution curves |
| `model_output.svg` | 6-panel energy-balance time series (Obs vs OPT vs EST) |
| `metrics_{EXP}.npy` | RMSE & Pearson r for each variable (OPT & EST vs observations) |

## Scripts

### `1.run_experiments.py`

Runs parameter optimisation for one or more (site, experiment) combinations.

```bash
# All experiments, all 3 sites
python "1.run_experiments.py" --all

# Single site, all experiments (default)
python "1.run_experiments.py" --site Huazhaizi

# Single experiment, single site
python "1.run_experiments.py" --exp ALL --site Huazhaizi

# Multiple experiments, all sites
python "1.run_experiments.py" --exp ALL RSL --all

# Custom epochs, disable T0 data assimilation
python "1.run_experiments.py" --site Huazhaizi --epochs 300 --no-da
```

### `2.batch_visualization.py`

Generates SVG figures and computes RMSE/Correlation metrics from completed experiments.

```bash
# All experiments, all sites (default)
python "2.batch_visualization.py"

# Single experiment, single site
python "2.batch_visualization.py" --exp ALL --site Huazhaizi

# Single experiment, all sites
python "2.batch_visualization.py" --exp RSL --all

# EST metrics only (no experiment data needed)
python "2.batch_visualization.py" --est-only

# Skip certain outputs
python "2.batch_visualization.py" --no-loss --no-energy
python "2.batch_visualization.py" --no-metrics
```

### `3.summary.py`

Generates grouped bar or line charts comparing RMSE, Correlation, Std Ratio, and Bias across all experiments and three sites (Huazhaizi, Ejin, Shenshawo). Each figure is a 3×2 grid for the six variables. Default output is bar charts; use `--lines` for line charts.

```bash
# RMSE & Correlation bar charts (default)
python 3.summary.py

# RMSE & Correlation line charts
python 3.summary.py --lines

# RMSE, Correlation, Std Ratio & Bias bar charts
python 3.summary.py --std-bias
python 3.summary.py --all               # same as --std-bias

# RMSE, Correlation, Std Ratio & Bias line charts
python 3.summary.py --lines --all
```

### `0.EST/run_estimation.py`

Runs a single forward pass through dLSEB using pre-estimated parameters (no optimisation). Output `EB_result_*.npy` and `T_result_*.npy` are consumed by `batch_visualization.py` as the EST baseline.

```bash
cd 0.EST
python run_estimation.py --site Huazhaizi
python run_estimation.py --all
```

## Workflow

1. **Estimate** — run `0.EST/run_estimation.py --all` to generate baseline simulations.
2. **Optimise** — run `python "1.run_experiments.py" --all` to perform parameter optimisation for all experiments and sites.
3. **Visualise** — run `python "2.batch_visualization.py"` to generate per-experiment figures and metrics. The output `RMSE_Cor_Param.log` provides a formatted comparison table across all sites and experiments.
4. **Summary charts** — run `python 3.summary.py --all` to generate grouped bar charts (`rmse_bars.svg`, `cor_bars.svg`, `std_bars.svg`, `bias_bars.svg`), or `python 3.summary.py --lines --all` for line charts (`rmse_lines.svg`, `cor_lines.svg`, ...), comparing all experiments across the three sites.
