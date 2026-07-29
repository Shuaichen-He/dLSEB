# Source Codes for Reviewers

## Paper Information

**Title:** Differentiable Land Surface Energy Balance Model under Hard Physical Constraints

**Authors:** Shuaichen He¹, Hui Zheng², Qidong Yang¹, Zifeng Wang³

1. Department of Atmospheric Sciences, Yunnan University, Kunming, Yunnan Province 650500, China.
2. Institute of Atmospheric Physics, Chinese Academy of Sciences, Beijing 100029, China.
3. Huawei Technologies Co., Ltd., Shenzhen 518129, China.

Co-first authors: Shuaichen He and Hui Zheng.

Co-corresponding authors: Qidong Yang (yangqd@ynu.edu.cn) and Hui Zheng (zhenghui@tea.ac.cn).

---

## Abstract

Land surface model parameters are difficult to estimate at scale, while machine learning often violates conservation laws. Differentiable modeling offers a framework in which process-based and data-driven approaches share the same computational graph and are optimized jointly under physical constraints. However, it remains unexplored whether strict physical constraints embedded in numerical solvers are preserved under gradient-based optimization, and whether such a framework can identify a physically consistent parameter set when only a subset of surface energy balance components is observed. To address these questions, we develop a differentiable land surface energy balance model (dLSEB) in JAX, coupling radiative transfer, turbulent transport, and soil heat diffusion under a hard energy balance constraint solved via Newton's method, with the soil heat equation discretized by the Crank–Nicolson scheme. We verify the framework through idealized experiments and apply dLSEB to plot-scale optimization at three desert sites under progressively expanding observational constraints. Idealized experiments confirm that embedded numerical solvers retain strict physical constraints under gradient-based optimization, and that physical parameters and neural network weights are mathematically equivalent optimizable nodes. The dLSEB optimization improves surface energy flux simulation over conventional component-wise estimation. Progressive-constraint experiments show that each observation directly improves the corresponding variable while unobserved variables respond through physical coupling; incorporating all observations yields a system-wide parameter optimum rather than the best fit for any single component. The surface emissivity is driven to its physically constrained upper limit and, when the constraint is removed, exceeds unity at two sites, attributed to observational energy non-closure rather than a physical property; the systematic cold bias in soil temperature further reflects redistribution of observational residuals across the coupled energy balance. This study demonstrates that differentiable computing unifies process-based and data-driven modeling within a single computational graph and can be deployed on multi-process coupled land surface models, providing a foundation for deeper integration of the two paradigms.

---

## Repository Structure

```
Source Codes for Reviewers/
├── dLSEB/                        # Core differentiable land surface energy balance model
├── data/                         # Data processing pipeline (raw data → training CSVs → parameter estimates)
├── Experiments/                  # Idealized experiments, parameter optimization experiments & discussion
├── pyproject.toml                # Project dependencies (Python ≥ 3.12)
├── uv.lock                       # Locked dependency versions (uv package manager)
└── README.md                     # This file
```

### `dLSEB/` — Core Model

The JAX-based differentiable land surface energy balance model. It computes turbulent fluxes via **Monin-Obukhov similarity theory**, solves soil heat transfer with a **Crank-Nicolson scheme**, and uses **optimistix** numerical solvers for efficient surface energy balance root-finding. The entire model is fully differentiable, enabling gradient-based parameter optimization and T₀ data assimilation.

Key modules:
- `config.py` — Physical constants, optimizer settings, training hyperparameters
- `data_loader.py` — Meteorological forcing & observation data loading
- `main.py` — Standalone single-site testing/training script
- `model/turbulence.py` — Turbulence module (Monin-Obukhov similarity theory)
- `model/energy_balence.py` — Energy balance & soil heat transfer (Crank-Nicolson, Newton root-finding)
- `model/loss_update.py` — Loss function, physical constraints, Adam optimization, T₀ data assimilation

> See `dLSEB/README.md` for detailed module descriptions.

### `data/` — Data Processing Pipeline

A three-step pipeline that prepares observational data from three desert stations (Ejin, Huazhaizi, Shenshawo) for land surface model parameter estimation:

1. **`1.origin/`** — Raw AWS (10-min) + EC (30-min) Excel data and valid-period search scripts. Identifies consecutive 10-day periods satisfying dLSEB requirements (no missing values, no precipitation, temperature > 0 °C, dry soil).
2. **`2.data_selection/`** — Best-period selection and training-data CSV export (meteorological variables, soil variables, eddy-covariance variables, air constants).
3. **`3.estimate_parameters/`** — Conventional component-wise parameter estimation (albedo, emissivity, roughness length, soil thermal conductivity, thermal diffusivity, heat capacity) with diagnostic figures.

> See `data/README.md` for full pipeline documentation and usage.

### `Experiments/` — Experiments

Contains three subdirectories:

| Subdirectory | Description |
|:-------------|:------------|
| `idea_experiments/` | **Idealized soil thermal diffusivity (λ) inversion** — validates that differentiable numerical solvers retain hard physical constraints and that physical parameters and neural network weights are equivalent optimizable nodes. Includes direct optimization and MLP-based approaches. |
| `parameters_optimization_expriments/` | **Progressive-constraint parameter optimization** — tests four loss-function configurations (RSL, RHS, RHT, ALL) across three desert sites, comparing dLSEB optimization against pre-estimated (EST) baselines. Includes batch visualization and summary metrics. |
| `discussion/` | **Discussion experiments** — effective emissivity calculation from IRT observations, and a sensitivity experiment relaxing the emissivity upper-bound constraint (σ ≤ 1.5) to investigate observational energy non-closure. |

> See each subdirectory's `README.md` for detailed experiment descriptions and usage instructions.

---

## Study Sites

| Station | Location | AWS Years | EC Years |
|---------|----------|-----------|----------|
| **Ejin** | 100.99°E, 42.11°N | 2022–2024 | 2022–2024 |
| **Huazhaizi** | 100.32°E, 38.77°N | 2018–2020 | 2018–2020 |
| **Shenshawo** | 100.49°E, 37.46°N | 2012–2014 | 2012–2014 |

---

## Dependencies

- **Python** ≥ 3.12
- **JAX** 0.10.0 — Automatic differentiation & GPU/TPU computation
- **optax** — Optimizer (Adam)
- **lineax** — Linear solver (tridiagonal, for Crank-Nicolson)
- **optimistix** — Numerical solvers (Newton root-finding, fixed-point iteration, Implicit Adjoint)
- **numpy**, **pandas**, **matplotlib**, **scikit-learn**, **seaborn**, **openpyxl**

Dependencies are managed via [uv](https://docs.astral.sh/uv/). Install with:

```bash
uv sync
```

---

## Quick Start

### 1. Data Processing

```bash
cd data/1.origin && bash run_find_periods.sh          # Step 1: Find valid 10-day periods
cd ../2.data_selection && bash run_select_best.sh      # Step 2: Select best period & export CSVs
cd ../3.estimate_parameters && python "3.parameter_estimate.py" --all  # Step 3: Estimate parameters
```

### 2. Idealized Experiments

```bash
cd Experiments/idea_experiments
python "a.direct_lmd.py"          # Direct λ inversion
python "b.nn_lmd.py"             # MLP-based λ inversion
python visualize.py --direct ./output/direct_lmd --nn ./output/nn_lmd --output ./output/comparison.svg
```

### 3. Parameter Optimization Experiments

```bash
cd Experiments/parameters_optimization_expriments
python "0.EST/run_estimation.py" --all          # Generate EST baseline
python "1.run_experiments.py" --all             # Run all optimization experiments
python "2.batch_visualization.py"               # Generate figures & metrics
python 3.summary.py --all                       # Generate summary bar charts
```

### 4. Single-Site Testing (dLSEB standalone)

```bash
cd dLSEB
python main.py    # Runs the complete training pipeline for a single site (default: Ejin)
```

---

## License

This project is for academic research purposes.
