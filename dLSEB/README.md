# dLSEB — Differentiable Land Surface Energy Balance Model

**Author:** Shuaichen He

---

## Overview

dLSEB is a JAX-based differentiable land surface energy balance model. It computes turbulent fluxes via **Monin-Obukhov similarity theory**, solves soil heat transfer with a **Crank-Nicolson scheme**, and uses **optimistix** numerical solvers for efficient surface energy balance root-finding. The entire model is fully differentiable, enabling gradient-based parameter optimization and T₀ data assimilation.

## Model Structure

```
dLSEB/
├── config.py              # Model configuration (physical parameters, optimizer, training hyperparameters)
├── data_loader.py         # Data loading module
├── main.py                # Standalone single-site testing/training script
├── model/
│   ├── __init__.py        # Submodule entry point
│   ├── turbulence.py      # Turbulence module (Monin-Obukhov similarity theory)
│   ├── energy_balence.py  # Energy balance & soil heat transfer module
│   └── loss_update.py     # Loss function & optimization module
└── output/                # Default output directory
```

## Module Descriptions

For detailed algorithm implementations, mathematical formulations, and parameter descriptions, please refer to the docstrings and inline comments in each source file.

### `config.py` — Model Configuration

Defines shared physical constants (albedo, emissivity, roughness length, soil thermal parameters, etc.), soil layer geometry, optimizer settings (three independent Adam optimizers for different parameter groups), T₀ data assimilation flags, and training hyperparameters (learning rates, number of epochs, spin-up days, etc.). All modules import from `config` to share these settings.

### `data_loader.py` — Data Loading Module

Reads meteorological forcing data (shortwave/longwave radiation, air temperature, wind speed, air pressure, etc.), soil temperature and moisture profiles, and eddy-covariance flux observations from CSV files, then assembles them into JAX arrays for model training and simulation. Currently defaults to loading data from the Ejin site.

### `model/turbulence.py` — Turbulence Module

Implements surface turbulent flux calculation based on **Monin-Obukhov similarity theory**. Key features include:
- Paulson (1970) stability correction functions (`ψ_m`, `ψ_h`)
- Roughness length for heat (`z₀ₕ`) via the Chirtwood-Zilitinkevich formula
- Friction velocity `u_*` and temperature scale `θ_*` computation
- Convective gustiness correction (wind enhancement under free convection)
- Coupled `(u_*, θ_{v*})` fixed-point iteration via `optimistix.fixed_point`, yielding sensible heat flux `H` and latent heat flux `LE`

### `model/energy_balence.py` — Energy Balance & Soil Heat Transfer

Implements surface energy balance solving and soil temperature simulation:
- **Net radiation** (`Net_Rad`): shortwave + longwave radiation budget
- **Ground heat flux** (`G_sfc`): based on Fourier's law of heat conduction
- **Crank-Nicolson scheme** (`Soil_Crank_Nicolson_diff`): implicit multi-layer soil heat transfer, solved with `lineax` tridiagonal linear solver
- **Energy balance solver** (`solving_EB_surface_T`): solves for surface temperature via `optimistix.root_find` (Newton's method + Implicit Adjoint) such that the energy balance residual vanishes
- **Spin-up** (`spin_up_soil_T`): repeated forward simulation to establish initial soil temperature profile
- **Time-series scanning** (`scan_training_periods`): efficient traversal over all time steps using `lax.scan`

### `model/loss_update.py` — Loss Function & Optimization Module

Defines parameter optimization loss functions and update strategies:
- **RMSE loss**: multi-variable constraint combining soil temperature (4 layers), reflected shortwave `R_su`, outgoing longwave `R_lu`, sensible heat flux `H`, and ground heat flux at 6 cm depth `G`
- **Physical constraints** (`apply_constraints`): enforces physically plausible bounds for albedo, emissivity, roughness length, thermal conductivity, and heat capacity
- **Three independent Adam optimizers**: separate learning rates for `[α, σ, z₀ₘ]`, `[k]`, and `[logC]` parameter groups, accounting for differing sensitivities to the loss function
- **NaN-safe update** (`safe_update`): automatically skips gradient updates when NaN/Inf gradients are detected
- **T₀ data assimilation** (`safe_DA_T0`): gradient-based assimilation of the initial soil temperature profile to improve simulation accuracy

---

## Relationship Between `main.py` and Experiment Scripts

**`main.py`** is a standalone single-site testing/training script that runs the complete training pipeline (data loading → spin-up → parameter optimization → T₀ assimilation → result saving) for a single site (default: Ejin). It is suitable for quick debugging and single-site validation.

- **`Experiments/parameters_optimization_expriments/`**: batch parameter optimization experiments supporting multiple sites (Huazhaizi, Ejin, Shenshawo) and multiple experiment configurations (RSL, RHS, RHT, ALL)

These experiment scripts **do not depend on `main.py`**. Instead, they directly import dLSEB module functions (e.g., `spin_up_soil_T`, `scan_training_periods`, `loss`, `safe_update`) and assemble their own training loops and data processing pipelines, enabling more flexible batch experiment control.

## Dependencies

- JAX / jax.numpy
- [optax](https://github.com/google-deepmind/optax) — Optimizer (Adam)
- [lineax](https://github.com/patrick-kidger/lineax) — Linear solver (tridiagonal)
- [optimistix](https://github.com/patrick-kidger/optimistix) — Numerical solvers (Newton root-finding, fixed-point iteration, Implicit Adjoint)
- pandas — CSV data reading

## License

This project is for academic research purposes.
