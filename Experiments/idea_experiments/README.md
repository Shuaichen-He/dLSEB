# Idea Experiments: Soil Thermal Diffusivity (λ) Inversion

This directory contains experiments for inverting soil thermal diffusivity λ from analytical
temperature fields using two complementary approaches.

## File Overview

| File | Role |
|:-----|------|
| `common.py` | Shared module: soil physics (`soil_alpha`), analytical temperature solution, Crank-Nicolson solver, loss functions, parameter grid builder, and result evaluation. |
| `a.direct_lmd.py` | **Direct approach**: optimizes each sample's λ as an independent scalar via mini-batch Adam gradient descent. Gradient chain: `loss → CN_solve → λ`. |
| `b.nn_lmd.py` | **MLP approach**: trains a neural network to learn the `(θ, n) → λ` mapping. Gradient chain: `loss → CN_solve → λ → MLP → weights`. |
| `visualize.py` | Generates a 5-panel comparison figure (True λ / Direct estimate / MLP estimate / Direct error / MLP error). |

## How It Works

1. A `100 × 100` grid of `(θ, n)` is built; only points with `θ ≤ n` are kept (physical constraint).
2. True λ is computed via the soil thermal diffusivity model `soil_alpha(θ, n)`.
3. For each true λ, the analytical soil temperature field (sinusoidal surface BC, 10 days) is generated.
4. The Crank-Nicolson solver uses the Dirichlet BC from the analytical solution and an estimated λ to
   produce a numerical temperature field. The loss is the RMSE between analytical and numerical temperatures.
5. Two inversion strategies minimize this loss:
   - **Direct**: each sample's λ is an independently optimized parameter.
   - **MLP**: a neural net `(θ, n) → λ` is trained so the mapping generalizes across the parameter space.

## Usage

### 1. Run the direct inversion
```bash
python "a.direct_lmd.py"
```
Outputs are saved to `./output/direct_lmd/` (4 `.npy` files).

### 2. Run the MLP inversion
```bash
python "b.nn_lmd.py"
```
Outputs are saved to `./output/nn_lmd/` (4 `.npy` files).

### 3. Visualize results
```bash
python visualize.py --direct ./output/direct_lmd --nn ./output/nn_lmd --output ./output/comparison.svg
```
Generates a 5-panel figure comparing true λ, both estimates, and their relative errors.

## Dependencies

- `jax`, `jax.numpy`
- `optax`
- `lineax`
- `numpy`, `matplotlib`
