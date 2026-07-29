"""
Common module: soil thermal diffusivity parameterization, space-time grid,
Crank-Nicolson solver, loss function, parameter grid construction, and result evaluation.

Shared by direct_lmd.py and nn_lmd.py.
"""

from functools import partial

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax
from jax import lax, vmap
import lineax as lx

# ============================================================
# 0. Global configuration
# ============================================================
DT = 1800  # time step [s]
Nz = 21  # number of spatial layers (incl. boundaries)
z_max = 2.0  # soil depth [m]
Nt = 480  # number of time steps (steps + 1 = time points)

# Spatial grid
z_full = jnp.linspace(0, z_max, Nz)  # (21,)
dz_edges = jnp.diff(z_full)  # (20,)
cz = (z_full[:-1] + z_full[1:]) / 2  # cell-center coordinates (20,)
dcz = jnp.diff(cz)  # cell-center spacing (19,)

# Time grid
t = jnp.linspace(0, 10 * 24 * 3600, Nt + 1)  # 10 days, 481 time points
T_mean, A_amp, omega, beta = 15.0, 8.0, 2 * jnp.pi / 86400, -jnp.pi / 2

# Parameter space & training config
n_theta, n_n = 100, 100
theta_lim = (0.0, 0.6)
n_lim = (0.02, 0.8)
batch_size = 1000
n_batches_per_epoch = 10


# ============================================================
# 1. Physical model: soil properties → thermal diffusivity (lambda / α)
#
#    lambda = k_eff / C_eff   [m^{2} s^{-1}]
#
#    where:
#      k_eff  — effective thermal conductivity [W m^{-1} K^{-1}]
#      C_eff  — effective volumetric heat capacity [J m^{-3} K^{-1}]
#
#    Volume fractions f (solid s / water w / air a):
#      f_s = 1 - n          (solid = 1 − porosity)
#      f_w = θ              (liquid = volumetric water content)
#      f_a = n - θ          (gas = porosity − water content)
#
#    k_eff — geometric mean model:
#      k_eff = k_s^{f_s} · k_w^{f_w} · k_a^{f_a}
#
#    C_eff — volume-weighted arithmetic mean:
#      C_eff = f_s·C_s + f_w·C_w + f_a·C_a
# ============================================================
@partial(jax.jit, static_argnames=("k_s", "k_w", "k_a",
                                      "C_s", "C_w", "C_a"))
def soil_alpha(
    theta,
    n,
    # --- thermal conductivity k [W m^{-1} K^{-1}] ---
    k_s=2.0,        # solid (soil particles)
    k_w=0.594,      # liquid (water)
    k_a=0.025,      # gas (air)
    # --- volumetric heat capacity C [J m^{-3} K^{-1}] ---
    C_s=2.0e6,      # solid (soil particles)
    C_w=4.18e6,     # liquid (water)
    C_a=1.25e3,     # gas (air)
):
    """
    Compute thermal diffusivity λ (lambda) from soil moisture θ and porosity n.

    Args:
        theta : volumetric water content, 0 ≤ θ ≤ n ≤ 1
        n     : porosity

    Returns:
        lambda = k_eff / C_eff   thermal diffusivity [m^{2} s^{-1}]
    """
    # Volume fractions of each phase
    fs = 1.0 - n            # solid
    fw = theta              # liquid (water)
    fa = n - theta          # gas (air)

    # Effective thermal conductivity — geometric mean model
    k_eff = (k_s**fs) * (k_w**fw) * (k_a**fa)

    # Effective volumetric heat capacity — volume-weighted arithmetic mean
    C_eff = fs * C_s + fw * C_w + fa * C_a

    # Thermal diffusivity = thermal conductivity / volumetric heat capacity
    lambda_ = k_eff / C_eff
    return lambda_


# ============================================================
# 2. Analytical temperature solution (heat equation under sinusoidal BC)
# ============================================================
@jax.jit
def analytical_T(lmd, z_grid, t_grid):
    """
    Generate full space-time analytical temperature field.
    lmd: (n_samples,) or scalar
    z_grid: (Nz,)
    t_grid: (Nt+1,)
    Returns: (n_samples, Nz, Nt+1)
    """
    Z, T = jnp.meshgrid(z_grid, t_grid, indexing="ij")  # (Nz, Nt+1)
    lmd_exp = jnp.expand_dims(lmd, axis=(1, 2))  # (n_samples, 1, 1)
    damping = jnp.sqrt(omega / (2 * lmd_exp))
    return T_mean + A_amp * jnp.exp(-Z * damping) * jnp.sin(
        omega * T + beta - Z * damping
    )


# ============================================================
# 3. Crank-Nicolson numerical solver (Dirichlet BC)
# ============================================================
@jax.jit
def solve_CN_step(lmd, T0, up_b_prev, low_b_prev, up_b_curr, low_b_curr):
    """
    Single CN time step.
    T0: interior temperature (19,)
    up_b_prev/low_b_prev: boundary values at previous time step
    up_b_curr/low_b_curr: boundary values at current time step
    """
    A = (lmd * DT) / (2 * dcz**2)  # (19,)

    # Tridiagonal matrix: (1+2A) on main diagonal, -A on sub/super diagonals
    operator = lx.TridiagonalLinearOperator(1 + 2 * A, -A[1:], -A[:-1])

    # Explicit RHS matrix
    tri_b = jnp.diag(1 - 2 * A) + jnp.diag(A[:-1], 1) + jnp.diag(A[1:], -1)

    # Boundary contribution terms
    b_boundary = jnp.concatenate(
        [
            jnp.array([A[0] * (up_b_curr + up_b_prev)]),
            jnp.zeros(len(T0) - 2),
            jnp.array([A[-1] * (low_b_curr + low_b_prev)]),
        ]
    )

    b = tri_b @ T0 + b_boundary
    solution = lx.linear_solve(operator, b, solver=lx.Tridiagonal())
    return solution.value


def CN_scan_step(carry, bc_input):
    """Step-advancing function for lax.scan."""
    lmd, T_prev, up_b_prev, low_b_prev = carry
    up_b_curr, low_b_curr = bc_input
    T_curr = solve_CN_step(lmd, T_prev, up_b_prev, low_b_prev, up_b_curr, low_b_curr)
    return (lmd, T_curr, up_b_curr, low_b_curr), T_curr


@jax.jit
def CN_solve(lmd, T_ana):
    """
    Use CN to extract BC/IC from the analytical temperature and solve numerically.
    lmd: scalar
    T_ana: (Nz, Nt+1) full space-time analytical temperature field
    Returns: (Nz-2, Nt) interior-node temperatures for all time steps
    """
    T0 = T_ana[1:-1, 0]  # (19,)
    bc_top = T_ana[0, :]  # (Nt+1,)
    bc_bot = T_ana[-1, :]  # (Nt+1,)

    init = (lmd, T0, bc_top[0], bc_bot[0])
    xs = (bc_top[1:], bc_bot[1:])  # each (Nt,)
    _, T_seq = lax.scan(CN_scan_step, init, xs)  # (Nt, 19)
    return T_seq.T  # (19, Nt)


# ============================================================
# 4. Loss function
# ============================================================
@jax.jit
def rmse(y_pred, y_true):
    return jnp.sqrt(jnp.mean((y_pred - y_true) ** 2))


@jax.jit
def loss_fn(lmd, T_ana):
    """RMSE between CN numerical solution and analytical solution."""
    T_num = CN_solve(lmd, T_ana)  # (19, Nt)
    T_ref = T_ana[1:-1, 1:]  # (19, Nt)
    return rmse(T_num, T_ref)


# ============================================================
# 5. Build valid parameter grid
# ============================================================
def build_valid_grid():
    """Build a (theta, n) grid and filter valid points (theta <= n)."""
    theta_vals = jnp.linspace(*theta_lim, n_theta)
    n_vals = jnp.linspace(*n_lim, n_n)
    TT, NN = jnp.meshgrid(theta_vals, n_vals)  # (100, 100)
    mask = TT <= NN

    idx = jnp.where(mask)
    theta_valid = TT[idx].flatten()
    n_valid = NN[idx].flatten()

    # Compute true lambda
    lmd_true = vmap(soil_alpha)(theta_valid, n_valid)

    return theta_valid, n_valid, lmd_true


# ============================================================
# 6. Result evaluation & output
# ============================================================
def evaluate(theta_arr, n_arr, lmd_true_arr, lmd_est_arr, output_dir="./output", tag=""):
    """
    Print inversion statistics and save data.

    output_dir: directory for saving npy files, e.g. "./output/direct_lmd", "./output/nn_lmd"
    tag: label suffix for statistics header, e.g. "(MLP)"
    """
    rel_err = jnp.abs(lmd_est_arr - lmd_true_arr) / lmd_true_arr
    abs_err = jnp.abs(lmd_est_arr - lmd_true_arr)

    sep = f" {tag}" if tag else ""
    print(f"\n{'='*60}")
    print(f"Inversion Result Statistics{sep}")
    print(f"{'='*60}")
    print(
        f"  True λ   range: [{float(jnp.min(lmd_true_arr)):.3e}, "
        f"{float(jnp.max(lmd_true_arr)):.3e}] m^{2} s^{-1}"
    )
    print(
        f"  Est  λ   range: [{float(jnp.min(lmd_est_arr)):.3e}, "
        f"{float(jnp.max(lmd_est_arr)):.3e}] m^{2} s^{-1}"
    )
    print(f"  Abs err  mean : {float(jnp.mean(abs_err)):.3e} m^{2} s^{-1}")
    print(f"  Rel err  mean : {float(jnp.mean(rel_err))*100:.2f} %")
    print(f"  Rel err median : {float(jnp.median(rel_err))*100:.2f} %")
    print(f"  Rel err  90%ile: {float(jnp.percentile(rel_err, 90))*100:.2f} %")

    # Save to specified directory
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "theta_valid.npy", np.array(theta_arr))
    np.save(out / "n_valid.npy", np.array(n_arr))
    np.save(out / "lmd_true.npy", np.array(lmd_true_arr))
    np.save(out / "lmd_est.npy", np.array(lmd_est_arr))
    print(
        f"\nData saved to: {out}/  (theta_valid.npy, n_valid.npy, lmd_true.npy, lmd_est.npy)"
    )
