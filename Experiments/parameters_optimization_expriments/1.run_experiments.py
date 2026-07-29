"""
Parameter Optimization Experiments for dLSEB
=============================================
Runs optimization experiments with different loss function configurations,
controlling which observation variables contribute to the loss
(progressively adding observations, ALL placed last for readability):

  1.RSL — Radiation only:                    Rsu + Rlu
  2.RHS — Radiation + Sensible heat:         Rsu + Rlu + H
  3.RHT — Radiation + H + Temperature:       Rsu + Rlu + H + T
  4.ALL — All loss terms:                    T + Rsu + Rlu + H + G(6cm)

Initial parameters are loaded from pre-estimated .npy files in
data/3.estimate_parameters/.

Output for each (site, experiment) is saved into:
  1.ALL/, 2.RSL/, 3.RHS/, 4.RHT/  respectively.

Usage
-----
All experiments, all sites (default):
    python "1.run_experiments.py"

Single site, all experiments:
    python "1.run_experiments.py" --site Huazhaizi

Single site, single experiment:
    python "1.run_experiments.py" --exp ALL --site Huazhaizi

All sites, single experiment:
    python "1.run_experiments.py" --exp ALL
"""

import os
import sys
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax


# =============================================================================
# Tee — duplicate stdout to a log file during an experiment run
# =============================================================================
class Tee:
    """Context manager that duplicates sys.stdout to a file.

    Usage:
        with Tee("/path/to/output.log"):
            print("this goes to both stdout and the log file")
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.stdout = None

    def __enter__(self):
        self.file = open(self.filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *args):
        sys.stdout = self.stdout
        if self.file:
            self.file.close()

    def write(self, data):
        self.stdout.write(data)
        if self.file:
            self.file.write(data)

    def flush(self):
        self.stdout.flush()
        if self.file:
            self.file.flush()


# =============================================================================
# Path resolution — must happen before dLSEB imports
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
DLSEB_DIR = os.path.join(REPO_ROOT, "dLSEB")

sys.path.insert(0, REPO_ROOT)  # enables  "from dLSEB.xxx import ..."
sys.path.insert(1, DLSEB_DIR)  # compat for dLSEB internal  "from config import ..."

if "EQX_ON_ERROR" not in os.environ:
    os.environ["EQX_ON_ERROR"] = "nan"

# ---------------------------------------------------------------------------
# Constants (mirror dLSEB/config.py, but kept local so experiments are
# self-contained and don't interfere with the main training config)
# ---------------------------------------------------------------------------
NUM_EPOCHS = 500
SPIN_UP_DAYS = 20

# Optimizer learning rates (three independent Adam groups)
BASE_LR_ASZ = 1e-2  # alpha, sigma, z0m
BASE_LR_K = 1e-1  # k
BASE_LR_C = 1e-1  # logC

# Per-parameter gradient scale: [alpha, sigma, z0m, k, logC]
#   - 1.0  → normal update (default, i.e. no scaling)
#   - 0    → freeze that parameter (zero gradient → no update)
#   - >1   → amplify learning / <1 → dampen learning for that parameter
GRAD_SCALE = [1.0, 1.0, 1.0, 1.0, 1.0]

# T0 data assimilation
DO_DA_T0 = True
T0_LR = 1

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_BASE = os.path.join(REPO_ROOT, "data", "2.data_selection")
PARAMS_DIR = os.path.join(REPO_ROOT, "data", "3.estimate_parameters")

SITES = ["Huazhaizi", "Ejin", "Shenshawo"]
# ── Experiment groups (progressively adding observations) ──
# RSL → RHS → RHT → ALL  (ALL placed last for progressive readability)
VALID_EXPS = ["RSL", "RHS", "RHT", "ALL"]

# Site-specific atmospheric reference height [m]
SITE_Z_ATM = {"Huazhaizi": 4.5, "Ejin": 4.5, "Shenshawo": 4.5}

# ---------------------------------------------------------------------------
# dlseb imports (after EQX_ON_ERROR and sys.path are set)
# ---------------------------------------------------------------------------
from dLSEB.model.energy_balence import spin_up_soil_T, scan_training_periods
from dLSEB.model.loss_update import rmse, apply_constraints


# =============================================================================
# Data Loader (site-aware, replaces dLSEB/data_loader.py which uses ../)
# =============================================================================
def load_site_data(site_name):
    """Load meteorological, soil, and EC data for a given site.

    Reads meteo_var.csv, soil.csv, ec_var.csv, air_constants.csv from
    DATA_BASE/{site_name}/ and assembles JAX arrays matching the dLSEB format.
    """
    import pandas as pd

    site_dir = os.path.join(DATA_BASE, site_name)

    data = pd.read_csv(os.path.join(site_dir, "meteo_var.csv"))
    soil_data = pd.read_csv(os.path.join(site_dir, "soil.csv"))
    flux = pd.read_csv(os.path.join(site_dir, "ec_var.csv"))
    air_constant = pd.read_csv(os.path.join(site_dir, "air_constants.csv"))

    rho = air_constant["rho"].astype(float).values

    # Meteorological forcing
    Rsd = jnp.where(
        data["DR"].astype(float).values > 0, data["DR"].astype(float).values, 0.0
    )
    Rld = data["DLR_Cor"].astype(float).values
    T_atm = data["Ta_5m"].astype(float).values + 273.15
    u = flux["Wnd"].astype(float).values
    P = data["Press"].astype(float).values
    q_atm = flux["H2O"].astype(float).values / 1000.0

    meteo_input = jnp.array([Rsd, Rld, T_atm, q_atm, u, rho, P])

    # Observations for loss
    Rsu = jnp.where(
        data["UR"].astype(float).values > 0, data["UR"].astype(float).values, 0.0
    )
    Rlu = data["ULR_Cor"].astype(float).values
    Hs = flux["Hs"].astype(float).values
    G = -1 * jnp.array(
        [
            soil_data["Gs_1"].astype(float).values,
            soil_data["Gs_2"].astype(float).values,
            soil_data["Gs_3"].astype(float).values,
        ]
    ).mean(axis=0)

    T_sfc = jnp.array(soil_data["Ts_0cm"].astype(float).values).T + 273.15

    Train_T = (
        jnp.array(
            [
                soil_data["Ts_2cm"].astype(float).values,
                soil_data["Ts_4cm"].astype(float).values,
                soil_data["Ts_10cm"].astype(float).values,
                soil_data["Ts_20cm"].astype(float).values,
            ]
        ).T
        + 273.15
    )

    return {
        "meteo_input": meteo_input,
        "Train_T": Train_T,
        "Rsu": Rsu,
        "Rlu": Rlu,
        "Hs": Hs,
        "G": G,
        "T_sfc": T_sfc,
    }


# =============================================================================
# Parameter Loader (from pre-estimated .npy files)
# =============================================================================
def load_initial_params(site_name):
    """Load pre-estimated parameters for a site.

    Returns raw (alpha, sigma, z0m, k, C) as Python floats,
    and the JAX-ready parameter vector  [alpha, sigma, z0m, k, log(C)].
    """
    npy_path = os.path.join(PARAMS_DIR, f"{site_name}.npy")
    params_dict = np.load(npy_path, allow_pickle=True).item()

    alpha = float(params_dict["alpha"])
    sigma = float(params_dict["sigma"])
    z0m = float(params_dict["z0m"])
    k = float(params_dict["k"])
    C = float(params_dict["C"])

    return alpha, sigma, z0m, k, C


# =============================================================================
# Loss Function Factory
# =============================================================================
def make_loss_fn(exp_mode):
    """Return a loss function that only includes terms specified by *exp_mode*.

    exp_mode  |  T (soil temp) | Rsu | Rlu | H | G (6cm)
    ----------+----------------+-----+-----+---+---------
    ALL       |      ✓         |  ✓  |  ✓  | ✓ |    ✓
    RSL       |                |  ✓  |  ✓  |   |
    RHS       |                |  ✓  |  ✓  | ✓ |
    RHT       |      ✓         |  ✓  |  ✓  | ✓ |
    """

    def loss_fn(params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc):
        alpha, sigma, z0m, k, C = params
        Rsu_hat, Rlu_hat, H_hat, G_hat, T_sfc_hat, T_hat = scan_training_periods(
            alpha, sigma, z0m, k, C, meteo_input, spin_up_T0
        )

        total = 0.0

        # Temperature term — used by ALL and RHT
        if exp_mode in ("ALL", "RHT"):
            total = total + rmse(T_hat[1:, :4], Train_T[1:, :4])

        # Radiation terms — used by ALL experiments
        if exp_mode in ("ALL", "RSL", "RHS", "RHT"):
            total = total + rmse(Rsu_hat, Rsu)
            total = total + rmse(Rlu_hat, Rlu)

        # Sensible heat term — used by ALL, RHS, RHT
        if exp_mode in ("ALL", "RHS", "RHT"):
            total = total + rmse(H_hat, Hs)

        # Ground heat flux at 6cm — ALL only (distinguishes ALL from RHT)
        if exp_mode == "ALL":
            total = total + rmse(k * (T_hat[0:, 2] - T_hat[0:, 1]) / 0.06, G[0:])

        return total

    return loss_fn


# =============================================================================
# Optimizer Factories
# =============================================================================
def make_optimizers():
    """Create three independent Adam optimizers (mirrors dLSEB/config.py)."""
    opt_asz = optax.adam(learning_rate=BASE_LR_ASZ)  # [alpha, sigma, z0m]
    opt_k = optax.adam(learning_rate=BASE_LR_K)  # [k]
    opt_C = optax.adam(learning_rate=BASE_LR_C)  # [logC]
    # t0_optimizer = optax.sgd(learning_rate=T0_LR*1e2)
    t0_optimizer = optax.adam(learning_rate=T0_LR)
    return opt_asz, opt_k, opt_C, t0_optimizer


# =============================================================================
# Custom Update Step (replicates safe_update, but with a configurable loss)
# =============================================================================
def make_update_fn(loss_fn, grad_scale=None):
    """Return a JIT-compiled safe_update-like function using *loss_fn*.

    Parameters
    ----------
    loss_fn : callable
        Loss function with signature ``(params, ...) -> scalar``.
    grad_scale : list or array of 5 floats, optional
        Per-parameter gradient scaling [α, σ, z₀ₘ, k, logC].
        Set an entry to 0 to freeze that parameter during optimisation.
        Defaults to ``[1, 1, 1, 1, 1]`` (no scaling).
    """
    if grad_scale is None:
        grad_scale = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
    else:
        grad_scale = jnp.array(grad_scale, dtype=jnp.float32)
    # Split into the three optimizer groups
    gs_asz = grad_scale[0:3]  # [α, σ, z₀ₘ]
    gs_k = grad_scale[3:4]  # [k]
    gs_C = grad_scale[4:5]  # [logC]

    optimizer_asz, optimizer_k, optimizer_C, _ = make_optimizers()

    @jax.jit
    def _update(
        params,
        opt_state_asz,
        opt_state_k,
        opt_state_C,
        meteo_input,
        Train_T,
        spin_up_T0,
        Rsu,
        Rlu,
        Hs,
        G,
        T_sfc,
    ):
        loss_val, grads = jax.value_and_grad(loss_fn)(
            params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
        )

        grad_ok = ~jnp.any(jnp.logical_or(jnp.isnan(grads), jnp.isinf(grads)))

        def do_update(_):
            params_asz = params[0:3]
            params_k = params[3:4]
            params_C = params[4:5]

            # Apply per-parameter gradient scale
            grads_asz = grads[0:3] * gs_asz
            grads_k = grads[3:4] * gs_k
            grads_C = grads[4:5] * gs_C

            updates_asz, new_opt_state_asz = optimizer_asz.update(
                grads_asz, opt_state_asz, params_asz
            )
            updates_k, new_opt_state_k = optimizer_k.update(
                grads_k, opt_state_k, params_k
            )
            updates_C, new_opt_state_C = optimizer_C.update(
                grads_C, opt_state_C, params_C
            )

            new_params_asz = optax.apply_updates(params_asz, updates_asz)
            new_params_k = optax.apply_updates(params_k, updates_k)
            new_params_C = optax.apply_updates(params_C, updates_C)

            new_params = jnp.concatenate([new_params_asz, new_params_k, new_params_C])
            return (
                apply_constraints(new_params),
                new_opt_state_asz,
                new_opt_state_k,
                new_opt_state_C,
            )

        def skip_update(_):
            return params, opt_state_asz, opt_state_k, opt_state_C

        new_params, new_opt_state_asz, new_opt_state_k, new_opt_state_C = lax.cond(
            grad_ok, do_update, skip_update, None
        )
        return (
            new_params,
            new_opt_state_asz,
            new_opt_state_k,
            new_opt_state_C,
            jnp.where(grad_ok, loss_val, jnp.nan),
        )

    return _update


def make_t0_update_fn(loss_fn):
    """Return a JIT-compiled T0 data-assimilation step using *loss_fn*."""
    _, _, _, t0_optimizer = make_optimizers()

    @jax.jit
    def _t0_update(
        params, spin_up_T0, meteo_input, Train_T, Rsu, Rlu, Hs, G, T_sfc, T0_opt_state
    ):
        T0_loss, T0_grads = jax.value_and_grad(loss_fn, argnums=3)(
            params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
        )

        grad_ok = ~jnp.any(jnp.logical_or(jnp.isnan(T0_grads), jnp.isinf(T0_grads)))

        def do_update(_):
            updates, new_T0_opt_state = t0_optimizer.update(
                T0_grads, T0_opt_state, spin_up_T0
            )
            new_T0 = optax.apply_updates(spin_up_T0, updates)
            return new_T0, new_T0_opt_state

        def skip_update(_):
            return spin_up_T0, T0_opt_state

        new_T0, new_T0_opt_state = lax.cond(grad_ok, do_update, skip_update, None)
        return new_T0, new_T0_opt_state

    return _t0_update


# =============================================================================
# Experiment Runner
# =============================================================================
def run_experiment(site_name, exp_mode):
    """Run one (site, experiment) optimisation and save results."""

    # --- Output directory & log file -------------------------------------
    exp_map = {"ALL": "1.ALL", "RSL": "2.RSL", "RHS": "3.RHS", "RHT": "4.RHT"}
    exp_dir = os.path.join(SCRIPT_DIR, exp_map[exp_mode])
    site_out_dir = os.path.join(exp_dir, site_name)
    os.makedirs(site_out_dir, exist_ok=True)
    log_path = os.path.join(site_out_dir, f"{exp_mode}.log")

    with Tee(log_path):
        result = _run_experiment_impl(site_name, exp_mode, site_out_dir)
    return result


def _run_experiment_impl(site_name, exp_mode, site_out_dir):
    """Core experiment logic (wrapped by Tee for log capture)."""
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {exp_mode}  |  Site: {site_name}")
    print(f"{'=' * 60}")

    # --- Set site-specific Z_ATM & clear JAX cache if changed -----------
    import dLSEB.config as dlseb_config

    z_atm = SITE_Z_ATM[site_name]
    if dlseb_config.Z_ATM != z_atm:
        dlseb_config.Z_ATM = z_atm
        jax.clear_caches()
        print(f"  [Config] Z_ATM set to {z_atm} m (JAX cache cleared)")

    # --- Load pre-estimated initial parameters ---------------------------
    alpha0, sigma0, z0m0, k0, C0 = load_initial_params(site_name)
    params = jnp.array([alpha0, sigma0, z0m0, k0, jnp.log(C0)])

    print(f"  Initial params (from estimate):")
    print(f"    α   = {alpha0:.4f},  σ = {sigma0:.4f},  z₀ₘ = {z0m0:.4e}")
    print(f"    k   = {k0:.4f},     C = {C0:.2e}")

    # --- Load data -------------------------------------------------------
    data = load_site_data(site_name)
    meteo_input = data["meteo_input"]
    Train_T = data["Train_T"]
    Rsu = data["Rsu"]
    Rlu = data["Rlu"]
    Hs = data["Hs"]
    G = data["G"]
    T_sfc = data["T_sfc"]

    # --- Build experiment-specific loss & update -------------------------
    loss_fn = make_loss_fn(exp_mode)
    update_fn = make_update_fn(loss_fn, GRAD_SCALE)
    t0_update_fn = make_t0_update_fn(loss_fn)

    # --- Initialise optimizers -------------------------------------------
    opt_asz, opt_k, opt_C, t0_opt = make_optimizers()
    opt_state_asz = opt_asz.init(params[0:3])
    opt_state_k = opt_k.init(params[3:4])
    opt_state_C = opt_C.init(params[4:5])

    # --- Spin-up ---------------------------------------------------------
    print(f"  Spin-up ({SPIN_UP_DAYS} iterations) ...")
    spin_up_T0 = spin_up_soil_T(
        alpha0, sigma0, z0m0, k0, jnp.log(C0), meteo_input, Train_T, SPIN_UP_DAYS
    )
    T0_opt_state = t0_opt.init(spin_up_T0)
    print(f"  Spin-up done.  T0[:4] = {spin_up_T0[:4]}")

    # --- Training Loop ---------------------------------------------------
    T_loss_values = []
    params_hat_values = []

    # Keep a copy of the last-good state in case the update produces NaN/Inf
    last_good = {
        "params": params,
        "opt_state_asz": opt_state_asz,
        "opt_state_k": opt_state_k,
        "opt_state_C": opt_state_C,
        "spin_up_T0": spin_up_T0,
        "T0_opt_state": T0_opt_state,
    }

    print(f"  Optimizing ({NUM_EPOCHS} epochs) ...")
    best_loss = float("inf")
    stale_epochs = 0
    prev_loss_2d = None
    flat_epochs = 0
    for epoch in range(NUM_EPOCHS):
        params, opt_state_asz, opt_state_k, opt_state_C, T_loss = update_fn(
            params,
            opt_state_asz,
            opt_state_k,
            opt_state_C,
            meteo_input,
            Train_T,
            spin_up_T0,
            Rsu,
            Rlu,
            Hs,
            G,
            T_sfc,
        )

        # ── Stop early if loss becomes NaN or Inf ──────────────────────
        loss_bad = jnp.logical_or(jnp.isnan(T_loss), jnp.isinf(T_loss))
        if loss_bad:
            T_loss = loss_fn(
                params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
            )
            loss_bad = jnp.logical_or(jnp.isnan(T_loss), jnp.isinf(T_loss))
            if loss_bad:
                print(
                    f"\n  [Early Stop] Loss NaN/Inf at epoch {epoch + 1}, "
                    f"reverting to last valid parameters."
                )
                # Restore last-good state
                params = last_good["params"]
                opt_state_asz = last_good["opt_state_asz"]
                opt_state_k = last_good["opt_state_k"]
                opt_state_C = last_good["opt_state_C"]
                spin_up_T0 = last_good["spin_up_T0"]
                T0_opt_state = last_good["T0_opt_state"]
                break

        # T0 data assimilation (every 10 epochs, only when loss is valid)
        if DO_DA_T0 and epoch % 10 == 0:
            spin_up_T0, T0_opt_state = t0_update_fn(
                params,
                spin_up_T0,
                meteo_input,
                Train_T,
                Rsu,
                Rlu,
                Hs,
                G,
                T_sfc,
                T0_opt_state,
            )

        # Record & save last-good snapshot
        loss_f = float(T_loss)
        params_hat_values.append(params)
        T_loss_values.append(loss_f)
        last_good = {
            "params": params,
            "opt_state_asz": opt_state_asz,
            "opt_state_k": opt_state_k,
            "opt_state_C": opt_state_C,
            "spin_up_T0": spin_up_T0,
            "T0_opt_state": T0_opt_state,
        }

        # ── Early stop conditions ────────────────────────────────────
        # (1) 0.1% relative tolerance plateau
        if loss_f < best_loss:
            best_loss = loss_f
            stale_epochs = 0
        elif abs(loss_f - best_loss) < 1e-6:  # essentially unchanged
            stale_epochs += 1
        else:  # loss is clearly worse → reset
            stale_epochs = 0

        # (2) Loss unchanged to 2 decimal places
        loss_2d = round(loss_f, 2)
        if prev_loss_2d is not None and loss_2d == prev_loss_2d:
            flat_epochs += 1
        else:
            flat_epochs = 0
        prev_loss_2d = loss_2d

        if stale_epochs >= 30 or flat_epochs >= 30:
            reason = (
                f"plateau (best_loss={best_loss:.6f})"
                if stale_epochs >= 30
                else f"flat to 3dp (loss≈{loss_2d:.3f})"
            )
            print(f"\n  [Early Stop] {reason} for 30 epochs " f"at epoch {epoch + 1}")
            break

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(
                f"    Epoch {epoch :>3d}/{NUM_EPOCHS}  "
                f"loss={loss_f:.6f}  "
                f"α={float(params[0]):.4f}  σ={float(params[1]):.4f}  "
                f"z₀ₘ={float(params[2]):.2e}  "
                f"k={float(params[3]):.4f}  C={float(jnp.exp(params[4])):.2e}  "
                f"T0={float(spin_up_T0[0]):.2f}"
            )

    # --- Save training history -------------------------------------------
    jnp.save(
        os.path.join(site_out_dir, "params_hat_values.npy"),
        jnp.array(params_hat_values),
    )
    jnp.save(
        os.path.join(site_out_dir, "T_loss_values.npy"),
        jnp.array(T_loss_values),
    )

    # --- Final simulation & save -----------------------------------------
    alpha_hat, sigma_hat, z0m_hat, k_hat, C_hat = params
    r_s, r_l, H, G_out, T_surface, T_result = scan_training_periods(
        alpha_hat, sigma_hat, z0m_hat, k_hat, C_hat, meteo_input, spin_up_T0
    )

    jnp.save(
        os.path.join(site_out_dir, "EB_result.npy"),
        jnp.array([r_s, r_l, H, G_out]),
    )
    jnp.save(
        os.path.join(site_out_dir, "T_result.npy"),
        jnp.array(T_result),
    )

    print(f"  [✓] Results saved →  {site_out_dir}/")
    final_loss = T_loss_values[-1] if T_loss_values else float("nan")
    print(f"      Final loss: {final_loss:.6f}")
    print(
        f"      Final α={float(params[0]):.4f}  σ={float(params[1]):.4f}  "
        f"z₀ₘ={float(params[2]):.2e}  k={float(params[3]):.4f}  "
        f"C={float(jnp.exp(params[4])):.2e}"
    )

    return {
        "site": site_name,
        "exp": exp_mode,
        "final_loss": T_loss_values[-1] if T_loss_values else float("nan"),
        "final_params": [float(v) for v in params],
    }


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="dLSEB parameter optimisation with configurable loss terms."
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        default=VALID_EXPS,
        choices=VALID_EXPS,
        help="Experiment mode(s): ALL, RSL, RHS, RHT (default: all 4)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--site",
        type=str,
        help=f"Single site name: {', '.join(SITES)}",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all 3 sites (Huazhaizi, Ejin, Shenshawo)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--no-da",
        action="store_true",
        help="Disable T0 data assimilation",
    )
    args = parser.parse_args()

    # Override globals from CLI
    NUM_EPOCHS = args.epochs
    if args.no_da:
        DO_DA_T0 = False

    experiments = args.exp
    if args.site:
        if args.site not in SITES:
            print(f"Error: unknown site '{args.site}'. Choose from {SITES}")
            sys.exit(1)
        sites = [args.site]
    else:
        sites = SITES  # default: all sites

    # Run all (site, experiment) combinations
    all_results = []
    total = len(sites) * len(experiments)
    count = 0

    t_start = time.time()
    for site in sites:
        for exp in experiments:
            count += 1
            print(f"\n{'~' * 60}")
            print(f"  [{count}/{total}]  {site} / {exp}")
            print(f"{'~' * 60}")
            result = run_experiment(site, exp)
            all_results.append(result)

    # -----------------------------------------------------------------
    # Summary (console + summary.log in the same directory)
    # -----------------------------------------------------------------
    summary_log_path = os.path.join(SCRIPT_DIR, "summary.log")
    with Tee(summary_log_path):
        print(f"\n{'=' * 60}")
        print(f"  ALL EXPERIMENTS COMPLETE")
        print(f"  Total time: {time.time() - t_start:.1f} s")
        print(f"{'=' * 60}")
        print(
            f"\n  {'Site':<15s} {'Exp':<6s} {'Final Loss':>12s}  "
            f"{'α':>8s} {'σ':>8s} {'z₀ₘ':>10s} {'k':>8s} {'C':>12s}"
        )
        print(
            f"  {'-' * 15} {'-' * 6} {'-' * 12}  "
            f"{'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 12}"
        )
        for r in all_results:
            p = r["final_params"]
            print(
                f"  {r['site']:<15s} {r['exp']:<6s} {r['final_loss']:>12.6f}  "
                f"{p[0]:>8.4f} {p[1]:>8.4f} {p[2]:>10.2e} "
                f"{p[3]:>8.4f} {jnp.exp(p[4]):>12.2e}"
            )
