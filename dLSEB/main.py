"""
dLSEB Main Training Script
----------------------------
Orchestrates data loading, soil spin-up, parameter optimization (Adam/SGD),
T0 data assimilation, and output saving. Uses three independent optimizers
for [alpha, sigma, z0m], [k], and [logC] parameter groups.

Dependencies: config, data_loader, model.energy_balence, model.loss_update
"""

import os
import jax.numpy as jnp
import time

from config import (
    ALPHA,
    SIGMA,
    Z0M,
    K,
    C,
    NUM_EPOCHS,
    SPIN_UP_DAYS,
    OUTPUT_DIR,
    DO_DA_T0,
    USE_TEMP_DA,
)
from data_loader import load_all_data
from model.energy_balence import spin_up_soil_T, scan_training_periods
from model.loss_update import (
    step,
    safe_DA_T0,
    DA_temp,
    loss,
    optimizer_asz,
    optimizer_k,
    optimizer_C,
    T0_optimizer,
)

# =============================================================================
# Physical Parameters
# =============================================================================

params = jnp.array([ALPHA, SIGMA, Z0M, K, jnp.log(C)])

# Optimizer states — three independent optimizers for different parameter groups
opt_state_asz = optimizer_asz.init(params[0:3])  # for [alpha, sigma, z0m]
opt_state_k = optimizer_k.init(params[3:4])  # for [k]
opt_state_C = optimizer_C.init(params[4:5])  # for [logC]

# =============================================================================
# Data Loading
# =============================================================================

data = load_all_data()
meteo_input = data["meteo_input"]
Train_T = data["Train_T"]
Rsu = data["Rsu"]
Rlu = data["Rlu"]
Hs = data["Hs"]
G = data["G"]
T_sfc = data["T_sfc"]

# =============================================================================
# Spin-up Initialization
# =============================================================================

spin_up_T0 = spin_up_soil_T(
    ALPHA, SIGMA, Z0M, K, jnp.log(C), meteo_input, Train_T, SPIN_UP_DAYS
)
T0_opt_state = T0_optimizer.init(spin_up_T0)

# =============================================================================
# Training Loop
# =============================================================================

T_loss_values = []
params_hat_values = []

for epoch in range(NUM_EPOCHS):
    start_time = time.time()

    # Update model parameters (auto-select based on OPTIMIZER_TYPE in config)
    # Uses three independent optimizers for [alpha,sigma,z0m], [k], and [logC]
    params, opt_state_asz, opt_state_k, opt_state_C, T_loss = step(
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
    # print(params, opt_state, T_loss)
    # Stop training if loss is NaN (gradient explosion / invalid state)
    if jnp.isnan(T_loss):
        T_loss = loss(params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc)
        if jnp.isnan(T_loss):
            print(
                f"\n[Early Stop] Loss is NaN at epoch {epoch + 1}, stopping training."
            )
            break

    # -------------------------------------------------------------------------
    # Data Assimilation for Initial Soil Temperature (optional)
    # -------------------------------------------------------------------------
    if DO_DA_T0:
        if USE_TEMP_DA:
            spin_up_T0 = DA_temp(
                params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
            )
        else:
            spin_up_T0, T0_opt_state = safe_DA_T0(
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

    # Record training history
    params_hat_values.append(params)
    T_loss_values.append(float(T_loss))

    end_time = time.time()

    # Print training progress
    print("#------------------------------------------#")
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Time: {end_time - start_time:.2f}s")
    print("#------------------------------------------#")
    print(f"T_loss: {T_loss:.6f}")
    print(f"T0 =  {spin_up_T0}")
    print("#------------------------------------------")
    print(f"alpha_hat: {params[0]:.6f}, sigma_hat: {params[1]:.6f}")
    print(f"z0m: {params[2]:.6f}")
    print(f"k_hat: {params[3]:.6f}, C_hat: {jnp.exp(params[4]):.6f}")
    print("#***********************************************#")

# =============================================================================
# Save Training Results
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

jnp.save(
    os.path.join(OUTPUT_DIR, "params_hat_values.npy"), jnp.array(params_hat_values)
)
jnp.save(os.path.join(OUTPUT_DIR, "T_loss_values.npy"), jnp.array(T_loss_values))

# =============================================================================
# Final Simulation and Save
# =============================================================================

alpha_hat, sigma_hat, z0m_hat, k_hat, C_hat = params

r_s, r_l, H, G_out, T_surface, T_result = scan_training_periods(
    alpha_hat, sigma_hat, z0m_hat, k_hat, C_hat, meteo_input, spin_up_T0
)

jnp.save(os.path.join(OUTPUT_DIR, "EB_result.npy"), jnp.array([r_s, r_l, H, G_out]))
jnp.save(os.path.join(OUTPUT_DIR, "T_result.npy"), jnp.array(T_result))
