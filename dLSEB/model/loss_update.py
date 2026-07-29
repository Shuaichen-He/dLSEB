"""
Loss Function and Optimization Module
---------------------------------------
Provides RMSE loss evaluation, physical parameter constraints (albedo,
emissivity, roughness, conductivity, heat capacity), NaN-safe gradient
updates with three independent Adam optimizers, and T0 data assimilation.

Dependencies: model.energy_balence, config, optax
"""

import jax
import jax.numpy as jnp
import optax
from jax import lax

# Import from energy_balance module for scan_training_periods
from model.energy_balence import scan_training_periods

# Optimizer instances - initialized with configuration
from config import (
    OPTIMIZER_TYPE,
    # BASE_LR,
    T0_LR,
    T0_DA_LR,
    grad_scale,
    BASE_LR_ASZ,
    BASE_LR_K,
    BASE_LR_C,
)

# optimizer = optax.adam(learning_rate=BASE_LR)  # legacy single optimizer
# Three independent optimizers for different parameter groups
optimizer_asz = optax.adam(learning_rate=BASE_LR_ASZ)  # for [alpha, sigma, z0m]
optimizer_k = optax.adam(learning_rate=BASE_LR_K)  # for [k]
optimizer_C = optax.adam(learning_rate=BASE_LR_C)  # for [logC]
T0_optimizer = optax.adam(learning_rate=T0_LR)


# =============================================================================
# Loss Function and Optimization
# =============================================================================
# Math: \mathcal{L} = \mathrm{RMSE}(R_{su}, \hat{R}_{su}) + \mathrm{RMSE}(R_{lu}, \hat{R}_{lu})
# Math: \mathrm{RMSE}(y, \hat{y}) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}


def rmse(y_pre, y_label):
    """
    Calculate Root Mean Square Error.

    Args:
        y_pre: Predicted values
        y_label: Observed/label values

    Returns:
        RMSE value
    """
    return jnp.sqrt(jnp.mean(jnp.power(y_pre - y_label, 2.0)))


def loss(params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc):
    """
    Calculate loss function for parameter estimation.

    Loss includes:
    - Soil temperature RMSE at multiple depths
    - Reflected shortwave radiation RMSE
    - Outgoing longwave radiation RMSE
    - Sensible heat flux RMSE
    - Ground heat flux (at 6cm depth) RMSE

    Args:
        params: Model parameters [alpha, sigma, z0m, k, C]
        meteo_input: Meteorological forcing
        Train_T: Training soil temperature observations [K]
        spin_up_T0: Initial soil temperature profile [K]
        Rsu: Observed reflected shortwave radiation [W m^-2]
        Rlu: Observed outgoing longwave radiation [W m^-2]
        Hs: Observed sensible heat flux [W m^-2]
        G: Observed ground heat flux [W m^-2]
        T_sfc: Observed surface temperature [K]

    Returns:
        Total loss value
    """
    alpha, sigma, z0m, k, C = params
    Rsu_hat, Rlu_hat, H_hat, G_hat, T_sfc_hat, T_hat = scan_training_periods(
        alpha, sigma, z0m, k, C, meteo_input, spin_up_T0
    )
    return (
        rmse(T_hat[1:, :2], Train_T[1:, :2])
        + rmse(Rsu_hat, Rsu)
        + rmse(Rlu_hat, Rlu)
        # + rmse(H_hat, Hs)
        # + rmse(k * (T_hat[0:, 2] - T_hat[0:, 1]) / 0.06, G[0:])  # G at 6cm
    )


# =============================================================================
# Parameter Constraint Functions
# =============================================================================


@jax.jit
def apply_constraints(params):
    """
    Apply physical constraints to model parameters after update.

    - alpha:  [0.01,   0.999]   (albedo)
    - sigma:  [0.01,   1.5]     (emissivity)
    - z0m:    ≥ 1e-6           (roughness length for momentum)
    - k:      ≥ 0.1            (thermal conductivity)
    - logC:   ≥ log(1e4)       (heat capacity in log-space)
    """
    alpha = jnp.clip(params[0], 0.01, 0.999)
    sigma = jnp.clip(params[1], 0.01, 0.999)
    # sigma = jnp.clip(params[1], 0.01, 1.5)
    z0m = jnp.maximum(params[2], 1e-6)
    k = jnp.maximum(params[3], 1e-2)
    C = jnp.maximum(params[4], jnp.log(1e4))
    return jnp.array([alpha, sigma, z0m, k, C])


@jax.jit
def safe_update(
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
    """
    Compute gradient and update parameters with NaN protection.
    Uses three independent Adam optimizers for different parameter groups.

    Args:
        params: Current model parameters [alpha, sigma, z0m, k, logC]
        opt_state_asz: Adam optimizer state for [alpha, sigma, z0m]
        opt_state_k:   Adam optimizer state for [k]
        opt_state_C:   Adam optimizer state for [logC]
        meteo_input: Meteorological forcing
        Train_T: Training soil temperature observations
        spin_up_T0: Initial soil temperature profile
        Rsu: Observed reflected shortwave radiation
        Rlu: Observed outgoing longwave radiation
        Hs: Observed sensible heat flux
        G: Observed ground heat flux
        T_sfc: Observed surface temperature

    Returns:
        new_params: Updated (and constrained) parameters
        new_opt_state_asz: Updated optimizer state for asz group
        new_opt_state_k:   Updated optimizer state for k
        new_opt_state_C:   Updated optimizer state for logC
        loss_val: Loss value (nan if gradient was invalid)
    """
    loss_val, grads = jax.value_and_grad(loss)(
        params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
    )

    # Check for NaN/Inf in gradients — skip update if detected
    grad_ok = ~jnp.any(jnp.logical_or(jnp.isnan(grads), jnp.isinf(grads)))

    # Only apply optimizer update when gradients are valid
    def do_update(_):
        # Split params and grads into three groups
        params_asz = params[0:3]  # [alpha, sigma, z0m]
        params_k = params[3:4]  # [k]
        params_C = params[4:5]  # [logC]
        grads_asz = grads[0:3]
        grads_k = grads[3:4]
        grads_C = grads[4:5]

        # Update each group with its own optimizer
        updates_asz, new_opt_state_asz = optimizer_asz.update(
            grads_asz, opt_state_asz, params_asz
        )
        updates_k, new_opt_state_k = optimizer_k.update(grads_k, opt_state_k, params_k)
        updates_C, new_opt_state_C = optimizer_C.update(grads_C, opt_state_C, params_C)

        # Scale gradients and apply updates
        new_params_asz = optax.apply_updates(
            params_asz,
            updates_asz,  # * jnp.array(grad_scale[0:3])
        )
        new_params_k = optax.apply_updates(
            params_k,
            updates_k,  # * jnp.array(grad_scale[3:4])
        )
        new_params_C = optax.apply_updates(
            params_C,
            updates_C,  # * jnp.array(grad_scale[4:5])
        )

        # Concatenate back and apply physical constraints
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


@jax.jit
def update(
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
    """Simple Adam update with three independent optimizers (no NaN guard)."""
    loss_val, grads = jax.value_and_grad(loss)(
        params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
    )
    # Split and update each group
    updates_asz, new_opt_state_asz = optimizer_asz.update(
        grads[0:3], opt_state_asz, params[0:3]
    )
    updates_k, new_opt_state_k = optimizer_k.update(
        grads[3:4], opt_state_k, params[3:4]
    )
    updates_C, new_opt_state_C = optimizer_C.update(
        grads[4:5], opt_state_C, params[4:5]
    )
    new_params = jnp.concatenate(
        [
            optax.apply_updates(params[0:3], updates_asz),
            optax.apply_updates(params[3:4], updates_k),
            optax.apply_updates(params[4:5], updates_C),
        ]
    )
    return new_params, new_opt_state_asz, new_opt_state_k, new_opt_state_C, loss_val


@jax.jit
def safe_DA_T0(
    params, spin_up_T0, meteo_input, Train_T, Rsu, Rlu, Hs, G, T_sfc, T0_opt_state
):
    """
    Data assimilation for initial soil temperature profile.
    Updates spin_up_T0 using gradient descent to better match observations.

    Args:
        params: Current model parameters
        spin_up_T0: Current soil temperature profile
        meteo_input: Meteorological forcing
        Train_T: Training soil temperature observations
        Rsu, Rlu, Hs, G, T_sfc: Observation data for loss
        T0_opt_state: Optimizer state for T0 update

    Returns:
        new_T0: Updated soil temperature profile
        new_T0_opt_state: Updated optimizer state
    """
    T0_loss, T0_grads = jax.value_and_grad(loss, argnums=3)(
        params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
    )

    grad_ok = ~jnp.any(jnp.logical_or(jnp.isnan(T0_grads), jnp.isinf(T0_grads)))

    def do_update(_):
        updates, new_T0_opt_state = T0_optimizer.update(
            T0_grads, T0_opt_state, spin_up_T0
        )
        new_T0 = optax.apply_updates(spin_up_T0, updates)
        return new_T0, new_T0_opt_state

    def skip_update(_):
        return spin_up_T0, T0_opt_state

    new_T0, new_T0_opt_state = lax.cond(grad_ok, do_update, skip_update, None)
    return new_T0, new_T0_opt_state


@jax.jit
def DA_T0(
    params, spin_up_T0, meteo_input, Train_T, Rsu, Rlu, Hs, G, T_sfc, T0_opt_state
):
    """
    T0 data assimilation with optax Adam (no NaN guard).

    Args:
        params: Current model parameters
        spin_up_T0: Current soil temperature profile
        meteo_input: Meteorological forcing
        Train_T: Training soil temperature observations
        Rsu, Rlu, Hs, G, T_sfc: Observation data for loss
        T0_opt_state: Optimizer state for T0 update

    Returns:
        new_T0: Updated soil temperature profile
        new_T0_opt_state: Updated optimizer state
    """
    T0_loss, T0_grads = jax.value_and_grad(loss, argnums=3)(
        params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
    )
    updates, new_T0_opt_state = T0_optimizer.update(T0_grads, T0_opt_state, spin_up_T0)
    new_T0 = optax.apply_updates(spin_up_T0, updates)
    return new_T0, new_T0_opt_state


# =============================================================================
# Manual SGD Update (alternative to Adam)
# =============================================================================


@jax.jit
def update_sgd(params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc):
    """
    Manual SGD update with NaN protection.

    Args:
        params: Model parameters [alpha, sigma, z0m, k, C]
        meteo_input: Meteorological forcing
        Train_T: Training soil temperature observations
        spin_up_T0: Initial soil temperature profile
        Rsu, Rlu, Hs, G, T_sfc: Observation data

    Returns:
        Updated parameters
    """
    params_grad = jax.grad(loss)(
        params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
    )

    def do_update(_):
        return [(p - g * lr) for p, g, lr in zip(params, params_grad, grad_scale)]

    def skip_update(_):
        return list(params)

    return lax.cond(
        jnp.any(jnp.isnan(jnp.array(params_grad))),
        skip_update,
        do_update,
        None,
    )


# =============================================================================
# Manual T0 Data Assimilation (alternative to Adam)
# =============================================================================


@jax.jit
def DA_temp(params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc):
    """
    Manual data assimilation for initial soil temperature profile.
    Updates spin_up_T0 using gradient descent with fixed learning rate.

    Args:
        params: Model parameters
        meteo_input: Meteorological forcing
        Train_T: Training soil temperature observations
        spin_up_T0: Current soil temperature profile
        Rsu, Rlu, Hs, G, T_sfc: Observation data

    Returns:
        Updated spin_up_T0
    """
    temp_grad = jax.grad(loss, argnums=3)(
        params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
    )
    return spin_up_T0 - temp_grad * T0_DA_LR


# =============================================================================
# Auto-select Update Functions based on OPTIMIZER_TYPE
# =============================================================================


@jax.jit
def step(
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
    """
    Auto-select update function based on OPTIMIZER_TYPE in config.

    Uses three independent Adam optimizers for different parameter groups:
      - opt_state_asz: for [alpha, sigma, z0m]
      - opt_state_k:   for [k]
      - opt_state_C:   for [logC]

    - "adam": Uses safe_update with optax Adam optimizer
    - "sgd": Uses update_sgd with manual gradient descent

    Args:
        params: Model parameters [alpha, sigma, z0m, k, logC]
        opt_state_asz: Optimizer state for [alpha, sigma, z0m]
        opt_state_k:   Optimizer state for [k]
        opt_state_C:   Optimizer state for [logC]
        meteo_input: Meteorological forcing
        Train_T: Training soil temperature observations
        spin_up_T0: Initial soil temperature profile
        Rsu, Rlu, Hs, G, T_sfc: Observation data

    Returns:
        new_params: Updated parameters
        new_opt_state_asz: Updated optimizer state for asz group
        new_opt_state_k:   Updated optimizer state for k
        new_opt_state_C:   Updated optimizer state for logC
        loss_val: Loss value
    """
    use_adam = OPTIMIZER_TYPE == "adam"

    def adam_step(_):
        return safe_update(
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

    def sgd_step(_):
        new_params = update_sgd(
            params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
        )
        new_params = jnp.array(new_params)
        loss_val = loss(
            new_params, meteo_input, Train_T, spin_up_T0, Rsu, Rlu, Hs, G, T_sfc
        )
        return new_params, opt_state_asz, opt_state_k, opt_state_C, loss_val

    return lax.cond(use_adam, adam_step, sgd_step, None)


@jax.jit
def step_T0(
    params, spin_up_T0, meteo_input, Train_T, Rsu, Rlu, Hs, G, T_sfc, T0_opt_state
):
    """
    T0 data assimilation using optax Adam optimizer.

    Args:
        params: Model parameters
        spin_up_T0: Current soil temperature profile
        meteo_input: Meteorological forcing
        Train_T: Training soil temperature observations
        Rsu, Rlu, Hs, G, T_sfc: Observation data
        T0_opt_state: Optimizer state for T0

    Returns:
        new_T0: Updated soil temperature profile
        new_T0_opt_state: Updated optimizer state
    """
    return DA_T0(
        params, spin_up_T0, meteo_input, Train_T, Rsu, Rlu, Hs, G, T_sfc, T0_opt_state
    )
