"""
MLP training: learn a function mapping (θ, n) → λ via a neural network.

Gradient chain: loss → CN_solve → λ → MLP → weights
"""

import jax
import jax.numpy as jnp
from jax import vmap, random
import optax

from common import (
    analytical_T,
    batch_size,
    build_valid_grid,
    evaluate,
    loss_fn,
    n_batches_per_epoch,
    t,
    z_full,
)


# ============================================================
# MLP definition: (θ, n) → λ
# ============================================================
def init_mlp_params(key, layer_sizes=(2, 32, 32, 16, 1)):
    """
    Initialize MLP weights and biases with uniform random distribution.

    Weights ~ U(-1, 1), biases ~ U(-0.1, 0.1).
    Simple initialization; the optimizer handles the rest.
    """
    params = []
    for i in range(len(layer_sizes) - 1):
        key, w_key, b_key = random.split(key, 3)
        W = random.uniform(w_key, (layer_sizes[i], layer_sizes[i + 1]),
                           minval=-1.0, maxval=1.0)
        b = random.uniform(b_key, (layer_sizes[i + 1],),
                           minval=-0.1, maxval=0.1)
        params.append((W, b))
    return params


def mlp_forward(params, x):
    """
    MLP forward pass.
    x: (batch, 2) → returns: (batch,)
    """
    for i, (W, b) in enumerate(params):
        x = jnp.dot(x, W) + b
        if i < len(params) - 1:
            x = jnp.tanh(x)
    return jnp.squeeze(x)


@jax.jit
def predict_lambda(params, theta, n):
    """
    MLP: (θ, n) → λ

    theta, n: (batch,) or scalar
    Returns: (batch,) or scalar, always > 0

    Design rationale: MLP outputs real → softplus guarantees positivity →
    multiply by a tunable scale. Avoids wide-range sigmoid log mapping,
    which would constrain the MLP to sigmoid's narrow linear region,
    thus fully unleashing the MLP's nonlinear expressivity.
    """
    # Input normalization to [-1, 1] for improved isotropy
    theta_norm = 2.0 * (theta - 0.3) / 0.6       # θ∈[0,0.6] → [-1,1]
    n_norm = 2.0 * (n - 0.46) / 0.88              # n∈[0.02,0.9] → [-1,1]

    x = jnp.stack([theta_norm, n_norm], axis=-1)  # (batch, 2)
    raw = mlp_forward(params, x)                   # (batch,)

    # softplus: ensures λ > 0, gradients nonzero everywhere (avoids ReLU dead zones)
    # Multiply by 1e-6 to place initial output near physical scale
    # (softplus(0) = ln2 ≈ 0.693)
    return jax.nn.softplus(raw) * 1e-6


# ============================================================
# MLP mini-batch training loop (optax.adam)
# ============================================================
def _batch_loss_fn(params, batch_theta, batch_n, batch_T_ana):
    """Batch loss helper, compiled via JIT."""
    lmd_pred = predict_lambda(params, batch_theta, batch_n)       # (B,)
    losses = vmap(loss_fn)(lmd_pred, batch_T_ana)                 # (B,)
    return jnp.mean(losses)


def train_mlp(theta_arr, n_arr, lmd_true_arr,
              lr=1e-3, n_epochs=100, seed=42):
    """
    Train MLP with optax.adam to learn the θ,n → λ mapping.

    Gradient chain:
      batch_loss = mean( RMSE( CN_solve(MLP(θ,n)), T_ana_ref ) )
      ∂loss/∂weights ← ∂CN/∂λ ← ∂MLP/∂weights

    theta_arr/n_arr: (N,) valid parameter points (θ≤n), inputs to the MLP
    lmd_true_arr: (N,) true λ values, used to generate training targets
    """
    N = len(theta_arr)
    key = random.PRNGKey(seed)

    # Initialize MLP parameters
    key, init_key = random.split(key)
    params = init_mlp_params(init_key)

    # Count parameters
    n_params = sum(w.size + b.size for w, b in params)
    # Dynamically build network structure description
    layer_sizes_desc = " → ".join(str(s) for s in [2, 32, 32, 16, 1])
    print(f"Valid parameter points (θ≤n): {N}")
    print(f"MLP architecture: {layer_sizes_desc} | Trainable params: {n_params}")
    print(f"Optimizer: optax.adam | LR: {lr}")
    print(f"Batch size: {batch_size}, batches per epoch: {n_batches_per_epoch}")
    print(f"Epochs: {n_epochs}")
    print(f"{'='*60}")

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    loss_epoch_history = []

    # Precompile batch loss + gradient: JIT + value_and_grad combined
    batch_value_and_grad = jax.jit(jax.value_and_grad(_batch_loss_fn))

    for epoch in range(n_epochs):
        # Shuffle all indices
        key, subkey = random.split(key)
        perm = random.permutation(subkey, N)

        epoch_T_losses = []
        epoch_lmd_losses = []

        for batch_i in range(n_batches_per_epoch):
            start = (batch_i * batch_size) % N
            idx_list = jnp.arange(start, start + batch_size) % N
            indices = perm[idx_list]

            batch_theta = theta_arr[indices]                    # (B,)
            batch_n = n_arr[indices]                            # (B,)
            batch_lmd_true = lmd_true_arr[indices]              # (B,)

            # Analytical temperature for current batch (generated with true λ as target)
            batch_T_ana = analytical_T(batch_lmd_true, z_full, t)  # (B, Nz, Nt+1)

            # Batch loss + gradient (JIT compiled, single forward+backward pass)
            loss_val, grads = batch_value_and_grad(
                params, batch_theta, batch_n, batch_T_ana
            )

            # optax update of MLP parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            epoch_T_losses.append(float(loss_val))
            # λ relative error RMSE
            batch_lmd_pred = predict_lambda(params, batch_theta, batch_n)
            epoch_lmd_losses.append(
                float(jnp.sqrt(jnp.mean(((batch_lmd_pred - batch_lmd_true) / batch_lmd_true) ** 2)))
            )

        avg_T_loss = float(jnp.mean(jnp.array(epoch_T_losses)))
        avg_lmd_loss = float(jnp.mean(jnp.array(epoch_lmd_losses)))
        loss_epoch_history.append(avg_T_loss)
        print(
            f"Epoch {epoch+1:3d}/{n_epochs}"
            f" | T_loss = {avg_T_loss:.6f} K"
            f" | λ_rRMSE = {avg_lmd_loss*100:.2f} %"
        )

    # Final prediction of λ for all valid parameter points
    lmd_est = predict_lambda(params, theta_arr, n_arr)

    return lmd_est, loss_epoch_history


# ============================================================
# Main entry
# ============================================================
if __name__ == "__main__":
    # ---- 1. Build 100×100 valid parameter grid ----
    theta_valid, n_valid, lmd_true = build_valid_grid()

    # ---- 2. Train MLP to map (θ, n) → λ ----
    lmd_est, loss_hist = train_mlp(
        theta_valid, n_valid, lmd_true,
        lr=1e-3, n_epochs=50, seed=42
    )

    # ---- 3. Print statistics & save data ----
    evaluate(theta_valid, n_valid, lmd_true, lmd_est, output_dir="./output/nn_lmd", tag="(MLP)")
