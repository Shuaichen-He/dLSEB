"""
Direct λ training: mini-batch gradient descent optimizes each sample's λ independently.

The optimization targets N independent λ scalars; (θ, n) are only used to generate
training targets (analytical temperature).
Gradient chain: loss → CN_solve → λ
"""

import jax
import jax.numpy as jnp
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
# Mini-batch training loop (optax.adam)
# ============================================================
def train_minibatch(theta_arr, n_arr, lmd_true_arr, lr=1e-7, n_epochs=20, seed=42):
    """
    optax.adam mini-batch training.
    lmd_true_arr: (N,) true λ values used to generate training targets (analytical temperature)
    theta_arr/n_arr: (N,) used only for evaluation/saving, not in gradient computation.
    Each epoch: randomly draw 10 batches, 1000 samples each, update corresponding λ values.
    """
    N = len(theta_arr)
    key = jax.random.PRNGKey(seed)

    # Randomly initialize all guessed λ values
    key, subkey = jax.random.split(key)
    lmd_guess = jax.random.uniform(subkey, (N,), minval=5e-8, maxval=1e-6)

    # optax Adam optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(lmd_guess)

    print(f"Number of samples: {N}")
    print(f"Optimizer: optax.adam | LR: {lr}")
    print(f"Batch size: {batch_size}, batches per epoch: {n_batches_per_epoch}")
    print(f"Epochs: {n_epochs}")
    print(f"{'='*60}")

    # Precompile batch gradient: JIT + vmap + value_and_grad combined
    batch_value_and_grad = jax.jit(
        jax.vmap(jax.value_and_grad(loss_fn), in_axes=(0, 0))
    )

    loss_epoch_history = []

    for epoch in range(n_epochs):
        # Shuffle all indices
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, N)

        epoch_T_losses = []
        epoch_lmd_losses = []

        for batch_i in range(n_batches_per_epoch):
            start = (batch_i * batch_size) % N
            idx_list = jnp.arange(start, start + batch_size) % N
            indices = perm[idx_list]

            # Current batch data
            batch_lmd_true = lmd_true_arr[indices]  # (B,)
            batch_lmd_guess = lmd_guess[indices]  # (B,)

            # Compute analytical temperature
            batch_T_ana = analytical_T(batch_lmd_true, z_full, t)  # (B, Nz, Nt+1)

            # Batched gradient + loss
            batch_losses, batch_grads = batch_value_and_grad(
                batch_lmd_guess, batch_T_ana
            )  # (B,), (B,)

            # Build full gradient vector (non-batch positions = 0)
            full_grads = jnp.zeros_like(lmd_guess)
            full_grads = full_grads.at[indices].set(batch_grads)

            # optax update
            updates, opt_state = optimizer.update(full_grads, opt_state, lmd_guess)
            lmd_guess = optax.apply_updates(lmd_guess, updates)
            lmd_guess = jnp.clip(lmd_guess, 1e-9, 1e-4)

            epoch_T_losses.append(float(jnp.mean(batch_losses)))
            # λ relative error RMSE
            batch_lmd_guess_updated = lmd_guess[indices]
            epoch_lmd_losses.append(
                float(jnp.sqrt(jnp.mean(((batch_lmd_guess_updated - batch_lmd_true) / batch_lmd_true) ** 2)))
            )

        avg_T_loss = float(jnp.mean(jnp.array(epoch_T_losses)))
        avg_lmd_loss = float(jnp.mean(jnp.array(epoch_lmd_losses)))
        loss_epoch_history.append(avg_T_loss)
        print(
            f"Epoch {epoch+1:3d}/{n_epochs}"
            f" | T_loss = {avg_T_loss:.6f} K"
            f" | λ_rRMSE = {avg_lmd_loss*100:.2f} %"
        )

    return lmd_guess, loss_epoch_history


# ============================================================
# Main entry
# ============================================================
if __name__ == "__main__":
    # ---- 1. Build parameter grid and compute true λ ----
    theta_valid, n_valid, lmd_true = build_valid_grid()

    # ---- 2. Mini-batch gradient descent training ----
    lmd_est, loss_hist = train_minibatch(
        theta_valid, n_valid, lmd_true, lr=1e-8, n_epochs=50, seed=42
    )

    # ---- 3. Print statistics & save data ----
    evaluate(theta_valid, n_valid, lmd_true, lmd_est, output_dir="./output/direct_lmd")
