"""Generic vmap-batched training loop.

`train` is physics-agnostic: pass an `scf_loss_fn` (e.g. `qex.scf.rks.rks_loss`
for closed-shell, later `uks_loss` for open-shell or `rks_loss_periodic` for
periodic systems). All SCF hyperparameters are forwarded via `scf_kwargs`.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax


def train(
    params: dict,
    training_data: list,
    optimizer: optax.GradientTransformation,
    *,
    scf_loss_fn: Callable,
    xc_eval_fn: Callable,
    n_iterations: int,
    log_every: int = 5,
    **scf_kwargs,
) -> dict:
    """vmap-batched SCF training over a fixed dataset.

    `training_data` is a list of tuples (energy, coords_and_density, precomputed, dm)
    where `precomputed` is (eri, ao_grid, grid_weights, s1e, h1e, energy_nuc, nelectron).
    `scf_loss_fn` takes (params, dm, eri, ao_grid, grid_weights, s1e, h1e,
    energy_nuc, nelectron, exact_energy, exact_density, exact_dm, **kwargs)
    and returns a scalar loss.
    """
    opt_state = optimizer.init(params)

    dms = jnp.stack([data[3] for data in training_data])
    eris = jnp.stack([data[2][0] for data in training_data])
    ao_grids = jnp.stack([data[2][1] for data in training_data])
    grid_weights = jnp.stack([data[2][2] for data in training_data])
    s1es = jnp.stack([data[2][3] for data in training_data])
    h1es = jnp.stack([data[2][4] for data in training_data])
    energy_nucs = jnp.array([data[2][5] for data in training_data])
    nelectrons = jnp.array([data[2][6] for data in training_data])
    exact_energies = jnp.array([data[0] for data in training_data])
    exact_densities = jnp.stack([data[1][:, 3] for data in training_data])
    exact_dms = dms  # reuse dm as exact_dm for now (matches legacy script)

    all_inputs = (
        dms,
        eris,
        ao_grids,
        grid_weights,
        s1es,
        h1es,
        energy_nucs,
        nelectrons,
        exact_energies,
        exact_densities,
        exact_dms,
    )
    all_inputs = jax.device_put(all_inputs)

    scf_fn = partial(scf_loss_fn, xc_eval_fn=xc_eval_fn, **scf_kwargs)

    @jax.jit
    def loss_fn(params):
        vmapped = jax.vmap(scf_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        losses = vmapped(params, *all_inputs)
        return jnp.mean(losses)

    @jax.jit
    def update_fn(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    for iteration in range(n_iterations):
        params, opt_state, loss = update_fn(params, opt_state)

        if iteration % log_every == 0:
            loss = jax.block_until_ready(loss)
            print(f"Iteration {iteration}: Loss {loss:.6f}")

    return params
