"""Generic vmap-batched training loop with optional periodic validation.

`train` is physics-agnostic: pass an `scf_loss_fn` (e.g. `qex.scf.rks.rks_loss`
for closed-shell, later `uks_loss` for open-shell or `rks_loss_periodic` for
periodic systems). All SCF hyperparameters are forwarded via `scf_kwargs`.

Validation is optional and cheap: pass `val_data` and a `n_val_iter` cadence and
the validation loss is evaluated through a *separate* jitted closure only every
`n_val_iter` steps, so the per-step training cost is unchanged. When validation
is on, the lowest-val-loss parameters are tracked and returned as ``best``, and
training can stop early once the val loss stops improving (`patience`).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm


@dataclass
class TrainHistory:
    """Per-run training history returned by :func:`train`.

    Attributes:
        train_iters / train_loss: training loss sampled every ``log_every`` steps.
        val_iters / val_loss: validation loss sampled every ``n_val_iter`` steps
            (empty if no validation set was provided).
        best_iter / best_val_loss: iteration and value of the lowest val loss
            (``None`` without validation).
        stopped_early: whether early stopping triggered.
    """

    train_iters: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_iters: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_iter: int | None = None
    best_val_loss: float | None = None
    stopped_early: bool = False


def _stack_inputs(data: list) -> tuple:
    """Stack a list of ``(energy, coords_density, core_inputs, features, dm)``.

    Returns ``(core_batched, features_batched)`` where:

    - ``core_batched`` is the tuple of batched core arrays consumed positionally
      by the vmapped SCF loss (``dm, eri, …, exact_dm``) — fixed and unchanged.
    - ``features_batched`` is the named model-feature bag with each array stacked
      along a new leading batch axis (a ``dict[str, Array]``, possibly empty).
      It is passed as the SCF loss's ``features`` argument; the network selects
      the subset it declares. Adding a feature needs no change here — every key
      the producer emits is stacked automatically.
    """
    # Transpose the list of per-sample tuples into columns we can batch.
    energies, coords_density, core_inputs, bags, dms = zip(*data)
    eri, ao_grid, weights, s1e, h1e, e_nuc, nelec = zip(*core_inputs)

    dm = jnp.stack(dms)
    core = (
        dm,                                          # dm
        jnp.stack(eri),                              # eri
        jnp.stack(ao_grid),                          # ao_grid
        jnp.stack(weights),                          # grid_weights
        jnp.stack(s1e),                              # s1e
        jnp.stack(h1e),                              # h1e
        jnp.array(e_nuc),                            # energy_nuc
        jnp.array(nelec),                            # nelectron
        jnp.array(energies),                         # exact_energy
        jnp.stack([cd[:, 3] for cd in coords_density]),  # exact_density
        dm,                                          # exact_dm (== dm for now)
    )

    # Stack the per-sample feature bags into one batched bag. All samples must
    # carry the same keys; assert that loudly rather than silently dropping a
    # feature for some samples.
    keys = tuple(bags[0])
    if any(tuple(bag) != keys for bag in bags):
        raise ValueError(
            f"Inconsistent feature keys across samples (first sample has {keys}). "
            f"Every datapoint in a split must carry the same features."
        )
    features = {k: jnp.stack([bag[k] for bag in bags]) for k in keys}

    return jax.device_put((core, features))


def _make_loss_fn(scf_loss_fn: Callable, xc_eval_fn: Callable, inputs: tuple, scf_kwargs):
    """Build a jitted ``params -> mean loss`` closure over fixed ``inputs``.

    ``inputs`` is ``(core_batched, features_batched)`` from :func:`_stack_inputs`.
    Core arrays are mapped over the batch axis (``in_axes=0``); the feature bag
    is one more vmapped argument whose ``in_axes`` is a dict ``{k: 0}`` (an empty
    bag is a trivial pytree, so a no-feature model maps over nothing).
    """
    core, features = inputs
    scf_fn = partial(scf_loss_fn, xc_eval_fn=xc_eval_fn, **scf_kwargs)
    core_axes = (0,) * len(core)
    feat_axes = {k: 0 for k in features}
    in_axes = (None, *core_axes, feat_axes)

    @jax.jit
    def loss_fn(params):
        losses = jax.vmap(scf_fn, in_axes=in_axes)(params, *core, features)
        return jnp.mean(losses)

    return loss_fn


def train(
    params: dict,
    training_data: list,
    optimizer: optax.GradientTransformation,
    *,
    scf_loss_fn: Callable,
    xc_eval_fn: Callable,
    n_iterations: int,
    log_every: int = 5,
    val_data: list | None = None,
    n_val_iter: int = 50,
    patience: int | None = None,
    return_history: bool = False,
    **scf_kwargs,
):
    """vmap-batched SCF training over a fixed dataset, with optional validation.

    `training_data` (and `val_data`) is a list of tuples
    (energy, coords_and_density, core_inputs, features, dm) where `core_inputs`
    is (eri, ao_grid, grid_weights, s1e, h1e, energy_nuc, nelectron) and
    `features` is the named model-feature bag (a possibly-empty dict, see
    qex.functionals.features). `scf_loss_fn` takes (params, dm, eri, ao_grid,
    grid_weights, s1e, h1e, energy_nuc, nelectron, exact_energy, exact_density,
    exact_dm, features, **kwargs) and returns a scalar loss.

    Args:
        val_data: Optional validation set. When given, validation loss is
            computed every `n_val_iter` steps via a separate jitted closure (the
            training step is untouched, so per-step cost is unchanged).
        n_val_iter: Validation cadence in iterations.
        patience: If set, stop early once the validation loss has not improved
            for `patience` consecutive validations. Requires `val_data`.
        return_history: If True, return ``(best_params, TrainHistory)``;
            otherwise return ``best_params`` (the lowest-val-loss params, or the
            final params when no validation set is provided).

    Returns:
        ``best_params`` or ``(best_params, history)`` depending on
        ``return_history``.
    """
    opt_state = optimizer.init(params)

    train_inputs = _stack_inputs(training_data)
    loss_fn = _make_loss_fn(scf_loss_fn, xc_eval_fn, train_inputs, scf_kwargs)

    val_loss_fn = None
    if val_data:
        val_inputs = _stack_inputs(val_data)
        val_loss_fn = _make_loss_fn(scf_loss_fn, xc_eval_fn, val_inputs, scf_kwargs)

    @jax.jit
    def update_fn(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    history = TrainHistory()
    best_params = params
    best_val = jnp.inf
    since_improved = 0
    last_train = float("nan")

    # Single tqdm bar; live metrics are shown via the postfix (updated in place,
    # no new line per step). `best` only appears once validation has run.
    pbar = tqdm(range(n_iterations), desc="train", dynamic_ncols=True)
    for iteration in pbar:
        params, opt_state, loss = update_fn(params, opt_state)

        if iteration % log_every == 0:
            last_train = float(jax.block_until_ready(loss))
            history.train_iters.append(iteration)
            history.train_loss.append(last_train)

        # Periodic validation (separate jitted closure; does not touch the step).
        ran_validation = val_loss_fn is not None and iteration % n_val_iter == 0
        if ran_validation:
            vloss = float(jax.block_until_ready(val_loss_fn(params)))
            history.val_iters.append(iteration)
            history.val_loss.append(vloss)

            if vloss < best_val:
                best_val = vloss
                best_params = params
                history.best_iter = iteration
                history.best_val_loss = vloss
                since_improved = 0
            else:
                since_improved += 1

        # Refresh the live readout on a log step or whenever validation ran.
        if iteration % log_every == 0 or ran_validation:
            postfix = {"loss": f"{last_train:.4e}"}
            if history.val_loss:
                postfix["val"] = f"{history.val_loss[-1]:.4e}"
                postfix["best"] = f"{best_val:.4e}"
            pbar.set_postfix(postfix)

        if ran_validation and patience is not None and since_improved >= patience:
            history.stopped_early = True
            pbar.write(
                f"Early stopping at iteration {iteration}: no val improvement "
                f"for {patience} validations "
                f"(best {best_val:.6f} @ {history.best_iter})."
            )
            break

    pbar.close()

    # Without a validation set, "best" is just the final params.
    if val_loss_fn is None:
        best_params = params

    if return_history:
        return best_params, history
    return best_params
