"""
This script contains the main training loop for one-dimensional (OD) KS DFT.
It defines the Kohn-Sham function, loss function, and training step, and includes
the main execution block for running the training process.
"""

import os
import pickle
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util
from jax_dft import jit_scf, losses, np_utils, utils
from jax_dft.scf import KohnShamState
from loguru import logger

from qedft.train.od import jit_scf as jit_scf_od
from qedft.train.od import losses_no_jit


def get_training_config(override_config: dict = None) -> dict:
    """Returns training configuration with default values.

    Args:
        override_config: Optional dictionary containing configuration overrides.

    Returns:
        Dictionary containing training configuration.

    Example:
        >>> config = get_training_config()
        >>> config['num_iterations']
        15
        >>> config = get_training_config({'num_iterations': 20})
        >>> config['num_iterations']
        20
    """
    defaults = {
        "num_iterations": 15,  # Number of SCF iterations
        "alpha": 0.5,  # Mixing parameter for density update
        "alpha_decay": 0.9,  # Decay rate for mixing parameter
        "num_mixing_iterations": 1,  # Number of density mixing iterations
        "density_mse_converge_tolerance": -1.0,  # Convergence tolerance for density
        "stop_gradient_step": -1,  # Step to stop gradient computation
        "save_every_n": 20,  # Save checkpoint every N steps
        "discount_factor": 0.9,  # Discount factor for trajectory loss
        "use_amplitude_encoding": False,  # Whether to use amplitude encoding
    }

    if override_config is not None:
        defaults.update(override_config)
    return defaults


def create_kohn_sham_fn(
    config: dict,
    dataset: Any,
    grids: jnp.ndarray,
    neural_xc_energy_density_fn: Callable,
    spec: Any,
) -> tuple[Callable, Callable]:
    """Creates the Kohn-Sham function and its batched version.

    Args:
        config: Dictionary of training configuration.
        dataset: Dataset containing electron configuration.
        grids: Grid points for numerical integration.
        neural_xc_energy_density_fn: Neural network for exchange-correlation energy.
        spec: Parameter specification for neural network.

    Returns:
        Tuple of (kohn_sham_fn, batch_kohn_sham_fn).
        kohn_sham_fn takes single inputs while batch_kohn_sham_fn is vectorized.

    Example:
        >>> kohn_sham_fn, batch_fn = create_kohn_sham_fn(config, dataset, grids, xc_fn, spec)
        >>> states = kohn_sham_fn(config, locations, charges, density)
        >>> batch_states = batch_fn(config, locations_batch, charges_batch, density_batch)
    """

    def _kohn_sham(flatten_params, locations, nuclear_charges, initial_density) -> KohnShamState:
        """Single example Kohn-Sham calculation.

        Args:
            flatten_params: Flattened neural network parameters.
            locations: Nuclear locations.
            nuclear_charges: Nuclear charges.
            initial_density: Initial electron density guess.

        Returns:
            KohnShamState containing converged density and energies.
        """
        common_params = {
            "locations": locations,
            "nuclear_charges": nuclear_charges,
            "num_electrons": dataset.num_electrons,
            "num_iterations": config["num_iterations"],
            "grids": grids,
            "xc_energy_density_fn": tree_util.Partial(
                neural_xc_energy_density_fn,
                params=np_utils.unflatten(spec, flatten_params),
            ),
            "interaction_fn": utils.exponential_coulomb,
            "initial_density": initial_density,
            "alpha": config["alpha"],
            "alpha_decay": config["alpha_decay"],
            "enforce_reflection_symmetry": True,
            "num_mixing_iterations": config["num_mixing_iterations"],
            "density_mse_converge_tolerance": config["density_mse_converge_tolerance"],
            "stop_gradient_step": config.get("stop_gradient_step", -1),
        }

        logger.info("Jitting kohn_sham_func")
        kohn_sham_func = (
            jit_scf_od.kohn_sham_amplitude_encoded
            if config["use_amplitude_encoding"]
            else jit_scf.kohn_sham
        )
        return kohn_sham_func(**common_params)

    # Batch Kohn-Sham function
    batch_kohn_sham = jax.vmap(_kohn_sham, in_axes=(None, 0, 0, 0))
    return _kohn_sham, batch_kohn_sham


def batch_from_states(states):
    """Combines a list of KohnShamState instances into a single batched state.

    Args:
        states: List of KohnShamState instances to combine.

    Returns:
        A single KohnShamState with batched fields.

    Example:
        >>> states = [state1, state2, state3]
        >>> batched_state = batch_from_states(states)
    """
    if not states:
        raise ValueError("Cannot batch empty list of states")

    # Use tree_map to combine all fields across the states
    # This stacks arrays along a new first dimension
    batched_state = tree_util.tree_map(
        lambda *xs: jnp.stack(xs),
        *states,
    )

    return batched_state


def create_kohn_sham_fn_non_jit(
    config: dict,
    dataset: Any,
    grids: jnp.ndarray,
    neural_xc_energy_density_fn: Callable,
    spec: Any,
) -> tuple[Callable, Callable]:
    """Creates the non-JIT Kohn-Sham function and its batched version.
    For compatibility with other non-JIT packages (e.g., Qiskit).

    Args:
        config: Dictionary of training configuration.
        dataset: Dataset containing electron configuration.
        grids: Grid points for numerical integration.
        neural_xc_energy_density_fn: Neural network for exchange-correlation energy.
        spec: Parameter specification for neural network.

    Returns:
        Tuple of (kohn_sham_fn, batch_kohn_sham_fn).
        kohn_sham_fn takes single inputs while batch_kohn_sham_fn is vectorized.

    Example:
        >>> kohn_sham_fn, batch_fn = create_kohn_sham_fn_non_jit(config, dataset, grids, xc_fn, spec)
        >>> states = kohn_sham_fn(config, locations, charges, density)
        >>> batch_states = batch_fn(config, locations_batch, charges_batch, density_batch)
    """

    def _kohn_sham(flatten_params, locations, nuclear_charges, initial_density) -> KohnShamState:
        """Single example Kohn-Sham calculation.

        Args:
            flatten_params: Flattened neural network parameters.
            locations: Nuclear locations.
            nuclear_charges: Nuclear charges.
            initial_density: Initial electron density guess.

        Returns:
            KohnShamState containing converged density and energies.
        """
        common_params = {
            "locations": locations,
            "nuclear_charges": nuclear_charges,
            "num_electrons": dataset.num_electrons,
            "num_iterations": config["num_iterations"],
            "grids": grids,
            "xc_energy_density_fn": tree_util.Partial(
                neural_xc_energy_density_fn,
                params=np_utils.unflatten(spec, flatten_params),
            ),
            "interaction_fn": utils.exponential_coulomb,
            "initial_density": initial_density,
            "alpha": config["alpha"],
            "alpha_decay": config["alpha_decay"],
            "enforce_reflection_symmetry": True,
            "num_mixing_iterations": config["num_mixing_iterations"],
            "density_mse_converge_tolerance": config["density_mse_converge_tolerance"],
        }

        logger.info("Using non-JIT kohn_sham_func")
        # Import the non-JIT versions
        from jax_dft import scf

        from qedft.train.od import scf as scf_od

        kohn_sham_func = (
            scf_od.kohn_sham_amplitude_encoded_no_jit
            if config["use_amplitude_encoding"]
            else scf.kohn_sham
        )
        return kohn_sham_func(**common_params)

    # Batch Kohn-Sham function (without jit.vmap)
    def batch_kohn_sham(flatten_params, locations, nuclear_charges, initial_density):
        """Batched version of Kohn-Sham calculation without using jax.vmap.

        Manually loops through each example in the batch.
        """
        batch_size = locations.shape[0]
        results = []

        for i in range(batch_size):
            result = _kohn_sham(
                flatten_params,
                locations[i],
                nuclear_charges[i],
                initial_density[i],
            )
            results.append(result)

        # Combine results into a single batch result
        return batch_from_states(results)

    return _kohn_sham, batch_kohn_sham


def create_loss_fn(
    batch_kohn_sham: Callable,
    grids: jnp.ndarray,
    dataset: Any,
    config: dict,
) -> Callable:
    """Creates the loss function for training.

    Args:
        batch_kohn_sham: Batched Kohn-Sham function.
        grids: Grid points for numerical integration.
        dataset: Dataset containing target values.
        config: Dictionary of training configuration.

    Returns:
        Loss function that takes parameters and data, returns scalar loss.

    Example:
        >>> loss_fn = create_loss_fn(batch_kohn_sham, grids, dataset, config)
        >>> loss = loss_fn(config, locations, charges, density, target_e, target_d)
    """
    grids_integration_factor = utils.get_dx(grids) * len(grids)

    def loss_fn(
        flatten_params,
        locations,
        nuclear_charges,
        initial_density,
        target_energy,
        target_density,
    ):
        """Computes loss between predictions and targets.

        Args:
            flatten_params: Flattened neural network parameters.
            locations: Nuclear locations.
            nuclear_charges: Nuclear charges.
            initial_density: Initial electron density guess.
            target_energy: Target total energy.
            target_density: Target electron density.

        Returns:
            Total loss combining energy and density terms.
        """
        states = batch_kohn_sham(
            flatten_params,
            locations,
            nuclear_charges,
            initial_density,
        )

        # Energy loss
        loss_value = (
            losses.trajectory_mse(
                target=target_energy,
                predict=states.total_energy[:, 10:],
                discount=config["discount_factor"],
            )
            / dataset.num_electrons
        )

        # Density loss
        loss_value += (
            losses.mean_square_error(
                target=target_density,
                predict=states.density[:, -1, :],
            )
            * grids_integration_factor
            / dataset.num_electrons
        )

        return loss_value

    return loss_fn


def create_loss_fn_no_jit(
    batch_kohn_sham: Callable,
    grids: jnp.ndarray,
    dataset: Any,
    config: dict,
) -> Callable:
    """Creates the loss function for training.

    Args:
        batch_kohn_sham: Batched Kohn-Sham function.
        grids: Grid points for numerical integration.
        dataset: Dataset containing target values.
        config: Dictionary of training configuration.

    Returns:
        Loss function that takes parameters and data, returns scalar loss.

    Example:
        >>> loss_fn = create_loss_fn(batch_kohn_sham, grids, dataset, config)
        >>> loss = loss_fn(config, locations, charges, density, target_e, target_d)
    """
    grids_integration_factor = utils.get_dx(grids) * len(grids)

    def loss_fn_no_jit(
        flatten_params,
        locations,
        nuclear_charges,
        initial_density,
        target_energy,
        target_density,
    ):
        """Computes loss between predictions and targets.

        Args:
            flatten_params: Flattened neural network parameters.
            locations: Nuclear locations.
            nuclear_charges: Nuclear charges.
            initial_density: Initial electron density guess.
            target_energy: Target total energy.
            target_density: Target electron density.

        Returns:
            Total loss combining energy and density terms.
        """
        states = batch_kohn_sham(
            flatten_params,
            locations,
            nuclear_charges,
            initial_density,
        )

        # Energy loss
        loss_value = (
            losses_no_jit.trajectory_mse(
                target=target_energy,
                predict=states.total_energy[:, 10:],
                discount=config["discount_factor"],
            )
            / dataset.num_electrons
        )

        # Density loss
        loss_value += (
            losses_no_jit.mean_square_error(
                target=target_density,
                predict=states.density[:, -1, :],
            )
            * grids_integration_factor
            / dataset.num_electrons
        )

        return loss_value

    return loss_fn_no_jit


def create_training_step(
    value_and_grad_fn: Callable,
    train_set: Any,
    initial_density: jnp.ndarray,
    save_every_n: int,
    initial_checkpoint_index: int = 0,
    checkpoint_dir: str | None = None,
    spec: Any = None,
    ckpt_prefix: str = "ckpt",
) -> Callable:
    """Creates training step function with checkpointing.

    Args:
        value_and_grad_fn: Function returning loss value and gradients.
        train_set: Training dataset.
        initial_density: Initial electron density guess.
        save_every_n: Save checkpoint every N steps.
        initial_checkpoint_index: Starting checkpoint index.
        checkpoint_dir: Directory to save checkpoints.
        spec: Parameter specification for neural network.

    Returns:
        Training step function compatible with optimizers.

    Example:
        >>> training_step = create_training_step(grad_fn, train_set, density, config)
        >>> loss, grads = training_step(model_params)
    """
    loss_record = []

    def np_value_and_grad_fn(flatten_params):
        """Computes loss and gradients for optimizer.

        Args:
            flatten_params: Flattened model parameters.

        Returns:
            Tuple of (loss, gradients).
        """

        start_time = time.time()
        train_set_loss, train_set_gradient = value_and_grad_fn(
            flatten_params,
            locations=train_set.locations,
            nuclear_charges=train_set.nuclear_charges,
            initial_density=initial_density,
            target_energy=train_set.total_energy,
            target_density=train_set.density,
        )

        step_time = time.time() - start_time
        step = initial_checkpoint_index + len(loss_record)
        logger.info(f"step {step}, loss {train_set_loss} in {step_time} sec")

        if checkpoint_dir and len(loss_record) % save_every_n == 0:
            checkpoint_path = f"{checkpoint_dir}/{ckpt_prefix}-{step:05d}"
            logger.info(f"Save checkpoint {checkpoint_path}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(checkpoint_path, "wb") as handle:
                pickle.dump(np_utils.unflatten(spec, flatten_params), handle)

        loss_record.append(train_set_loss)
        return train_set_loss, jnp.array(train_set_gradient)

    return np_value_and_grad_fn


if __name__ == "__main__":

    """Example of how to run the training script."""

    import os
    from pathlib import Path

    import numpy as np
    import scipy
    from jax import random
    from jax_dft import scf

    import qedft
    from qedft.config.config import Config
    from qedft.data_io.dataset_loader import load_molecular_datasets_from_config
    from qedft.models.classical.global_ksr_model import create_ksr_model_from_config

    # Set up JAX to use 64-bit precision
    # NaNs in loss if not using 64-bit precision
    jax.config.update("jax_enable_x64", True)

    seed = random.PRNGKey(0)

    # Initialize configuration
    project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
    config = Config(
        config_path=project_path / "qedft" / "config" / "train_config.yaml",
    )
    config_dict = config.config

    # Load config with amplitude encoding option
    # config = get_training_config()

    # Load dataset
    base_path = project_path / "data" / "od"
    list_datasets = load_molecular_datasets_from_config(
        config_dict,
        base_path,
        check_grid_centering=True,
    )
    dataset, train_set = list_datasets[0]  # First molecule

    # Get initial density and grids
    initial_density = scf.get_initial_density(
        train_set,
        method="noninteracting",
    )
    logger.info(f"initial_density: {initial_density}")
    grids = dataset.grids

    # Create and initialize KSR model
    network, init_fn, neural_xc_energy_density_fn = create_ksr_model_from_config(grids)

    spec, flatten_init_params = np_utils.flatten(init_fn(seed))

    neural_xc_energy_density_fn = jax.jit(neural_xc_energy_density_fn)

    # Create Kohn-Sham function and its batched version
    kohn_sham_fn, batch_kohn_sham = create_kohn_sham_fn(
        config,
        dataset,
        grids,
        neural_xc_energy_density_fn,
        spec,
    )
    logger.info(
        f"kohn_sham_fn: {kohn_sham_fn}, batch_kohn_sham: {batch_kohn_sham}",
    )

    # Create and initialize loss function
    loss_fn = create_loss_fn(batch_kohn_sham, grids, dataset, config)
    logger.info(f"loss_fn: {loss_fn}")

    # Create value and gradient function
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    logger.info(f"value_and_grad_fn: {value_and_grad_fn}")

    # Create training step function
    training_step = create_training_step(
        value_and_grad_fn,
        train_set,
        initial_density,
        save_every_n=1,
        initial_checkpoint_index=0,
        checkpoint_dir=project_path / "tests" / "ckpts",
        spec=spec,
        ckpt_prefix="ckpt",
    )
    logger.info(f"training_step: {training_step}")

    # Optimize using L-BFGS-B
    _, _, info = scipy.optimize.fmin_l_bfgs_b(
        training_step,
        x0=np.array(flatten_init_params),
        maxfun=1,
        factr=1,
        m=2,
        pgtol=1e-14,
    )
