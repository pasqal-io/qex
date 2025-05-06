"""
Evaluation of trained models.
Producing the states of the Kohn-Sham calculation for a given distance.
Used for plotting.
This code is based on the code from jax_dft library.
"""

import pickle
from collections.abc import Callable
from functools import partial
from typing import Any

import jax.numpy as jnp
from jax import tree_util
from jax_dft import scf as scf_jax_dft
from jax_dft import xc
from jax_dft.utils import exponential_coulomb
from loguru import logger

from qedft.train.od import scf as scf_qedft


def kohn_sham(
    params: Any,
    locations: jnp.ndarray,
    nuclear_charges: jnp.ndarray,
    num_electrons: int,
    num_iterations: int,
    grids: jnp.ndarray,
    neural_xc_energy_density_fn: Callable,
    use_amplitude_encoding: bool,
    initial_density: jnp.ndarray | None = None,
    use_lda: bool = False,
    alpha: float = 0.5,
    alpha_decay: float = 0.9,
    num_mixing_iterations: int = 2,
    density_mse_converge_tolerance: float = -1.0,
    enforce_reflection_symmetry: bool = True,
):
    """
    Kohn-Sham calculation using either total XC functional
    or the usual XC energy density per grid point.

    Args:
        params: Model parameters.
        locations: Locations of the atoms.
        nuclear_charges: Nuclear charges of the atoms.
        num_electrons: Number of electrons in the system.
        num_iterations: Number of iterations.
        grids: Grids of the system.
        neural_xc_energy_density_fn: Neural network for the exchange-correlation energy density.
        use_amplitude_encoding: Whether to use amplitude encoding.
        initial_density: Initial density of the system.
        use_lda: Whether to use LDA.
        alpha: Mixing parameter.
        alpha_decay: Mixing parameter decay.
        num_mixing_iterations: Number of mixing iterations.
        density_mse_converge_tolerance: Density convergence tolerance.
        enforce_reflection_symmetry: Whether to enforce reflection symmetry.

    Returns:
        Kohn-Sham state.
    """

    if use_amplitude_encoding:
        # If the output of the neural network is the total XC energy
        kohn_sham_fn = scf_qedft.kohn_sham_amplitude_encoded
    else:
        # If the output of the neural network is the XC energy density per grid point
        kohn_sham_fn = scf_jax_dft.kohn_sham

    return kohn_sham_fn(
        locations=locations,
        nuclear_charges=nuclear_charges,
        num_electrons=num_electrons,
        num_iterations=num_iterations,
        grids=grids,
        xc_energy_density_fn=tree_util.Partial(
            xc.get_lda_xc_energy_density_fn() if use_lda else neural_xc_energy_density_fn,
            params=params,
        ),
        interaction_fn=exponential_coulomb,
        initial_density=initial_density,
        alpha=alpha,
        alpha_decay=alpha_decay,
        enforce_reflection_symmetry=enforce_reflection_symmetry,
        num_mixing_iterations=num_mixing_iterations,
        density_mse_converge_tolerance=density_mse_converge_tolerance,
    )


def load_model_params(ckpt_path: str) -> Any:
    """
    Load model parameters from a checkpoint file.

    Args:
        ckpt_path: Path to the checkpoint file.

    Returns:
        Model parameters.
    """
    logger.info(f"Loading checkpoint from {ckpt_path}")
    with open(ckpt_path, "rb") as handle:
        return pickle.load(handle)


def get_states(
    ckpt_path: str,
    kohn_sham_fn: Callable,
    plot_distances: list,
    plot_set: Any,
    plot_initial_density: list,
):
    """
    Get the states of the Kohn-Sham calculation.

    Args:
        ckpt_path: Path to the checkpoint.
        kohn_sham_fn: Kohn-Sham function evaluated at the checkpoint and other parameters.
        plot_distances: Distances to plot.
        plot_set: Set of points to plot.
        plot_initial_density: Initial density of the system.

    Returns:
        States of the Kohn-Sham calculation.
    """
    params = load_model_params(ckpt_path)
    states = []

    for i in range(len(plot_distances)):
        logger.info(f"Processing distance {plot_distances[i]}")
        states.append(
            kohn_sham_fn(
                params=params,
                locations=plot_set.locations[i],
                nuclear_charges=plot_set.nuclear_charges[i],
                initial_density=plot_initial_density[i],
            ),
        )
    return tree_util.tree_map(lambda *x: jnp.stack(x), *states)


def save_states(states: Any, output_path: str):
    """
    Save the states to a file.

    Args:
        states: States to save.
        output_path: Path to save the states.
    """
    import os

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info(f"Saving states to {output_path}")
    with open(output_path, "wb") as handle:
        pickle.dump(states, handle)


def eval_trained_model(
    ckpt_path: str,
    plot_distances: list,
    plot_set: Any,
    plot_initial_density: list,
    num_electrons: int,
    num_iterations: int,
    grids: jnp.ndarray,
    neural_xc_energy_density_fn: Callable,
    use_amplitude_encoding: bool,
    use_lda: bool = False,
    alpha: float = 0.5,
    alpha_decay: float = 0.9,
    num_mixing_iterations: int = 2,
    density_mse_converge_tolerance: float = -1.0,
    enforce_reflection_symmetry: bool = True,
    output_path: str = None,
):
    """
    Evaluate the trained model.

    Args:
        ckpt_path: Path to the checkpoint.
        plot_distances: Distances to plot.
        plot_set: Set of points to plot.
        plot_initial_density: Initial density of the system.
        num_electrons: Number of electrons in the system.
        num_iterations: Number of iterations.
        grids: Grids of the system.
        neural_xc_energy_density_fn: Neural network for the exchange-correlation energy density.
        use_amplitude_encoding: Whether to use amplitude encoding.
        use_lda: Whether to use LDA.
        alpha: Mixing parameter.
        alpha_decay: Mixing parameter decay.
        num_mixing_iterations: Number of mixing iterations.
        density_mse_converge_tolerance: Density convergence tolerance.
        enforce_reflection_symmetry: Whether to enforce reflection symmetry.
        output_path: Path to save the states. If None, states are not saved.

    Returns:
        States of the Kohn-Sham calculation.
    """
    kohn_sham_eval_fn = partial(
        kohn_sham,
        num_electrons=num_electrons,
        num_iterations=num_iterations,
        grids=grids,
        neural_xc_energy_density_fn=neural_xc_energy_density_fn,
        use_amplitude_encoding=use_amplitude_encoding,
        use_lda=use_lda,
        alpha=alpha,
        alpha_decay=alpha_decay,
        num_mixing_iterations=num_mixing_iterations,
        density_mse_converge_tolerance=density_mse_converge_tolerance,
        enforce_reflection_symmetry=enforce_reflection_symmetry,
    )
    states = get_states(
        ckpt_path=ckpt_path,
        kohn_sham_fn=kohn_sham_eval_fn,
        plot_distances=plot_distances,
        plot_set=plot_set,
        plot_initial_density=plot_initial_density,
    )

    if output_path is not None:
        save_states(states, output_path)

    return states
