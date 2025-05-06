"""
This module contains functions for performing self-consistent field (SCF) calculations.
We use the existing jax_dft codebase as a reference and modify it to fit
non-local representation of XC functional.

SCF calculations are used in computational chemistry and physics to find the
electron density and energy of a quantum system in a self-consistent manner.
This code is based on the code from jax_dft library.
"""

import typing
from typing import Union

import jax
import jax.numpy as jnp
from jax import tree_util
from jax_dft import utils
from jax_dft.scf import (
    KohnShamState,
    discrete_laplacian,
    get_external_potential_energy,
    get_gap,
    get_hartree_energy,
    get_hartree_potential,
    get_total_eigen_energies,
    solve_noninteracting_system,
)

ArrayLike = Union[float, bool, jnp.ndarray]


def get_xc_energy_amplitude_encoded(
    density: jnp.ndarray,
    xc_energy_density_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    grids: jnp.ndarray,
) -> jnp.ndarray:
    """Gets exchange-correlation (xc) energy.

    Args:
      density: Float numpy array with shape (num_grids,).
      xc_energy_density_fn: Function that takes density and returns float
          numpy array with shape (num_grids,).
      grids: Float numpy array with shape (num_grids,).

    Returns:
      Float: The exchange-correlation energy.
    """
    # Return summed energy
    # xc_energy_density_fn(density) should be a scalar anyway
    return jnp.sum(xc_energy_density_fn(density)) * utils.get_dx(grids)


def get_xc_potential_amplitude_encoded(
    density: jnp.ndarray,
    xc_energy_density_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    grids: jnp.ndarray,
) -> jnp.ndarray:
    """Gets exchange-correlation (xc) potential.

    Args:
      density: Float numpy array with shape (num_grids,).
      xc_energy_density_fn: Function that takes density and returns float
          numpy array with shape (num_grids,).
      grids: Float numpy array with shape (num_grids,).

    Returns:
      Float numpy array with shape (num_grids,): The exchange-correlation potential.
    """
    # Take gradient of total energy with respect to density
    return jax.grad(
        lambda d: get_xc_energy_amplitude_encoded(
            d,
            xc_energy_density_fn,
            grids,
        ),
    )(density)


def kohn_sham_iteration_amplitude_encoded(
    state: KohnShamState,
    num_electrons: int,
    xc_energy_density_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    interaction_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    enforce_reflection_symmetry: bool,
) -> KohnShamState:
    """One iteration of Kohn-Sham calculation.

    Note xc_energy_density_fn must be wrapped by jax.tree_util.Partial so
    this function can take a callable. When the arguments of this callable
    changes, e.g. the parameters of the neural network, kohn_sham_iteration()
    will not be recompiled.

    Args:
      state: KohnShamState.
      num_electrons: Integer, the number of electrons in the system. The first
          num_electrons states are occupied.
      xc_energy_density_fn: Function that takes density (num_grids,) and
          returns the energy density (num_grids,).
      interaction_fn: Function that takes displacements and returns float
          numpy array with the same shape of displacements.
      enforce_reflection_symmetry: Boolean, whether to enforce reflection
          symmetry. If True, the system is symmetric with respect to the
          center.

    Returns:
      KohnShamState: The next state of Kohn-Sham iteration.
    """
    # Calculate Hartree potential
    hartree_potential = get_hartree_potential(
        density=state.density,
        grids=state.grids,
        interaction_fn=interaction_fn,
    )

    # Calculate exchange-correlation potential
    xc_potential = get_xc_potential_amplitude_encoded(
        density=state.density,
        xc_energy_density_fn=xc_energy_density_fn,
        grids=state.grids,
    )

    # Calculate Kohn-Sham potential
    ks_potential = hartree_potential + xc_potential + state.external_potential

    # Calculate exchange-correlation energy density
    xc_energy_density = xc_energy_density_fn(state.density)

    # Solve Kohn-Sham equation to get new density, total eigen energies, and
    # gap
    density, total_eigen_energies, gap = solve_noninteracting_system(
        external_potential=ks_potential,
        num_electrons=num_electrons,
        grids=state.grids,
    )

    # Calculate total energy
    total_energy = (
        total_eigen_energies
        - get_external_potential_energy(
            external_potential=ks_potential,
            density=density,
            grids=state.grids,
        )
        + get_hartree_energy(
            density=density,
            grids=state.grids,
            interaction_fn=interaction_fn,
        )
        + get_xc_energy_amplitude_encoded(
            density=density,
            xc_energy_density_fn=xc_energy_density_fn,
            grids=state.grids,
        )
        + get_external_potential_energy(
            external_potential=state.external_potential,
            density=density,
            grids=state.grids,
        )
    )

    # Enforce reflection symmetry if required
    if enforce_reflection_symmetry:
        density = utils.flip_and_average(
            locations=state.locations,
            grids=state.grids,
            array=density,
        )

    # Return the updated state
    return state._replace(
        density=density,
        total_energy=total_energy,
        hartree_potential=hartree_potential,
        xc_potential=xc_potential,
        xc_energy_density=xc_energy_density,
        gap=gap,
    )


################################################################################
# Adding the no jit version of the functions
################################################################################


def get_kinetic_matrix_no_jit(grids):
    """Gets kinetic matrix.

    Args:
        grids: Float numpy array with shape (num_grids,).

    Returns:
        Float numpy array with shape (num_grids, num_grids).
    """
    dx = utils.get_dx(grids)
    return -0.5 * discrete_laplacian(grids.size) / (dx * dx)


def _wavefunctions_to_density_no_jit(num_electrons, wavefunctions, grids):
    """Converts wavefunctions to density."""
    # Reduce the amount of computation by removing most of the unoccupid states.
    wavefunctions = wavefunctions[:num_electrons]
    # Normalize the wavefunctions.
    wavefunctions = wavefunctions / jnp.sqrt(
        jnp.sum(
            wavefunctions**2,
            axis=1,
            keepdims=True,
        )
        * utils.get_dx(grids),
    )
    # Each eigenstate has spin up and spin down.
    intensities = jnp.repeat(wavefunctions**2, repeats=2, axis=0)
    return jnp.sum(intensities[:num_electrons], axis=0)


def wavefunctions_to_density_no_jit(num_electrons, wavefunctions, grids):
    """Converts wavefunctions to density.

    Note each eigenstate contains two states: spin up and spin down.

    Args:
        num_electrons: Integer, the number of electrons in the system. The first
            num_electrons states are occupid.
        wavefunctions: Float numpy array with shape (num_eigen_states, num_grids).
        grids: Float numpy array with shape (num_grids,).

    Returns:
        Float numpy array with shape (num_grids,).
    """
    return _wavefunctions_to_density_no_jit(num_electrons, wavefunctions, grids)


def _solve_noninteracting_system_no_jit(external_potential, num_electrons, grids):
    """Solves noninteracting system."""
    eigen_energies, wavefunctions_transpose = jnp.linalg.eigh(
        # Hamiltonian matrix.
        get_kinetic_matrix_no_jit(grids)
        + jnp.diag(external_potential),
    )
    density = wavefunctions_to_density_no_jit(
        num_electrons,
        jnp.transpose(wavefunctions_transpose),
        grids,
    )
    total_eigen_energies = get_total_eigen_energies(
        num_electrons=num_electrons,
        eigen_energies=eigen_energies,
    )
    gap = get_gap(num_electrons, eigen_energies)
    return density, total_eigen_energies, gap


def solve_noninteracting_system_no_jit(external_potential, num_electrons, grids):
    """Solves noninteracting system.

    Args:
        external_potential: Float numpy array with shape (num_grids,).
        num_electrons: Integer, the number of electrons in the system. The first
            num_electrons states are occupid.
        grids: Float numpy array with shape (num_grids,).

    Returns:
        density: Float numpy array with shape (num_grids,).
            The ground state density.
        total_eigen_energies: Float, the total energy of the eigen states.
        gap: Float, the HOMO–LUMO gap.
    """
    return _solve_noninteracting_system_no_jit(external_potential, num_electrons, grids)


def _get_hartree_energy_no_jit(density, grids, interaction_fn):
    """Gets the Hartree energy."""
    n1 = jnp.expand_dims(density, axis=0)
    n2 = jnp.expand_dims(density, axis=1)
    r1 = jnp.expand_dims(grids, axis=0)
    r2 = jnp.expand_dims(grids, axis=1)
    return (
        0.5
        * jnp.sum(
            n1 * n2 * interaction_fn(r1 - r2),
        )
        * utils.get_dx(grids) ** 2
    )


def get_hartree_energy_no_jit(density, grids, interaction_fn):
    r"""Gets the Hartree energy.

    U[n] = 0.5 \int dx \int dx' n(x) n(x') / \sqrt{(x - x')^2 + 1}

    Args:
        density: Float numpy array with shape (num_grids,).
        grids: Float numpy array with shape (num_grids,).
        interaction_fn: function takes displacements and returns
            float numpy array with the same shape of displacements.

    Returns:
        Float.
    """
    return _get_hartree_energy_no_jit(density, grids, interaction_fn)


def _get_hartree_potential_no_jit(density, grids, interaction_fn):
    """Gets the Hartree potential."""
    n1 = jnp.expand_dims(density, axis=0)
    r1 = jnp.expand_dims(grids, axis=0)
    r2 = jnp.expand_dims(grids, axis=1)
    return jnp.sum(n1 * interaction_fn(r1 - r2), axis=1) * utils.get_dx(grids)


def get_hartree_potential_no_jit(density, grids, interaction_fn):
    r"""Gets the Hartree potential.

    v_H(x) = \int dx' n(x') / \sqrt{(x - x')^2 + 1}

    Args:
        density: Float numpy array with shape (num_grids,).
        grids: Float numpy array with shape (num_grids,).
        interaction_fn: function takes displacements and returns
            float numpy array with the same shape of displacements.

    Returns:
        Float numpy array with shape (num_grids,).
    """
    return _get_hartree_potential_no_jit(density, grids, interaction_fn)


def kohn_sham_iteration_amplitude_encoded_no_jit(
    state: KohnShamState,
    num_electrons: int,
    xc_energy_density_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    interaction_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    enforce_reflection_symmetry: bool,
) -> KohnShamState:
    """One iteration of Kohn-Sham calculation.

    Note xc_energy_density_fn must be wrapped by jax.tree_util.Partial so
    this function can take a callable. When the arguments of this callable
    changes, e.g. the parameters of the neural network, kohn_sham_iteration()
    will not be recompiled.

    Args:
        state: KohnShamState.
        num_electrons: Integer, the number of electrons in the system. The first
            num_electrons states are occupied.
        xc_energy_density_fn: Function that takes density (num_grids,) and
            returns the energy density (num_grids,).
        interaction_fn: Function that takes displacements and returns float
            numpy array with the same shape of displacements.
        enforce_reflection_symmetry: Boolean, whether to enforce reflection
            symmetry. If True, the system is symmetric with respect to the
            center.

    Returns:
        KohnShamState: The next state of Kohn-Sham iteration.
    """
    # Calculate Hartree potential
    hartree_potential = get_hartree_potential_no_jit(
        density=state.density,
        grids=state.grids,
        interaction_fn=interaction_fn,
    )

    # Calculate exchange-correlation potential
    xc_potential = get_xc_potential_amplitude_encoded(
        density=state.density,
        xc_energy_density_fn=xc_energy_density_fn,
        grids=state.grids,
    )

    # Calculate Kohn-Sham potential
    ks_potential = hartree_potential + xc_potential + state.external_potential

    # Calculate exchange-correlation energy density
    xc_energy_density = xc_energy_density_fn(state.density)

    # Solve Kohn-Sham equation to get new density, total eigen energies, and
    # gap
    density, total_eigen_energies, gap = solve_noninteracting_system_no_jit(
        external_potential=ks_potential,
        num_electrons=num_electrons,
        grids=state.grids,
    )

    # Calculate total energy
    total_energy = (
        total_eigen_energies
        - get_external_potential_energy(
            external_potential=ks_potential,
            density=density,
            grids=state.grids,
        )
        + get_hartree_energy_no_jit(
            density=density,
            grids=state.grids,
            interaction_fn=interaction_fn,
        )
        + get_xc_energy_amplitude_encoded(
            density=density,
            xc_energy_density_fn=xc_energy_density_fn,
            grids=state.grids,
        )
        + get_external_potential_energy(
            external_potential=state.external_potential,
            density=density,
            grids=state.grids,
        )
    )

    # Enforce reflection symmetry if required
    if enforce_reflection_symmetry:
        density = utils.flip_and_average(
            locations=state.locations,
            grids=state.grids,
            array=density,
        )

    # Return the updated state
    return state._replace(
        density=density,
        total_energy=total_energy,
        hartree_potential=hartree_potential,
        xc_potential=xc_potential,
        xc_energy_density=xc_energy_density,
        gap=gap,
    )


################################################################################
# End of adding the no jit version of the functions
################################################################################


def kohn_sham_amplitude_encoded(
    locations: jnp.ndarray,
    nuclear_charges: jnp.ndarray,
    num_electrons: int,
    num_iterations: int,
    grids: jnp.ndarray,
    xc_energy_density_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    interaction_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    initial_density: jnp.ndarray | None = None,
    alpha: float = 0.5,
    alpha_decay: float = 0.9,
    enforce_reflection_symmetry: bool = False,
    num_mixing_iterations: int = 2,
    density_mse_converge_tolerance: float = -1.0,
) -> KohnShamState:
    """Runs Kohn-Sham to solve ground state of external potential.

    Args:
        locations: Float numpy array with shape (num_nuclei,), the locations of
            atoms.
        nuclear_charges: Float numpy array with shape (num_nuclei,), the nuclear
            charges.
        num_electrons: Integer, the number of electrons in the system. The first
            num_electrons states are occupied.
        num_iterations: Integer, the number of Kohn-Sham iterations.
        grids: Float numpy array with shape (num_grids,).
        xc_energy_density_fn: Function that takes density (num_grids,) and
            returns the energy density (num_grids,).
        interaction_fn: Function that takes displacements and returns float
            numpy array with the same shape of displacements.
        initial_density: Float numpy array with shape (num_grids,), initial guess
            of the density for Kohn-Sham calculation. Default None, the initial
            density is non-interacting solution from the external_potential.
        alpha: Float between 0 and 1, density linear mixing factor, the fraction
            of the output of the k-th Kohn-Sham iteration. If 0, the input
            density to the k-th Kohn-Sham iteration is fed into the (k+1)-th
            iteration. The output of the k-th Kohn-Sham iteration is completely
            ignored. If 1, the output density from the k-th Kohn-Sham iteration
            is fed into the (k+1)-th iteration, equivalent to no density mixing.
        alpha_decay: Float between 0 and 1, the decay factor of alpha. The mixing
            factor after k-th iteration is alpha * alpha_decay ** k.
        enforce_reflection_symmetry: Boolean, whether to enforce reflection
            symmetry. If True, the density is symmetric with respect to the
            center.
        num_mixing_iterations: Integer, the number of density differences in the
            previous iterations to mix the density.
        density_mse_converge_tolerance: Float, the stopping criteria. When the
            MSE density difference between two iterations is smaller than this
            value, the Kohn Sham iterations finish. The outputs of the rest of
            the steps are padded by the output of the converged step. Set this
            value to negative to disable early stopping.

        Returns:
        KohnShamState: The states of all the Kohn-Sham iteration steps.
    """
    # Calculate external potential
    external_potential = utils.get_atomic_chain_potential(
        grids=grids,
        locations=locations,
        nuclear_charges=nuclear_charges,
        interaction_fn=interaction_fn,
    )

    # If initial density is not provided, use the non-interacting solution
    # from the external potential as initial guess
    if initial_density is None:
        initial_density, _, _ = solve_noninteracting_system(
            external_potential=external_potential,
            num_electrons=num_electrons,
            grids=grids,
        )

    # Create initial state
    state = KohnShamState(
        density=initial_density,
        total_energy=jnp.inf,
        locations=locations,
        nuclear_charges=nuclear_charges,
        external_potential=external_potential,
        grids=grids,
        num_electrons=num_electrons,
    )

    states = []
    differences = None
    converged = False

    for _ in range(num_iterations):
        if converged:
            states.append(state)
            continue

        old_state = state
        state = kohn_sham_iteration_amplitude_encoded(
            state=old_state,
            num_electrons=num_electrons,
            xc_energy_density_fn=xc_energy_density_fn,
            interaction_fn=interaction_fn,
            enforce_reflection_symmetry=enforce_reflection_symmetry,
        )

        density_difference = state.density - old_state.density

        if differences is None:
            differences = jnp.array([density_difference])
        else:
            differences = jnp.vstack([differences, density_difference])

        if jnp.mean(jnp.square(density_difference)) < density_mse_converge_tolerance:
            converged = True

        state = state._replace(converged=converged)

        # Density mixing
        state = state._replace(
            density=old_state.density
            + alpha
            * jnp.mean(
                differences[-num_mixing_iterations:],
                axis=0,
            ),
        )

        states.append(state)
        alpha *= alpha_decay

    return tree_util.tree_map(lambda *x: jnp.stack(x), *states)


# Version without jit


def kohn_sham_amplitude_encoded_no_jit(
    locations: jnp.ndarray,
    nuclear_charges: jnp.ndarray,
    num_electrons: int,
    num_iterations: int,
    grids: jnp.ndarray,
    xc_energy_density_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    interaction_fn: typing.Callable[[jnp.ndarray], jnp.ndarray],
    initial_density: jnp.ndarray | None = None,
    alpha: float = 0.5,
    alpha_decay: float = 0.9,
    enforce_reflection_symmetry: bool = False,
    num_mixing_iterations: int = 2,
    density_mse_converge_tolerance: float = -1.0,
) -> KohnShamState:
    """Runs Kohn-Sham to solve ground state of external potential.

    Args:
        locations: Float numpy array with shape (num_nuclei,), the locations of
            atoms.
        nuclear_charges: Float numpy array with shape (num_nuclei,), the nuclear
            charges.
        num_electrons: Integer, the number of electrons in the system. The first
            num_electrons states are occupied.
        num_iterations: Integer, the number of Kohn-Sham iterations.
        grids: Float numpy array with shape (num_grids,).
        xc_energy_density_fn: Function that takes density (num_grids,) and
            returns the energy density (num_grids,).
        interaction_fn: Function that takes displacements and returns float
            numpy array with the same shape of displacements.
        initial_density: Float numpy array with shape (num_grids,), initial guess
            of the density for Kohn-Sham calculation. Default None, the initial
            density is non-interacting solution from the external_potential.
        alpha: Float between 0 and 1, density linear mixing factor, the fraction
            of the output of the k-th Kohn-Sham iteration. If 0, the input
            density to the k-th Kohn-Sham iteration is fed into the (k+1)-th
            iteration. The output of the k-th Kohn-Sham iteration is completely
            ignored. If 1, the output density from the k-th Kohn-Sham iteration
            is fed into the (k+1)-th iteration, equivalent to no density mixing.
        alpha_decay: Float between 0 and 1, the decay factor of alpha. The mixing
            factor after k-th iteration is alpha * alpha_decay ** k.
        enforce_reflection_symmetry: Boolean, whether to enforce reflection
            symmetry. If True, the density is symmetric with respect to the
            center.
        num_mixing_iterations: Integer, the number of density differences in the
            previous iterations to mix the density.
        density_mse_converge_tolerance: Float, the stopping criteria. When the
            MSE density difference between two iterations is smaller than this
            value, the Kohn Sham iterations finish. The outputs of the rest of
            the steps are padded by the output of the converged step. Set this
            value to negative to disable early stopping.

        Returns:
        KohnShamState: The states of all the Kohn-Sham iteration steps.
    """
    # Calculate external potential
    external_potential = utils.get_atomic_chain_potential(
        grids=grids,
        locations=locations,
        nuclear_charges=nuclear_charges,
        interaction_fn=interaction_fn,
    )

    # If initial density is not provided, use the non-interacting solution
    # from the external potential as initial guess
    if initial_density is None:
        initial_density, _, _ = solve_noninteracting_system_no_jit(
            external_potential=external_potential,
            num_electrons=num_electrons,
            grids=grids,
        )

    # Create initial state
    state = KohnShamState(
        density=initial_density,
        total_energy=jnp.inf,
        locations=locations,
        nuclear_charges=nuclear_charges,
        external_potential=external_potential,
        grids=grids,
        num_electrons=num_electrons,
    )

    states = []
    differences = None
    converged = False

    for _ in range(num_iterations):
        if converged:
            states.append(state)
            continue

        old_state = state
        state = kohn_sham_iteration_amplitude_encoded_no_jit(
            state=old_state,
            num_electrons=num_electrons,
            xc_energy_density_fn=xc_energy_density_fn,
            interaction_fn=interaction_fn,
            enforce_reflection_symmetry=enforce_reflection_symmetry,
        )

        density_difference = state.density - old_state.density

        if differences is None:
            differences = jnp.array([density_difference])
        else:
            differences = jnp.vstack([differences, density_difference])

        if jnp.mean(jnp.square(density_difference)) < density_mse_converge_tolerance:
            converged = True

        state = state._replace(converged=converged)

        # Density mixing
        state = state._replace(
            density=old_state.density
            + alpha
            * jnp.mean(
                differences[-num_mixing_iterations:],
                axis=0,
            ),
        )

        states.append(state)
        alpha *= alpha_decay

    return tree_util.tree_map(lambda *x: jnp.stack(x), *states)
