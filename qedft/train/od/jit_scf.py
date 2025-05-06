"""Functions for self-consistent field calculation simplified for jit.
This code is based on the code from jax_dft library.
"""

import functools

import jax
import jax.numpy as jnp
from jax_dft import jit_scf, scf, utils

import qedft.train.od.scf as scf_qedft


@functools.partial(jax.jit, static_argnums=(3, 5, 6))
def _kohn_sham_iteration_amplitude_encoded(
    density,
    external_potential,
    grids,
    num_electrons,
    xc_energy_density_fn,
    interaction_fn,
    enforce_reflection_symmetry,
):
    """One iteration of Kohn-Sham calculation."""

    hartree_potential = scf.get_hartree_potential(
        density=density,
        interaction_fn=interaction_fn,
        grids=grids,
    )
    xc_potential = scf_qedft.get_xc_potential_amplitude_encoded(
        density=density,
        grids=grids,
        xc_energy_density_fn=xc_energy_density_fn,
    )
    ks_potential = hartree_potential + xc_potential + external_potential
    xc_energy_density = xc_energy_density_fn(density)

    # Solve Kohn-Sham equation.
    density, total_eigen_energies, gap = scf.solve_noninteracting_system(
        external_potential=ks_potential,
        num_electrons=num_electrons,
        grids=grids,
    )

    total_energy = (
        # kinetic energy = total_eigen_energies - external_potential_energy
        total_eigen_energies
        - scf.get_external_potential_energy(
            external_potential=ks_potential,
            density=density,
            grids=grids,
        )
        # Hartree energy
        + scf.get_hartree_energy(
            density=density,
            grids=grids,
            interaction_fn=interaction_fn,
        )
        # xc energy
        + scf_qedft.get_xc_energy_amplitude_encoded(
            density=density,
            xc_energy_density_fn=xc_energy_density_fn,
            grids=grids,
        )
        # external energy
        + scf.get_external_potential_energy(
            external_potential=external_potential,
            density=density,
            grids=grids,
        )
    )

    # This is however still switched on!
    # This was crucial to converge at large bond distances.
    if enforce_reflection_symmetry:
        density = jit_scf._flip_and_average_on_center(density)

    return (
        density,
        total_energy,
        hartree_potential,
        xc_potential,
        xc_energy_density,
        gap,
    )


def kohn_sham_iteration_amplitude_encoded(
    state,
    num_electrons,
    xc_energy_density_fn,
    interaction_fn,
    enforce_reflection_symmetry,
):
    """One iteration of Kohn-Sham calculation.

    Note xc_energy_density_fn must be wrapped by jax.tree_util.Partial so this
    function can take a callable. When the arguments of this callable changes,
    e.g. the parameters of the neural network, kohn_sham_iteration() will not be
    recompiled.

    Args:
      state: KohnShamState.
      num_electrons: Integer, the number of electrons in the system. The first
          num_electrons states are occupid.
      xc_energy_density_fn: function takes density (num_grids,) and returns
          the energy density (num_grids,).
      interaction_fn: function takes displacements and returns
          float numpy array with the same shape of displacements.
      enforce_reflection_symmetry: Boolean, whether to enforce reflection
          symmetry. If True, the system are symmetric respecting to the center.

    Returns:
      KohnShamState, the next state of Kohn-Sham iteration.
    """
    (
        density,
        total_energy,
        hartree_potential,
        xc_potential,
        xc_energy_density,
        gap,
    ) = _kohn_sham_iteration_amplitude_encoded(
        state.density,
        state.external_potential,
        state.grids,
        num_electrons,
        xc_energy_density_fn,
        interaction_fn,
        enforce_reflection_symmetry,
    )
    return state._replace(
        density=density,
        total_energy=total_energy,
        hartree_potential=hartree_potential,
        xc_potential=xc_potential,
        xc_energy_density=xc_energy_density,
        gap=gap,
    )


@functools.partial(jax.jit, static_argnums=(2, 3, 6, 8, 9, 10, 11, 12, 13))
def _kohn_sham_amplitude_encoded(
    locations,
    nuclear_charges,
    num_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density,
    alpha,
    alpha_decay,
    enforce_reflection_symmetry,
    num_mixing_iterations,
    density_mse_converge_tolerance,
    stop_gradient_step,
):
    """Jit-able Kohn Sham calculation."""
    num_grids = grids.shape[0]
    weights = jit_scf._connection_weights(num_iterations, num_mixing_iterations)

    def _converged_kohn_sham_iteration(old_state_differences):
        old_state, differences = old_state_differences
        return old_state._replace(converged=True), differences

    def _uncoveraged_kohn_sham_iteration(idx_old_state_alpha_differences):
        idx, old_state, alpha, differences = idx_old_state_alpha_differences
        state = kohn_sham_iteration_amplitude_encoded(
            state=old_state,
            num_electrons=num_electrons,
            xc_energy_density_fn=xc_energy_density_fn,
            interaction_fn=interaction_fn,
            enforce_reflection_symmetry=enforce_reflection_symmetry,
        )
        differences = differences.at[idx].set(state.density - old_state.density)
        # Density mixing.
        state = state._replace(
            density=old_state.density + alpha * jnp.dot(weights[idx], differences),
        )
        return state, differences

    def _single_kohn_sham_iteration(carry, inputs):
        del inputs
        idx, old_state, alpha, converged, differences = carry

        state, differences = jax.lax.cond(
            converged,
            true_operand=(old_state, differences),
            true_fun=_converged_kohn_sham_iteration,
            false_operand=(idx, old_state, alpha, differences),
            false_fun=_uncoveraged_kohn_sham_iteration,
        )
        converged = (
            jnp.mean(jnp.square(state.density - old_state.density))
            < density_mse_converge_tolerance
        )
        state = jax.lax.cond(
            idx <= stop_gradient_step,
            true_fun=jax.lax.stop_gradient,
            false_fun=lambda x: x,
            operand=state,
        )
        return (
            idx + 1,
            state,
            alpha * alpha_decay,
            converged,
            differences,
        ), state

    # Create initial state.
    state = scf.KohnShamState(
        density=initial_density,
        total_energy=jnp.inf,
        locations=locations,
        nuclear_charges=nuclear_charges,
        external_potential=utils.get_atomic_chain_potential(
            grids=grids,
            locations=locations,
            nuclear_charges=nuclear_charges,
            interaction_fn=interaction_fn,
        ),
        grids=grids,
        num_electrons=num_electrons,
        # Add dummy fields so the input and output of lax.scan have the same type
        # structure.
        hartree_potential=jnp.zeros_like(grids),
        xc_potential=jnp.zeros_like(grids),
        xc_energy_density=jnp.zeros(shape=()),
        gap=0.0,
        converged=False,
    )
    # Initialize the density differences with all zeros since the carry in
    # lax.scan must keep the same shape.
    differences = jnp.zeros((num_iterations, num_grids))

    _, states = jax.lax.scan(
        _single_kohn_sham_iteration,
        init=(0, state, alpha, state.converged, differences),
        xs=jnp.arange(num_iterations),
    )
    return states


def kohn_sham_amplitude_encoded(
    locations,
    nuclear_charges,
    num_electrons,
    num_iterations,
    grids,
    xc_energy_density_fn,
    interaction_fn,
    initial_density,
    alpha=0.5,
    alpha_decay=0.9,
    enforce_reflection_symmetry=False,
    num_mixing_iterations=2,
    density_mse_converge_tolerance=-1.0,
    stop_gradient_step=-1,
):
    """Jit-able Kohn Sham calculation.

    In order to make it jit-able. The following options are removed from
    kohn_sham():

      * There is no default initial density.
      * There is no convergence criteria and early stopping.
      * Reflection symmetry flip density at the center of the grids, not
          locations.

    Besides, the for loop is replaced by jax.lax.scan so it is jit friendly.
    Otherwise, jit on GPU runs into issues for big amount of for loop steps.

    Args:
      locations: Float numpy array with shape (num_nuclei,), the locations of
          atoms.
      nuclear_charges: Float numpy array with shape (num_nuclei,), the nuclear
          charges.
      num_electrons: Integer, the number of electrons in the system. The first
          num_electrons states are occupid.
      num_iterations: Integer, the number of Kohn-Sham iterations.
      grids: Float numpy array with shape (num_grids,).
      xc_energy_density_fn: function takes density (num_grids,) and returns
          the energy density (num_grids,).
      interaction_fn: function takes displacements and returns
          float numpy array with the same shape of displacements.
      initial_density: Float numpy array with shape (num_grids,), initial guess
          of the density for Kohn-Sham calculation.
      alpha: Float between 0 and 1, density linear mixing factor, the fraction
          of the output of the k-th Kohn-Sham iteration.
          If 0, the input density to the k-th Kohn-Sham iteration is fed into
          the (k+1)-th iteration. The output of the k-th Kohn-Sham iteration is
          completely ignored.
          If 1, the output density from the k-th Kohn-Sham iteration is fed into
          the (k+1)-th iteration, equivalent to no density mixing.
      alpha_decay: Float between 0 and 1, the decay factor of alpha. The mixing
          factor after k-th iteration is alpha * alpha_decay ** k.
      enforce_reflection_symmetry: Boolean, whether to enforce reflection
          symmetry.
      num_mixing_iterations: Integer, the number of density differences in the
          previous iterations to mix the density.
      density_mse_converge_tolerance: Float, the stopping criteria. When the MSE
          density difference between two iterations is smaller than this value,
          the Kohn Sham iterations finish. The outputs of the rest of the steps
          are padded by the output of the converged step. Set this value to
          negative to disable early stopping.
      stop_gradient_step: Integer, apply stop gradient on the output state of
          this step and all steps before. The first KS step is indexed as 0.

    Returns:
      KohnShamState, the states of all the Kohn-Sham iteration steps.
    """
    return _kohn_sham_amplitude_encoded(
        locations,
        nuclear_charges,
        num_electrons,
        num_iterations,
        grids,
        xc_energy_density_fn,
        interaction_fn,
        initial_density,
        alpha,
        alpha_decay,
        enforce_reflection_symmetry,
        num_mixing_iterations,
        density_mse_converge_tolerance,
        stop_gradient_step,
    )
