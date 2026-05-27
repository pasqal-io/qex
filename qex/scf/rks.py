"""SCF training/evaluation loops with optional fractional occupation.

Two entry points:
    - `rks_loss(...)`: differentiable SCF loop, returns weighted (energy + density) loss.
    - `rks_energy(...)`: differentiable SCF loop, returns final total energy (for eval).

Both accept `xc_eval_fn` (closure over a network) so the same loop drives any XC.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from chex import Array

from qex.linalg.generalized_eigensolver import generalized_eigh
from qex.scf.fermi import get_fractional_occupations_jax
from qex.scf.jax_diis import apply_diis, initialize_diis
from qex.scf.operators import (
    energy_tot,
    get_occ,
    get_veff,
    make_rdm1_custom,
)


def _occ_step(mo_energy: Array, nelectron: int, frac_enabled: int, **frac_kwargs):
    """Choose between fractional Fermi-Dirac occupations and aufbau."""

    def frac_branch(_):
        mo_occ, energy_entr, _mu, _loss = get_fractional_occupations_jax(
            mo_energy=mo_energy,
            n_electrons=nelectron,
            theta=frac_kwargs["frac_theta"],
            frac_mu=frac_kwargs["frac_mu"],
            frac_mu_shift=frac_kwargs["frac_mu_shift"],
            frac_step_grad=frac_kwargs["frac_step_grad"],
            frac_max_steps=frac_kwargs["frac_max_steps"],
        )
        return mo_occ, energy_entr

    def integer_branch(_):
        return get_occ(nelectron, mo_energy), jnp.array(0.0)

    return jax.lax.cond(frac_enabled == 1, frac_branch, integer_branch, None)


@partial(
    jax.jit,
    static_argnames=(
        "xc_eval_fn",
        "encoding",
        "max_cycle",
        "diis_max_vec",
        "diis_min_vec",
        "diis_start_cycle",
        "diis_damping",
        "ignore_ks_iter",
        "energy_weight",
        "density_weight",
        "frac_enabled",
        "frac_theta",
        "frac_mu",
        "frac_mu_shift",
        "frac_step_grad",
        "frac_max_steps",
    ),
)
def rks_loss(
    params: dict,
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    s1e: Array,
    h1e: Array,
    energy_nuc: float,
    nelectron: int,
    exact_energy: float,
    exact_density: Array,
    exact_dm: Array,
    grid_coords: Array | None = None,
    atom_coords: Array | None = None,
    *,
    xc_eval_fn: Callable,
    encoding: str = "local",
    max_cycle: int = 15,
    diis_max_vec: int = 15,
    diis_min_vec: int = 2,
    diis_start_cycle: int = 1,
    diis_damping: float = 0.0,
    ignore_ks_iter: int = 5,
    discount_coeffs: Array | None = None,
    energy_weight: float = 1.0,
    density_weight: float = 1.0,
    frac_enabled: int = 1,
    frac_theta: float = 0.04,
    frac_mu: float | None = None,
    frac_mu_shift: float = 0.001,
    frac_step_grad: float = 0.6,
    frac_max_steps: int = 100,
) -> Array:
    """Differentiable SCF loop, returns weighted (energy + density) loss."""

    print("Compiling/executing SCF loop")

    vhf, exc_energy, J = get_veff(
        dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
        encoding=encoding, grid_coords=grid_coords, atom_coords=atom_coords,
    )
    e_tot = energy_tot(dm, h1e, J, exc_energy, energy_nuc)

    diis_state = initialize_diis(diis_max_vec)
    loss = 0.0

    frac_kwargs = dict(
        frac_theta=frac_theta,
        frac_mu=frac_mu,
        frac_mu_shift=frac_mu_shift,
        frac_step_grad=frac_step_grad,
        frac_max_steps=frac_max_steps,
    )

    for cycle in range(max_cycle):
        fock = h1e + vhf

        if cycle >= diis_start_cycle:
            fock, diis_state = apply_diis(
                diis_state,
                fock,
                dm,
                s1e,
                diis_max_vec,
                diis_min_vec,
                diis_damping,
            )

        mo_energy, mo_coeff = generalized_eigh(fock, s1e)
        mo_occ, energy_entr = _occ_step(mo_energy, nelectron, frac_enabled, **frac_kwargs)

        dm = make_rdm1_custom(mo_coeff, mo_occ)

        vhf, exc_energy, J = get_veff(
            dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
            encoding=encoding,
            grid_coords=grid_coords, atom_coords=atom_coords,
        )
        e_tot = energy_tot(dm, h1e, J, exc_energy, energy_nuc) + energy_entr

        if cycle > ignore_ks_iter:
            if discount_coeffs is not None:
                loss += discount_coeffs[cycle] * (e_tot - exact_energy) ** 2
            else:
                loss += (e_tot - exact_energy) ** 2

    rho = jnp.einsum("gi,ij,gj->g", ao_grid, dm, ao_grid)
    rho_loss = jnp.mean((rho - exact_density) ** 2)

    return energy_weight * loss + density_weight * rho_loss


def rks_energy(
    params: dict,
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    s1e: Array,
    h1e: Array,
    energy_nuc: float,
    nelectron: int,
    grid_coords: Array | None = None,
    atom_coords: Array | None = None,
    *,
    xc_eval_fn: Callable,
    encoding: str = "local",
    max_cycle: int = 15,
    diis_max_vec: int = 15,
    diis_min_vec: int = 2,
    diis_start_cycle: int = 1,
    diis_damping: float = 0.0,
    frac_enabled: int = 1,
    frac_theta: float = 0.04,
    frac_mu: float | None = None,
    frac_mu_shift: float = 0.001,
    frac_step_grad: float = 0.6,
    frac_max_steps: int = 100,
) -> Array:
    """Differentiable SCF loop, returns final total energy."""
    vhf, exc_energy, J = get_veff(
        dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
        encoding=encoding, grid_coords=grid_coords, atom_coords=atom_coords,
    )
    e_tot = energy_tot(dm, h1e, J, exc_energy, energy_nuc)

    diis_state = initialize_diis(diis_max_vec)

    frac_kwargs = dict(
        frac_theta=frac_theta,
        frac_mu=frac_mu,
        frac_mu_shift=frac_mu_shift,
        frac_step_grad=frac_step_grad,
        frac_max_steps=frac_max_steps,
    )

    for cycle in range(max_cycle):
        fock = h1e + vhf

        if cycle >= diis_start_cycle:
            fock, diis_state = apply_diis(
                diis_state,
                fock,
                dm,
                s1e,
                diis_max_vec,
                diis_min_vec,
                diis_damping,
            )

        mo_energy, mo_coeff = generalized_eigh(fock, s1e)
        mo_occ, energy_entr = _occ_step(mo_energy, nelectron, frac_enabled, **frac_kwargs)

        dm = make_rdm1_custom(mo_coeff, mo_occ)

        vhf, exc_energy, J = get_veff(
            dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
            encoding=encoding,
            grid_coords=grid_coords, atom_coords=atom_coords,
        )
        e_tot = energy_tot(dm, h1e, J, exc_energy, energy_nuc) + energy_entr

    return e_tot
