"""SCF operators: effective potential, total energy, AO values, occupations, density matrices.

Two XC encodings are supported:

- **local**: the network returns a per-grid energy density per electron ε_xc(r).
  The total XC energy is assembled outside the network as E_xc = Σ ε_xc(r) ρ(r) w(r).
  This is the classical LDA/GGA / PySCF convention.

- **global**: the network returns the already-integrated scalar E_xc[ρ].
  No outer ρ·w multiplication is applied.

Pick the matching `get_veff_*` for the encoding produced by your network.
"""

from collections.abc import Callable

import jax.numpy as jnp
from chex import Array


def get_veff_local(
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    params: dict,
    xc_eval_fn: Callable,
) -> tuple[Array, Array, Array]:
    """Effective KS potential for LOCAL encoding (ε_xc(r) per electron).

    The network returns a per-grid energy density per electron, and we
    integrate explicitly: E_xc = Σ ε_xc(r) ρ(r) w(r).

    Returns (Vhf = J + Vxc, exc_energy, J).
    """
    J = jnp.einsum("ijkl,kl->ij", eri, dm)
    rho = jnp.einsum("gi,ij,gj->g", ao_grid, dm, ao_grid)
    exc, (vrho, _, _, _), _, _ = xc_eval_fn(
        "", rho, params=params, grid_weights=grid_weights,
    )
    Vxc = jnp.einsum("gi,g,gj->ij", ao_grid, grid_weights * vrho, ao_grid)
    return J + Vxc, jnp.sum(exc * rho * grid_weights), J


def get_veff_global(
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    params: dict,
    xc_eval_fn: Callable,
    grid_coords: Array | None = None,
    atom_coords: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Effective KS potential for GLOBAL encoding (scalar E_xc[ρ]).

    The network returns the already-integrated scalar XC energy, so we do
    NOT multiply by ρ·w again — that would double-integrate.

    Descriptor-style networks (e.g. `DescriptorXC`) also need `grid_coords`
    and `atom_coords` to evaluate atom-centered features; pass them through.
    Plain `GlobalMLP` ignores both.

    Returns (Vhf = J + Vxc, exc_energy, J).
    """
    J = jnp.einsum("ijkl,kl->ij", eri, dm)
    rho = jnp.einsum("gi,ij,gj->g", ao_grid, dm, ao_grid)
    network_extra_args = ()
    if grid_coords is not None and atom_coords is not None:
        # `DescriptorXC.__call__(rho, grid_coords, grid_weights, atom_coords)`
        network_extra_args = (grid_coords, grid_weights, atom_coords)
    exc, (vrho, _, _, _), _, _ = xc_eval_fn(
        "", rho,
        params=params,
        grid_weights=grid_weights,
        network_extra_args=network_extra_args,
    )
    Vxc = jnp.einsum("gi,g,gj->ij", ao_grid, grid_weights * vrho, ao_grid)
    return J + Vxc, exc, J


def get_veff(
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    params: dict,
    xc_eval_fn: Callable,
    encoding: str = "local",
    grid_coords: Array | None = None,
    atom_coords: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Dispatch to `get_veff_local` or `get_veff_global` based on `encoding`.

    `grid_coords`/`atom_coords` are forwarded to the global path for
    descriptor-style networks; ignored by the local path.
    """
    if encoding == "local":
        return get_veff_local(dm, eri, ao_grid, grid_weights, params, xc_eval_fn)
    if encoding == "global":
        return get_veff_global(
            dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
            grid_coords=grid_coords, atom_coords=atom_coords,
        )
    raise ValueError(f"Unknown XC encoding: {encoding!r} (expected 'local' or 'global').")


def energy_tot(dm: Array, h1e: Array, J: Array, exc_energy: Array, energy_nuc: float) -> Array:
    """KS total energy: one-electron + Hartree + XC + nuclear."""
    e_one = jnp.einsum("ij,ji->", dm, h1e)
    e_hartree = 0.5 * jnp.einsum("ij,ij->", dm, J)
    return e_one + e_hartree + exc_energy + energy_nuc


def get_ao_value(mol, coords: Array) -> Array:
    """Evaluate AO basis on a grid (PySCF mole helper)."""
    deriv = 0
    feval = "GTOval_cart_deriv%d" % deriv if mol.cart else "GTOval_sph_deriv%d" % deriv
    return mol.eval_gto(feval, coords)


def get_occ(nelectron: int, mo_energy: Array) -> Array:
    """Integer aufbau occupation (closed-shell): 2 in lowest n/2 MOs, 0 elsewhere."""
    e_idx = jnp.argsort(mo_energy)
    nocc = nelectron // 2

    idx = jnp.arange(mo_energy.shape[0])
    mo_occ = jnp.where(idx < nocc, 2.0, 0.0)
    return mo_occ[jnp.argsort(e_idx)]


def make_rdm1(mo_coeff: Array, mo_occ: Array) -> Array:
    """1-RDM from MO coefficients and integer occupations."""
    return jnp.einsum("ij,j,kj->ik", mo_coeff, mo_occ, mo_coeff)


def make_rdm1_custom(mo_coeff: Array, mo_occ: Array) -> Array:
    """1-RDM allowing fractional occupations (dm = C·diag(occ)·C^T)."""
    weighted_mo = mo_coeff * jnp.sqrt(mo_occ)
    return jnp.dot(weighted_mo, weighted_mo.T)
