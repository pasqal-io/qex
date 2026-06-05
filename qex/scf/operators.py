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
    # `optimize=True` lets opt_einsum/XLA pick the contraction order. It is
    # numerically identical to the naive order (verified in tests) and never
    # slower, so it is on for every contraction in this module.
    J = jnp.einsum("ijkl,kl->ij", eri, dm, optimize=True)
    rho = jnp.einsum("gi,ij,gj->g", ao_grid, dm, ao_grid, optimize=True)
    exc, (vrho, _, _, _), _, _ = xc_eval_fn(
        "", rho, params=params, grid_weights=grid_weights,
    )
    Vxc = jnp.einsum("gi,g,gj->ij", ao_grid, grid_weights * vrho, ao_grid, optimize=True)
    return J + Vxc, jnp.sum(exc * rho * grid_weights), J


def get_veff_global(
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    params: dict,
    xc_eval_fn: Callable,
    features: dict | None = None,
) -> tuple[Array, Array, Array]:
    """Effective KS potential for GLOBAL encoding (scalar E_xc[ρ]).

    The network returns the already-integrated scalar XC energy, so we do
    NOT multiply by ρ·w again — that would double-integrate.

    `features` is the named model-feature bag (see `qex.functionals.features`),
    already narrowed to the network's `required_features`. It is forwarded to
    `xc_eval_fn` verbatim and handed to the network by keyword; this function
    does not interpret it, so adding a new feature never touches the SCF math.
    Descriptor-style networks consume e.g. `grid_coords`/`atom_coords`; a plain
    `GlobalMLP` requests `()` and gets an empty bag.

    Returns (Vhf = J + Vxc, exc_energy, J).
    """
    # See `get_veff_local`: `optimize=True` is numerically identical and never
    # slower; on for every contraction here.
    J = jnp.einsum("ijkl,kl->ij", eri, dm, optimize=True)
    rho = jnp.einsum("gi,ij,gj->g", ao_grid, dm, ao_grid, optimize=True)
    exc, (vrho, _, _, _), _, _ = xc_eval_fn(
        "", rho,
        params=params,
        grid_weights=grid_weights,
        features=features,
    )
    Vxc = jnp.einsum("gi,g,gj->ij", ao_grid, grid_weights * vrho, ao_grid, optimize=True)
    return J + Vxc, exc, J


def get_veff(
    dm: Array,
    eri: Array,
    ao_grid: Array,
    grid_weights: Array,
    params: dict,
    xc_eval_fn: Callable,
    encoding: str = "local",
    features: dict | None = None,
) -> tuple[Array, Array, Array]:
    """Dispatch to `get_veff_local` or `get_veff_global` based on `encoding`.

    `features` is the named model-feature bag (see `qex.functionals.features`),
    forwarded to the global path for descriptor-style networks; ignored by the
    local path (ε_xc(r) depends on ρ alone).
    """
    if encoding == "local":
        return get_veff_local(dm, eri, ao_grid, grid_weights, params, xc_eval_fn)
    if encoding == "global":
        return get_veff_global(
            dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
            features=features,
        )
    if encoding == "libxc":
        # Reference path: a standard PySCF/libxc functional (LDA/GGA/mGGA/hybrid)
        # driven through this same SCF loop to validate it against PySCF. The
        # ingredients (AO value+gradient arrays, xc code, hybrid coeff) ride in
        # `params`; see `qex.functionals.libxc_veff.make_libxc_ingredients`.
        from qex.functionals.libxc_veff import get_veff_libxc

        return get_veff_libxc(dm, eri, ao_grid, grid_weights, params, xc_eval_fn)
    raise ValueError(
        f"Unknown XC encoding: {encoding!r} (expected 'local', 'global', or 'libxc')."
    )


def energy_tot(dm: Array, h1e: Array, J: Array, exc_energy: Array, energy_nuc: float) -> Array:
    """KS total energy: one-electron + Hartree + XC + nuclear."""
    e_one = jnp.einsum("ij,ji->", dm, h1e, optimize=True)
    e_hartree = 0.5 * jnp.einsum("ij,ij->", dm, J, optimize=True)
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
