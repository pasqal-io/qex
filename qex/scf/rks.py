"""SCF training/evaluation loops with optional fractional occupation.

Entry points:
    - `rks_loss(...)`:     Python-unrolled SCF + DIIS, returns weighted loss.
    - `rks_loss_scan(...)`: lax.scan SCF, no DIIS, returns the same loss.
    - `rks_energy(...)`:   Python-unrolled SCF + DIIS, returns final energy.

All three accept `xc_eval_fn` (closure over a network) so the same loop drives
any XC.

Which loss to use
-----------------
- `rks_loss` is the historical default. DIIS gives faster SCF convergence at
  the cost of a graph that scales linearly with `max_cycle` — fine for cheap
  XC networks (MLPs), but with the QCNN the unrolled `value_and_grad` trace
  pushes XLA compile time into many minutes.
- `rks_loss_scan` folds the SCF cycles with `lax.scan`, so compile cost is
  roughly constant in `max_cycle`. It currently drops DIIS because the
  existing `jax_diis.py` keeps a Python list of error/Fock vectors that
  grows each iteration — not scan-compatible without a fixed-buffer rewrite.
  Use this for QCNN training; revisit once DIIS is ported to fixed buffers
  (then `rks_loss_scan` can take an optional `use_diis` flag, mirroring
  how `frac_enabled` is already toggleable).
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from chex import Array

from qex.linalg.generalized_eigensolver import generalized_eigh
from qex.scf.fermi import get_fractional_occupations_jax
from qex.scf.jax_diis import apply_diis, initialize_diis
from qex.scf.jax_diis_scan import (
    apply_diis_scan,
    initialize_diis_scan,
)
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


@partial(
    jax.jit,
    static_argnames=(
        "xc_eval_fn",
        "encoding",
        "max_cycle",
        "ignore_ks_iter",
        "energy_weight",
        "density_weight",
        "frac_enabled",
        "frac_theta",
        "frac_mu",
        "frac_mu_shift",
        "frac_step_grad",
        "frac_max_steps",
        "use_diis",
        "diis_max_vec",
        "diis_min_vec",
        "diis_start_cycle",
        "diis_damping",
    ),
)
def rks_loss_scan(
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
    ignore_ks_iter: int = 5,
    energy_weight: float = 1.0,
    density_weight: float = 1.0,
    frac_enabled: int = 1,
    frac_theta: float = 0.04,
    frac_mu: float | None = None,
    frac_mu_shift: float = 0.001,
    frac_step_grad: float = 0.6,
    frac_max_steps: int = 100,
    use_diis: bool = False,
    diis_max_vec: int = 15,
    diis_min_vec: int = 2,
    diis_start_cycle: int = 1,
    diis_damping: float = 0.0,
) -> Array:
    """Scan-based SCF loss — same loss as `rks_loss`, DIIS optional.

    `rks_loss` Python-unrolls `max_cycle` iterations inside `@jax.jit`. Each
    cycle re-traces the XC network; with the QCNN that pushes XLA compile
    times into minutes for max_cycle=15. `lax.scan` reuses a single trace of
    the body, so compile cost is roughly independent of `max_cycle` (~3s
    instead of ~40s in our H2 benchmarks).

    DIIS is opt-in via `use_diis`. When enabled, it uses the fixed-buffer
    `jax_diis_scan.apply_diis_scan`, which is numerically equivalent to the
    legacy `jax_diis.apply_diis` (machine precision) but stores history in
    pre-allocated arrays so the scan carry has a static shape.

    Fractional occupations and the `ignore_ks_iter` mask are preserved.
    """

    print(
        f"Compiling/executing SCF loop (scan, "
        f"{'with DIIS' if use_diis else 'no DIIS'})",
    )

    frac_kwargs = dict(
        frac_theta=frac_theta,
        frac_mu=frac_mu,
        frac_mu_shift=frac_mu_shift,
        frac_step_grad=frac_step_grad,
        frac_max_steps=frac_max_steps,
    )

    fock_size = h1e.size
    if use_diis:
        diis_init = initialize_diis_scan(diis_max_vec, fock_size)
    else:
        diis_init = None

    def step(carry, cycle):
        if use_diis:
            dm_c, loss_c, diis_state = carry
        else:
            dm_c, loss_c = carry

        vhf, exc_energy, J = get_veff(
            dm_c, eri, ao_grid, grid_weights, params, xc_eval_fn,
            encoding=encoding, grid_coords=grid_coords, atom_coords=atom_coords,
        )
        fock = h1e + vhf

        if use_diis:
            # Match `rks_loss`: only extrapolate from `diis_start_cycle` on.
            do_diis = cycle >= diis_start_cycle
            fock_extrap, diis_state_new = apply_diis_scan(
                diis_state, fock, dm_c, s1e,
                max_vec=diis_max_vec,
                min_vecs=diis_min_vec,
                damping=diis_damping,
            )
            fock = jnp.where(do_diis, fock_extrap, fock)
            # Keep buffer in sync regardless of whether we used the extrap.
            diis_state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(do_diis, new, old),
                diis_state_new,
                diis_state,
            )

        mo_energy, mo_coeff = generalized_eigh(fock, s1e)
        mo_occ, energy_entr = _occ_step(
            mo_energy, nelectron, frac_enabled, **frac_kwargs,
        )
        dm_new = make_rdm1_custom(mo_coeff, mo_occ)
        vhf, exc_energy, J = get_veff(
            dm_new, eri, ao_grid, grid_weights, params, xc_eval_fn,
            encoding=encoding, grid_coords=grid_coords, atom_coords=atom_coords,
        )
        e_tot = energy_tot(dm_new, h1e, J, exc_energy, energy_nuc) + energy_entr
        include = (cycle > ignore_ks_iter).astype(e_tot.dtype)
        loss_new = loss_c + include * (e_tot - exact_energy) ** 2

        if use_diis:
            return (dm_new, loss_new, diis_state), None
        return (dm_new, loss_new), None

    cycles = jnp.arange(max_cycle)
    init = (dm, jnp.float64(0.0))
    if use_diis:
        init = (*init, diis_init)
    final_carry, _ = jax.lax.scan(step, init, cycles)
    dm_final, loss = final_carry[0], final_carry[1]

    rho = jnp.einsum("gi,ij,gj->g", ao_grid, dm_final, ao_grid)
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
