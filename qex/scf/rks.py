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
from loguru import logger

from qex.linalg.generalized_eigensolver import generalized_eigh
from qex.linalg.ks_solvers import lobpcg_solve, mcweeny_purify
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


def _solve_density(
    fock: Array,
    s1e: Array,
    nelectron: int,
    frac_enabled: int,
    solver: str,
    frac_kwargs: dict,
):
    """Build the next 1-RDM from the Fock matrix via the chosen KS solver.

    Returns ``(dm, energy_entr)``. Unifies the dense / LOBPCG eigen-path (which
    yields ``(mo_energy, mo_coeff)`` and goes through occupations + make_rdm1)
    with the purification path (which yields ``dm`` directly, no eigenvectors).

    Solver / occupation compatibility (enforced below with clear errors, never
    silent wrong numbers):

    - ``"dense"``  : `generalized_eigh`. The historical default — UNCHANGED, and
      the only solver supporting fractional occupation under the training vmap.
      Works for any size and any `frac_enabled`. Traces to the exact same ops as
      the original inline solve, so the default path is byte-for-byte preserved.

    - ``"purify"`` : McWeeny density-matrix purification (GEMM-only, scales best
      for large single systems). Builds the closed-shell *idempotent* density
      directly, so it is INTEGER AUFBAU ONLY — `frac_enabled=1` is rejected
      (purification cannot represent fractional/smeared occupations). Accepts a
      traced `n_occ`, so it runs inside the jitted/vmapped loop.

    - ``"lobpcg"`` : lowest-`n_occ` iterative eigensolve (same eigen-contract).
      Its search-block width is an array SHAPE built from `n_occ`, so `n_occ`
      must be a compile-time int. In the training loop `nelectron` is traced
      (vmap over samples) -> rejected with a clear error; use it for direct
      solves. With fractional occupation it additionally needs computed states
      ABOVE the HOMO (a real LUMO) to smear against, i.e. a basis large enough
      that `5*(n_occ+buffer) < n`; otherwise rejected.

    `energy_entr` is 0 on the purify path (no smearing entropy).
    """
    n_occ = nelectron // 2

    if solver == "purify":
        # frac_enabled is a static (Python) arg in both SCF loops, so this
        # branch is resolved at trace time — no runtime cost.
        if frac_enabled == 1:
            raise ValueError(
                "solver='purify' supports integer aufbau only; set "
                "frac_enabled=0 (or use solver='dense'/'lobpcg' for "
                "fractional occupation)."
            )
        dm = mcweeny_purify(fock, s1e, n_occ=n_occ)
        return dm, jnp.array(0.0)

    if solver == "lobpcg":
        # LOBPCG's search-block width k = n_occ + buffer sets an ARRAY SHAPE, so
        # n_occ must be a concrete Python int at trace time. Inside the jitted
        # SCF loop `nelectron` is traced, so n_occ here is a tracer -> LOBPCG
        # cannot be used unless `nelectron` is made a static argument. Detect
        # that up front and explain it, instead of letting JAX fail later with
        # an opaque "unhashable DynamicJaxprTracer" from the solver's jit.
        if isinstance(n_occ, jax.core.Tracer):
            raise TypeError(
                "solver='lobpcg' needs a static (compile-time) electron count, "
                "but `nelectron` is traced inside the SCF loop, so n_occ is a "
                "tracer. LOBPCG sizes its search block from n_occ, which must "
                "be a Python int. Use solver='dense' or 'purify' for the "
                "jitted loop (both accept a traced n_occ); LOBPCG is intended "
                "for calling the solver directly on a fixed-size system."
            )
        # Fractional (Fermi-Dirac) occupation smears charge across the gap, so
        # it needs computed states ABOVE the HOMO (a real LUMO). LOBPCG's block
        # is capped by 5*k < n; when that leaves no room above n_occ there is no
        # LUMO to smear and the chemical-potential solve diverges to NaN. Reject
        # that combination up front rather than returning silent NaNs.
        n_basis = fock.shape[-1]
        k_max = (n_basis - 1) // 5
        if frac_enabled == 1 and k_max <= n_occ:
            raise ValueError(
                f"solver='lobpcg' with fractional occupation needs LOBPCG to "
                f"compute states above the HOMO, but the basis is too small: "
                f"n={n_basis}, n_occ={n_occ} allow at most k={k_max} states "
                f"(5*k < n), leaving no LUMO buffer. Use a larger basis, "
                f"solver='dense', or set frac_enabled=0."
            )
        mo_energy, mo_coeff = lobpcg_solve(fock, s1e, n_occ=n_occ)
    else:  # "dense"
        mo_energy, mo_coeff = generalized_eigh(fock, s1e)

    mo_occ, energy_entr = _occ_step(
        mo_energy, nelectron, frac_enabled, **frac_kwargs,
    )
    return make_rdm1_custom(mo_coeff, mo_occ), energy_entr


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
    features: dict | None = None,
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
    """Differentiable SCF loop, returns weighted (energy + density) loss.

    `features` is the named model-feature bag (see `qex.functionals.features`),
    passed straight through to `get_veff` and consumed by the network. The SCF
    math never reads it, so adding a feature does not touch this loop.
    """

    logger.info("Compiling/executing SCF loop (rks_loss, unrolled + DIIS)")

    vhf, exc_energy, J = get_veff(
        dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
        encoding=encoding, features=features,
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
            encoding=encoding, features=features,
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
        "solver",
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
    features: dict | None = None,
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
    solver: str = "dense",
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

    logger.info(
        "Compiling/executing SCF loop (scan, {}, {})".format(
            "with DIIS" if use_diis else "no DIIS",
            "with frac. occ." if frac_enabled else "no frac. occ.",
        )
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
            encoding=encoding, features=features,
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

        dm_new, energy_entr = _solve_density(
            fock, s1e, nelectron, frac_enabled, solver, frac_kwargs,
        )
        vhf, exc_energy, J = get_veff(
            dm_new, eri, ao_grid, grid_weights, params, xc_eval_fn,
            encoding=encoding, features=features,
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
    features: dict | None = None,
    *,
    xc_eval_fn: Callable,
    encoding: str = "local",
    max_cycle: int = 15,
    use_diis: bool = True,
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
    solver: str = "dense",
) -> Array:
    """Differentiable SCF loop, returns final total energy.

    `use_diis` toggles Fock extrapolation so this evaluator can be configured
    to match whichever SCF path was used at training time (see `rks_loss_scan`)
    — train/eval mismatch on DIIS would otherwise drift the dissociation
    profile away from the converged training fixed point.

    `features` is the named model-feature bag (see `qex.functionals.features`),
    forwarded to `get_veff` and consumed by the network; the SCF math ignores it.
    """
    vhf, exc_energy, J = get_veff(
        dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
        encoding=encoding, features=features,
    )
    e_tot = energy_tot(dm, h1e, J, exc_energy, energy_nuc)

    diis_state = initialize_diis(diis_max_vec) if use_diis else None

    frac_kwargs = dict(
        frac_theta=frac_theta,
        frac_mu=frac_mu,
        frac_mu_shift=frac_mu_shift,
        frac_step_grad=frac_step_grad,
        frac_max_steps=frac_max_steps,
    )

    for cycle in range(max_cycle):
        fock = h1e + vhf

        if use_diis and cycle >= diis_start_cycle:
            fock, diis_state = apply_diis(
                diis_state,
                fock,
                dm,
                s1e,
                diis_max_vec,
                diis_min_vec,
                diis_damping,
            )

        dm, energy_entr = _solve_density(
            fock, s1e, nelectron, frac_enabled, solver, frac_kwargs,
        )

        vhf, exc_energy, J = get_veff(
            dm, eri, ao_grid, grid_weights, params, xc_eval_fn,
            encoding=encoding, features=features,
        )
        e_tot = energy_tot(dm, h1e, J, exc_energy, energy_nuc) + energy_entr

    return e_tot
