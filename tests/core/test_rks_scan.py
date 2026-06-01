"""Equivalence tests: legacy `rks_loss` (Python-unroll + DIIS) vs
`rks_loss_scan` (lax.scan, DIIS optional).

Two regimes covered:

- **No DIIS, both losses must agree numerically.** `rks_loss` is forced into
  the no-DIIS branch by pushing `diis_start_cycle` past `max_cycle`.
- **With DIIS, both losses must agree numerically.** Uses the matching
  parameters on both sides; the scan version is backed by the fixed-buffer
  `jax_diis_scan`, which `test_jax_diis_scan.py` already proves matches
  `jax_diis` per-step.

Fixture is built in-code (no pickled artifacts) to stay version-robust:
H2/STO-3G, a single grid point cluster, a small `GlobalMLP` XC. Whole file
runs in a few seconds on CPU.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import dft, gto

from qex.functionals.mlp import GlobalMLP
from qex.functionals.xc import make_eval_xc_global
from qex.scf.operators import get_ao_value
from qex.scf.rks import rks_loss, rks_loss_scan

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def h2_inputs():
    """All arrays `rks_loss(_scan)` needs for one tiny H2 point.

    Built fresh in-process; nothing serialized to disk. Cheap (well under a
    second) and shared across every test in this module.
    """
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = dft.RKS(mol)
    mf.xc = "lda"
    # Tiny grid keeps the test fast; we don't need converged energies.
    mf.grids.level = 0
    mf.grids.build()
    mf.kernel()

    network = GlobalMLP(features=[16], act_fn=nn.gelu)
    n_grid = mf.grids.coords.shape[0]
    params = network.init(jax.random.PRNGKey(0), jnp.ones(n_grid))
    xc_eval_fn = make_eval_xc_global(network, vxc_grad_scale=1.0)

    dm = jnp.asarray(mf.make_rdm1())
    rho = jnp.einsum(
        "gi,ij,gj->g",
        jnp.asarray(get_ao_value(mol, mf.grids.coords)),
        dm,
        jnp.asarray(get_ao_value(mol, mf.grids.coords)),
    )

    return dict(
        params=params,
        xc_eval_fn=xc_eval_fn,
        dm=dm,
        eri=jnp.asarray(mol.intor("int2e", aosym="s1")),
        ao_grid=jnp.asarray(get_ao_value(mol, mf.grids.coords)),
        grid_weights=jnp.asarray(mf.grids.weights),
        s1e=jnp.asarray(mf.get_ovlp(mol)),
        h1e=jnp.asarray(mf.get_hcore(mol)),
        energy_nuc=float(mol.energy_nuc()),
        nelectron=int(mol.nelectron),
        exact_energy=float(mf.e_tot),
        exact_density=rho,
        n_grid=n_grid,
    )


# Frac-occ kwargs that both losses accept; we leave them on (matches train_h2)
# so the test exercises the same `_occ_step` path used in real training.
_FRAC = dict(
    frac_enabled=1,
    frac_theta=0.04,
    frac_mu=None,
    frac_mu_shift=0.001,
    frac_step_grad=0.6,
    frac_max_steps=100,
)


def _call_rks_loss(inputs, **kw):
    return float(rks_loss(
        inputs["params"], inputs["dm"], inputs["eri"], inputs["ao_grid"],
        inputs["grid_weights"], inputs["s1e"], inputs["h1e"],
        inputs["energy_nuc"], inputs["nelectron"],
        inputs["exact_energy"], inputs["exact_density"], inputs["dm"],
        xc_eval_fn=inputs["xc_eval_fn"], encoding="global",
        **_FRAC, **kw,
    ))


def _call_rks_loss_scan(inputs, **kw):
    return float(rks_loss_scan(
        inputs["params"], inputs["dm"], inputs["eri"], inputs["ao_grid"],
        inputs["grid_weights"], inputs["s1e"], inputs["h1e"],
        inputs["energy_nuc"], inputs["nelectron"],
        inputs["exact_energy"], inputs["exact_density"], inputs["dm"],
        xc_eval_fn=inputs["xc_eval_fn"], encoding="global",
        **_FRAC, **kw,
    ))


def test_rks_no_diis_legacy_vs_scan(h2_inputs):
    """With DIIS disabled, both losses must produce the same number."""
    max_cycle = 4

    # Legacy `rks_loss` doesn't expose a `use_diis` switch; we disable DIIS
    # by pushing `diis_start_cycle` past the last cycle.
    loss_legacy = _call_rks_loss(
        h2_inputs,
        max_cycle=max_cycle,
        diis_max_vec=15, diis_min_vec=2,
        diis_start_cycle=max_cycle + 1, diis_damping=0.0,
        ignore_ks_iter=1,
    )
    loss_scan = _call_rks_loss_scan(
        h2_inputs,
        max_cycle=max_cycle, ignore_ks_iter=1,
        use_diis=False,
    )

    np.testing.assert_allclose(loss_legacy, loss_scan, atol=1e-10, rtol=1e-10)


def test_rks_with_diis_legacy_vs_scan(h2_inputs):
    """With DIIS enabled, both losses must agree (within DIIS extrapolation
    numerics, which are machine-precision equivalent per `test_jax_diis_scan`)."""
    max_cycle = 4
    diis_kwargs = dict(
        diis_max_vec=6, diis_min_vec=2,
        diis_start_cycle=1, diis_damping=0.0,
    )

    loss_legacy = _call_rks_loss(
        h2_inputs,
        max_cycle=max_cycle, ignore_ks_iter=1,
        **diis_kwargs,
    )
    loss_scan = _call_rks_loss_scan(
        h2_inputs,
        max_cycle=max_cycle, ignore_ks_iter=1,
        use_diis=True, **diis_kwargs,
    )

    np.testing.assert_allclose(loss_legacy, loss_scan, atol=1e-9, rtol=1e-9)


def test_rks_loss_scan_runs_under_grad(h2_inputs):
    """Smoke check: the scan loss must be differentiable end-to-end."""
    def loss_of_params(p):
        inp = {**h2_inputs, "params": p}
        return rks_loss_scan(
            inp["params"], inp["dm"], inp["eri"], inp["ao_grid"],
            inp["grid_weights"], inp["s1e"], inp["h1e"],
            inp["energy_nuc"], inp["nelectron"],
            inp["exact_energy"], inp["exact_density"], inp["dm"],
            xc_eval_fn=inp["xc_eval_fn"], encoding="global",
            max_cycle=3, ignore_ks_iter=0,
            use_diis=True, diis_max_vec=4, diis_min_vec=2,
            diis_start_cycle=1, diis_damping=0.0,
            **_FRAC,
        )

    grads = jax.grad(loss_of_params)(h2_inputs["params"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)
