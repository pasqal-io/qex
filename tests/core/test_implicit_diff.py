"""Implicit differentiation of the SCF fixed point.

Covers `qex.scf.implicit_diff.custom_fixed_point` and the `differentiation`
switch on `qex.scf.rks.rks_loss_scan`:

- The implicit and unrolled losses agree on the *value* at a converged fixed
  point (both read the same `dm*`, only the gradient path differs).
- Their gradients w.r.t. the XC params agree at convergence. The comparison
  uses a density-only loss (`energy_weight=0`) so both modes differentiate the
  exact same scalar function of `dm*` — the unrolled loss otherwise sums energy
  errors across cycles while the implicit loss reads only the converged density.
- Both gradients are finite, and the wrapper differentiates in isolation.

Fixture mirrors `test_rks_scan.py`: H2/STO-3G, a tiny grid, a small `GlobalMLP`
XC. The whole file runs in a couple of seconds on CPU.
"""

import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import dft, gto

from qex.functionals.mlp import GlobalMLP
from qex.functionals.xc import make_eval_xc_global
from qex.scf.implicit_diff import custom_fixed_point
from qex.scf.operators import get_ao_value
from qex.scf.rks import rks_loss_scan

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def h2_inputs():
    """All arrays `rks_loss_scan` needs for one tiny H2 point (built in-process)."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = dft.RKS(mol)
    mf.xc = "lda"
    mf.grids.level = 0
    mf.grids.build()
    mf.kernel()

    network = GlobalMLP(features=[16], act_fn=nn.gelu)
    n_grid = mf.grids.coords.shape[0]
    params = network.init(jax.random.PRNGKey(0), jnp.ones(n_grid))
    xc_eval_fn = make_eval_xc_global(network, vxc_grad_scale=1.0)

    ao = jnp.asarray(get_ao_value(mol, mf.grids.coords))
    dm = jnp.asarray(mf.make_rdm1())
    rho = jnp.einsum("gi,ij,gj->g", ao, dm, ao)

    return dict(
        params=params,
        xc_eval_fn=xc_eval_fn,
        dm=dm,
        eri=jnp.asarray(mol.intor("int2e", aosym="s1")),
        ao_grid=ao,
        grid_weights=jnp.asarray(mf.grids.weights),
        s1e=jnp.asarray(mf.get_ovlp(mol)),
        h1e=jnp.asarray(mf.get_hcore(mol)),
        energy_nuc=float(mol.energy_nuc()),
        nelectron=int(mol.nelectron),
        exact_energy=float(mf.e_tot),
        exact_density=rho,
        n_grid=n_grid,
    )


# Integer aufbau keeps the fixed point clean; frac-occ adds an inner solve that
# only blurs the unroll-vs-implicit comparison without exercising new code here.
_FRAC = dict(
    frac_enabled=0,
    frac_theta=0.04,
    frac_mu=None,
    frac_mu_shift=0.001,
    frac_step_grad=0.6,
    frac_max_steps=100,
)


def _loss(inputs, params, **kw):
    return rks_loss_scan(
        params, inputs["dm"], inputs["eri"], inputs["ao_grid"],
        inputs["grid_weights"], inputs["s1e"], inputs["h1e"],
        inputs["energy_nuc"], inputs["nelectron"],
        inputs["exact_energy"], inputs["exact_density"], inputs["dm"],
        xc_eval_fn=inputs["xc_eval_fn"], encoding="global",
        **_FRAC, **kw,
    )


def test_unknown_mode_raises(h2_inputs):
    with pytest.raises(ValueError, match="differentiation"):
        _loss(h2_inputs, h2_inputs["params"], max_cycle=3, differentiation="bogus")


def test_value_matches_at_convergence(h2_inputs):
    """Same loss value: both modes read the converged `dm*`.

    Density-only (`energy_weight=0`) so the value is a pure function of `dm*`,
    which is identical regardless of how gradients are taken.
    """
    common = dict(max_cycle=30, energy_weight=0.0, density_weight=1.0)
    loss_unroll = float(_loss(
        h2_inputs, h2_inputs["params"],
        use_diis=False, ignore_ks_iter=0, differentiation="unroll", **common,
    ))
    loss_impl = float(_loss(
        h2_inputs, h2_inputs["params"], differentiation="implicit", **common,
    ))
    np.testing.assert_allclose(loss_unroll, loss_impl, atol=1e-8, rtol=1e-8)


def test_grad_matches_at_convergence(h2_inputs):
    """Implicit and unrolled param-gradients agree at a converged fixed point."""
    common = dict(max_cycle=40, energy_weight=0.0, density_weight=1.0)

    def loss_unroll(p):
        return _loss(
            h2_inputs, p, use_diis=False, ignore_ks_iter=0,
            differentiation="unroll", **common,
        )

    def loss_impl(p):
        return _loss(h2_inputs, p, differentiation="implicit", **common)

    g_unroll = jax.grad(loss_unroll)(h2_inputs["params"])
    g_impl = jax.grad(loss_impl)(h2_inputs["params"])

    leaves_u = jax.tree_util.tree_leaves(g_unroll)
    leaves_i = jax.tree_util.tree_leaves(g_impl)
    assert len(leaves_u) == len(leaves_i)
    for gu, gi in zip(leaves_u, leaves_i):
        assert jnp.all(jnp.isfinite(gu))
        assert jnp.all(jnp.isfinite(gi))
        # Loose tolerance: the unrolled solve is only converged to ~1e-7 in dm,
        # so its gradient carries a small finite-iteration bias relative to the
        # exact implicit gradient. They should still agree to a few digits.
        np.testing.assert_allclose(
            np.asarray(gu), np.asarray(gi), atol=1e-4, rtol=1e-3,
        )


def test_implicit_grad_is_finite_with_energy_term(h2_inputs):
    """End-to-end smoke check: the implicit loss (energy + density) differentiates."""
    def loss_of_params(p):
        return _loss(
            h2_inputs, p, max_cycle=20,
            energy_weight=1.0, density_weight=1.0, differentiation="implicit",
        )

    grads = jax.grad(loss_of_params)(h2_inputs["params"])
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(jnp.all(jnp.isfinite(g)) for g in leaves)


def test_custom_fixed_point_unit():
    """`custom_fixed_point` solves a toy linear fixed point and differentiates it.

    T(x, a) = 0.5 x + a  has the closed-form fixed point x* = 2a, so
    d x*/d a = 2 and d (sum x*)/d a = 2 * dim. Checks the forward solution and
    the implicit-function-theorem VJP against the analytic answer.
    """
    def step_fn(x, a):
        return 0.5 * x + a

    solver = custom_fixed_point(step_fn, max_cycle=200)

    a = jnp.array([1.0, -2.0, 0.5])
    x0 = jnp.zeros_like(a)

    x_star = solver(x0, a)
    np.testing.assert_allclose(np.asarray(x_star), 2.0 * np.asarray(a), atol=1e-9)

    g = jax.grad(lambda aa: jnp.sum(solver(x0, aa)))(a)
    np.testing.assert_allclose(np.asarray(g), np.full(a.shape, 2.0), atol=1e-6)


def test_runs_fast(h2_inputs):
    """Guardrail: a grad call (including compile) stays well under a minute."""
    def loss_of_params(p):
        return _loss(h2_inputs, p, max_cycle=20, differentiation="implicit")

    t0 = time.perf_counter()
    g = jax.grad(loss_of_params)(h2_inputs["params"])
    jax.block_until_ready(g)
    assert time.perf_counter() - t0 < 60.0


def test_implicit_grad_with_descriptor_features(h2_inputs):
    """A descriptor network carries a non-empty `features` bag (grid_coords,
    atom_coords). Those arrays must be threaded through the fixed-point `theta`,
    not closed over — otherwise they leak into the jitted implicit-backward as a
    captured constant and JAX rejects them ("not a valid JAX type"). This is the
    case the H2 CLI run (model.type=descriptor) hit; the MLP tests above use an
    empty bag and would not catch it.
    """
    from qex.functionals.descriptor import DescriptorXC

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = dft.RKS(mol)
    mf.xc = "lda"
    mf.grids.level = 0
    mf.grids.build()
    mf.kernel()

    grid_coords = jnp.asarray(mf.grids.coords)
    atom_coords = jnp.asarray(mol.atom_coords())
    n_grid = grid_coords.shape[0]

    net = DescriptorXC(hidden=(16,), alphas=(0.5, 1.0, 2.0, 4.0))
    params = net.init(
        jax.random.PRNGKey(0), jnp.ones(n_grid), jnp.asarray(mf.grids.weights),
        grid_coords=grid_coords, atom_coords=atom_coords,
    )
    xc_eval_fn = make_eval_xc_global(net, vxc_grad_scale=1.0)
    features = {"grid_coords": grid_coords, "atom_coords": atom_coords}

    ao = jnp.asarray(get_ao_value(mol, mf.grids.coords))
    dm = jnp.asarray(mf.make_rdm1())
    rho = jnp.einsum("gi,ij,gj->g", ao, dm, ao)

    def loss_of_params(p):
        return rks_loss_scan(
            p, dm, jnp.asarray(mol.intor("int2e", aosym="s1")), ao,
            jnp.asarray(mf.grids.weights), jnp.asarray(mf.get_ovlp(mol)),
            jnp.asarray(mf.get_hcore(mol)), float(mol.energy_nuc()),
            int(mol.nelectron), float(mf.e_tot), rho, dm, features,
            xc_eval_fn=xc_eval_fn, encoding="global",
            max_cycle=20, energy_weight=1.0, density_weight=1.0,
            differentiation="implicit", **_FRAC,
        )

    grads = jax.grad(loss_of_params)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves and all(jnp.all(jnp.isfinite(g)) for g in leaves)
