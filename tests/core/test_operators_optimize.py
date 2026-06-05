"""Equivalence of the `optimize=True` contractions in `qex.scf.operators`.

`get_veff`/`energy_tot` pass `optimize=True` to every `jnp.einsum`, letting
opt_einsum/XLA choose the contraction order. That is a performance-only change:
the result must be numerically identical to the naive left-to-right order. These
tests pin that — both the operator outputs and the gradient through them — so a
future contraction-order tweak can never silently shift the physics.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qex.functionals.mlp import GlobalMLP
from qex.functionals.xc import make_eval_xc_global
from qex.scf.operators import energy_tot, get_veff

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def veff_inputs():
    rng = np.random.RandomState(0)
    nao, n_grid = 6, 200
    dm = rng.randn(nao, nao)
    dm = jnp.asarray(dm + dm.T)
    eri = jnp.asarray(rng.randn(nao, nao, nao, nao))
    ao = jnp.asarray(rng.randn(n_grid, nao))
    w = jnp.asarray(np.abs(rng.randn(n_grid)))
    net = GlobalMLP(features=[16], act_fn=nn.gelu)
    params = net.init(jax.random.PRNGKey(0), jnp.ones(n_grid))
    xc = make_eval_xc_global(net, vxc_grad_scale=1.0)
    return dict(dm=dm, eri=eri, ao=ao, w=w, params=params, xc=xc, nao=nao)


def _naive_veff_global(dm, eri, ao, w, params, xc):
    """The byte-for-byte naive (optimize=False) reference contraction order."""
    J = jnp.einsum("ijkl,kl->ij", eri, dm, optimize=False)
    rho = jnp.einsum("gi,ij,gj->g", ao, dm, ao, optimize=False)
    exc, (vrho, _, _, _), _, _ = xc("", rho, params=params, grid_weights=w, features={})
    Vxc = jnp.einsum("gi,g,gj->ij", ao, w * vrho, ao, optimize=False)
    return J + Vxc, exc, J


def test_get_veff_matches_naive_order(veff_inputs):
    d = veff_inputs
    vhf, exc, J = get_veff(
        d["dm"], d["eri"], d["ao"], d["w"], d["params"], d["xc"],
        encoding="global", features={},
    )
    vhf0, exc0, J0 = _naive_veff_global(
        d["dm"], d["eri"], d["ao"], d["w"], d["params"], d["xc"],
    )
    # Same contraction, different evaluation order -> identical to ~fp64 eps.
    np.testing.assert_allclose(np.asarray(J), np.asarray(J0), rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.asarray(vhf), np.asarray(vhf0), rtol=0, atol=1e-13)
    np.testing.assert_allclose(float(exc), float(exc0), rtol=0, atol=1e-13)


def test_energy_tot_matches_naive_order(veff_inputs):
    d = veff_inputs
    h1e = jnp.asarray(np.eye(d["nao"]))
    _, _, J = get_veff(
        d["dm"], d["eri"], d["ao"], d["w"], d["params"], d["xc"],
        encoding="global", features={},
    )
    e = energy_tot(d["dm"], h1e, J, jnp.asarray(1.5), 2.0)
    e_one = jnp.einsum("ij,ji->", d["dm"], h1e, optimize=False)
    e_h = 0.5 * jnp.einsum("ij,ij->", d["dm"], J, optimize=False)
    e0 = e_one + e_h + 1.5 + 2.0
    np.testing.assert_allclose(float(e), float(e0), rtol=0, atol=1e-12)


def test_grad_matches_naive_order(veff_inputs):
    """The gradient w.r.t. params must also be order-independent."""
    d = veff_inputs

    def loss(p, naive):
        fn = _naive_veff_global if naive else (
            lambda dm, eri, ao, w, params, xc: get_veff(
                dm, eri, ao, w, params, xc, encoding="global", features={})
        )
        vhf, exc, _ = fn(d["dm"], d["eri"], d["ao"], d["w"], p, d["xc"])
        return jnp.sum(vhf**2) + exc**2

    g_opt = jax.grad(lambda p: loss(p, naive=False))(d["params"])
    g_naive = jax.grad(lambda p: loss(p, naive=True))(d["params"])
    for a, b in zip(jax.tree_util.tree_leaves(g_opt),
                    jax.tree_util.tree_leaves(g_naive)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=1e-12)
