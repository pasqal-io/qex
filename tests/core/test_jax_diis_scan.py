"""Equivalence tests: legacy `jax_diis.apply_diis` vs scan-friendly
`jax_diis_scan.apply_diis_scan`.

Both implement the same math; the only difference is that the scan version
stores history in fixed-size buffers so it can run inside `lax.scan`. The
test drives a few SCF steps on a tiny H2/STO-3G problem and checks the
extrapolated Fock matches at machine precision every cycle.

Kept small (no full SCF convergence, just 4 DIIS applications) so it runs
in well under a second on CPU.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, scf

from qex.scf.jax_diis import apply_diis as apply_diis_legacy
from qex.scf.jax_diis import initialize_diis as init_diis_legacy
from qex.scf.jax_diis_scan import apply_diis_scan, initialize_diis_scan

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def h2_seed():
    """Reference H2/STO-3G state used as DIIS input."""
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mf = scf.RHF(mol)
    mf.kernel()
    return {
        "mf": mf,
        "ovlp": jnp.asarray(mf.get_ovlp()),
        "dm0": jnp.asarray(mf.make_rdm1()),
        "fock0": jnp.asarray(mf.get_fock()),
    }


@pytest.mark.parametrize("max_vec,min_vecs", [(6, 2), (10, 1)])
def test_scan_diis_matches_legacy(h2_seed, max_vec, min_vecs):
    """After each apply_diis call the two extrapolated Focks must match."""
    mf = h2_seed["mf"]
    ovlp = h2_seed["ovlp"]
    dm = h2_seed["dm0"]
    fock = h2_seed["fock0"]
    fock_size = fock.size

    legacy_state = init_diis_legacy(max_vec=max_vec)
    scan_state = initialize_diis_scan(max_vec=max_vec, fock_size=fock_size)

    for _ in range(4):
        f_legacy, legacy_state = apply_diis_legacy(
            legacy_state, fock, dm, ovlp,
            max_vec=max_vec, min_vecs=min_vecs,
        )
        f_scan, scan_state = apply_diis_scan(
            scan_state, fock, dm, ovlp,
            max_vec=max_vec, min_vecs=min_vecs,
        )
        np.testing.assert_allclose(
            np.asarray(f_legacy), np.asarray(f_scan), atol=1e-12, rtol=1e-12,
        )
        # Step forward with the new fock so the next call sees fresh data.
        fock = f_scan
        dm = jnp.asarray(mf.make_rdm1(np.asarray(fock)))


def test_scan_diis_runs_under_lax_scan(h2_seed):
    """The whole point: the scan version must trace inside `lax.scan`."""
    ovlp = h2_seed["ovlp"]
    dm = h2_seed["dm0"]
    fock0 = h2_seed["fock0"]
    max_vec = 6

    init_state = initialize_diis_scan(max_vec=max_vec, fock_size=fock0.size)

    def step(carry, _):
        fock, state = carry
        fock_new, state_new = apply_diis_scan(
            state, fock, dm, ovlp, max_vec=max_vec, min_vecs=2,
        )
        return (fock_new, state_new), fock_new

    (final_fock, _), focks = jax.jit(
        lambda f, s: jax.lax.scan(step, (f, s), None, length=3),
    )(fock0, init_state)

    assert focks.shape == (3, *fock0.shape)
    assert jnp.all(jnp.isfinite(final_fock))


# ---------------------------------------------------------------------------
# Failure-mode regression tests.
#
# Both DIIS variants build a Gram matrix B = E E^T of error vectors and solve
# a small linear system for the extrapolation coefficients. As error vectors
# become collinear (which happens precisely as SCF converges well) B becomes
# rank-deficient, and `jnp.linalg.solve` returns NaN/Inf silently — the
# `try/except` in the legacy version never fires under JIT. The tests below
# pin down two consequences:
#
#   1. Forward pass must stay finite on a near-singular history.
#   2. The gradient of the extrapolated Fock w.r.t. the input Fock must be
#      finite even when B is severely ill-conditioned. This is the exact
#      failure mode that produces NaN parameter updates when training the
#      QCNN through `rks_loss` with DIIS enabled.
# ---------------------------------------------------------------------------


def _near_converged_focks(fock_shape, n: int, scales=None):
    """Build a Fock sequence that mimics late-stage SCF convergence.

    Each Fock is a fixed `base` plus a tiny shrinking perturbation, so the
    DIIS error vectors form a near-degenerate set — exactly the regime
    where DIIS B-matrix conditioning collapses during training.
    """
    rng = np.random.default_rng(0)
    if scales is None:
        scales = [1e-6 * 0.5 ** i for i in range(n)]
    base = jnp.eye(fock_shape[0]) * jnp.linspace(1.0, 2.0, fock_shape[0])
    return [
        base + s * jnp.asarray(rng.standard_normal(fock_shape))
        for s in scales
    ]


def _run_diis_chain_legacy(fock_in, dm, ovlp, prior_focks, max_vec, min_vecs):
    state = init_diis_legacy(max_vec=max_vec)
    for prior in prior_focks:
        _, state = apply_diis_legacy(
            state, prior, dm, ovlp, max_vec=max_vec, min_vecs=min_vecs,
        )
    fock_out, _ = apply_diis_legacy(
        state, fock_in, dm, ovlp, max_vec=max_vec, min_vecs=min_vecs,
    )
    return jnp.sum(fock_out ** 2)


def _run_diis_chain_scan(fock_in, dm, ovlp, prior_focks, max_vec, min_vecs):
    fock_size = fock_in.size
    state = initialize_diis_scan(max_vec=max_vec, fock_size=fock_size)
    for prior in prior_focks:
        _, state = apply_diis_scan(
            state, prior, dm, ovlp, max_vec=max_vec, min_vecs=min_vecs,
        )
    fock_out, _ = apply_diis_scan(
        state, fock_in, dm, ovlp, max_vec=max_vec, min_vecs=min_vecs,
    )
    return jnp.sum(fock_out ** 2)


# Gradient-magnitude ceiling. Without `stop_gradient` on the DIIS coefficients,
# `∂F_extrap/∂F_in` carries `B^{-1}` factors that explode to O(1e5+) on a
# near-converged history — the exact mechanism that turns into NaN parameter
# updates once compounded through `max_cycle=15` SCF iterations and reverse-mode
# through a QCNN. The fix is to stop gradients through `c = solve(B, rhs)`.
# 100 is well above any healthy gradient (typical scale is O(1)) and well
# below the O(1e5) blowup, so it cleanly separates fixed vs. unfixed.
_GRAD_MAGNITUDE_CEILING = 1e2


def test_legacy_diis_forward_finite_on_near_converged_history(h2_seed):
    ovlp = h2_seed["ovlp"]
    dm = h2_seed["dm0"]
    fock_shape = h2_seed["fock0"].shape

    max_vec = 6
    focks = _near_converged_focks(fock_shape, n=max_vec)
    state = init_diis_legacy(max_vec=max_vec)
    for fock in focks:
        fock_out, state = apply_diis_legacy(
            state, fock, dm, ovlp, max_vec=max_vec, min_vecs=2,
        )
    assert jnp.all(jnp.isfinite(fock_out))


def test_scan_diis_forward_finite_on_near_converged_history(h2_seed):
    ovlp = h2_seed["ovlp"]
    dm = h2_seed["dm0"]
    fock_shape = h2_seed["fock0"].shape
    fock_size = h2_seed["fock0"].size

    max_vec = 6
    focks = _near_converged_focks(fock_shape, n=max_vec)
    state = initialize_diis_scan(max_vec=max_vec, fock_size=fock_size)
    for fock in focks:
        fock_out, state = apply_diis_scan(
            state, fock, dm, ovlp, max_vec=max_vec, min_vecs=2,
        )
    assert jnp.all(jnp.isfinite(fock_out))


def test_legacy_diis_gradient_bounded_on_near_converged_history(h2_seed):
    """Reproduces the QCNN-training NaN mechanism.

    On a near-converged Fock history the DIIS B-matrix is ill-conditioned, so
    `c = B^{-1} rhs` and its derivative w.r.t. B blow up as 1/σ². Without
    `stop_gradient` on `c`, that blowup propagates into `∂F_extrap/∂F_in` and
    becomes O(1e5+) — which then compounds through `max_cycle` SCF cycles and
    a QCNN backward pass into the NaN parameter updates we observe in
    `train_h2.py` with DIIS on.
    """
    ovlp = h2_seed["ovlp"]
    dm = h2_seed["dm0"]
    fock_shape = h2_seed["fock0"].shape

    max_vec = 6
    prior = _near_converged_focks(fock_shape, n=max_vec)
    fock_in = prior[-1] + 1e-3 * jnp.ones(fock_shape)

    grad = jax.grad(_run_diis_chain_legacy)(
        fock_in, dm, ovlp, prior, max_vec, 2,
    )
    grad_max = float(jnp.max(jnp.abs(grad)))
    assert jnp.all(jnp.isfinite(grad))
    assert grad_max < _GRAD_MAGNITUDE_CEILING, (
        f"legacy DIIS gradient exploded to {grad_max:.2e} on near-converged "
        "history (expected stop_gradient on c to keep it bounded)"
    )


def test_scan_diis_gradient_bounded_on_near_converged_history(h2_seed):
    ovlp = h2_seed["ovlp"]
    dm = h2_seed["dm0"]
    fock_shape = h2_seed["fock0"].shape

    max_vec = 6
    prior = _near_converged_focks(fock_shape, n=max_vec)
    fock_in = prior[-1] + 1e-3 * jnp.ones(fock_shape)

    grad = jax.grad(_run_diis_chain_scan)(
        fock_in, dm, ovlp, prior, max_vec, 2,
    )
    grad_max = float(jnp.max(jnp.abs(grad)))
    assert jnp.all(jnp.isfinite(grad))
    assert grad_max < _GRAD_MAGNITUDE_CEILING, (
        f"scan DIIS gradient exploded to {grad_max:.2e} on near-converged history"
    )


def test_scan_diis_gradient_bounded_under_jit(h2_seed):
    """Same as the scan gradient test but exercises the jit path."""
    ovlp = h2_seed["ovlp"]
    dm = h2_seed["dm0"]
    fock_shape = h2_seed["fock0"].shape

    max_vec = 6
    prior = _near_converged_focks(fock_shape, n=max_vec)
    fock_in = prior[-1] + 1e-3 * jnp.ones(fock_shape)

    grad_fn = jax.jit(
        jax.grad(_run_diis_chain_scan),
        static_argnames=("max_vec", "min_vecs"),
    )
    grad = grad_fn(fock_in, dm, ovlp, prior, max_vec, 2)
    grad_max = float(jnp.max(jnp.abs(grad)))
    assert jnp.all(jnp.isfinite(grad))
    assert grad_max < _GRAD_MAGNITUDE_CEILING
