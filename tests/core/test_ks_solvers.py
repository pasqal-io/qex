"""Fast unit tests for the alternative KS solvers (LOBPCG, McWeeny purify).

These check the *solve* step in isolation: build one converged Fock/overlap
pair from a quick PySCF run, then assert each solver reproduces the dense
`generalized_eigh` result on that single matrix. No SCF loop, no `cc-pVTZ`, no
full ERI build — so this runs in well under a second per case and belongs in
the regular suite.

The end-to-end SCF-loop validation against PySCF energies (LDA / B3LYP across
several systems) is intentionally NOT here — it costs minutes (full O(N^4) ERI
tensor + JIT compile per case). It lives in `compare_ks_solvers.py`, a
standalone script run on demand, matching the repo's `compare_*.py` convention.

What these catch: the two real bugs found during development — LOBPCG returning
the largest eigenvalues instead of the lowest, and purification's Cholesky
back-transform using L^-T where L^-1 was needed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import gto, scf

from qex.linalg.generalized_eigensolver import generalized_eigh
from qex.linalg.ks_solvers import lobpcg_solve, mcweeny_purify
from qex.scf.operators import get_occ, make_rdm1_custom

jax.config.update("jax_enable_x64", True)


def _fock_overlap(atom: str, basis: str):
    """One converged Fock + overlap from a cheap PySCF RHF run."""
    mol = gto.M(atom=atom, basis=basis)
    mf = scf.RHF(mol)
    mf.kernel()
    F = jnp.asarray(mf.get_fock())
    S = jnp.asarray(mf.get_ovlp())
    return F, S, int(mol.nelectron)


def _dense_dm(F, S, nelectron):
    """Reference closed-shell 1-RDM via the dense generalized eigensolver."""
    w, V = generalized_eigh(F, S)
    return make_rdm1_custom(V, get_occ(nelectron, w)), w


# Purification works at any size — include tiny matrices where LOBPCG cannot
# run, to prove it has no size floor.
_PURIFY_SYSTEMS = [
    ("H2_sto3g", "H 0 0 0; H 0 0 0.74", "sto-3g"),
    ("H2O_631g", "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "631g"),
]

# LOBPCG needs 5*n_occ < nao, so pick small-n_occ systems with enough basis:
# LiH/631g (nao=11, n_occ=2) and H2/cc-pVDZ (nao=10, n_occ=1). Still tiny/fast.
_LOBPCG_SYSTEMS = [
    ("LiH_631g", "Li 0 0 0; H 0 0 1.60", "631g"),
    ("H2_ccpvdz", "H 0 0 0; H 0 0 0.74", "cc-pVDZ"),
]


@pytest.mark.parametrize(
    "name,atom,basis", _PURIFY_SYSTEMS, ids=[s[0] for s in _PURIFY_SYSTEMS]
)
def test_purify_matches_dense(name, atom, basis):
    """McWeeny purification reproduces the dense 1-RDM (works at any size)."""
    F, S, nelectron = _fock_overlap(atom, basis)
    dm_ref, _ = _dense_dm(F, S, nelectron)
    dm = mcweeny_purify(F, S, n_occ=nelectron // 2)

    np.testing.assert_allclose(
        np.asarray(dm), np.asarray(dm_ref), atol=1e-9, rtol=0.0,
        err_msg=f"{name}: purify dm != dense dm",
    )
    # Idempotency / electron count: trace(S @ dm) == nelectron.
    trace_sdm = float(jnp.einsum("ij,ji->", S, dm))
    assert trace_sdm == pytest.approx(nelectron, abs=1e-9)


@pytest.mark.parametrize(
    "name,atom,basis", _LOBPCG_SYSTEMS, ids=[s[0] for s in _LOBPCG_SYSTEMS]
)
def test_lobpcg_matches_dense(name, atom, basis):
    """LOBPCG yields the lowest eigenpairs / same 1-RDM as the dense solver."""
    F, S, nelectron = _fock_overlap(atom, basis)
    dm_ref, w_ref = _dense_dm(F, S, nelectron)

    n_occ = nelectron // 2
    w, V = lobpcg_solve(F, S, n_occ=n_occ)
    dm = make_rdm1_custom(V, get_occ(nelectron, w))

    # LOBPCG is iterative, so it matches the direct dense solver only to its
    # convergence tolerance — not machine precision. Eigenvalues converge
    # tighter than the density matrix; assert each at a realistic level.
    np.testing.assert_allclose(
        np.asarray(w[:n_occ]), np.asarray(w_ref[:n_occ]), atol=1e-7, rtol=0.0,
        err_msg=f"{name}: lobpcg lowest eigenvalues != dense",
    )
    np.testing.assert_allclose(
        np.asarray(dm), np.asarray(dm_ref), atol=1e-5, rtol=0.0,
        err_msg=f"{name}: lobpcg dm != dense dm",
    )


def test_lobpcg_rejects_too_small_system():
    """LOBPCG raises a clear error when 5*n_occ >= n (e.g. STO-3G H2)."""
    F, S, nelectron = _fock_overlap("H 0 0 0; H 0 0 0.74", "sto-3g")
    with pytest.raises(ValueError, match="too small for LOBPCG"):
        lobpcg_solve(F, S, n_occ=nelectron // 2)


def test_purify_accepts_traced_n_occ():
    """purify works with a traced n_occ (the SCF loop passes nelectron//2 as a
    JAX value). Regression test for the 'unhashable DynamicJaxprTracer' crash
    from when n_occ was a static argname."""
    F, S, nelectron = _fock_overlap(
        "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "631g"
    )

    @jax.jit
    def run(F, S, ne):  # ne is traced under jit -> n_occ is a tracer
        return mcweeny_purify(F, S, n_occ=ne // 2)

    dm = run(F, S, nelectron)
    assert not bool(jnp.any(jnp.isnan(dm)))
    assert float(jnp.einsum("ij,ji->", S, dm)) == pytest.approx(
        nelectron, abs=1e-7
    )


def test_entropy_safe_at_occupation_limits():
    """Shared fermi.entropy must give 0 (not NaN) at f=0 and f=1, while staying
    bit-identical for real occupations 0<f<1 (the only values the original
    dense+frac path ever produces). Guards the xlogy fix that lets LOBPCG's
    padded states feed the Fermi solve without poisoning it."""
    from qex.scf.fermi import entropy

    real = jnp.array([0.9, 0.5, 0.1, 1e-3])
    # Reference: the original plain-log formula on real occupations.
    ref = real * jnp.log(real) + (1 - real) * jnp.log(1 - real)
    np.testing.assert_allclose(np.asarray(entropy(real)), np.asarray(ref))
    # Limits: finite (0), not NaN.
    assert float(entropy(jnp.array(0.0))) == 0.0
    assert float(entropy(jnp.array(1.0))) == 0.0
