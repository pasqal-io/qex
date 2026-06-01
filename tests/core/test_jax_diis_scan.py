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
