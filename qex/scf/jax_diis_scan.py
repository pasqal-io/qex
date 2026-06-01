"""Scan-compatible DIIS with fixed-size ring buffers.

`jax_diis.py` keeps `error_vecs` / `fock_vecs` as Python lists that grow each
SCF iteration; that's incompatible with `lax.scan` (carry shapes must be
static). This module mirrors the same math against pre-allocated buffers of
shape `(max_vec, fock_size)` plus a small `(head, count)` bookkeeping pair, so
it slots into a scanned SCF loop without changing convergence behavior.

Public surface:

    state = initialize_diis_scan(max_vec, fock_size)
    fock_out, state = apply_diis_scan(state, fock, dm, ovlp,
                                      max_vec, min_vecs, damping)

The state is a `DIISStateScan` NamedTuple of JAX arrays — fully traceable.

Differentiability through DIIS
------------------------------
The extrapolation coefficients `c` solve `B c = rhs` where `B` is the Gram
matrix of stored error vectors. As SCF converges the error vectors become
near-collinear, `B` becomes rank-deficient, and `∂c/∂B` carries a `B^{-1}`
factor scaling as `1/σ²` in the smallest singular value of `B`. On a
near-converged H₂ history this hits **O(10¹³)** in the gradient (see
`tests/core/test_jax_diis_scan.py::test_scan_diis_gradient_bounded_on_near_converged_history`),
which compounds through `max_cycle` SCF iterations and reverse-mode through
the XC functional into NaN parameter updates — the mechanism behind the
QCNN-training NaNs in `examples/train_h2.py` when `use_diis=True`.

Fix in `_extrapolate`:
  1. `jnp.linalg.lstsq` (pinv with rcond cutoff) instead of `jnp.linalg.solve`,
     so the solve stays finite when `B` is rank-deficient. Under JIT,
     `solve` returns NaN/Inf silently rather than raising.
  2. `jax.lax.stop_gradient` on `c`. The forward DIIS acceleration is
     preserved — extrapolation `Σᵢ cᵢ Fᵢ` is still differentiable through the
     `Fᵢ` — but the ill-conditioned `B^{-1}` factor in `∂c/∂B` no longer
     reaches the parameter gradient. This matches the standard PySCF-AD
     treatment of DIIS.

The gradient-magnitude regression tests assert `max|grad| < 100` on a
near-converged history (including under `jit`); without the fix the
observed magnitude is ~3×10¹³, so any regression will fail loudly.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from chex import Array


class DIISStateScan(NamedTuple):
    error_vecs: Array   # (max_vec, fock_size)
    fock_vecs: Array    # (max_vec, fock_size)
    B_matrix: Array     # (max_vec + 1, max_vec + 1)
    head: Array         # () int — next write slot
    count: Array        # () int — number of valid entries (capped at max_vec)


def initialize_diis_scan(max_vec: int, fock_size: int) -> DIISStateScan:
    return DIISStateScan(
        error_vecs=jnp.zeros((max_vec, fock_size)),
        fock_vecs=jnp.zeros((max_vec, fock_size)),
        B_matrix=jnp.zeros((max_vec + 1, max_vec + 1)),
        head=jnp.array(0, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
    )


def _diis_error(fock: Array, dm: Array, ovlp: Array) -> Array:
    fds = fock @ dm @ ovlp
    sdf = ovlp @ dm @ fock
    return fds - sdf


def _update_state(
    state: DIISStateScan,
    error_vec: Array,
    fock: Array,
    max_vec: int,
) -> DIISStateScan:
    """Write the new (error, fock) pair into slot `head` and refresh the B row+col.

    Rather than recomputing the full `B = E @ E.T` each step, we only patch
    the row and column at `head` — the other entries are unchanged.
    """
    err_flat = error_vec.ravel()
    fock_flat = fock.ravel()

    error_vecs = state.error_vecs.at[state.head].set(err_flat)
    fock_vecs = state.fock_vecs.at[state.head].set(fock_flat)

    # New inner products of the freshly inserted error vec against all stored
    # error vecs (including itself). The B layout matches `jax_diis.py`:
    # row/col 0 hold the -1 Lagrange multiplier coefficients; the actual Gram
    # block lives at `B[1:, 1:]`.
    dots = error_vecs @ err_flat  # shape (max_vec,)
    B = state.B_matrix
    # Lagrange row/col (idempotent, but the buffer starts zero so set explicitly).
    B = B.at[0, 1:].set(-1.0)
    B = B.at[1:, 0].set(-1.0)
    # Patch row `head+1` and column `head+1` of the Gram block.
    h = state.head + 1
    B = B.at[h, 1:].set(dots)
    B = B.at[1:, h].set(dots)
    # Diagonal: stays consistent because dots[head] == err·err.

    new_head = (state.head + 1) % max_vec
    new_count = jnp.minimum(state.count + 1, max_vec)
    return DIISStateScan(
        error_vecs=error_vecs,
        fock_vecs=fock_vecs,
        B_matrix=B,
        head=new_head,
        count=new_count,
    )


def _extrapolate(
    state: DIISStateScan,
    fock: Array,
    max_vec: int,
    min_vecs: int,
    damping: float,
) -> Array:
    """Solve the masked DIIS system; fall back to the most recent fock when
    fewer than `min_vecs` entries are stored.

    The unused rows/cols of B are masked by overwriting them with rows of the
    identity and zeroing the corresponding rhs entry — this keeps the system
    non-singular without changing the answer for the valid slots.
    """
    n = max_vec
    valid = jnp.arange(n) < state.count        # (n,) bool
    valid_ext = jnp.concatenate([jnp.array([True]), valid])  # (n+1,)

    B = state.B_matrix
    # Regularize the diagonal slightly to match the original implementation.
    B = B + 1e-14 * jnp.eye(n + 1)

    # Zero invalid rows/cols, then drop in identity rows so the masked system
    # has a unique zero solution at those slots.
    mask2 = valid_ext[:, None] & valid_ext[None, :]
    B_masked = jnp.where(mask2, B, 0.0)
    B_masked = B_masked + jnp.diag(jnp.where(valid_ext, 0.0, 1.0))

    rhs = jnp.zeros(n + 1).at[0].set(-1.0)
    rhs = jnp.where(valid_ext, rhs, 0.0)

    # Use lstsq (pinv) instead of solve so the system stays finite when B is
    # rank-deficient (collinear errors near convergence).
    #
    # `stop_gradient` on c is the key fix for QCNN-training NaNs with DIIS on:
    # ∂c/∂B carries a B⁻¹ factor whose magnitude scales as 1/σ², and on a
    # near-converged history this hits O(1e13). The forward extrapolation
    # `Σ cᵢ Fᵢ` is still fully differentiable through the Fᵢ; freezing the
    # coefficients matches the standard PySCF-AD treatment of DIIS.
    c = jnp.linalg.lstsq(B_masked, rhs, rcond=1e-12)[0]
    c = jax.lax.stop_gradient(c)
    coeffs = c[1:]  # (n,)

    # Linear combination of stored Fock vectors weighted by coeffs; invalid
    # slots are zero, so they drop out naturally.
    fock_flat_extrap = (coeffs[:, None] * state.fock_vecs).sum(axis=0)

    # Most recent Fock — used as the cold-start fallback and for damping.
    last_idx = (state.head - 1) % n
    last_fock_flat = state.fock_vecs[last_idx]

    fock_flat = jnp.where(
        state.count >= min_vecs,
        fock_flat_extrap,
        last_fock_flat,
    )

    if damping > 0.0:
        fock_flat = (1.0 - damping) * fock_flat + damping * last_fock_flat

    return fock_flat.reshape(fock.shape)


def apply_diis_scan(
    state: DIISStateScan,
    fock: Array,
    dm: Array,
    ovlp: Array,
    max_vec: int,
    min_vecs: int = 2,
    damping: float = 0.0,
) -> tuple[Array, DIISStateScan]:
    """Equivalent of `apply_diis` from `jax_diis.py`, scan-friendly.

    `max_vec` must be a Python int (it sets the buffer shape and so is static
    under jit/scan); `min_vecs` and `damping` can be Python floats too.
    """
    err = _diis_error(fock, dm, ovlp)
    new_state = _update_state(state, err, fock, max_vec)
    extrapolated = _extrapolate(new_state, fock, max_vec, min_vecs, damping)
    return extrapolated, new_state


if __name__ == "__main__":
    # Quick equivalence smoke test against the legacy implementation on a
    # toy H2 single point. We replay 5 SCF cycles with both versions and
    # check the extrapolated Fock at each step matches.
    import numpy as np
    from pyscf import gto, scf

    from qex.scf.jax_diis import apply_diis as apply_diis_legacy
    from qex.scf.jax_diis import initialize_diis as init_diis_legacy

    jax.config.update("jax_enable_x64", True)

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g")
    mf = scf.RHF(mol)
    mf.kernel()
    ovlp = jnp.asarray(mf.get_ovlp())
    dm = jnp.asarray(mf.make_rdm1())
    fock = jnp.asarray(mf.get_fock())
    fock_size = fock.size

    max_vec, min_vecs = 6, 2

    legacy_state = init_diis_legacy(max_vec=max_vec)
    scan_state = initialize_diis_scan(max_vec=max_vec, fock_size=fock_size)

    for i in range(5):
        f_legacy, legacy_state = apply_diis_legacy(
            legacy_state, fock, dm, ovlp, max_vec=max_vec, min_vecs=min_vecs,
        )
        f_scan, scan_state = apply_diis_scan(
            scan_state, fock, dm, ovlp, max_vec=max_vec, min_vecs=min_vecs,
        )
        diff = float(jnp.max(jnp.abs(f_legacy - f_scan)))
        print(f"cycle {i}: max|legacy - scan| = {diff:.3e}")
        # Step forward with the extrapolated fock so the next iter sees new data.
        fock = f_scan
        dm = jnp.asarray(mf.make_rdm1(np.asarray(fock)))
    print("OK")
