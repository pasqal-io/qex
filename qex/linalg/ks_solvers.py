"""Alternative Kohn-Sham "solve" steps that avoid full dense diagonalization.

The dense path (`generalized_eigh`) solves the full generalized eigenproblem
`F C = S C ε` and is O(N^3) with no way to chunk it. For larger single systems
that wall dominates. This module provides two drop-in alternatives, both on
GPU-friendly JAX primitives:

- `lobpcg_solve`:  iterative eigensolver for only the lowest `n_occ` (+buffer)
  states. Fits the existing ``(mo_energy, mo_coeff)`` contract exactly, so the
  occupation/`make_rdm1` machinery downstream is unchanged. Cost is dominated
  by the block matvec `F @ X`, which is map-reducible / GPU-ideal.

- `mcweeny_purify`:  density-matrix purification. Iterates the (orthonormal-
  basis) density matrix to the idempotent ground state with `D <- 3D^2 - 2D^3`.
  Only GEMMs — no eigendecomposition — so it is cleanly differentiable (no
  `1/(eps_i - eps_j)` term blowing up on near-degeneracies) and GPU-ideal. It
  produces the 1-RDM *directly*, so it bypasses the `_occ_step` / `make_rdm1`
  path; the SCF loop handles that with a small branch.

Both work in the Loewdin/Cholesky-orthonormalized basis built the same way as
`generalized_eigh`: `B = L L^T`, `C_tilde = L^-1 A L^-T`. This keeps numerical
behavior consistent with the dense path.

Integer aufbau occupation only (closed shell): the two solvers here target the
gapped, closed-shell small systems used for validation against PySCF LDA/B3LYP.
Fractional occupation would need a Fermi-operator / grand-canonical variant and
is intentionally out of scope here.
"""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp


def _orthonormalize(A: jax.Array, B: jax.Array, eps: float, dtype):
    """Reduce `A v = lambda B v` to a standard problem `C u = lambda u`.

    Mirrors `generalized_eigh`: symmetrize, SPD-shift B, Cholesky `B = L L^T`,
    then `C = L^-1 A L^-T` via triangular solves. Returns `(C, L)` so callers
    can back-transform vectors with `v = L^-T u`.
    """
    A = (A + A.T.conj()) * 0.5
    B = (B + B.T.conj()) * 0.5
    A = A.astype(dtype)
    B = B.astype(dtype)

    lam_min = jnp.min(jnp.linalg.eigvalsh(B))
    shift = jnp.where(lam_min < eps, eps - lam_min, 0.0)
    B = B + shift * jnp.eye(B.shape[-1], dtype=dtype)

    L = jnp.linalg.cholesky(B)
    Y = jsp.solve_triangular(L, A, lower=True, trans="N")
    C = jsp.solve_triangular(L, Y.T, lower=True, trans="N").T
    C = (C + C.T.conj()) * 0.5
    return C, L


@partial(
    jax.jit,
    static_argnames=("n_occ", "n_extra", "max_iter", "tol", "eps", "dtype"),
)
def lobpcg_solve(
    fock: jax.Array,
    s1e: jax.Array,
    *,
    n_occ: int,
    n_extra: int = 8,
    max_iter: int = 100,
    tol: float = 1e-8,
    eps: float = 1.0e-12,
    dtype=jnp.float64,
):
    """Lowest-`n_occ` generalized eigenpairs via LOBPCG.

    Solves only for the `k = n_occ + n_extra` lowest states (you never need the
    virtual space to build the density), then pads back to full length so the
    return matches `generalized_eigh`'s `(w, V)` shapes and the downstream
    occupation code is unchanged. The `n_extra` buffer states improve LOBPCG
    convergence near the HOMO-LUMO gap; they get zero occupation.

    Returns
    -------
    w : [n]     eigenvalues ascending; the trailing `n - k` are padded large
                so aufbau never occupies them.
    V : [n, n]  eigenvectors in columns; only the first `k` are meaningful, the
                rest are zero. With integer aufbau (occupies lowest `n_occ`),
                only the first `n_occ` are ever used.
    """
    from jax.experimental.sparse.linalg import lobpcg_standard

    n = fock.shape[-1]
    C, L = _orthonormalize(fock, s1e, eps, dtype)

    # JAX's lobpcg_standard requires the search block k to satisfy 5*k < n.
    # We need at least n_occ states; clamp the buffer so the constraint holds,
    # and fail loudly (at trace time) if the matrix is simply too small for the
    # occupied space — LOBPCG is meaningless when 5*n_occ >= n (use "dense").
    k_max = (n - 1) // 5
    if k_max < n_occ:
        raise ValueError(
            f"LOBPCG needs 5*k < n with k >= n_occ; got n={n}, n_occ={n_occ} "
            f"(max usable k={k_max}). System too small for LOBPCG — use "
            f"solver='dense' or 'purify'."
        )
    k = min(n_occ + n_extra, k_max)

    # Deterministic random initial block. A plain identity/canonical block can
    # be (near-)invariant under C and makes LOBPCG break down (NaNs); a random
    # orthonormal block is the standard robust start. Fixed PRNGKey keeps it
    # repeatable and jit-safe (no Math.random / host RNG).
    X0 = jax.random.normal(jax.random.PRNGKey(0), (n, k), dtype=dtype)

    # jax's lobpcg_standard finds the LARGEST eigenpairs; we want the lowest
    # occupied states, so solve on -C and negate the eigenvalues back.
    neg_w_k, U_k, _ = lobpcg_standard(-C, X0, m=max_iter, tol=tol)
    w_k = -neg_w_k

    # Back-transform: v = L^-T u
    V_k = jsp.solve_triangular(L.T, U_k, lower=False, trans="N")

    # Pad the unconverged high-lying eigenvalues to full width so the return
    # shape matches the dense solver's contract.
    #
    # The pad value is delicate when fractional (Fermi-Dirac) occupation is
    # used: the chemical-potential solve in qex.scf.fermi differentiates
    # through fermi_function = 1/(1+exp((e-mu)/theta)). If a padded energy sits
    # too far above mu, exp((e-mu)/theta) overflows and its GRADIENT becomes
    # inf*0 = NaN (autodiff through exp(huge)), poisoning the whole solve. So we
    # pad only ~`pad_gap` Ha above the highest computed state: large enough that
    # occupation there is ~exp(-pad_gap/theta) ≈ 0 (theta ~ 0.04 -> exp(-25)),
    # but small enough that exp stays finite and differentiable. The entropy
    # 0*log0 limit is handled separately (xlogy in qex.scf.fermi.entropy).
    #
    # NOTE: meaningful Fermi smearing still needs computed states ABOVE the HOMO
    # (a real LUMO), i.e. k > n_occ. On bases too small for that buffer (5*k < n
    # leaves k == n_occ) there is no LUMO to smear against; use solver='dense'.
    pad_gap = 1.0
    big = jnp.max(w_k) + pad_gap
    w = jnp.full((n,), big, dtype=dtype).at[:k].set(w_k)
    V = jnp.zeros((n, n), dtype=dtype).at[:, :k].set(V_k)
    return w, V


@partial(jax.jit, static_argnames=("max_iter", "eps", "dtype"))
def mcweeny_purify(
    fock: jax.Array,
    s1e: jax.Array,
    *,
    n_occ,
    max_iter: int = 100,
    eps: float = 1.0e-12,
    dtype=jnp.float64,
):
    """Closed-shell density matrix via McWeeny purification (no eigensolve).

    Builds the idempotent ground-state density in the orthonormal basis by
    iterating `P <- 3 P^2 - 2 P^3`, which drives eigenvalues of `P` toward 0/1
    while preserving the trace (electron count) near the fixed point. The
    initial `P` is a shifted/scaled Fock that puts the `n_occ` lowest states in
    [0.5, 1] and the rest in [0, 0.5], so purification sorts them to 1/0.

    Returns the *closed-shell* 1-RDM in the original AO basis
    (`dm = 2 * L^-T P L^-1`), i.e. directly usable as `dm` — there is no
    `(mo_energy, mo_coeff)` here.

    Notes
    -----
    Only matrix multiplies and triangular solves — GPU-ideal and cleanly
    differentiable (no eigenvector-gradient `1/(eps_i-eps_j)` singularity).
    Fixed `max_iter` (no data-dependent early stop) keeps it scan/jit-clean;
    purification converges quadratically, so ~30-50 iters is plenty for the
    gapped small systems validated here.
    """
    n = fock.shape[-1]
    C, L = _orthonormalize(fock, s1e, eps, dtype)
    eye = jnp.eye(n, dtype=dtype)

    # Initial guess: a linear map of C into [0,1] with eigenvalues reversed
    # (lowest-energy state -> 1, highest -> 0), shifted by a chemical potential
    # mu placed in the HOMO-LUMO gap so that exactly n_occ eigenvalues land
    # above 0.5. McWeeny's 3P^2-2P^3 then drives each toward its nearest of
    # {0,1}, yielding the idempotent rank-n_occ projector.
    #
    # mu = midpoint of the HOMO/LUMO eigenvalues. eigvalsh gives only the
    # spectrum (no eigenvectors): cheap, and avoids the degenerate-vector
    # gradient pathology that motivates purification in the first place.
    evals = jnp.linalg.eigvalsh(C)  # ascending
    lam_min = evals[0]
    lam_max = evals[-1]
    # HOMO/LUMO midpoint. n_occ may be a traced scalar (the SCF loop passes
    # nelectron//2 as a JAX value), so index with dynamic gathers rather than
    # static slicing — only the scalar value, never an array shape, depends on
    # n_occ here, so purification stays fully jit-able with a traced n_occ.
    n_occ = jnp.asarray(n_occ)
    homo = jnp.take(evals, n_occ - 1)
    lumo = jnp.take(evals, n_occ)
    mu = 0.5 * (homo + lumo)

    # Scale so the full spectrum stays within [0,1] after the mu shift:
    # P0 = 0.5*I - (C - mu*I)/scale, scale = 2*max(mu-lam_min, lam_max-mu).
    scale = 2.0 * jnp.maximum(mu - lam_min, lam_max - mu)
    P = 0.5 * eye - (C - mu * eye) / scale

    def body(P, _):
        P2 = P @ P
        P3 = P2 @ P
        return 3.0 * P2 - 2.0 * P3, None

    P, _ = jax.lax.scan(body, P, xs=None, length=max_iter)

    # Back-transform to AO basis and scale for closed-shell double occupancy.
    # P is the projector in the orthonormal basis; dm_AO = L^-T P L^-1
    # (consistent with generalized_eigh's vector back-transform V = L^-T U).
    #   P L^-1 : solve X L = P  ->  X^T = solve_triangular(L^T, P^T, upper)
    #   L^-T M : solve L^T X = M -> X   = solve_triangular(L^T, M,   upper)
    P_Linv = jsp.solve_triangular(L.T, P.T, lower=False, trans="N").T
    dm_orth_to_ao = jsp.solve_triangular(L.T, P_Linv, lower=False, trans="N")
    return 2.0 * dm_orth_to_ao
