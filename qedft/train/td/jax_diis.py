"""
JAX-based implementation of DIIS.
"""

from typing import NamedTuple

import jax.numpy as jnp


class DIISState(NamedTuple):
    error_vecs: list[jnp.ndarray]
    fock_vecs: list[jnp.ndarray]
    B_matrix: jnp.ndarray
    iteration: int


def initialize_diis(max_vec: int = 6) -> DIISState:
    return DIISState(
        error_vecs=[],
        fock_vecs=[],
        B_matrix=jnp.zeros((max_vec + 1, max_vec + 1)),
        iteration=0,
    )


def update_diis_state(
    state: DIISState,
    error_vec: jnp.ndarray,
    fock: jnp.ndarray,
    max_vec: int = 6,
) -> DIISState:
    error_vec_flat = error_vec.ravel()
    fock_flat = fock.ravel()

    error_vecs = state.error_vecs.copy()
    fock_vecs = state.fock_vecs.copy()

    error_vecs.append(error_vec_flat)
    fock_vecs.append(fock_flat)

    if len(error_vecs) > max_vec:
        error_vecs = error_vecs[-max_vec:]
        fock_vecs = fock_vecs[-max_vec:]

    n_vecs = len(error_vecs)
    B = jnp.zeros((n_vecs + 1, n_vecs + 1))

    # The -1 is critical here, +1 breaks convergence of SCF.
    B = B.at[0, 1:].set(-1.0)
    B = B.at[1:, 0].set(-1.0)

    for i in range(n_vecs):
        for j in range(n_vecs):
            B = B.at[i + 1, j + 1].set(jnp.dot(error_vecs[i], error_vecs[j]))

    return DIISState(
        error_vecs=error_vecs,
        fock_vecs=fock_vecs,
        B_matrix=B,
        iteration=state.iteration + 1,
    )


def extrapolate_fock(
    state: DIISState,
    fock_shape: tuple[int, ...],
    min_vecs: int = 2,
    condition_threshold: float = 1e-12,
    damping: float = 0.0,
) -> jnp.ndarray:
    n_vecs = len(state.fock_vecs)

    if n_vecs < min_vecs:
        if n_vecs > 0:
            return state.fock_vecs[-1].reshape(fock_shape)
        return jnp.zeros(fock_shape)

    rhs = jnp.zeros(n_vecs + 1)
    rhs = rhs.at[0].set(-1.0)

    B = state.B_matrix[: n_vecs + 1, : n_vecs + 1]

    diag_idx = jnp.diag_indices(n_vecs + 1)
    B_reg = B.at[diag_idx].add(1e-14)

    try:
        c = jnp.linalg.solve(B_reg, rhs)
    except:
        U, s, Vh = jnp.linalg.svd(B_reg, full_matrices=False)
        s_inv = jnp.where(s > condition_threshold, 1.0 / s, 0.0)
        c = Vh.T @ (s_inv[:, None] * (U.T @ rhs[:, None]))
        c = c.flatten()

    coeffs = c[1:]

    fock_flat = jnp.zeros_like(state.fock_vecs[0])
    for i, coeff in enumerate(coeffs):
        fock_flat = fock_flat + coeff * state.fock_vecs[i]

    if damping > 0.0 and n_vecs > 0:
        last_fock = state.fock_vecs[-1]
        fock_flat = (1.0 - damping) * fock_flat + damping * last_fock

    return fock_flat.reshape(fock_shape)


def get_diis_error(
    fock: jnp.ndarray,
    dm: jnp.ndarray,
    ovlp: jnp.ndarray,
) -> jnp.ndarray:
    fds = jnp.matmul(fock, jnp.matmul(dm, ovlp))
    sdf = jnp.matmul(ovlp, jnp.matmul(dm, fock))
    return fds - sdf


def apply_diis(
    state: DIISState,
    fock: jnp.ndarray,
    dm: jnp.ndarray,
    ovlp: jnp.ndarray,
    max_vec: int = 6,
    min_vecs: int = 2,
    damping: float = 0.0,
) -> tuple[jnp.ndarray, DIISState]:
    error_vec = get_diis_error(fock, dm, ovlp)
    new_state = update_diis_state(state, error_vec, fock, max_vec=max_vec)
    extrapolated_fock = extrapolate_fock(
        new_state,
        fock.shape,
        min_vecs=min_vecs,
        damping=damping,
    )
    return extrapolated_fock, new_state


if __name__ == "__main__":
    from pyscf import gto, scf

    # Simple molecule hard to converge
    # mol = gto.M(atom="H 0 0 0; H 0 0 0.3", basis="6-31g")
    # Harder to converge
    # mol = gto.M(atom="H 0 0 0;  H 0 0 4.5", basis="6-31g")
    # Complex molecule
    # mol = gto.M(atom="H 0 0 0;  H 0 0 4.5; H 0 0 9.0; H 0 0 13.5", basis="6-31g")
    # N2 molecule, hard to converge
    mol = gto.M(atom="N 0 0 0; N 0 0 4.1", basis="sto-3g")

    mf = scf.RHF(mol)
    mf.kernel()

    # Our JAX DIIS
    # JAX DIIS is better with more vectors
    max_vec = 20
    min_vecs = 20
    damping = 0.0
    # With max = 6 and min = 1 can get NaNs

    print("Our JAX DIIS")
    diis_state = initialize_diis(max_vec=max_vec)
    for i in range(15):
        if i == 0:
            dm = mf.make_rdm1()
            fock = mf.get_fock()
        else:
            extrapolated_fock, diis_state = apply_diis(
                diis_state,
                fock,
                dm,
                mf.get_ovlp(),
                max_vec=max_vec,
                min_vecs=min_vecs,
                damping=damping,
            )
            dm = mf.make_rdm1(extrapolated_fock)
            fock = extrapolated_fock
        print(f"E_{i} = {mf.energy_tot(dm, fock, mf.get_veff(mol, dm)):.12f}")
    print("-" * 50)
    # Same with PySCF DIIS
    from pyscf import lib

    # Initialize PySCF DIIS
    print("PySCF DIIS")
    adiis = lib.diis.DIIS()
    for i in range(15):
        dm = mf.make_rdm1()
        fock = mf.get_fock()
        extrapolated_fock = adiis.update(fock, dm)
        dm = mf.make_rdm1(extrapolated_fock)
        fock = extrapolated_fock
        print(f"E_{i} = {mf.energy_tot(dm, fock, mf.get_veff(mol, dm)):.12f}")
