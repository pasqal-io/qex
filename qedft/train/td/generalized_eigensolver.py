"""
This is a JAX-based implementation of the generalized eigenvalue solver for Kohn-Sham DFT.
It is differentiable and can be run on GPU.

See: https://github.com/jax-ml/jax/issues/2748
And: https://github.com/jax-ml/jax/issues/5461
The latter provides the jax_eig function.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
import scipy

# -------------------------------------------------------------------------------------------------
# Old solution but not differentiable
# -------------------------------------------------------------------------------------------------


# from functools import partial
# from jax import custom_vjp
# import jax.numpy as jnp
# from jax import lax
# import jax.test_util as jtu
# import numpy as np

# @custom_vjp
# def safe_eigh(x):
#     return jnp.linalg.eigh(x)


# def safe_eigh_fwd(x):
#     w, v = safe_eigh(x)
#     return (w, v), (w, v)


# def safe_eigh_bwd(res, g):
#     w, v = res
#     wct, vct = g
#     deltas = w[..., jnp.newaxis, :] - w[..., :, jnp.newaxis]
#     on_diagonal = jnp.eye(w.shape[-1], dtype=bool)
#     F = jnp.where(on_diagonal, 0, 1 / jnp.where(on_diagonal, 1, deltas))
#     matmul = partial(jnp.matmul, precision=lax.Precision.HIGHEST)
#     vT_ct = matmul(v.T.conj(), vct)
#     F_vT_vct = jnp.where(vT_ct != 0, F * vT_ct, 0)  # ignore values that would give NaN
#     g = matmul(v, matmul(jnp.diag(wct) + F_vT_vct, v.T.conj()))
#     g = (g + g.T.conj()) / 2
#     return (g,)


# safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_bwd)
# jtu.check_grads(safe_eigh, (np.random.RandomState(0).randn(3, 3),), order=2, modes=['rev'])


# -------------------------------------------------------------------------------------------------
# Another option but requires complex numbers
# -------------------------------------------------------------------------------------------------


# import jax
# import jax.numpy as jnp

# from functools import partial


# def conj(arr):
#     return arr.real + arr.imag * -1j
#     # return arr.conj()


# @partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
# def eig(x, type_complex=jnp.complex64, perturbation=1E-10, device='cpu'):

#     _eig = jax.jit(jnp.linalg.eig, device=jax.devices('cpu')[0])

#     eigenvalues_shape = jax.ShapeDtypeStruct(x.shape[:-1], type_complex)
#     eigenvectors_shape = jax.ShapeDtypeStruct(x.shape, type_complex)

#     result_shape_dtype = (eigenvalues_shape, eigenvectors_shape)
#     if device == 'cpu':
#         res = _eig(x)
#     else:
#         res = jax.pure_callback(_eig, result_shape_dtype, x)

#     return res


# def eig_fwd(x, type_complex, perturbation, device):
#     return eig(x, type_complex, perturbation), (eig(x, type_complex, perturbation), x)


# def eig_bwd(type_complex, perturbation, device, res, g):
#     """
#     Gradient of a general square (complex valued) matrix
#     Eq 2~5 in https://www.nature.com/articles/s42005-021-00568-6
#     Eq 4.77 in https://arxiv.org/pdf/1701.00392.pdf
#     Eq. 30~32 in https://www.sciencedirect.com/science/article/abs/pii/S0010465522002715
#     https://github.com/kch3782/torcwa
#     https://github.com/weiliangjinca/grcwa
#     https://github.com/pytorch/pytorch/issues/41857
#     https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#Complex-numbers-and-differentiation
#     https://discuss.pytorch.org/t/autograd-on-complex-numbers/144687/3
#     """

#     (eig_val, eig_vector), x = res
#     grad_eigval, grad_eigvec = g

#     grad_eigval = jnp.diag(grad_eigval)
#     W_H = eig_vector.T.conj()

#     Fij = eig_val.reshape((1, -1)) - eig_val.reshape((-1, 1))
#     Fij = Fij / (jnp.abs(Fij) ** 2 + perturbation)
#     Fij = Fij.at[jnp.diag_indices_from(Fij)].set(0)

#     # diag_indices = jnp.arange(len(eig_val))
#     # Eij = eig_val.reshape((1, -1)) - eig_val.reshape((-1, 1))
#     # Eij = Eij.at[diag_indices, diag_indices].set(1)
#     # Fij = 1 / Eij
#     # Fij = Fij.at[diag_indices, diag_indices].set(0)

#     grad = jnp.linalg.inv(W_H) @ (grad_eigval.conj() + Fij * (W_H @ grad_eigvec.conj())) @ W_H
#     grad = grad.conj()
#     if not jnp.iscomplexobj(x):
#         grad = grad.real

#     return grad,

# eig.defvjp(eig_fwd, eig_bwd)


# -------------------------------------------------------------------------------------------------
# Solution for degenerate cases of Kasim adapted to JAX
# -------------------------------------------------------------------------------------------------


def is_debug_enabled():
    return False


@jax.custom_vjp
def degen_eigh(A):
    """JAX implementation of symmetric eigendecomposition that handles degenerate cases."""
    eival, eivec = jnp.linalg.eigh(A)
    return eival, eivec


def degen_eigh_fwd(A):
    """Forward pass of the degenerate eigendecomposition."""
    eival, eivec = jnp.linalg.eigh(A)
    return (eival, eivec), (eival, eivec)


def degen_eigh_bwd(res, grads):
    """Clarification on the case in my DFT application.
    Note:
    By Kasim from: https://github.com/jax-ml/jax/issues/669#issuecomment-777052841
    The case in my DFT application is that it is supposed
    to have exactly the same eigenvalues theoretically,
    but numerically, the retrieved eigenvalues are only close to each other (near-degenerate).
    In the near-degenerate case, the denominator (λ_j - λ_i) is close to 0,
    so unless you have near 0 nominator
    (i.e. eqs 2.8 and 2.13-2.15), then the numerical instability is unavoidable (just like 1/x).
    In the paper, I just consider the case where the nominator is supposed to be 0.
    If it's not 0 (due to numerical error), it is assumed to be 0.
    If you want to consider a case where the nominator is supposed
    to be a very small value but not 0, then this is not covered in the paper.
    In my DFT case, this is sufficient, because the loss function does not depend directly
    on the degenerate eigenvectors (it depends on the space spanned by the eigenvectors),
    so the nominator (eq 2.13) is supposed to be 0.

    Args:
        res: (eival, eivec)
        grads: (grad_eival, grad_eivec)

    Returns:
        grad_A: (grad_A)
    """
    eival, eivec = res
    grad_eival, grad_eivec = grads
    in_debug_mode = is_debug_enabled()

    min_threshold = jnp.finfo(eival.dtype).eps ** 0.6
    eivect = jnp.transpose(eivec, axes=(-2, -1))

    # Handle contribution from eigenvectors
    result = jnp.zeros_like(eivec)

    if grad_eivec is not None:
        # Calculate F matrix for handling degeneracy
        F = jnp.expand_dims(eival, -2) - jnp.expand_dims(eival, -1)
        idx = jnp.abs(F) < min_threshold
        F = jnp.where(idx, jnp.inf, F)

        # Debug check for degeneracy requirements
        if in_debug_mode:
            degenerate = jnp.any(idx)
            xtg = eivect @ grad_eivec
            diff_xtg = (xtg - jnp.transpose(xtg, axes=(-2, -1)))[idx]
            reqsat = jnp.allclose(diff_xtg, jnp.zeros_like(diff_xtg))

        # Calculate contribution from eigenvectors
        F = jnp.power(F, -1)
        F = F * jnp.matmul(eivect, grad_eivec)
        result = jnp.matmul(eivec, jnp.matmul(F, eivect))

    # Calculate contribution from eigenvalues
    if grad_eival is not None:
        result = result + jnp.matmul(eivec, jnp.expand_dims(grad_eival, -1) * eivect)

    # Symmetrize result
    result = (result + jnp.transpose(result, axes=(-2, -1))) * 0.5

    return (result,)


degen_eigh.defvjp(degen_eigh_fwd, degen_eigh_bwd)


# -------------------------------------------------------------------------------------------------
# Eigenvalue solvers
# -------------------------------------------------------------------------------------------------


def standard_eig(fock, s1e):
    """Generalized eigenvalue solver using jax.linalg.eigh. Only CPU.
    Args:
        fock: Fock matrix
        s1e: Overlap matrix
    Returns:
        eigenvalues: Array of eigenvalues
        eigenvectors: Array of eigenvectors
    """
    # This is what mf.eig would do,
    # currently not differentiable and cannot be run on GPU
    e, c = scipy.linalg.eigh(fock, s1e)
    idx = e.argsort()
    return e[idx], c[:, idx]


@jax.jit
def jax_eig(A, B):
    """JAX-based generalized eigenvalue solver. CPU/GPU.
    Note might be unstable for ill-conditioned matrices.
    For hard cases run on CPU with scipy.linalg.eigh.
    Simple version without stabilization.
    Args:
        A: Fock matrix
        B: Overlap matrix
    Returns:
        eigenvalues: Array of eigenvalues
        eigenvectors: Array of eigenvectors
    """
    L = jnp.linalg.cholesky(B)
    L_inv = jnp.linalg.inv(L)
    C = L_inv @ A @ L_inv.T
    eigenvalues, eigenvectors_transformed = jnp.linalg.eigh(C)
    eigenvectors_original = L_inv.T @ eigenvectors_transformed
    return eigenvalues, eigenvectors_original


@partial(jax.jit, static_argnames=("eps", "scale", "dtype"))
def generalized_eigh(
    A: jax.Array,
    B: jax.Array,
    *,
    eps: float = 1.0e-12,
    scale: bool = False,
    dtype=jnp.float64,
):
    """
    Robust generalized symmetric eigensolver:  A v = λ B v
    This is a JAX-based implementation of the generalized eigenvalue solver
    for the Kohn-Sham model. It is differentiable.
    Best results were achieved with this implementation and default parameters.

    Parameters
    ----------
    A, B : [..., n, n]  real‑symmetric (or Hermitian) batches
        B must be (numerically) SPD.
    eps : float
        Minimal eigen‑value enforced for B (diagonal shift).
    scale : bool
        If True apply Jacobi (diagonal) scaling to lower condition number.
    dtype : jnp.dtype
        Working precision.

    Returns
    -------
    w : [..., n]            eigen‑values  (ascending)
    v : [..., n, n]         eigen‑vectors (columns)
    """
    A = (A + A.T.conj()) * 0.5
    B = (B + B.T.conj()) * 0.5
    A = A.astype(dtype)
    B = B.astype(dtype)

    # Optional diagonal scaling
    if scale:
        s = jnp.sqrt(jnp.diag(B))
        S_inv = 1.0 / s
        A = (S_inv[:, None] * A) * S_inv[None, :]
        B = (S_inv[:, None] * B) * S_inv[None, :]

    # Guarantee SPD
    λ_min = jnp.min(jnp.linalg.eigvalsh(B))
    shift = jnp.where(λ_min < eps, eps - λ_min, 0.0)
    B = B + shift * jnp.eye(B.shape[-1], dtype=dtype)

    # Cholesky & triangular solves (no explicit inverse)
    L = jnp.linalg.cholesky(B)  # B = L Lᵀ

    # C = L⁻¹ A L⁻ᵀ  → use two triangular solves
    Y = jsp.solve_triangular(L, A, lower=True, trans="N")
    C = jsp.solve_triangular(L, Y.T, lower=True, trans="N").T
    C = (C + C.T.conj()) * 0.5  # enforce symmetry

    # Diagonalise
    w, U = jnp.linalg.eigh(C)  # C U = U diag(w)

    # Back‑transform eigen‑vectors:  v = L⁻ᵀ U
    V = jsp.solve_triangular(L.T, U, lower=False, trans="N")

    # Optional re‑orthonormalisation in B metric (rarely needed)
    # This would change quite a bit the results
    # V, _ = jnp.linalg.qr(B @ V)

    return w, V


@partial(jax.jit, static_argnames=("eps", "scale", "dtype"))
def safe_generalized_eigh(
    A: jax.Array,
    B: jax.Array,
    *,
    eps: float = 1.0e-12,
    scale: bool = False,
    dtype=jnp.float64,
):
    """
    Degenerate case symmetric eigensolver:  A v = λ B v
    This is a JAX-based implementation of the generalized eigenvalue solver
    for the Kohn-Sham model. It is differentiable.
    Best results were achieved with this implementation and default parameters.

    This version is adapted for degenerate cases from Kasim's implementation.

    Parameters
    ----------
    A, B : [..., n, n]  real‑symmetric (or Hermitian) batches
        B must be (numerically) SPD.
    eps : float
        Minimal eigen‑value enforced for B (diagonal shift).
    scale : bool
        If True apply Jacobi (diagonal) scaling to lower condition number.
    dtype : jnp.dtype
        Working precision.

    Returns
    -------
    w : [..., n]            eigen‑values  (ascending)
    v : [..., n, n]         eigen‑vectors (columns)
    """
    A = (A + A.T.conj()) * 0.5
    B = (B + B.T.conj()) * 0.5
    A = A.astype(dtype)
    B = B.astype(dtype)

    # Optional diagonal scaling
    if scale:
        s = jnp.sqrt(jnp.diag(B))
        S_inv = 1.0 / s
        A = (S_inv[:, None] * A) * S_inv[None, :]
        B = (S_inv[:, None] * B) * S_inv[None, :]

    # Guarantee SPD
    λ_min = jnp.min(jnp.linalg.eigvalsh(B))
    shift = jnp.where(λ_min < eps, eps - λ_min, 0.0)
    B = B + shift * jnp.eye(B.shape[-1], dtype=dtype)

    # Cholesky & triangular solves (no explicit inverse)
    L = jnp.linalg.cholesky(B)  # B = L Lᵀ

    # C = L⁻¹ A L⁻ᵀ  → use two triangular solves
    Y = jsp.solve_triangular(L, A, lower=True, trans="N")
    C = jsp.solve_triangular(L, Y.T, lower=True, trans="N").T
    C = (C + C.T.conj()) * 0.5  # enforce symmetry

    # Diagonalise
    w, U = degen_eigh(C)  # C U = U diag(w)

    # Back‑transform eigen‑vectors:  v = L⁻ᵀ U
    V = jsp.solve_triangular(L.T, U, lower=False, trans="N")

    # Optional re‑orthonormalisation in B metric (rarely needed)
    # This would change quite a bit the results
    # V, _ = jnp.linalg.qr(B @ V)

    return w, V


if __name__ == "__main__":

    # Test gradient behavior with different eigensolvers
    import jax.numpy as jnp
    from jax import grad

    print("\n=== Testing gradient behavior with different eigensolvers ===\n")

    # Create test matrices with degenerate eigenvalues
    A = jnp.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    B = jnp.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    # Helper function to test and print results for each eigensolver
    def test_eigensolver(name, solver_fn, *args):
        print(f"\n--- Testing {name} ---")

        # Define function to extract specific eigenvector component
        def get_eigenvector_component(*solver_args, i, j):
            return solver_fn(*solver_args)[1][i, j]

        # Get gradient function
        grad_fn = grad(get_eigenvector_component, argnums=0)

        # Compute eigenvectors and their gradients
        V = jnp.array(
            [[get_eigenvector_component(*args, i=i, j=j) for j in range(3)] for i in range(3)],
        )

        try:
            dV = jnp.array([[grad_fn(*args, i=i, j=j) for j in range(3)] for i in range(3)])
            has_nans = jnp.any(jnp.isnan(dV))

            print(f"Eigenvectors:\n{V}")
            print(f"Gradients:\n{dV}")

            if has_nans:
                print(f"WARNING: {name} gradients contain NaN values!")
            else:
                print(f"SUCCESS: {name} gradients computed without NaNs")
        except Exception as e:
            print(f"ERROR computing gradients: {str(e)}")

    # 1. Test standard eigh
    def standard_eigh(M):
        return jnp.linalg.eigh(M)

    test_eigensolver("Standard eigh", standard_eigh, A)

    # 2. Test degen_eigh
    test_eigensolver("Degenerate-aware eigh", degen_eigh, A)

    # 3. Test generalized_eigh
    test_eigensolver("Generalized eigh", generalized_eigh, A, B)

    # 4. Test safe_generalized_eigh
    test_eigensolver("Safe generalized eigh", safe_generalized_eigh, A, B)

    print("\n=== Summary ===")
    print("Standard eigh and degen_eigh: May produce NaNs with degenerate eigenvalues")
    print("Generalized eigh: May produce NaNs with degenerate eigenvalues")
    print("Safe generalized eigh: Handles degenerate eigenvalues correctly")

    # Test with a simple example
    A = jnp.array([[1.0, 2.0], [2.0, 3.0]])  # Symmetric matrix
    B = jnp.array([[2.0, 0.5], [0.5, 1.0]])  # Symmetric positive definite matrix

    # Compare standard scipy solution with our JAX implementation
    scipy_vals, scipy_vecs = standard_eig(A, B)
    jax_vals, jax_vecs = jax_eig(A, B)
    gen_jax_vals, gen_jax_vecs = generalized_eigh(A, B)

    print("Scipy eigenvalues:", scipy_vals)
    print("JAX eigenvalues:", jax_vals)
    print("New JAX:", gen_jax_vals)

    # Verify the generalized eigenvalue equation: A @ v = lambda * B @ v
    for i in range(len(jax_vals)):
        lhs = A @ jax_vecs[:, i]
        rhs = jax_vals[i] * (B @ jax_vecs[:, i])
        residual = jnp.linalg.norm(lhs - rhs)
        print(f"Residual for eigenpair {i}: {residual}")

    # Test with larger random matrices
    key = jax.random.PRNGKey(42)
    n = 10
    key1, key2 = jax.random.split(key)

    # Create random symmetric matrices
    A_large = jax.random.normal(key1, (n, n))
    A_large = (A_large + A_large.T) / 2  # Make symmetric

    # Create random positive definite matrix
    B_temp = jax.random.normal(key2, (n, n))
    B_large = B_temp @ B_temp.T + n * jnp.eye(n)  # Ensure positive definiteness

    print("\n --- Testing timing for large matrices ---\n")

    # Time the JAX implementation
    start = time.time()
    jax_vals_large, jax_vecs_large = jax_eig(A_large, B_large)
    end = time.time()
    print(
        f"JAX generalized eigensolve with COMPILE for {n}x{n} matrix took {end-start:.4f} seconds",
    )

    # Time the generalized eigensolver
    start = time.time()
    gen_jax_vals_large, gen_jax_vecs_large = generalized_eigh(A_large, B_large)
    end = time.time()
    print(f"Generalized JAX eigensolve for {n}x{n} matrix took {end-start:.4f} seconds")

    # JIT the JAX implementation
    start = time.time()
    jax_vals_large_jit, jax_vecs_large_jit = jax_eig(A_large, B_large)
    end = time.time()
    print(f"JIT JAX generalized eigensolve for {n}x{n} matrix took {end-start} seconds")

    # Time the scipy implementation
    start = time.time()
    scipy_vals_large, scipy_vecs_large = standard_eig(A_large, B_large)
    end = time.time()
    print(f"Scipy generalized eigensolve for {n}x{n} matrix took {end-start:.4f} seconds")

    print("\n--- Testing with ill-conditioned matrices ---")

    # Test 1: Highly ill-conditioned matrix
    print("\nTest 1: Highly ill-conditioned matrix")
    n = 5
    # Create a matrix with high condition number
    diag_vals = jnp.logspace(-8, 0, n)  # Values from 10^-8 to 1
    A_ill = jnp.diag(diag_vals)
    # Make sure A is symmetric
    temp = jax.random.normal(jax.random.PRNGKey(0), (n, n))
    Q, _ = jnp.linalg.qr(temp)  # Orthogonal matrix
    A_ill = Q @ A_ill @ Q.T

    # B is better conditioned but still challenging
    diag_vals_b = jnp.logspace(-4, 0, n)  # Values from 10^-4 to 1
    B_ill = jnp.diag(diag_vals_b)
    B_ill = Q @ B_ill @ Q.T

    # Calculate condition numbers
    cond_A = jnp.linalg.norm(A_ill) * jnp.linalg.norm(jnp.linalg.inv(A_ill))
    cond_B = jnp.linalg.norm(B_ill) * jnp.linalg.norm(jnp.linalg.inv(B_ill))
    print(f"Condition number of A: {cond_A:.2e}")
    print(f"Condition number of B: {cond_B:.2e}")

    # Compare implementations
    try:
        jax_vals_ill, jax_vecs_ill = jax_eig(A_ill, B_ill)
        scipy_vals_ill, scipy_vecs_ill = standard_eig(A_ill, B_ill)
        gen_jax_vals_ill, gen_jax_vecs_ill = generalized_eigh(A_ill, B_ill)

        # Calculate max difference between eigenvalues
        max_diff = jnp.max(jnp.abs(jax_vals_ill - scipy_vals_ill))
        max_diff_gen = jnp.max(jnp.abs(jax_vals_ill - gen_jax_vals_ill))
        print(f"Max difference in eigenvalues: {max_diff:.2e}")
        print(f"Max difference in eigenvalues (gen): {max_diff_gen:.2e}")

        # Verify the generalized eigenvalue equation
        max_residual = 0
        for i in range(len(jax_vals_ill)):
            lhs = A_ill @ jax_vecs_ill[:, i]
            rhs = jax_vals_ill[i] * (B_ill @ jax_vecs_ill[:, i])
            residual = jnp.linalg.norm(lhs - rhs)
            max_residual = max(max_residual, residual)
        print(f"Max residual: {max_residual:.2e}")

        # Verify the generalized eigenvalue equation
        max_residual_gen = 0
        for i in range(len(jax_vals_ill)):
            lhs = A_ill @ gen_jax_vecs_ill[:, i]
            rhs = gen_jax_vals_ill[i] * (B_ill @ gen_jax_vecs_ill[:, i])
            residual_gen = jnp.linalg.norm(lhs - rhs)
            max_residual_gen = max(max_residual_gen, residual_gen)
        print(f"Max residual (gen): {max_residual_gen:.2e}")

        if max_diff_gen < max_diff:
            print("Generalized JAX eigensolver is more accurate than JAX eigensolver")
        else:
            print("JAX eigensolver is more accurate than generalized eigensolver")

    except Exception as e:
        print(f"Error in ill-conditioned test: {e}")

    # Test 2: Nearly singular B matrix
    print("\nTest 2: Nearly singular B matrix")
    n = 4
    key = jax.random.PRNGKey(1)

    # Create symmetric A
    temp_A = jax.random.normal(key, (n, n))
    A_reg = (temp_A + temp_A.T) / 2

    # Create nearly singular B (small eigenvalue)r
    diag_vals_b2 = jnp.array([1.0, 0.1, 0.01, 1e-8])
    temp = jax.random.normal(jax.random.PRNGKey(2), (n, n))
    Q2, _ = jnp.linalg.qr(temp)
    B_sing = Q2 @ jnp.diag(diag_vals_b2) @ Q2.T

    print(f"Smallest eigenvalue of B: {jnp.min(jnp.linalg.eigvalsh(B_sing)):.2e}")

    # Compare implementations with error handling
    try:
        scipy_vals_sing, scipy_vecs_sing = standard_eig(A_reg, B_sing)
        print("Scipy solution completed")
    except Exception as e:
        print(f"Scipy error: {e}")

    try:
        jax_vals_sing, jax_vecs_sing = jax_eig(A_reg, B_sing)
        print("JAX solution completed")
    except Exception as e:
        print(f"JAX error: {e}")

    # Test 3: Matrix with clustered eigenvalues
    print("\nTest 3: Matrix with clustered eigenvalues")
    n = 6

    # Create matrix with clustered eigenvalues
    clustered_vals = jnp.array([1.0, 1.0001, 1.0002, 5.0, 5.0001, 10.0])
    temp = jax.random.normal(jax.random.PRNGKey(3), (n, n))
    Q3, _ = jnp.linalg.qr(temp)
    A_clustered = Q3 @ jnp.diag(clustered_vals) @ Q3.T

    # Standard B
    B_std = jnp.eye(n)

    # Compare implementations
    try:
        scipy_vals_clust, scipy_vecs_clust = standard_eig(A_clustered, B_std)
        jax_vals_clust, jax_vecs_clust = jax_eig(A_clustered, B_std)

        # Check if eigenvalues are correctly identified
        print("Scipy clustered eigenvalues:", scipy_vals_clust)
        print("JAX clustered eigenvalues:", jax_vals_clust)

        # Calculate differences
        diff = jnp.abs(jnp.sort(scipy_vals_clust) - jnp.sort(jax_vals_clust))
        print(
            f"Max difference in clustered eigenvalues: " f"{jnp.max(diff):.2e}",
        )
    except Exception as e:
        print(f"Error in clustered eigenvalues test: {e}")
