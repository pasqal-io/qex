"""
This is a JAX-based implementation of the generalized eigenvalue solver for Kohn-Sham DFT.
It is differentiable and can be run on GPU.

See: https://github.com/jax-ml/jax/issues/2748
And: https://github.com/jax-ml/jax/issues/5461
The latter provides the jax_eig function.
"""

import time

import jax
import jax.numpy as jnp
from jax import Array, random

from qedft.train.td.generalized_eigensolver import generalized_eigh, safe_generalized_eigh


@jax.jit
def masked_generalized_eigh(fock: Array, s1e: Array, mask: Array) -> tuple[Array, Array]:
    """
    Generalized eigenvalue solver that handles padded matrices using mask arrays.
    Returns eigenvalues in ascending order.

    Args:
        fock: Fock matrix, possibly padded
        s1e: Overlap matrix, possibly padded
        mask: Boolean mask indicating real values (True) vs padding (False)
             Shape should be (n,) where n is the matrix dimension

    Returns:
        mo_energy: Eigenvalues, with padding zeroed out and properly ordered (ascending)
        mo_coeff: Eigenvectors, with padding zeroed out and properly ordered
    """
    n = fock.shape[0]

    # Create 2D mask for the matrices
    mask_2d = mask[:, None] & mask[None, :]

    # Enforce numerical stability in padded regions
    fock_masked = jnp.where(mask_2d, fock, 0.0)

    # For padded diagonal elements of overlap, use small positive values for stability
    padded_diag = jnp.where(
        ~mask & jnp.eye(n, dtype=bool),
        jnp.ones_like(s1e) * 1e-12,
        0.0,
    )
    s1e_masked = jnp.where(mask_2d, s1e, padded_diag)

    # In padded region of fock, add identity to ensure proper eigenvalues
    fock_padded_diag = jnp.where(
        ~mask & jnp.eye(n, dtype=bool),
        jnp.ones_like(fock) * 1e-12,
        0.0,
    )
    fock_masked = fock_masked + fock_padded_diag

    # Solve generalized eigenvalue problem
    # The safe version is adapted for the degenerate case
    # Standard eigh cannot be differentiated for degenerate cases
    mo_energy, mo_coeff = safe_generalized_eigh(fock_masked, s1e_masked)

    # FIXED SORTING LOGIC:
    # Create a sorting key that:
    # 1. Ensures real eigenvalues come before padded ones
    # 2. Sorts real eigenvalues in ascending order

    # For real values, use the actual eigenvalue
    # For padded values, use a large positive value to push them to the end
    real_part_indicator = jnp.where(mask, 1.0, 0.0)
    sort_key = jnp.where(
        real_part_indicator > 0,
        mo_energy,
        jnp.ones_like(mo_energy) * 1e12,
    )

    # Sort by this key to get ascending real eigenvalues followed by padded ones
    sort_indices = jnp.argsort(sort_key)

    # Apply sorting to eigenvalues and eigenvectors
    mo_energy_sorted = jnp.take(mo_energy, sort_indices)
    mo_coeff_sorted = mo_coeff[:, sort_indices]

    # Zero out padded regions in results
    mo_energy_final = mo_energy_sorted * jnp.take(mask, sort_indices)
    mo_coeff_final = jnp.where(mask_2d[:, sort_indices], mo_coeff_sorted, 0.0)

    return mo_energy_final, mo_coeff_final


def test_masked_generalized_eigh():
    """Test the masked generalized eigenvalue solver."""
    print("Testing masked_generalized_eigh...")

    # Create test matrices
    real_size, pad_size = 3, 5
    rng = random.PRNGKey(42)
    key1, key2 = random.split(rng)

    # Create test matrices
    f_small = random.normal(key1, (real_size, real_size))
    f_small = f_small @ f_small.T  # Make symmetric
    s_small = random.normal(key2, (real_size, real_size))
    s_small = (s_small @ s_small.T) + jnp.eye(real_size) * 5.0  # Make positive definite
    print(f"f_small: {f_small}")
    # Get reference results
    ref_energy, ref_coeff = generalized_eigh(f_small, s_small)

    # Create padded matrices
    f_padded = jnp.zeros((pad_size, pad_size))
    s_padded = jnp.zeros((pad_size, pad_size))
    f_padded = f_padded.at[:real_size, :real_size].set(f_small)
    s_padded = s_padded.at[:real_size, :real_size].set(s_small)
    print(f"f_padded: {f_padded}")
    # Create boolean mask
    mask = jnp.zeros(pad_size, dtype=bool)
    mask = mask.at[:real_size].set(True)

    # Test with zero padding
    time_start = time.time()
    test_energy, test_coeff = masked_generalized_eigh(f_padded, s_padded, mask)
    time_end = time.time()
    print(f"Test with zero padding - Time: {time_end - time_start:.6f}s")

    # Test with different padding values
    pad_shape = (pad_size - real_size, pad_size - real_size)
    f_padded2 = f_padded.at[real_size:, real_size:].set(jnp.ones(pad_shape) * 100.0)
    s_padded2 = s_padded.at[real_size:, real_size:].set(jnp.ones(pad_shape) * 100.0)

    time_start = time.time()
    test_energy2, test_coeff2 = masked_generalized_eigh(f_padded2, s_padded2, mask)
    time_end = time.time()
    print(f"Test with different padding values - Time: {time_end - time_start:.6f}s")

    # Test with scaled matrix
    f_padded3 = f_padded * 0.2
    s_padded3 = s_padded * 0.2

    time_start = time.time()
    test_energy3, test_coeff3 = masked_generalized_eigh(f_padded3, s_padded3, mask)
    time_end = time.time()
    print(f"Test with scaled matrix - Time: {time_end - time_start:.6f}s")

    # Compare results
    energy_match = jnp.allclose(test_energy[:real_size], ref_energy)
    coeff_match = jnp.allclose(test_coeff[:real_size, :real_size], ref_coeff)
    coeff_match_2 = jnp.allclose(test_coeff2[:real_size, :real_size], ref_coeff)
    padding_independent = jnp.allclose(
        test_energy[:real_size],
        test_energy2[:real_size],
    )

    # Print comparison results
    print("\nValidation Results:")
    print(f"  Energy matches reference: {energy_match}")
    print(f"  Coefficients match reference: {coeff_match}")
    print(f"  Coefficients match with different padding: {coeff_match_2}")
    print(f"  Results independent of padding: {padding_independent}")

    # Show eigenvalues
    print("\nEigenvalues:")
    print(f"  Reference: {ref_energy}")
    print(f"  Zero padding: {test_energy[:real_size]}")
    print(f"  Zero padding, full output: {test_energy[:]}")
    print(f"  Custom padding: {test_energy2[:real_size]}")

    # Check differences
    energy_diff = jnp.abs(test_energy[:real_size] - test_energy2[:real_size])
    print(f"  Absolute differences: {energy_diff}")

    # Relaxed tolerance check
    padding_independent_relaxed = jnp.allclose(
        test_energy[:real_size],
        test_energy2[:real_size],
        rtol=1e-5,
        atol=1e-5,
    )
    print(f"  Results match with relaxed tolerance: {padding_independent_relaxed}")

    # Test gradient computation
    @jax.jit
    def loss(fock, s1e, mask):
        energy, coeff = masked_generalized_eigh(fock, s1e, mask)
        return jnp.sum(energy**2)

    grad = jax.grad(loss)(f_padded, s_padded, mask)
    print("\nGradient computation successful")

    return energy_match and coeff_match and (padding_independent or padding_independent_relaxed)


if __name__ == "__main__":
    basic_test = test_masked_generalized_eigh()
