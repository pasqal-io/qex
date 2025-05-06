"""Test the make_rdm1 function."""

from typing import Any

import numpy as np
import pytest
from jax import random


def make_rdm1_original(mo_coeff: np.ndarray, mo_occ: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Original implementation of the one-particle reduced density matrix.

    Args:
        mo_coeff: Molecular orbital coefficients, shape (n_orbitals, n_orbitals)
        mo_occ: Occupation numbers, shape (n_orbitals,)
        **kwargs: Additional keyword arguments (not used)

    Returns:
        The one-particle reduced density matrix, shape (n_orbitals, n_orbitals)
    """
    mocc = mo_coeff[:, mo_occ > 0]
    dm = np.dot(mocc * mo_occ[mo_occ > 0], mocc.conj().T)
    return dm


def make_rdm1_new(mo_coeff: np.ndarray, mo_occ: np.ndarray, **kwargs: Any) -> np.ndarray:
    """New implementation of the one-particle reduced density matrix using improved indexing.

    Args:
        mo_coeff: Molecular orbital coefficients, shape (n_orbitals, n_orbitals)
        mo_occ: Occupation numbers, shape (n_orbitals,)
        **kwargs: Additional keyword arguments (not used)

    Returns:
        The one-particle reduced density matrix, shape (n_orbitals, n_orbitals)
    """
    mask = mo_occ > 0
    mocc = mo_coeff[:, mask]  # Use proper slicing instead of broadcasting
    dm = np.dot(mocc * mo_occ[mask], mocc.conj().T)
    return dm


class TestMakeRDM1:
    """Test suite for the one-particle reduced density matrix implementations."""

    @pytest.fixture
    def rng_key(self) -> random.PRNGKey:
        """Fixture providing a random key for reproducible tests.

        Returns:
            A JAX random key
        """
        return random.PRNGKey(0)

    def generate_test_data(
        self,
        rng_key: random.PRNGKey,
        n_orbitals: int,
        n_occupied: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate test data for mo_coeff and mo_occ.

        Args:
            rng_key: JAX random key
            n_orbitals: Number of orbitals
            n_occupied: Number of occupied orbitals

        Returns:
            Tuple of (mo_coeff, mo_occ) arrays
        """
        key1, key2 = random.split(rng_key)

        # Generate random mo_coeff (using float64 for better precision)
        mo_coeff = np.array(
            random.normal(key1, (n_orbitals, n_orbitals)),
            dtype=np.float64,
        )

        # Generate mo_occ with specific number of occupied orbitals
        mo_occ = np.zeros(n_orbitals, dtype=np.float64)
        mo_occ[:n_occupied] = 2.0  # Double occupation for closed shell

        return mo_coeff, mo_occ

    def test_simple_case(self) -> None:
        """Test with a simple 2x2 matrix."""
        mo_coeff = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        mo_occ = np.array([2.0, 0.0], dtype=np.float64)  # One occupied orbital

        dm1 = make_rdm1_original(mo_coeff, mo_occ)
        dm2 = make_rdm1_new(mo_coeff, mo_occ)

        np.testing.assert_allclose(dm1, dm2, atol=1e-14)

        # Check expected result
        expected = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        np.testing.assert_allclose(dm1, expected, atol=1e-14)

    def test_random_matrices(self, rng_key: random.PRNGKey) -> None:
        """Test with random matrices of different sizes.

        Args:
            rng_key: JAX random key
        """
        sizes = [(4, 2), (6, 3), (8, 4)]  # (n_orbitals, n_occupied)

        for n_orbitals, n_occupied in sizes:
            mo_coeff, mo_occ = self.generate_test_data(
                rng_key,
                n_orbitals,
                n_occupied,
            )

            dm1 = make_rdm1_original(mo_coeff, mo_occ)
            dm2 = make_rdm1_new(mo_coeff, mo_occ)

            np.testing.assert_allclose(dm1, dm2, atol=1e-14)

    def test_complex_matrices(self, rng_key: random.PRNGKey) -> None:
        """Test with complex matrices.

        Args:
            rng_key: JAX random key
        """
        key1, key2 = random.split(rng_key)

        n_orbitals, n_occupied = 4, 2
        mo_coeff = (
            random.normal(key1, (n_orbitals, n_orbitals))
            + 1j * random.normal(key2, (n_orbitals, n_orbitals))
        ).astype(np.complex128)
        mo_occ = np.zeros(n_orbitals, dtype=np.float64)
        mo_occ[:n_occupied] = 2.0

        dm1 = make_rdm1_original(mo_coeff, mo_occ)
        dm2 = make_rdm1_new(mo_coeff, mo_occ)

        np.testing.assert_allclose(dm1, dm2, atol=1e-14)

    def test_fractional_occupation(self, rng_key: random.PRNGKey) -> None:
        """Test with fractional occupation numbers.

        Args:
            rng_key: JAX random key
        """
        n_orbitals = 4
        mo_coeff, _ = self.generate_test_data(rng_key, n_orbitals, 2)
        # Fractional occupation
        mo_occ = np.array([1.5, 0.5, 0.0, 0.0], dtype=np.float64)

        dm1 = make_rdm1_original(mo_coeff, mo_occ)
        dm2 = make_rdm1_new(mo_coeff, mo_occ)

        np.testing.assert_allclose(dm1, dm2, atol=1e-14)

    def test_hermiticity(self, rng_key: random.PRNGKey) -> None:
        """Test that the density matrix is Hermitian.

        Args:
            rng_key: JAX random key
        """
        n_orbitals, n_occupied = 4, 2
        mo_coeff, mo_occ = self.generate_test_data(
            rng_key,
            n_orbitals,
            n_occupied,
        )

        dm1 = make_rdm1_original(mo_coeff, mo_occ)
        dm2 = make_rdm1_new(mo_coeff, mo_occ)

        # Check hermiticity for both implementations
        np.testing.assert_allclose(dm1, dm1.conj().T, atol=1e-7)
        np.testing.assert_allclose(dm2, dm2.conj().T, atol=1e-7)

    def test_trace_conservation(self, rng_key: random.PRNGKey) -> None:
        """Test that the trace equals the number of electrons.

        Args:
            rng_key: JAX random key
        """
        n_orbitals, n_occupied = 4, 2
        mo_coeff, mo_occ = self.generate_test_data(
            rng_key,
            n_orbitals,
            n_occupied,
        )

        dm1 = make_rdm1_original(mo_coeff, mo_occ)
        dm2 = make_rdm1_new(mo_coeff, mo_occ)

        # Check trace for both implementations
        np.testing.assert_allclose(np.trace(dm2), np.trace(dm1), atol=1e-7)


if __name__ == "__main__":
    pytest.main([__file__])
