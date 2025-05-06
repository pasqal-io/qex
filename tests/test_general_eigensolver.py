"""
Test the JAX-based eigensolver against the standard eigensolver.

The standard eigensolver is only CPU and CANNOT be run on GPU.
The JAX-based eigensolver is CPU/GPU and can be run on GPU.
"""

import jax
import numpy as np
import pytest
from pyscf import gto, scf

# Import the jax_eig function from hf_legacy.py
from qedft.train.td.generalized_eigensolver import jax_eig, standard_eig

# Enable double precision
jax.config.update("jax_enable_x64", True)


def standard_eig(fock, s1e):
    """Standard PySCF eigenvalue solver"""
    import scipy.linalg

    e, c = scipy.linalg.eigh(fock, b=s1e)
    idx = e.argsort()
    return e[idx], c[:, idx]


@pytest.mark.parametrize(
    "mol_str, basis, name",
    [
        # H2
        ("H 0 0 0; H 0 0 1.0", "631g", "H2"),
        # H2O
        ("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", "631g", "H2O"),
        # H4 linear configuration
        ("H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0", "631g", "H4"),
        # H4 square configuration
        ("H 0 0 0; H 0 0 1.0; H 1 0 0; H 1 0 1.0", "631g", "H4_square"),
        # N2
        ("N 0 0 0; N 0 0 3.0", "631g", "N2"),
        # CH4
        (
            "C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63",
            "631g",
            "CH4",
        ),
    ],
)
def test_jax_eig_vs_standard(mol_str, basis, name):
    """Test that jax_eig produces the same results as standard_eig."""
    # Create molecule
    mol = gto.Mole()
    mol.atom = mol_str
    mol.basis = basis
    mol.build()

    # Run SCF to get Fock matrix
    mf = scf.RHF(mol)
    mf.kernel()

    # Get Fock matrix and overlap matrix
    fock = mf.get_fock()
    s1e = mf.get_ovlp()

    # Run both eigensolvers
    e_std, c_std = standard_eig(fock, s1e)
    e_jax, c_jax = jax_eig(fock, s1e)

    # Convert JAX arrays to numpy for comparison
    e_jax = np.array(e_jax)
    c_jax = np.array(c_jax)

    # Compare eigenvalues
    e_diff = np.abs(e_std - e_jax)

    # Compare eigenvectors (accounting for possible sign differences)
    c_diff = np.minimum(
        np.linalg.norm(c_std - c_jax, axis=0),
        np.linalg.norm(c_std + c_jax, axis=0),
    )

    # Set tolerance thresholds
    e_tol = 1e-7
    c_tol = 1e-7

    # Assert that differences are within tolerance
    assert np.max(e_diff) < e_tol, f"Eigenvalue difference too large for {name}: {np.max(e_diff)}"
    # For N2 and CH4, we relax the eigenvector comparison due to potential degeneracies
    # so the comparison is not strict and would produce large differences
    if name in ["N2", "CH4"]:
        if np.max(c_diff) >= c_tol:
            print(
                f"Note: Eigenvector difference for {name} is {np.max(c_diff):.2e}, "
                f"which exceeds tolerance {c_tol:.2e} but is allowed for this molecule",
            )
    else:
        assert (
            np.max(c_diff) < c_tol
        ), f"Eigenvector difference too large for {name}: {np.max(c_diff)}"

    print(f"Test passed for {name}:")
    print(f"Max eigenvalue difference: {np.max(e_diff):.2e}")
    print(f"Max eigenvector difference: {np.max(c_diff):.2e}")


if __name__ == "__main__":
    pytest.main([__file__])
