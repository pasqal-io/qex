"""
Compare the two eigensolvers for a given molecule.

The two eigensolvers are:
1. Standard eigenvalue solver, only CPU
2. Equivalent JAX-based eigensolver that runs on GPU

The standard JAX generalized eigenvalue solver cannot be run on GPU.
We rewrite the same computation using two eighs, which is equivalent to
the standard PySCF eigenvalue solver.

This is differentiable and can be run on GPU.
Necessary to JIT the SCF loop.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf

from qedft.train.td.generalized_eigensolver import jax_eig, standard_eig

# Enable double precision
jax.config.update("jax_enable_x64", True)


def compare_eigensolvers(mol_str, basis="631g"):
    """Compare the two eigensolvers for a given molecule"""
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

    return {
        "eigenvalues_std": e_std,
        "eigenvalues_jax": e_jax,
        "eigenvalues_diff": e_diff,
        "eigenvectors_diff": c_diff,
        "max_eigenvalue_diff": np.max(e_diff),
        "max_eigenvector_diff": np.max(c_diff),
    }


def plot_comparison(results):
    """Plot the comparison results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot eigenvalues
    ax1.plot(results["eigenvalues_std"], label="Standard")
    ax1.plot(results["eigenvalues_jax"], "o", markersize=3, label="JAX")
    ax1.set_title("Eigenvalues Comparison")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Energy (a.u.)")
    ax1.legend()

    # Plot differences
    ax2.semilogy(results["eigenvalues_diff"], label="Eigenvalues")
    ax2.semilogy(results["eigenvectors_diff"], label="Eigenvectors")
    ax2.set_title("Differences (log scale)")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Absolute Difference")
    ax2.legend()
    plt.savefig("./eigensolver_comparison.png")
    plt.tight_layout()
    return fig


# Test with different molecules
molecules = {
    "H2": "H 0 0 0; H 0 0 1.0",
    "H2O": "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
}

if __name__ == "__main__":
    print("Comparing eigensolvers for different molecules:")

    for name, mol_str in molecules.items():
        print(f"\nMolecule: {name}")
        results = compare_eigensolvers(mol_str)

        print(f"Max eigenvalue difference: {results['max_eigenvalue_diff']:.2e}")
        print(f"Max eigenvector difference: {results['max_eigenvector_diff']:.2e}")

        # Plot and save the comparison
        fig = plot_comparison(results)
        fig.suptitle(f"Eigensolver Comparison for {name}")
        plt.savefig(f"eigensolver_comparison_{name}.png")
        plt.close(fig)

    print("\nComparison complete. Check the generated plots for details.")
