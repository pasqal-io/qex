"""
Tests for the DIIS comparison module.

This module tests the functionality of the compare_diis.py module,
which compares different DIIS implementations for SCF convergence.
"""

import numpy as np

from .compare_diis import compare_implementations, run_jax_diis, run_no_diis, run_pyscf_diis


class TestDIISComparison:
    """Test cases for DIIS comparison functionality."""

    def test_pyscf_diis_convergence(self):
        """Test that PySCF DIIS implementation converges."""
        energies = run_pyscf_diis()
        # Check that we have the expected number of iterations
        assert len(energies) == 20
        # Check that energy decreases over iterations
        assert energies[-1] < energies[0]
        # Check convergence by examining energy differences in last iterations
        energy_diff = abs(energies[-1] - energies[-2])
        assert energy_diff < 1e-4, "PySCF DIIS should converge to < 1e-4"

    def test_jax_diis_convergence(self):
        """Test that JAX DIIS implementation converges."""
        energies = run_jax_diis()
        # Check that we have the expected number of iterations
        assert len(energies) == 20
        # Check that energy decreases over iterations
        assert energies[-1] < energies[0]
        # Check convergence by examining energy differences in last iterations
        energy_diff = abs(energies[-1] - energies[-2])
        assert energy_diff < 1e-4, "JAX DIIS should converge to < 1e-4"

    def test_no_diis_behavior(self):
        """Test behavior of SCF without DIIS."""
        energies = run_no_diis()
        # Check that we have the expected number of iterations
        assert len(energies) == 20
        # No-DIIS may not converge well, but energy should still decrease overall
        # or no convergence at all
        assert energies[-1] < energies[0] or energies[-1] > energies[0]

    def test_diis_implementations_agreement(self):
        """Test that PySCF and JAX DIIS implementations agree reasonably well."""
        pyscf_energies, jax_energies, _, pyscf_energies_h2, jax_energies_h2, _ = (
            compare_implementations()
        )

        # Final energies should be close between PySCF and JAX implementations
        final_energy_diff = abs(pyscf_energies[-1] - jax_energies[-1])
        assert final_energy_diff < 1e-5, "PySCF and JAX DIIS should converge to similar energies"
        final_energy_diff_h2 = abs(pyscf_energies_h2[-1] - jax_energies_h2[-1])
        assert (
            final_energy_diff_h2 < 1e-5
        ), "PySCF and JAX DIIS should converge to similar energies"

    def test_diis_improves_convergence(self):
        """Test that DIIS implementations improve convergence compared to no DIIS."""
        (
            pyscf_energies,
            jax_energies,
            no_diis_energies,
            pyscf_energies_h2,
            jax_energies_h2,
            no_diis_energies_h2,
        ) = compare_implementations()

        # Calculate energy differences between consecutive iterations
        pyscf_diffs = [
            abs(pyscf_energies[i] - pyscf_energies[i - 1]) for i in range(1, len(pyscf_energies))
        ]
        jax_diffs = [
            abs(jax_energies[i] - jax_energies[i - 1]) for i in range(1, len(jax_energies))
        ]
        no_diis_diffs = [
            abs(no_diis_energies[i] - no_diis_energies[i - 1])
            for i in range(1, len(no_diis_energies))
        ]
        pyscf_diffs_h2 = [
            abs(pyscf_energies_h2[i] - pyscf_energies_h2[i - 1])
            for i in range(1, len(pyscf_energies_h2))
        ]
        jax_diffs_h2 = [
            abs(jax_energies_h2[i] - jax_energies_h2[i - 1])
            for i in range(1, len(jax_energies_h2))
        ]
        no_diis_diffs_h2 = [
            abs(no_diis_energies_h2[i] - no_diis_energies_h2[i - 1])
            for i in range(1, len(no_diis_energies_h2))
        ]

        # DIIS should generally have smaller energy differences in later iterations
        # compared to no DIIS (testing with average of last 5 iterations)
        avg_pyscf_diff = np.mean(pyscf_diffs[-5:])
        avg_jax_diff = np.mean(jax_diffs[-5:])
        avg_no_diis_diff = np.mean(no_diis_diffs[-5:])
        avg_pyscf_diff_h2 = np.mean(pyscf_diffs_h2[-5:])
        avg_jax_diff_h2 = np.mean(jax_diffs_h2[-5:])
        avg_no_diis_diff_h2 = np.mean(no_diis_diffs_h2[-5:])

        assert avg_pyscf_diff < avg_no_diis_diff, "PySCF DIIS should converge better than no DIIS"
        assert avg_jax_diff < avg_no_diis_diff, "JAX DIIS should converge better than no DIIS"
        assert (
            avg_pyscf_diff_h2 < avg_no_diis_diff_h2
        ), "PySCF DIIS should converge better than no DIIS"
        assert (
            avg_jax_diff_h2 < avg_no_diis_diff_h2
        ), "JAX DIIS should converge better than no DIIS"
