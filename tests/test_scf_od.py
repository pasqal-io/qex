"""Tests for self-consistent field calculations.

This module tests the implementation of orbital-dependent Kohn-Sham DFT calculations,
including:
- Exchange-correlation energy and potential calculations
- Kohn-Sham iterations and convergence
- State validation and properties
"""

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from jax import config, tree_util
from jax_dft import utils

from qedft.train.od import scf as scf_od

# Enable double precision
config.update("jax_enable_x64", True)


class ScfOdTest(parameterized.TestCase):
    """Tests for SCF calculations.

    Tests the core functionality of the SCF implementation,
    including energy calculations, potential evaluations, and convergence behavior.
    """

    def setUp(self):
        """Sets up common test parameters."""
        super().setUp()
        self.grids = jnp.linspace(-5, 5, 101)
        self.num_electrons = 2
        self.locations = jnp.array([-0.5, 0.5])
        self.nuclear_charges = jnp.array([1, 1])

    def test_get_xc_energy_amplitude_encoded(self):
        """Tests exchange-correlation energy calculation with amplitude encoding.

        Uses 3D LDA exchange functional (E_x = -0.73855 ∫ n^(4/3) dx) with a
        Gaussian test density to verify energy calculations.
        """
        grids = jnp.linspace(-5, 5, 10001)

        def xc_energy_density_fn(density):
            """Computes exchange energy density using LDA functional."""
            return jnp.sum(-0.73855 * density ** (1 / 3))

        density = jnp.exp(-((grids - 1) ** 2))

        self.assertAlmostEqual(
            float(
                scf_od.get_xc_energy_amplitude_encoded(
                    density=density,
                    xc_energy_density_fn=xc_energy_density_fn,
                    grids=grids,
                ),
            ),
            -2.2660978689082367,  # Reference value for given density
            places=5,
        )

    def test_get_xc_potential_amplitude_encoded(self):
        """Tests exchange-correlation potential calculation with amplitude encoding.

        Verifies that the functional derivative of the XC energy yields the
        correct XC potential for the LDA functional.
        """
        grids = jnp.linspace(-5, 5, 10001)

        def xc_energy_density_fn(density):
            """Computes exchange energy density using LDA functional."""
            return jnp.sum(-0.73855 * density ** (1 / 3))

        density = jnp.exp(-((grids - 1) ** 2))

        # XC potential should be δE_xc/δn = -0.73855 * (4/3) * n^(1/3)
        np.testing.assert_allclose(
            scf_od.get_xc_potential_amplitude_encoded(
                density=density,
                xc_energy_density_fn=xc_energy_density_fn,
                grids=grids,
            ),
            -0.73855 * (1 / 3) * density ** (-2 / 3) * utils.get_dx(grids),
        )

    def _test_state(self, state, external_potential):
        """Validates properties of a KohnShamState.

        Args:
            state: KohnShamState to validate
            external_potential: Expected external potential for comparison

        Tests normalization, energy finiteness, array shapes, and state consistency.
        """
        # Check density normalization
        self.assertAlmostEqual(
            float(jnp.sum(state.density) * utils.get_dx(self.grids)),
            self.num_electrons,
        )

        # Verify energy is well-defined
        self.assertTrue(jnp.isfinite(state.total_energy))

        # Validate potential array dimensions
        self.assertLen(state.hartree_potential, len(state.grids))
        self.assertLen(state.xc_potential, len(state.grids))

        # Verify state consistency
        np.testing.assert_allclose(state.locations, self.locations)
        np.testing.assert_allclose(state.nuclear_charges, self.nuclear_charges)
        np.testing.assert_allclose(state.external_potential, external_potential)
        np.testing.assert_allclose(state.grids, self.grids)
        self.assertEqual(state.num_electrons, self.num_electrons)
        self.assertGreater(state.gap, 0)

    @parameterized.parameters(utils.soft_coulomb, utils.exponential_coulomb)
    def test_kohn_sham_iteration_amplitude_encoded(self, interaction_fn):
        """Tests single iteration of Kohn-Sham calculation.

        Args:
            interaction_fn: Electron-electron interaction potential function
        """
        initial_state = self._create_initial_state(interaction_fn)

        def xc_energy_density_fn(density):
            """Regularized LDA exchange functional."""
            return -0.73855 * (density + 1e-10) ** (1 / 3)

        next_state = scf_od.kohn_sham_iteration_amplitude_encoded(
            state=initial_state,
            num_electrons=self.num_electrons,
            xc_energy_density_fn=tree_util.Partial(xc_energy_density_fn),
            interaction_fn=interaction_fn,
            enforce_reflection_symmetry=False,
        )

        self._test_state(
            next_state,
            self._create_external_potential(interaction_fn),
        )

    def _create_initial_state(self, interaction_fn):
        """Creates initial KohnShamState for testing.

        Args:
            interaction_fn: Electron-electron interaction potential function

        Returns:
            KohnShamState: Initial state with Gaussian density distribution
        """
        return scf_od.KohnShamState(
            density=self.num_electrons
            * utils.gaussian(
                grids=self.grids,
                center=0.0,
                sigma=1.0,
            ),
            total_energy=jnp.inf,
            locations=self.locations,
            nuclear_charges=self.nuclear_charges,
            external_potential=self._create_external_potential(interaction_fn),
            grids=self.grids,
            num_electrons=self.num_electrons,
        )

    def _create_external_potential(self, interaction_fn):
        """Creates external potential for testing.

        Args:
            interaction_fn: Electron-electron interaction potential function

        Returns:
            Array: External potential on simulation grid
        """
        return utils.get_atomic_chain_potential(
            grids=self.grids,
            locations=self.locations,
            nuclear_charges=self.nuclear_charges,
            interaction_fn=interaction_fn,
        )

    @parameterized.parameters(
        (-1.0, [False, False, False]),
        (jnp.inf, [True, True, True]),
    )
    def test_kohn_sham_amplitude_encoded_convergence(
        self,
        density_mse_converge_tolerance,
        expected_converged,
    ):
        """Tests convergence behavior of Kohn-Sham calculation.

        Args:
            density_mse_converge_tolerance: Convergence threshold for density
            expected_converged: Expected convergence status for each iteration
        """

        def xc_energy_density_fn(density):
            """LDA exchange functional."""
            return -0.73855 * density ** (1 / 3)

        state = scf_od.kohn_sham_amplitude_encoded(
            locations=self.locations,
            nuclear_charges=self.nuclear_charges,
            num_electrons=self.num_electrons,
            num_iterations=3,
            grids=self.grids,
            xc_energy_density_fn=tree_util.Partial(xc_energy_density_fn),
            interaction_fn=utils.exponential_coulomb,
            density_mse_converge_tolerance=density_mse_converge_tolerance,
        )

        np.testing.assert_allclose(state.converged, expected_converged)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
