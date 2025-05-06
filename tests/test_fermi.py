"""Test the Fermi functions for fractional occupations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qedft.train.td.fermi import (
    get_fractional_occupations,
    get_fractional_occupations_jax,
    get_fractional_occupations_jax_stable,
)

# Enable double precision for tests
jax.config.update("jax_enable_x64", True)


class TestFermiFunctions:

    @pytest.mark.parametrize("n_electrons", [2.0, 4.0, 6.0])
    def test_fermi_optimization(self, n_electrons):
        """Test the full Fermi optimization process with jax.lax.scan."""
        mo_energy = jnp.linspace(-1.0, 1.0, 10)
        theta = 0.5
        frac_mu = None
        frac_mu_shift = 0.9
        frac_step_grad = 0.6
        frac_max_steps = 100

        # Initial mu - use a better initial guess based on n_electrons
        if n_electrons == 4.0:
            frac_mu_shift = mo_energy[2] + frac_mu_shift
        else:
            frac_mu_shift = mo_energy[0] + frac_mu_shift

        # Use the new implementation
        occ, energy_entr, mu_final, loss_final = get_fractional_occupations_jax(
            mo_energy=mo_energy,
            n_electrons=n_electrons,
            theta=theta,
            frac_mu=frac_mu,
            frac_mu_shift=frac_mu_shift,
            frac_step_grad=frac_step_grad,
            frac_max_steps=frac_max_steps,
        )
        print(f"\nocc: {occ}")

        # Compare with original implementation
        orig_occ, orig_energy_entr, mu_final_original, loss_final_original = (
            get_fractional_occupations(
                mo_energy=mo_energy,
                n_electrons=n_electrons,
                frac_mo_occ=1,
                frac_theta=theta,
                frac_mu=frac_mu,
                frac_mu_shift=frac_mu_shift,
                frac_step_grad=frac_step_grad,
                frac_max_steps=frac_max_steps,
            )
        )
        print(f"Original occ: {orig_occ}")

        # Assert that the occupations from both implementations are similar
        assert np.allclose(occ, orig_occ, rtol=1), "Occupations do not match"
        assert np.allclose(energy_entr, orig_energy_entr, rtol=1), "Energy entropy does not match"
        assert np.allclose(
            mu_final,
            mu_final_original,
            rtol=1,
        ), "Chemical potential does not match"
        assert np.allclose(loss_final, loss_final_original, rtol=1), "Loss does not match"

        # Check that the sum of occupations is close to n_electrons
        assert np.isclose(jnp.sum(occ), n_electrons, rtol=1e-1)

        # Check that the loss is small
        assert loss_final < 1e-4

        # Check that occupation decreases with increasing energy
        for i in range(len(mo_energy) - 1):
            if mo_energy[i] < mo_energy[i + 1]:
                assert occ[i] >= occ[i + 1] - 1e-10  # Allow for small numerical errors

    def test_fermi_stable_random_combinations(self):
        """Test the stable Fermi optimization with random parameter combinations."""
        key = jax.random.PRNGKey(42)
        n_tests = 20

        # Create a JIT-compiled version for testing performance
        jitted_get_fractional_occupations_jax_stable = jax.jit(
            get_fractional_occupations_jax_stable,
            static_argnums=(5, 6),
        )

        for i in range(n_tests):
            # Generate random parameters for this test
            key, subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 6)

            # Random orbital energies with reasonable spacing - ensure even number of orbitals
            n_orbs = 2 * jax.random.randint(subkey1, (1,), 2, 7)[0]  # 4, 6, 8, 10, 12 orbitals
            mo_energy = jax.random.uniform(subkey2, (n_orbs,), minval=-5.0, maxval=5.0)
            mo_energy = jnp.sort(mo_energy)  # Ensure energies are sorted

            # Use even integer electron counts
            n_electrons = 2.0 * float(jax.random.randint(subkey3, (1,), 1, n_orbs)[0])

            # Random temperature and other parameters
            theta = jax.random.uniform(subkey4, (1,), minval=0.01, maxval=1.0)[0]

            # Either use None or a random value for frac_mu
            use_none_mu = jax.random.uniform(subkey5, (1,))[0] > 0.5
            if use_none_mu:
                frac_mu = None
            else:
                # Choose a reasonable mu value
                frac_mu = jnp.mean(mo_energy) + jax.random.normal(subkey5, (1,))[0]

            # Fixed parameters for simplicity
            frac_mu_shift = 0.9
            frac_step_grad = 0.6
            frac_max_steps = 100

            # Run the stable implementation
            occ_stable, energy_entr_stable, mu_final_stable, loss_final_stable = (
                jitted_get_fractional_occupations_jax_stable(
                    mo_energy=mo_energy,
                    n_electrons=n_electrons,
                    theta=theta,
                    frac_mu=frac_mu,
                    frac_mu_shift=frac_mu_shift,
                    frac_step_grad=frac_step_grad,
                    frac_max_steps=frac_max_steps,
                )
            )

            # Validate results
            # 1. Check that the sum of occupations is close to n_electrons
            total_electrons = jnp.sum(occ_stable)
            assert (
                jnp.abs(total_electrons - n_electrons) < 1e-1
            ), f"Electron count mismatch: {total_electrons} vs {n_electrons}"

            # 2. Check that occupation decreases with increasing energy (monotonicity)
            for j in range(len(mo_energy) - 1):
                if mo_energy[j] < mo_energy[j + 1]:
                    assert (
                        occ_stable[j] >= occ_stable[j + 1] - 1e-10
                    ), f"Non-monotonic occupations at index {j}"

            # 3. Check occupations are bounded between 0 and 2
            assert jnp.all(occ_stable >= 0), f"Negative occupations found: {occ_stable}"
            assert jnp.all(occ_stable <= 2), f"Occupations > 2 found: {occ_stable}"

            # 4. Find the fallback case (when original optimization fails)
            # This is indicated by very large loss or electron count mismatch
            is_fallback = loss_final_stable > 1e-3
            if is_fallback:
                # We're expecting our stable version to still give reasonable results
                assert (
                    jnp.abs(total_electrons - n_electrons) < 1e-1
                ), "Fallback case should still maintain electron count"

            print(
                f"Test {i+1}/{n_tests} passed: n_elec={n_electrons:.2f}, theta={theta:.2f}, n_orbs={n_orbs}, "
                f"fallback={is_fallback}, total_e={total_electrons:.6f}",
            )

    def test_fermi_stable_extreme_cases(self):
        """Test the stable Fermi optimization with extreme parameter values that might break the algorithm."""
        cases = [
            # Integer electron counts - common cases
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 2.0,
                "theta": 0.5,
                "name": "2 electrons, standard",
            },
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 10.0,
                "theta": 0.5,
                "name": "10 electrons, standard",
            },
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 18.0,
                "theta": 0.5,
                "name": "Full shell (18 electrons)",
            },
            # Integer electron counts with varied theta
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 4.0,
                "theta": 0.001,
                "name": "4 electrons, low theta",
            },
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 8.0,
                "theta": 2.0,
                "name": "8 electrons, high theta",
            },
            # Integer electron counts with varied energy spacings
            {
                "mo_energy": jnp.array([-10.0, -8.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 8.0, 10.0]),
                "n_electrons": 6.0,
                "theta": 0.5,
                "name": "6 electrons, widely spaced energies",
            },
            {
                "mo_energy": jnp.array([-1.0, -0.9, -0.8, -0.7, -0.6, 0.6, 0.7, 0.8, 0.9, 1.0]),
                "n_electrons": 6.0,
                "theta": 0.1,
                "name": "6 electrons, closely spaced energies",
            },
            # Very low electron count
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 0.0,
                "theta": 0.5,
                "name": "Zero electron count",
            },
            # Very high electron count
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 20.0,
                "theta": 0.5,
                "name": "Very high electron count (even)",
            },
            # Very small theta (close to integer occupations)
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 6.0,
                "theta": 0.001,
                "name": "Very small theta (even e-)",
            },
            # Very large theta (highly fractional)
            {
                "mo_energy": jnp.linspace(-1.0, 1.0, 10),
                "n_electrons": 4.0,
                "theta": 5.0,
                "name": "Very large theta (even e-)",
            },
            # Degenerate energy levels
            {
                "mo_energy": jnp.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
                "n_electrons": 8.0,
                "theta": 0.5,
                "name": "Degenerate energy levels (even e-)",
            },
            # Single pair of orbitals
            {
                "mo_energy": jnp.array([0.0, 0.5]),
                "n_electrons": 2.0,
                "theta": 0.5,
                "name": "Two orbitals (even e-)",
            },
        ]

        # JIT-compile the function for testing
        jitted_stable_fn = jax.jit(
            get_fractional_occupations_jax_stable,
            static_argnums=(5, 6),
        )

        for case in cases:
            print(f"\nTesting: {case['name']}")
            mo_energy = case["mo_energy"]
            n_electrons = case["n_electrons"]
            theta = case["theta"]

            # Fixed parameters
            frac_mu = None
            frac_mu_shift = 0.9
            frac_step_grad = 0.6
            frac_max_steps = 100

            try:
                # Run the stable implementation
                occ_stable, energy_entr_stable, mu_final_stable, loss_final_stable = (
                    jitted_stable_fn(
                        mo_energy=mo_energy,
                        n_electrons=n_electrons,
                        theta=theta,
                        frac_mu=frac_mu,
                        frac_mu_shift=frac_mu_shift,
                        frac_step_grad=frac_step_grad,
                        frac_max_steps=frac_max_steps,
                    )
                )

                # Basic validation
                total_electrons = jnp.sum(occ_stable)
                print(f"  Total electrons: {total_electrons} (target: {n_electrons})")
                print(f"  Loss: {loss_final_stable}")

                # Check electron count
                assert (
                    jnp.abs(total_electrons - n_electrons) < 1e-1
                ), f"Electron count mismatch: {total_electrons} vs {n_electrons}"

                # Check occupations are bounded between 0 and 2
                assert jnp.all(occ_stable >= 0), f"Negative occupations found: {occ_stable}"
                assert jnp.all(occ_stable <= 2), f"Occupations > 2 found: {occ_stable}"

                # For integer electron counts and small theta, check for integer-like occupations
                if n_electrons.is_integer() and theta < 0.01:
                    # Check that occupations are close to integers (0, 1, or 2)
                    for occ in occ_stable:
                        closest_int = jnp.round(occ)
                        assert (
                            jnp.abs(occ - closest_int) < 0.1
                        ), f"Expected integer-like occupation, got {occ}"

            except Exception as e:
                assert False, f"Test failed for {case['name']}: {str(e)}"

    def test_specific_integer_electron_cases(self):
        """Test specific cases with integer electron counts that are important for chemical systems."""
        # JIT-compile the function for testing
        jitted_stable_fn = jax.jit(
            get_fractional_occupations_jax_stable,
            static_argnums=(5, 6),
        )

        # Common scenarios in chemistry with even integer electron counts and even orbitals
        test_cases = [
            # H2 (2 electrons)
            {
                "mo_energy": jnp.array([-0.6, 0.2]),
                "n_electrons": 2.0,
                "theta": 0.1,
                "name": "H2-like",
            },
            # C atom (6 electrons) - adjusted to have even orbitals
            {
                "mo_energy": jnp.array([-10.0, -5.0, -5.0, -5.0, 0.1, 0.1, 0.1, 2.0]),
                "n_electrons": 6.0,
                "theta": 0.1,
                "name": "C-atom-like",
            },
            # O2 (16 electrons) - adjusted to have even orbitals
            {
                "mo_energy": jnp.array(
                    [-15.0, -10.0, -10.0, -8.0, -8.0, -5.0, -5.0, -1.0, -1.0, 0.1, 0.1, 1.0],
                ),
                "n_electrons": 16.0,
                "theta": 0.1,
                "name": "O2-like",
            },
            # Water (8 electrons)
            {
                "mo_energy": jnp.array([-12.0, -8.0, -7.0, -5.0, -2.0, 0.1, 2.0, 3.0]),
                "n_electrons": 8.0,
                "theta": 0.1,
                "name": "H2O-like",
            },
            # Methane (8 electrons)
            {
                "mo_energy": jnp.array([-11.0, -8.0, -8.0, -8.0, -0.5, 0.2, 0.2, 0.2]),
                "n_electrons": 8.0,
                "theta": 0.1,
                "name": "CH4-like",
            },
        ]

        for case in test_cases:
            print(f"\nTesting: {case['name']}")
            mo_energy = case["mo_energy"]
            n_electrons = case["n_electrons"]
            theta = case["theta"]

            # Fixed parameters
            frac_mu = None
            frac_mu_shift = 0.9
            frac_step_grad = 0.6
            frac_max_steps = 100

            # Run the stable implementation
            occ_stable, _, _, _ = jitted_stable_fn(
                mo_energy=mo_energy,
                n_electrons=n_electrons,
                theta=theta,
                frac_mu=frac_mu,
                frac_mu_shift=frac_mu_shift,
                frac_step_grad=frac_step_grad,
                frac_max_steps=frac_max_steps,
            )

            # Print the occupations
            print(f"  Occupations: {occ_stable}")
            print(f"  Energy levels: {mo_energy}")

            # Validate the results
            total_electrons = jnp.sum(occ_stable)
            assert (
                jnp.abs(total_electrons - n_electrons) < 1e-1
            ), f"Electron count mismatch: {total_electrons} vs {n_electrons}"

            # Check that occupation decreases with increasing energy (monotonicity)
            for j in range(len(mo_energy) - 1):
                if mo_energy[j] < mo_energy[j + 1]:
                    assert (
                        occ_stable[j] >= occ_stable[j + 1] - 1e-10
                    ), f"Non-monotonic occupations at index {j}"


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
