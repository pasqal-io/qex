"""Tests for hardware_ansatz.py."""

import jax.numpy as jnp
import pytest
from horqrux import zero_state
from horqrux.apply import apply_gates
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
from horqrux.primitives.parametric import RX, RY, RZ

from qedft.models.quantum.hardware_ansatz import advanced_hea, hea, hea_legacy


@pytest.fixture
def basic_circuit_params():
    """Fixture providing basic circuit parameters."""
    return {
        "n_qubits": 2,
        "n_layers": 3,
    }


def test_hea_creates_valid_circuit(basic_circuit_params):
    """Test that hea creates a valid circuit with expected properties."""
    n_qubits = basic_circuit_params["n_qubits"]
    n_layers = basic_circuit_params["n_layers"]

    ansatz = hea(n_qubits, n_layers)

    # Check that we have gates
    assert len(ansatz) > 0

    # Test circuit execution
    state = zero_state(n_qubits)
    params = {op.param: 0.01 for op in ansatz if hasattr(op, "param")}
    new_state = apply_gates(state, ansatz, params)

    # Should be a pure state with correct dimensions
    assert new_state.shape == (2,) * n_qubits


def test_hea_with_noise(basic_circuit_params):
    """Test that hea with noise creates a valid noisy circuit."""
    n_qubits = basic_circuit_params["n_qubits"]
    n_layers = basic_circuit_params["n_layers"]

    # Create noisy circuit
    noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.001),)
    noisy_ansatz = hea(n_qubits, n_layers, noise=noise)

    # Test circuit execution
    state = zero_state(n_qubits)
    params = {op.param: 0.01 for op in noisy_ansatz if hasattr(op, "param")}
    noisy_state = apply_gates(state, noisy_ansatz, params)

    # Should be a density matrix with correct dimensions for noisy circuit
    assert noisy_state.array.shape == (2, 2) * n_qubits


def test_different_noise_levels_produce_different_states(basic_circuit_params):
    """Test that different noise levels produce different quantum states."""
    n_qubits = basic_circuit_params["n_qubits"]
    n_layers = basic_circuit_params["n_layers"]

    # Create initial state
    state = zero_state(n_qubits)

    # Create circuits with different noise levels
    noise1 = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.001),)
    noise2 = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.1),)

    noisy_ansatz1 = hea(n_qubits, n_layers, noise=noise1)
    noisy_ansatz2 = hea(n_qubits, n_layers, noise=noise2)

    # Use same parameters for both circuits
    params1 = {op.param: 0.01 for op in noisy_ansatz1 if hasattr(op, "param")}
    params2 = {op.param: 0.01 for op in noisy_ansatz2 if hasattr(op, "param")}

    new_state1 = apply_gates(state, noisy_ansatz1, params1)
    new_state2 = apply_gates(state, noisy_ansatz2, params2)

    # States should be different due to different noise levels
    assert not jnp.allclose(new_state1.array, new_state2.array)


def test_hea_and_legacy_have_same_gate_count(basic_circuit_params):
    """Test that hea and hea_legacy have the same number of gates."""
    n_qubits = basic_circuit_params["n_qubits"]
    n_layers = basic_circuit_params["n_layers"]

    legacy_ansatz = hea_legacy(n_qubits, n_layers)
    new_ansatz = hea(n_qubits, n_layers)

    # Count parametrized gates
    rotation_gates_legacy = [op for op in legacy_ansatz if hasattr(op, "param")]
    rotation_gates = [op for op in new_ansatz if hasattr(op, "param")]
    assert len(rotation_gates) == len(rotation_gates_legacy)

    # Count non-parametrized gates
    no_params_legacy = [op for op in legacy_ansatz if not hasattr(op, "param")]
    no_params = [op for op in new_ansatz if not hasattr(op, "param")]
    assert len(no_params) == len(no_params_legacy)

    # Total gate count should be the same
    assert len(legacy_ansatz) == len(new_ansatz)


def test_advanced_hea_creates_valid_circuit(basic_circuit_params):
    """Test that advanced_hea creates a valid circuit with expected properties."""
    n_qubits = basic_circuit_params["n_qubits"]
    n_layers = basic_circuit_params["n_layers"]

    # Test with different entangling block types
    for block_type in ["circular", "linear", "full", "pairwise"]:
        ansatz = advanced_hea(n_qubits, n_layers, entangling_block_type=block_type)

        # Check that we have gates
        assert len(ansatz) > 0

        # Test circuit execution
        state = zero_state(n_qubits)
        params = {op.param: 0.01 for op in ansatz if hasattr(op, "param")}
        new_state = apply_gates(state, ansatz, params)

        # Should be a pure state with correct dimensions
        assert new_state.shape == (2,) * n_qubits


def test_custom_rotations():
    """Test that custom rotation functions work correctly."""
    n_qubits = 2
    n_layers = 2

    # Test with custom rotation functions
    custom_rots = [RZ, RX, RY]
    ansatz = hea(n_qubits, n_layers, rot_fns=custom_rots)

    # Count the number of each type of rotation gate
    rx_count = sum(1 for op in ansatz if hasattr(op, "name") and op.name == "RX")
    ry_count = sum(1 for op in ansatz if hasattr(op, "name") and op.name == "RY")
    rz_count = sum(1 for op in ansatz if hasattr(op, "name") and op.name == "RZ")

    # With [RZ, RX, RZ], we should have RX and RZ gates and RY gates
    assert rx_count > 0
    assert rz_count > 0
    assert ry_count > 0


def test_custom_parameter_prefix():
    """Test that custom parameter prefix works correctly."""
    n_qubits = 2
    n_layers = 2
    custom_prefix = "theta_"

    ansatz = hea(n_qubits, n_layers, variational_param_prefix=custom_prefix)

    # Check that all parameters have the correct prefix
    for op in ansatz:
        if hasattr(op, "param"):
            assert op.param.startswith(custom_prefix)


def test_advanced_hea_with_noise():
    """Test that advanced_hea with noise creates a valid noisy circuit."""
    n_qubits = 2
    n_layers = 2

    # Create noisy circuit
    noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.01),)
    noisy_ansatz = advanced_hea(n_qubits, n_layers, noise=noise)

    # Test circuit execution
    state = zero_state(n_qubits)
    params = {op.param: 0.01 for op in noisy_ansatz if hasattr(op, "param")}
    noisy_state = apply_gates(state, noisy_ansatz, params)

    # Should be a density matrix with correct dimensions for noisy circuit
    assert noisy_state.array.shape == (2, 2) * n_qubits


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
