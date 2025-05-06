"""Tests for quantum measurement functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from horqrux import QuantumCircuit, X, apply_gates, expectation, random_state, zero_state
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
from horqrux.primitives.parametric import RX

from qedft.models.quantum.measurement import (
    qubit_magnetization,
    total_magnetization,
    total_magnetization_ops,
    total_magnetization_via_inner_product,
)


@pytest.fixture
def zero_state_2qubits():
    """Return a 2-qubit zero state |00⟩."""
    return zero_state(2)


@pytest.fixture
def random_state_4qubits():
    """Return a 4-qubit random state with fixed seed for reproducibility."""
    return random_state(4)


def test_qubit_magnetization_zero_state(zero_state_2qubits):
    """Test qubit magnetization on zero state."""
    mag = qubit_magnetization(zero_state_2qubits)

    # In |00⟩ state, both qubits should have +1 magnetization
    assert mag.shape == (2,)
    np.testing.assert_allclose(mag, jnp.array([1.0, 1.0]), rtol=1e-5)


def test_qubit_magnetization_after_x_gate(zero_state_2qubits):
    """Test qubit magnetization after applying X gate to first qubit."""
    # Apply X gate to first qubit to get |10⟩ state
    flipped_state = apply_gates(zero_state_2qubits, X(0))
    mag = qubit_magnetization(flipped_state)

    # First qubit should have -1 magnetization, second still +1
    assert mag.shape == (2,)
    np.testing.assert_allclose(mag, jnp.array([-1.0, 1.0]), rtol=1e-5)


def test_total_magnetization(zero_state_2qubits):
    """Test total magnetization calculation."""
    # For |00⟩ state, total magnetization should be 2
    mag = total_magnetization(zero_state_2qubits)
    assert mag.shape == (1,)
    np.testing.assert_allclose(mag, jnp.array([2.0]), rtol=1e-5)

    # Test with n_out=2 (split output)
    mag_split = total_magnetization(zero_state_2qubits, n_out=2)
    assert mag_split.shape == (2,)
    np.testing.assert_allclose(mag_split, jnp.array([1.0, 1.0]), rtol=1e-5)


def test_total_magnetization_via_inner_product(zero_state_2qubits):
    """Test total magnetization calculation via inner product method."""
    mag_fn = total_magnetization_via_inner_product(2)
    mag = mag_fn(zero_state_2qubits, {})

    # For |00⟩ state, total magnetization should be 2
    np.testing.assert_allclose(mag, 2.0, rtol=1e-5)


def test_total_magnetization_ops():
    """Test creation of magnetization operators."""
    ops = total_magnetization_ops(3)

    # Should return 3 operators (one Z for each qubit)
    assert len(ops) == 3


def test_magnetization_methods_consistency(random_state_4qubits):
    """Test that different magnetization methods give consistent results."""
    # Method 1: Direct total magnetization
    mag_direct = total_magnetization(random_state_4qubits, n_out=1)

    # Method 2: Inner product approach
    mag_fn = total_magnetization_via_inner_product(4)
    mag_inner = mag_fn(random_state_4qubits, {})

    # Method 3: Using Pauli Z operators with expectation values
    z_ops = total_magnetization_ops(4)
    gates = [RX("x", 0), RX("y", 1)]
    circuit = QuantumCircuit(4, gates)  # Empty circuit
    values = {"x": jnp.array([0.0]), "y": jnp.array([0.0])}

    mag_expectation = expectation(
        random_state_4qubits,
        circuit,
        z_ops,
        values,
        key=jax.random.PRNGKey(0),
    )

    # All methods should give the same result
    np.testing.assert_allclose(mag_direct[0], mag_inner, rtol=1e-5)
    np.testing.assert_allclose(
        mag_direct[0],
        jnp.sum(mag_expectation),
        rtol=1e-5,
    )


def test_noise_effects():
    """Test the effects of noise on magnetization measurements."""
    # Create noise instances
    low_noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.0),)
    high_noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.01),)

    # Create circuits with different noise levels
    gates_no_noise = [RX("x", 0), RX("y", 1)]
    gates_low_noise = [RX("x", 0, noise=low_noise), RX("y", 1, noise=low_noise)]
    gates_high_noise = [RX("x", 0, noise=high_noise), RX("y", 1, noise=high_noise)]

    circuit_no_noise = QuantumCircuit(4, gates_no_noise)
    circuit_low_noise = QuantumCircuit(4, gates_low_noise)
    circuit_high_noise = QuantumCircuit(4, gates_high_noise)

    # Create measurement operators
    z_ops = total_magnetization_ops(4)

    # Parameters for parametric gates
    values = {"x": jnp.array([0.0]), "y": jnp.array([0.0])}

    # Test with zero state
    state = zero_state(4)
    key = jax.random.PRNGKey(0)

    mag_no_noise = expectation(state, circuit_no_noise, z_ops, values, key=key)
    mag_low_noise = expectation(state, circuit_low_noise, z_ops, values, key=key)
    mag_high_noise = expectation(state, circuit_high_noise, z_ops, values, key=key)

    # Zero noise should match no noise
    np.testing.assert_allclose(jnp.sum(mag_no_noise), jnp.sum(mag_low_noise), rtol=1e-5)

    # High noise should be different from no noise
    assert not np.allclose(jnp.sum(mag_no_noise), jnp.sum(mag_high_noise), rtol=1e-5)


def test_jax_transformations(random_state_4qubits):
    """Test that magnetization functions work with JAX transformations."""
    # Test gradient of total magnetization
    grad_fn = jax.grad(lambda s: total_magnetization(s, n_out=1)[0])
    grad = grad_fn(random_state_4qubits)

    # Gradient should have same shape as input state
    assert grad.shape == random_state_4qubits.shape

    # Test vmap with qubit magnetization
    batch_size = 3
    batch_states = jnp.stack([random_state_4qubits for _ in range(batch_size)])
    batch_mag_fn = jax.vmap(qubit_magnetization, in_axes=0)
    batch_mag = batch_mag_fn(batch_states)

    # Should return magnetization for each qubit in each batch element
    assert batch_mag.shape == (batch_size, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
