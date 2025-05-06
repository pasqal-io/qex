"""Test the amplitude encoding function."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qedft.models.quantum.quantum_models import amplitude_encode


@pytest.mark.parametrize(
    "n_qubits,input_shape",
    [
        (2, (4,)),  # Exact fit
        (3, (8,)),  # Exact fit
        (2, (2,)),  # Smaller than Hilbert space
        (3, (4,)),  # Smaller than Hilbert space
        (2, (6,)),  # Larger than Hilbert space
        (3, (10,)),  # Larger than Hilbert space
        (2, (2, 2)),  # 2D input
        (3, (2, 4)),  # 2D input
    ],
)
def test_amplitude_encode_shapes(n_qubits, input_shape):
    """Test that amplitude_encode produces correctly shaped outputs."""
    x = jnp.ones(input_shape)

    # Calculate the dimension of the Hilbert space
    hilbert_dim = 2**n_qubits
    # Get the data dimension
    data_dim = np.prod(input_shape)

    state = amplitude_encode(x, n_qubits, hilbert_dim, data_dim)

    # Check output shape is correct (2,) * n_qubits
    expected_shape = tuple([2] * n_qubits)
    assert state.shape == expected_shape

    # Check total size is 2^n_qubits
    assert np.prod(state.shape) == 2**n_qubits


def test_amplitude_encode_normalization():
    """Test that amplitude_encode correctly normalizes the output state."""
    # Test with various inputs
    test_cases = [
        jnp.array([1.0, 2.0, 3.0, 4.0]),  # Regular array
        jnp.array([0.1, 0.2]),  # Small values
        jnp.array([10.0, 20.0]),  # Large values
        jnp.zeros((4,)),  # All zeros
    ]

    for x in test_cases:
        n_qubits = 2  # 2 qubits = 4 amplitudes
        hilbert_dim = 2**n_qubits
        data_dim = x.size

        state = amplitude_encode(x, n_qubits, hilbert_dim, data_dim)

        # Flatten the state to compute norm
        flat_state = state.reshape(-1)
        norm = jnp.linalg.norm(flat_state)

        # Check if the state is normalized (allowing for numerical precision)
        if jnp.all(x == 0):
            # For all zeros input, output should be all zeros (unnormalized)
            # Should be all nans in a shape of the state
            assert jnp.all(jnp.isnan(flat_state))
            # assert jnp.all(flat_state == jnp.zeros_like(flat_state))
        else:
            # For non-zero inputs, output should be normalized
            assert jnp.isclose(norm, 1.0, atol=1e-6)


def test_amplitude_encode_padding():
    """Test that amplitude_encode correctly pads smaller inputs."""
    x = jnp.array([1.0, 2.0])
    n_qubits = 2  # 2 qubits = 4 amplitudes
    hilbert_dim = 2**n_qubits
    data_dim = x.size

    state = amplitude_encode(x, n_qubits, hilbert_dim, data_dim)
    flat_state = state.reshape(-1)

    # First two values should be proportional to input, rest should be zeros
    norm = jnp.linalg.norm(jnp.array([1.0, 2.0]))
    expected = jnp.array([1.0 / norm, 2.0 / norm, 0.0, 0.0])

    assert jnp.allclose(flat_state, expected, atol=1e-6)


def test_amplitude_encode_truncation():
    """Test that amplitude_encode correctly truncates larger inputs."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    n_qubits = 2  # 2 qubits = 4 amplitudes
    hilbert_dim = 2**n_qubits
    data_dim = x.size

    state = amplitude_encode(x, n_qubits, hilbert_dim, data_dim)
    flat_state = state.reshape(-1)

    # Only first 4 values should be used
    norm = jnp.linalg.norm(jnp.array([1.0, 2.0, 3.0, 4.0]))
    expected = jnp.array([1.0 / norm, 2.0 / norm, 3.0 / norm, 4.0 / norm])

    assert jnp.allclose(flat_state, expected, atol=1e-6)


def test_amplitude_encode_jit_compatibility():
    """Test that amplitude_encode works with JAX JIT compilation."""
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n_qubits = 2
    hilbert_dim = 2**n_qubits
    data_dim = x.size

    # Define a JIT-compiled version of amplitude_encode
    # We need to make n_qubits static for JIT to work with the reshape operation
    jitted_encode = jax.jit(
        lambda x: amplitude_encode(x, n_qubits, hilbert_dim, data_dim),
    )

    # Run both versions
    normal_result = amplitude_encode(x, n_qubits, hilbert_dim, data_dim)
    jitted_result = jitted_encode(x)

    # Results should be identical
    assert jnp.allclose(normal_result, jitted_result)


def test_amplitude_encode_with_random_data():
    """Test amplitude_encode with random data of different sizes."""
    key = jax.random.PRNGKey(42)

    for n_qubits in [2, 3, 4]:
        hilbert_dim = 2**n_qubits

        # Test with exact fit
        x = jax.random.normal(key, (hilbert_dim,))
        data_dim = x.size
        state = amplitude_encode(x, n_qubits, hilbert_dim, data_dim)
        assert state.shape == tuple([2] * n_qubits)

        # Test with smaller input
        key, subkey = jax.random.split(key)
        x_small = jax.random.normal(subkey, (hilbert_dim // 2,))
        data_dim_small = x_small.size
        state_small = amplitude_encode(x_small, n_qubits, hilbert_dim, data_dim_small)
        assert state_small.shape == tuple([2] * n_qubits)

        # Test with larger input
        key, subkey = jax.random.split(key)
        x_large = jax.random.normal(subkey, (hilbert_dim * 2,))
        data_dim_large = x_large.size
        state_large = amplitude_encode(x_large, n_qubits, hilbert_dim, data_dim_large)
        assert state_large.shape == tuple([2] * n_qubits)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
