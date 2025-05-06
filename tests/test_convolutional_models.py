"""Test the convolutional models."""

import jax
import jax.numpy as jnp
import pytest
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

from qedft.models.quantum.convolutional_models import (
    build_conv_amplitude_encoding_qnn,
    build_conv_mlp,
    build_conv_qnn,
    compute_kernel_width_per_layer,
    construct_convolutional_model,
)


@pytest.fixture
def test_config():
    """Basic configuration for testing convolutional models."""
    return {
        "input_dimension": 64,
        "n_qubits": 4,
        "largest_kernel_width": 4,
        "n_var_layers": 2,
        "n_out": 1,
        "max_number_conv_layers": 2,
    }


def test_compute_kernel_width_per_layer(test_config):
    """Test that kernel width computation works correctly."""
    list_kernel_dimensions, list_outputs_per_conv_layer = compute_kernel_width_per_layer(
        input_dimension=test_config["input_dimension"],
        largest_kernel_dimension=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
    )

    # Check that we got some valid results
    assert len(list_kernel_dimensions) > 0
    assert len(list_outputs_per_conv_layer) > 0
    assert len(list_kernel_dimensions) == len(list_outputs_per_conv_layer)

    # Check that the dimensions make sense
    for kernel_dim, output_dim in zip(list_kernel_dimensions, list_outputs_per_conv_layer):
        assert kernel_dim > 0
        assert output_dim > 0
        assert kernel_dim <= test_config["largest_kernel_width"]


def test_construct_convolutional_model(test_config):
    """Test that model construction works."""
    list_conv_layers = construct_convolutional_model(
        n_qubits=test_config["n_qubits"],
        n_var_layers=test_config["n_var_layers"],
        n_out=test_config["n_out"],
        input_dimension=test_config["input_dimension"],
        largest_kernel_width=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
    )

    # Check that we got some layers
    assert len(list_conv_layers) > 0

    # Each layer should have an init_fn and apply_fn
    for layer in list_conv_layers:
        assert len(layer) == 2
        init_fn, apply_fn = layer
        assert callable(init_fn)
        assert callable(apply_fn)


def test_build_conv_qnn(test_config):
    """Test building a convolutional QNN."""
    init_fn, apply_fn = build_conv_qnn(
        n_qubits=test_config["n_qubits"],
        n_var_layers=test_config["n_var_layers"],
        n_out=test_config["n_out"],
        input_dimension=test_config["input_dimension"],
        largest_kernel_width=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
    )

    # Check that we got valid functions
    assert callable(init_fn)
    assert callable(apply_fn)

    # Test initialization and forward pass
    key = jax.random.PRNGKey(0)
    output_shape, params = init_fn(key, (test_config["input_dimension"],))

    # Create test input
    test_input = jnp.ones(test_config["input_dimension"])

    # Test forward pass
    output = apply_fn(params, test_input)

    # Output should be a scalar for each batch element
    assert output.ndim == 0 or (output.ndim == 1 and output.shape[0] == 1)


def test_build_conv_mlp(test_config):
    """Test building a convolutional MLP."""
    init_fn, apply_fn = build_conv_mlp(
        n_qubits=test_config["n_qubits"],
        n_var_layers=test_config["n_var_layers"],
        n_out=test_config["n_out"],
        input_dimension=test_config["input_dimension"],
        largest_kernel_width=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
    )

    # Check that we got valid functions
    assert callable(init_fn)
    assert callable(apply_fn)

    # Test initialization and forward pass
    key = jax.random.PRNGKey(0)
    output_shape, params = init_fn(key, (test_config["input_dimension"],))

    # Create test input
    test_input = jnp.ones(test_config["input_dimension"])

    # Test forward pass
    output = apply_fn(params, test_input)

    # Output should be a scalar for each batch element
    assert output.ndim == 0 or (output.ndim == 1 and output.shape[0] == 1)


def test_build_conv_amplitude_encoding_qnn(test_config):
    """Test building a convolutional QNN with amplitude encoding."""
    init_fn, apply_fn = build_conv_amplitude_encoding_qnn(
        n_qubits=test_config["n_qubits"],
        n_var_layers=test_config["n_var_layers"],
        n_out=test_config["n_out"],
        input_dimension=test_config["input_dimension"],
        largest_kernel_width=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
    )

    # Check that we got valid functions
    assert callable(init_fn)
    assert callable(apply_fn)

    # Test initialization and forward pass
    key = jax.random.PRNGKey(0)
    output_shape, params = init_fn(key, (test_config["input_dimension"],))

    # Create test input
    test_input = jnp.ones(test_config["input_dimension"])

    # Test forward pass
    output = apply_fn(params, test_input)

    # Output should be a scalar for each batch element
    assert output.ndim == 0 or (output.ndim == 1 and output.shape[0] == 1)


def test_amplitude_encoding_qnn_with_noise(test_config):
    """Test building a convolutional amplitude encoding QNN with noise."""
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.01),)

    init_fn, apply_fn = build_conv_amplitude_encoding_qnn(
        n_qubits=test_config["n_qubits"],
        n_var_layers=test_config["n_var_layers"],
        n_out=test_config["n_out"],
        input_dimension=test_config["input_dimension"],
        largest_kernel_width=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
        noise=noise,
    )

    # Test initialization and forward pass
    key = jax.random.PRNGKey(0)
    output_shape, params = init_fn(key, (test_config["input_dimension"],))

    # Create test input
    test_input = jnp.ones(test_config["input_dimension"])

    # Test forward pass
    output = apply_fn(params, test_input)

    # Output should be a scalar for each batch element
    assert output.ndim == 0 or (output.ndim == 1 and output.shape[0] == 1)


def test_amplitude_encoding_qnn_jit_compatibility(test_config):
    """Test that the amplitude encoding QNN is compatible with JAX JIT compilation."""
    init_fn, apply_fn = build_conv_amplitude_encoding_qnn(
        n_qubits=test_config["n_qubits"],
        n_var_layers=test_config["n_var_layers"],
        n_out=test_config["n_out"],
        input_dimension=test_config["input_dimension"],
        largest_kernel_width=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    output_shape, params = init_fn(key, (test_config["input_dimension"],))

    # Create test input
    test_input = jnp.ones(test_config["input_dimension"])

    # JIT compile the apply function
    jit_apply_fn = jax.jit(apply_fn)

    # Test that JIT compilation works
    jit_output = jit_apply_fn(params, test_input)
    normal_output = apply_fn(params, test_input)

    # Outputs should match
    assert jnp.allclose(jit_output, normal_output)


def test_noisy_conv_qnn(test_config):
    """Test building a convolutional QNN with noise."""
    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.01),)

    init_fn, apply_fn = build_conv_qnn(
        n_qubits=test_config["n_qubits"],
        n_var_layers=test_config["n_var_layers"],
        n_out=test_config["n_out"],
        input_dimension=test_config["input_dimension"],
        largest_kernel_width=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
        noise=noise,
    )

    # Test initialization and forward pass
    key = jax.random.PRNGKey(0)
    output_shape, params = init_fn(key, (test_config["input_dimension"],))

    # Create test input
    test_input = jnp.ones(test_config["input_dimension"])

    # Test forward pass
    output = apply_fn(params, test_input)

    # Output should be a scalar for each batch element
    assert output.ndim == 0 or (output.ndim == 1 and output.shape[0] == 1)


def test_jit_compatibility(test_config):
    """Test that the models are compatible with JAX JIT compilation."""
    init_fn, apply_fn = build_conv_qnn(
        n_qubits=test_config["n_qubits"],
        n_var_layers=test_config["n_var_layers"],
        n_out=test_config["n_out"],
        input_dimension=test_config["input_dimension"],
        largest_kernel_width=test_config["largest_kernel_width"],
        max_number_conv_layers=test_config["max_number_conv_layers"],
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    output_shape, params = init_fn(key, (test_config["input_dimension"],))

    # Create test input
    test_input = jnp.ones(test_config["input_dimension"])

    # JIT compile the apply function
    jit_apply_fn = jax.jit(apply_fn)

    # Test that JIT compilation works
    jit_output = jit_apply_fn(params, test_input)
    normal_output = apply_fn(params, test_input)

    # Outputs should match
    assert jnp.allclose(jit_output, normal_output)


def test_different_kernel_widths(test_config):
    """Test that models work with different kernel widths."""
    # Try with a smaller kernel width
    small_kernel_config = test_config.copy()
    small_kernel_config["largest_kernel_width"] = 2

    init_fn, apply_fn = build_conv_amplitude_encoding_qnn(
        n_qubits=small_kernel_config["n_qubits"],
        n_var_layers=small_kernel_config["n_var_layers"],
        n_out=small_kernel_config["n_out"],
        input_dimension=small_kernel_config["input_dimension"],
        largest_kernel_width=small_kernel_config["largest_kernel_width"],
        max_number_conv_layers=small_kernel_config["max_number_conv_layers"],
    )

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    output_shape, params = init_fn(key, (small_kernel_config["input_dimension"],))

    # Create test input
    test_input = jnp.ones(small_kernel_config["input_dimension"])

    # Test forward pass
    output = apply_fn(params, test_input)

    # Output should be a scalar for each batch element
    assert output.ndim == 0 or (output.ndim == 1 and output.shape[0] == 1)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
