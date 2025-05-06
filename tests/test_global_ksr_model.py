"""Tests for the Kohn-Sham Response (KSR) model using pytest."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest

from qedft.models.classical.global_ksr_model import create_ksr_model_from_config


@pytest.fixture
def test_grid():
    """Create a small grid for testing."""
    return np.linspace(0.0, 1.0, 513)


@pytest.fixture
def test_config():
    """Default test configuration."""
    return {
        "num_global_filters": 8,
        "num_local_filters": 8,
        "num_local_conv_layers": 2,
        "activation": "swish",
        "minval": 0.1,
        "maxval": 2.0,
        "downsample_factor": 0,
    }


@pytest.fixture
def random_key():
    """Create a fixed random key for reproducibility."""
    return random.PRNGKey(42)


def test_model_creation(test_grid):
    """Test that the model can be created with default config."""
    network, init_fn, apply_fn = create_ksr_model_from_config(test_grid)
    assert network is not None
    assert init_fn is not None
    assert apply_fn is not None


def test_model_creation_with_custom_config(test_grid, test_config):
    """Test that the model can be created with custom config."""
    network, init_fn, apply_fn = create_ksr_model_from_config(
        test_grid,
        test_config,
    )
    assert network is not None
    assert init_fn is not None
    assert apply_fn is not None


def test_parameter_initialization(test_grid, test_config, random_key):
    """Test parameter initialization."""
    _, init_fn, _ = create_ksr_model_from_config(test_grid, test_config)
    params = init_fn(random_key)
    assert params is not None

    # Verify we can flatten the parameters
    from jax_dft import np_utils

    spec, flattened_params = np_utils.flatten(params)
    assert isinstance(flattened_params, np.ndarray)
    assert len(flattened_params) > 0


def test_model_application(test_grid, test_config, random_key):
    """Test applying the model to inputs."""
    _, init_fn, apply_fn = create_ksr_model_from_config(test_grid, test_config)
    params = init_fn(random_key)

    # Create test input (density)
    test_input = jnp.ones_like(test_grid)

    # Apply model
    output = apply_fn(test_input, params)

    # Check output shape matches input shape
    assert output.shape == test_input.shape


@pytest.mark.parametrize("size", [2**4 + 1, 2**5 + 1, 2**6 + 1])
def test_model_with_different_grid_sizes(size, test_config, random_key):
    """Test model with different grid sizes."""
    # Grid sizes that are powers of 2 plus 1 only allowed
    grid = np.linspace(0.0, 1.0, size)
    _, init_fn, apply_fn = create_ksr_model_from_config(grid, test_config)
    params = init_fn(random_key)

    # Create test input
    test_input = jnp.ones(size)

    # Apply model
    output = apply_fn(test_input, params)

    # Check output shape
    assert output.shape == (size,)


@pytest.mark.parametrize("size", [2**4, 2**5, 2**6])
def test_model_with_wrong_grid_sizes(size, test_config, random_key):
    """Test model with different grid sizes."""
    # Grid sizes that are not powers of 2 plus 1 only allowed
    grid = np.linspace(0.0, 1.0, size)
    with pytest.raises(ValueError):
        _, init_fn, apply_fn = create_ksr_model_from_config(grid, test_config)


def test_model_gradients(test_grid, test_config, random_key):
    """Test that gradients can be computed through the model."""
    _, init_fn, apply_fn = create_ksr_model_from_config(test_grid, test_config)
    params = init_fn(random_key)

    # Create test input
    test_input = jnp.ones_like(test_grid)

    # Define a loss function (mean of outputs)
    def loss_fn(params):
        outputs = apply_fn(test_input, params)
        return jnp.mean(outputs)

    # Compute gradients
    grads = jax.grad(loss_fn)(params)

    # Check that gradients exist
    assert grads is not None


def test_model_with_random_inputs(test_grid, test_config, random_key):
    """Test model with random inputs."""
    _, init_fn, apply_fn = create_ksr_model_from_config(test_grid, test_config)
    params = init_fn(random_key)

    # Generate random inputs
    input_key = random.PRNGKey(123)
    random_input = random.uniform(
        input_key,
        shape=(len(test_grid),),
        minval=0.0,
        maxval=1.0,
    )

    # Apply model
    output = apply_fn(random_input, params)

    # Check output shape
    assert output.shape == random_input.shape


if __name__ == "__main__":
    pytest.main([__file__])
