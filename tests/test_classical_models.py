"""Tests for classical neural network models."""

import jax
import jax.numpy as jnp
import pytest

from qedft.models.classical.classical_models import (
    build_batched_global_mlp,
    build_global_mlp,
    build_local_mlp,
    create_mlp_model,
)


@pytest.fixture
def test_grids():
    """Create a simple grid for testing."""
    return jnp.linspace(0.0, 1.0, 10)


@pytest.fixture
def test_density(test_grids):
    """Create a test density array."""
    return jnp.ones_like(test_grids)


@pytest.fixture
def test_batch_density():
    """Create a batch of test density arrays."""
    return jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


@pytest.fixture
def rng_key():
    """Create a JAX random key for testing."""
    return jax.random.PRNGKey(42)


def test_local_mlp(test_grids, test_density, rng_key):
    """Test local MLP initialization and application."""
    init_fn, apply_fn = build_local_mlp(
        n_neurons=32,
        n_layers=2,
        activation="tanh",
        grids=test_grids,
    )

    _, params = init_fn(rng_key, input_shape=None)
    output = apply_fn(params, test_density)

    assert output.shape == test_density.shape
    assert not jnp.isnan(output).any()


def test_global_mlp(test_grids, test_density, rng_key):
    """Test global MLP initialization and application."""
    init_fn, apply_fn = build_global_mlp(
        n_neurons=32,
        n_layers=2,
        activation="relu",
        grids=test_grids,
    )

    _, params = init_fn(rng_key, input_shape=None)
    output = apply_fn(params, test_density)

    assert output.shape == (1,)
    assert not jnp.isnan(output).any()


def test_batched_global_mlp(test_batch_density, rng_key):
    """Test batched global MLP initialization and application."""
    init_fn, apply_fn = build_batched_global_mlp(
        n_neurons=32,
        n_layers=2,
        activation="tanh",
        n_features=3,
    )

    _, params = init_fn(rng_key, input_shape=None)
    output = apply_fn(params, test_batch_density)

    assert output.shape == (2, 1)
    assert not jnp.isnan(output).any()


def test_create_mlp_model_local(test_grids, test_density, rng_key):
    """Test model factory with local MLP configuration."""
    config = {
        "network_type": "mlp",
        "n_neurons": 32,
        "n_layers": 2,
        "activation": "tanh",
    }

    init_fn, apply_fn = create_mlp_model(config, grids=test_grids)
    _, params = init_fn(rng_key, input_shape=None)
    output = apply_fn(params, test_density)

    assert output.shape == test_density.shape


def test_create_mlp_model_global(test_grids, test_density, rng_key):
    """Test model factory with global MLP configuration."""
    config = {
        "network_type": "mlp_ksr",
        "n_neurons": 32,
        "n_layers": 2,
        "activation": "relu",
    }

    init_fn, apply_fn = create_mlp_model(config, grids=test_grids)
    _, params = init_fn(rng_key, input_shape=None)
    output = apply_fn(params, test_density)

    assert output.shape == (1,)


def test_activation_functions(test_grids, test_density, rng_key):
    """Test different activation functions."""
    activations = ["tanh", "relu", "sigmoid", "elu", "gelu"]

    for activation in activations:
        init_fn, apply_fn = build_local_mlp(
            n_neurons=32,
            n_layers=2,
            activation=activation,
            grids=test_grids,
        )

        _, params = init_fn(rng_key, input_shape=None)
        output = apply_fn(params, test_density)

        assert output.shape == test_density.shape
        assert not jnp.isnan(output).any()


def test_invalid_activation():
    """Test error handling for invalid activation function."""
    with pytest.raises(ValueError):
        build_local_mlp(
            n_neurons=32,
            n_layers=2,
            activation="invalid_activation",
            grids=jnp.linspace(0, 1, 10),
        )


def test_invalid_network_type(test_grids):
    """Test error handling for invalid network type."""
    config = {
        "network_type": "invalid_type",
        "n_neurons": 32,
        "n_layers": 2,
    }

    with pytest.raises(ValueError):
        create_mlp_model(config, grids=test_grids)


if __name__ == "__main__":
    pytest.main([__file__])
