"""Tests for the wrappers module.

This module contains tests for the wrappers module, which provides wrappers for
neural network models to be used in DFT calculations. The tests verify:

- Neural network wrapper initialization and forward pass
- Local and global encoding schemes
- Exchange-correlation functional wrappers
- Network parameter initialization
- Density normalization and grid handling

The tests ensure proper integration of neural networks into the DFT framework
through the wrapper interfaces.
"""

import jax
import jax.numpy as jnp
import pytest

from qedft.models.classical.classical_models import build_global_mlp, build_local_mlp
from qedft.models.wrappers import neural_xc_functional, wrap_network


def setup_test_network(network_type="global"):
    """Helper function to create a test network and grids."""
    grids = jnp.linspace(0.0, 1.0, 513)  # Small grid for testing
    if network_type == "global":
        network = build_global_mlp(
            n_neurons=10,
            n_layers=2,
            activation="tanh",
            n_outputs=1,
            density_normalization_factor=2.0,
            grids=grids,
        )
    else:  # local
        network = build_local_mlp(
            n_neurons=10,
            n_layers=2,
            activation="tanh",
            n_outputs=1,
            density_normalization_factor=2.0,
            grids=grids,
        )
    return network, grids


def test_create_network_functional_local():
    """Test create_network_functional with local encoding."""
    network, grids = setup_test_network(network_type="local")

    init_fn, xc_fn = neural_xc_functional(
        network=network,
        grids=grids,
        network_type="mlp",
        encoding="local",
    )

    # Test initialization
    key = jax.random.PRNGKey(0)
    params = init_fn(key)

    # Test forward pass
    density = jnp.ones(grids.shape)
    result = xc_fn(density, params)

    assert result.shape == grids.shape, f"Expected shape {grids.shape}, got {result.shape}"


def test_create_network_functional_amplitude():
    """Test create_network_functional with amplitude encoding."""
    network, grids = setup_test_network()

    init_fn, xc_fn = neural_xc_functional(
        network=network,
        grids=grids,
        network_type="mlp_ksr",
        encoding="amplitude",
    )

    # Test initialization
    key = jax.random.PRNGKey(0)
    params = init_fn(key, input_shape=(grids.shape[0],))

    # Test forward pass
    density = jnp.ones(grids.shape)
    result = xc_fn(density, params)

    assert result.shape == (), f"Expected shape {1}, got {result.shape}"


def test_wrap_network_local():
    """Test wrap_network with local model."""
    network, grids = setup_test_network(network_type="local")

    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp",
        wrap_self_interaction=False,
    )

    # Test initialization
    key = jax.random.PRNGKey(0)
    params = init_fn(key)

    # Test forward pass
    density = jnp.ones(grids.shape)
    result = apply_fn(density, params)

    assert result.shape == grids.shape, f"Expected shape {grids.shape}, got {result.shape}"


def test_wrap_with_self_interaction_network_local():
    """Test wrap_network with local model and self-interaction."""
    network, grids = setup_test_network(network_type="local")

    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp",
        wrap_self_interaction=True,
    )

    # Test initialization
    key = jax.random.PRNGKey(0)
    params = init_fn(key)

    # Test forward pass
    density = jnp.ones(grids.shape)
    result = apply_fn(density, params)

    assert result.shape == grids.shape, f"Expected shape {grids.shape}, got {result.shape}"


def test_wrap_network_global_no_self_interaction():
    """Test wrap_network with global model."""
    network, grids = setup_test_network(network_type="global")

    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp_ksr",
        wrap_self_interaction=False,
        wrap_with_negative_transform=False,
    )

    # Test initialization
    key = jax.random.PRNGKey(0)
    params = init_fn(key)

    # Test forward pass
    density = jnp.ones(grids.shape)
    result = apply_fn(density, params)

    assert result.shape == (), f"Expected shape {1}, got {result.shape}"

    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp_ksr",
        wrap_self_interaction=False,
        wrap_with_negative_transform=True,
    )

    # Test initialization
    params = init_fn(key)
    result = apply_fn(density, params)

    assert result.shape == (), f"Expected shape {1}, got {result.shape}"


def test_wrap_network_global():
    """Test wrap_network with global model."""
    network, grids = setup_test_network(network_type="global")

    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp_ksr",
        wrap_self_interaction=True,
        wrap_with_negative_transform=True,
    )

    # Test initialization
    key = jax.random.PRNGKey(0)
    params = init_fn(key)

    # Test forward pass
    density = jnp.ones(grids.shape)
    result = apply_fn(density, params)

    assert result.shape == (), f"Expected shape {1}, got {result.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
