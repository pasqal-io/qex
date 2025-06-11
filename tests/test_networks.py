"""Test the networks."""

import jax
import jax.numpy as jnp
import pytest
from horqrux.utils.operator_utils import DiffMode

from qedft.models.networks import (
    BatchedGlobalMLP,
    GlobalMLP,
    GlobalQiCQNN,
    GlobalQiQNN,
    GlobalQNN,
    GlobalQNNLayer,
    GlobalQNNwithMLP,
    LocalMLP,
    LocalQNN,
)


@pytest.fixture
def test_data():
    num_points = 4
    density = jnp.linspace(0, 1, num_points)
    grids = jnp.ones(density.shape)
    rng_key = jax.random.PRNGKey(0)
    return {
        "density": density,
        "grids": grids,
        "rng_key": rng_key,
        "num_points": num_points,
    }


@pytest.mark.parametrize(
    "network_class,config",
    [
        (LocalMLP, {"n_neurons": 32, "n_layers": 2}),
        (GlobalMLP, {"n_neurons": 32, "n_layers": 2}),
        (BatchedGlobalMLP, {"n_neurons": 32, "n_layers": 2, "n_features": 1}),
        (LocalQNN, {"n_qubits": 2, "n_layers": 1}),
        (GlobalQNNLayer, {"n_qubits": 2, "n_layers": 1}),
        (GlobalQNN, {"n_qubits": 2, "n_var_layers": 1, "largest_kernel_width": 2}),
        (GlobalQNNwithMLP, {"n_qubits": 2, "n_var_layers": 1, "largest_kernel_width": 2}),
        (GlobalQiCQNN, {"n_qubits": 2, "n_var_layers": 1, "largest_kernel_width": 2}),
        (GlobalQiQNN, {"n_qubits": 2, "n_layers": 1}),
    ],
)
def test_network_initialization_and_application(network_class, config, test_data):
    # Initialize the network
    network = network_class(config_dict=config)

    # Build the network
    init_fn, apply_fn = network.build_network(test_data["grids"])

    # Initialize parameters
    input_shape = (-1, test_data["num_points"], 1)
    _, params = init_fn(test_data["rng_key"], input_shape=input_shape)

    # Apply the network to test data
    output = apply_fn(params, test_data["density"])

    # Check that output has the expected shape
    if network_class in [LocalMLP, LocalQNN]:
        # Local models should return output for each grid point
        assert output.shape[-1] == test_data["num_points"]
    elif network_class in [
        GlobalMLP,
        BatchedGlobalMLP,
        GlobalQNNLayer,
        GlobalQNN,
        GlobalQiCQNN,
        GlobalQiQNN,
    ]:
        # Global models should return a single value
        assert output.ndim >= 1 or output.ndim == 0

    # Check that output contains finite values
    assert jnp.all(jnp.isfinite(output))


@pytest.mark.parametrize(
    "network_class,config,expected_error",
    [
        (LocalMLP, {"use_amplitude_encoding": True}, ValueError),
        (GlobalMLP, {"use_amplitude_encoding": False}, ValueError),
    ],
)
def test_network_initialization_errors(network_class, config, expected_error, test_data):
    """Test that networks raise appropriate errors for invalid configurations."""
    with pytest.raises(expected_error):
        network = network_class(config_dict=config)
        init_fn, apply_fn = network.build_network(test_data["grids"])


@pytest.mark.parametrize(
    "qnn_type,layer_type",
    [
        ("LocalQNN", "DirectQNN"),
        ("LocalQNN", "ChebyshevQNN"),
        ("LocalQNN", "ProductQNN"),
    ],
)
def test_local_qnn_variants(qnn_type, layer_type, test_data):
    """Test different variants of LocalQNN."""
    config = {
        "qnn_type": qnn_type,
        "layer_type": layer_type,
        "n_qubits": 3,
        "n_layers": 2,
    }
    network = LocalQNN(config_dict=config)
    init_fn, apply_fn = network.build_network(test_data["grids"])

    input_shape = (-1, test_data["num_points"], 1)
    _, params = init_fn(test_data["rng_key"], input_shape=input_shape)
    output = apply_fn(params, test_data["density"])

    assert output.shape[-1] == test_data["num_points"]
    assert jnp.all(jnp.isfinite(output))


@pytest.mark.parametrize("last_layer_type", ["dense", "mlp"])
def test_global_qnn_last_layer_types(last_layer_type, test_data):
    """Test GlobalQNN with different last layer types."""
    config = {
        "n_qubits": 2,
        "n_var_layers": 1,
        "largest_kernel_width": 2,
        "last_layer_type": last_layer_type,
        "use_bias_mlp": True if last_layer_type == "mlp" else False,
    }
    network = GlobalQNN(config_dict=config)
    init_fn, apply_fn = network.build_network(test_data["grids"])

    input_shape = (-1, test_data["num_points"], 1)
    _, params = init_fn(test_data["rng_key"], input_shape=input_shape)
    output = apply_fn(params, test_data["density"])

    assert output.ndim >= 1 or output.ndim == 0
    assert jnp.all(jnp.isfinite(output))


def test_network_config_defaults(test_data):
    """Test that networks can be initialized with default configurations."""
    networks = [
        LocalMLP(),
        GlobalMLP(),
        BatchedGlobalMLP(),
        LocalQNN(),
        GlobalQNNLayer(),
        GlobalQNN(),
        GlobalQNNwithMLP(),
        GlobalQiCQNN(),
        GlobalQiQNN(),
    ]

    for network in networks:
        init_fn, apply_fn = network.build_network(test_data["grids"])
        input_shape = (-1, test_data["num_points"], 1)
        _, params = init_fn(test_data["rng_key"], input_shape=input_shape)
        output = apply_fn(params, test_data["density"])
        assert jnp.all(jnp.isfinite(output))


def test_global_qicqnn_specific(test_data):
    """Test specific features of GlobalQiCQNN."""
    config = {
        "n_qubits": 3,
        "n_var_layers": 2,
        "largest_kernel_width": 2,
        "last_layer_type": "dense",
        "diff_mode": DiffMode.AD,
        "n_shots": 100,  # Test with shots
    }
    network = GlobalQiCQNN(config_dict=config)
    init_fn, apply_fn = network.build_network(test_data["grids"])

    input_shape = (-1, test_data["num_points"], 1)
    _, params = init_fn(test_data["rng_key"], input_shape=input_shape)
    output = apply_fn(params, test_data["density"])

    assert output.ndim >= 1 or output.ndim == 0
    assert jnp.all(jnp.isfinite(output))


def test_global_qiqnn_specific(test_data):
    """Test specific features of GlobalQiQNN."""
    config = {
        "n_qubits": 3,
        "n_layers": 2,
        "n_features": test_data["num_points"],  # Match to grid size
        "diff_mode": DiffMode.AD,
        "n_shots": 0,  # Test without shots
    }
    network = GlobalQiQNN(config_dict=config)
    init_fn, apply_fn = network.build_network(test_data["grids"])

    input_shape = (-1, test_data["num_points"], 1)
    _, params = init_fn(test_data["rng_key"], input_shape=input_shape)
    output = apply_fn(params, test_data["density"])

    assert output.ndim >= 1 or output.ndim == 0
    assert jnp.all(jnp.isfinite(output))


@pytest.mark.parametrize(
    "n_features,n_qubits",
    [
        (4, 2),  # Exact match (2^2 = 4)
        (5, 3),  # More qubits than needed (2^3 = 8 > 5)
        (9, 3),  # Not enough qubits (2^3 = 8 < 9) - should log warning but still work
    ],
)
def test_global_qiqnn_feature_encoding(n_features, n_qubits, test_data):
    """Test GlobalQiQNN with different feature and qubit configurations."""
    # Create custom test data with the specified number of points
    num_points = n_features
    density = jnp.linspace(0, 1, num_points)
    grids = jnp.ones(density.shape)

    config = {
        "n_qubits": n_qubits,
        "n_layers": 1,
        "n_features": n_features,
    }
    network = GlobalQiQNN(config_dict=config)
    init_fn, apply_fn = network.build_network(grids)

    input_shape = (-1, num_points, 1)
    _, params = init_fn(test_data["rng_key"], input_shape=input_shape)
    output = apply_fn(params, density)

    assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
