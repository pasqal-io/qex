"""Neural network models for density functional theory.

This module implements multilayer perceptron (MLP) architectures for DFT calculations:
- LocalMLP: Processes each grid point independently
- GlobalMLP: Processes the entire density vector as one input

Key Features:
- JAX-based implementation for automatic differentiation
- Configurable architecture (neurons, layers, activations)
- Normalized inputs for better training stability
- Support for both local and global processing

Example:
    >>> config = {
    ...     "network_type": "mlp",  # "mlp" for local, "mlp_ksr" for global
    ...     "n_neurons": 64,        # Width of hidden layers
    ...     "n_layers": 3,          # Number of hidden layers
    ...     "activation": "tanh"    # Activation function
    ... }
    >>> grids = jnp.linspace(0, 1, 100)  # Spatial discretization
    >>> init_fn, apply_fn = create_mlp_model(config, grids)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.example_libraries import stax

# Default configuration values
DEFAULT_N_NEURONS = 64  # Width of hidden layers
DEFAULT_N_LAYERS = 3  # Number of hidden layers
DEFAULT_ACTIVATION = "tanh"  # Default activation function
DEFAULT_N_OUTPUTS = 1  # Single output per point by default
DEFAULT_DENSITY_NORM = 2.0  # Typical scale for electron density
DEFAULT_N_FEATURES = 3  # Number of features in the density vector

# Supported activation functions
ACTIVATION_MAP = {
    "tanh": stax.Tanh,  # Hyperbolic tangent
    "relu": stax.Relu,  # Rectified linear unit
    "softmax": stax.Softmax,  # Normalized exponential
    "softplus": stax.Softplus,  # Smooth ReLU variant
    "sigmoid": stax.Sigmoid,  # Logistic function
    "elu": stax.Elu,  # Exponential linear unit
    "leaky_relu": stax.LeakyRelu,  # ReLU with small negative slope
    "selu": stax.Selu,  # Scaled exponential linear unit
    "gelu": stax.Gelu,  # Gaussian error linear unit
}


def build_activation_layer(activation: str) -> Callable:
    """Creates an activation layer from the stax library.

    Args:
        activation: Name of activation function from ACTIVATION_MAP

    Returns:
        Stax activation layer constructor

    Raises:
        ValueError: If activation name is not in ACTIVATION_MAP

    Example:
        >>> activation_fn = build_activation_layer("relu")
        >>> # Use in network: layers.extend([Dense(64), activation_fn])
    """
    if activation not in ACTIVATION_MAP:
        raise ValueError(
            f"Unknown activation '{activation}'. Valid options: {list(ACTIVATION_MAP.keys())}",
        )
    return ACTIVATION_MAP[activation]


def build_mlp_layers(
    n_neurons: int,
    n_layers: int,
    activation: str,
    n_outputs: int = DEFAULT_N_OUTPUTS,
) -> list:
    """Constructs layers for a multilayer perceptron.

    Architecture:
        Input -> [Dense(n_neurons) -> Activation] x n_layers -> Dense(n_outputs)

    Args:
        n_neurons: Number of neurons in each hidden layer
        n_layers: Number of hidden layers (Dense + Activation pairs)
        activation: Name of activation function to use
        n_outputs: Size of final output layer (default: 1)

    Returns:
        List of stax layer constructors defining the network

    Raises:
        ValueError: If n_neurons or n_layers is not positive
    """
    if n_neurons <= 0:
        raise ValueError("n_neurons must be positive")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")

    activation_layer = build_activation_layer(activation)

    layers = []
    for _ in range(n_layers):
        layers.extend(
            [
                stax.Dense(n_neurons),
                activation_layer,
            ],
        )

    layers.append(stax.Dense(n_outputs))
    return layers


# ================================================================
# Factory function
# ================================================================


def build_local_mlp(
    n_neurons: int = DEFAULT_N_NEURONS,
    n_layers: int = DEFAULT_N_LAYERS,
    activation: str = DEFAULT_ACTIVATION,
    n_outputs: int = DEFAULT_N_OUTPUTS,
    density_normalization_factor: float = DEFAULT_DENSITY_NORM,
    grids: jnp.ndarray | None = None,
    **kwargs,
) -> tuple[Callable, Callable]:
    """Builds a local MLP that processes each spatial point independently.

    The local MLP applies the same network to each grid point using JAX's vmap.
    This approach maintains translational invariance and scales well with grid size.

    Architecture:
        For each grid point i:
            density[i] -> normalize -> MLP -> output[i]

    Args:
        n_neurons: Width of hidden layers (default: 64)
        n_layers: Number of hidden layers (default: 3)
        activation: Activation function name (default: "tanh")
        n_outputs: Number of outputs per point (default: 1)
        density_normalization_factor: Scale factor for input normalization (default: 2.0)
        grids: Grid points (required for shape info)
        **kwargs: Additional arguments (unused)

    Returns:
        (init_fn, apply_fn) tuple of network functions

    Raises:
        ValueError: If grids is None
    """
    if grids is None:
        raise ValueError("grids must be provided for local MLP")

    layers = build_mlp_layers(n_neurons, n_layers, activation, n_outputs)
    network = stax.serial(*layers)
    num_grids = grids.shape[0]

    def init_fn(rng, input_shape):
        # input_shape has to be provided to work in stax
        del input_shape
        return network[0](rng, input_shape=(-1, num_grids, 1))

    def apply_fn(params, inputs, **kwargs):
        del kwargs
        normalized_inputs = inputs / density_normalization_factor
        batched_fn = jax.vmap(lambda x: network[1](params, x))
        return batched_fn(normalized_inputs).squeeze()

    return init_fn, apply_fn


def build_global_mlp(
    n_neurons: int = DEFAULT_N_NEURONS,
    n_layers: int = DEFAULT_N_LAYERS,
    activation: str = DEFAULT_ACTIVATION,
    n_outputs: int = DEFAULT_N_OUTPUTS,
    density_normalization_factor: float = DEFAULT_DENSITY_NORM,
    grids: jnp.ndarray | None = None,
    **kwargs,
) -> tuple[Callable, Callable]:
    """Builds an MLP that processes the entire density vector globally.

    Architecture:
        full_density -> normalize -> MLP -> scalar_output

    Args:
        n_neurons: Width of hidden layers (default: 64)
        n_layers: Number of hidden layers (default: 3)
        activation: Activation function name (default: "tanh")
        n_outputs: Number of outputs (default: 1)
        density_normalization_factor: Scale for input normalization (default: 2.0)
        grids: Grid points defining spatial discretization
        **kwargs: Additional arguments (unused)

    Returns:
        (init_fn, apply_fn) tuple of network functions

    Raises:
        ValueError: If grids is None
    """
    if grids is None:
        raise ValueError("grids must be provided for global MLP")

    num_grids = grids.shape[0]
    layers = build_mlp_layers(n_neurons, n_layers, activation, n_outputs)
    network = stax.serial(*layers)

    def init_fn(rng, input_shape):
        # input_shape has to be provided to work in stax
        del input_shape
        return network[0](rng, input_shape=(num_grids,))

    def apply_fn(params, inputs, **kwargs):
        del kwargs
        normalized_inputs = inputs / density_normalization_factor
        # ? squeeze() here like in local MLP
        return network[1](params, normalized_inputs)

    return init_fn, apply_fn


def build_batched_global_mlp(
    n_neurons: int = DEFAULT_N_NEURONS,
    n_layers: int = DEFAULT_N_LAYERS,
    activation: str = DEFAULT_ACTIVATION,
    n_outputs: int = DEFAULT_N_OUTPUTS,
    density_normalization_factor: float = DEFAULT_DENSITY_NORM,
    n_features: int = DEFAULT_N_FEATURES,
    **kwargs,
) -> tuple[Callable, Callable]:
    """Builds a batched global MLP that processes multiple density vectors independently.

    Architecture:
        For each density vector in batch:
            density -> normalize -> MLP -> scalar_output

    Args:
        n_neurons: Width of hidden layers (default: 64)
        n_layers: Number of hidden layers (default: 3)
        activation: Activation function name (default: "tanh")
        n_outputs: Number of outputs (default: 1)
        density_normalization_factor: Scale for input normalization (default: 2.0)
        n_features: Number of features in the density vector (default: 1)
        **kwargs: Additional arguments (unused)

    Returns:
        (init_fn, apply_fn) tuple of network functions

    Raises:
        ValueError: If grids is None
    """
    if n_features is None:
        raise ValueError("grids must be provided for batched global MLP")

    layers = build_mlp_layers(n_neurons, n_layers, activation, n_outputs)
    network = stax.serial(*layers)

    def init_fn(rng, input_shape):
        # input_shape has to be provided to work in stax
        del input_shape
        return network[0](rng, input_shape=(-1, n_features))

    def apply_fn(params, inputs, **kwargs):
        del kwargs
        normalized_inputs = inputs / density_normalization_factor
        # Use vmap to apply the network to each input in the batch
        normalized_inputs = normalized_inputs.reshape(-1, n_features)
        print("normalized_inputs.shape: ", normalized_inputs.shape)
        batched_fn = jax.vmap(lambda x: network[1](params, x))
        return batched_fn(normalized_inputs)

    return init_fn, apply_fn


def create_mlp_model(
    config: dict,
    grids: jnp.ndarray | None = None,
) -> tuple[Callable, Callable]:
    """Factory function to create MLP models based on configuration.

    Args:
        config: Dictionary containing model configuration:
            - network_type: "mlp" for local or "mlp_ksr" for global processing
            - n_neurons: Width of hidden layers (default: 64)
            - n_layers: Number of hidden layers (default: 3)
            - activation: Activation function name (default: "tanh")
            - n_outputs: Number of outputs (default: 1)
            - density_normalization_factor: Input normalization scale (default: 2.0)
        grids: Grid points for spatial discretization (required)

    Returns:
        (init_fn, apply_fn) tuple for the requested MLP variant

    Raises:
        ValueError: If network_type is invalid or grids is None
    """
    if grids is None:
        raise ValueError("grids must be provided")

    common_args = {
        "n_neurons": config.get("n_neurons", DEFAULT_N_NEURONS),
        "n_layers": config.get("n_layers", DEFAULT_N_LAYERS),
        "activation": config.get("activation", DEFAULT_ACTIVATION),
        "n_outputs": config.get("n_outputs", DEFAULT_N_OUTPUTS),
        "density_normalization_factor": config.get(
            "density_normalization_factor",
            DEFAULT_DENSITY_NORM,
        ),
        "grids": grids,
    }

    builders = {
        "mlp": build_local_mlp,
        "mlp_ksr": build_global_mlp,
        "batched_mlp_ksr": build_batched_global_mlp,
    }

    builder = builders.get(config["network_type"])
    if builder is None:
        raise ValueError(f"Unknown network type: {config['network_type']}")

    return builder(**common_args)


if __name__ == "__main__":

    # Example usage with config
    config_local = {
        "network_type": "mlp",
        "n_neurons": 513,
        "n_layers": 2,
        "activation": "tanh",
    }

    config_global = {
        "network_type": "mlp_ksr",
        "n_neurons": 513,
        "n_layers": 2,
        "activation": "tanh",
    }

    grids = jnp.linspace(0.0, 1.0, 513)
    density = jnp.ones(grids.shape)

    # Create and test both network types with config
    for config in [config_local, config_global]:
        network = create_mlp_model(config, grids=grids)
        init_fn, apply_fn = network
        rng_key = jax.random.PRNGKey(0)
        _, params = init_fn(rng_key, input_shape=(-1, 513, 1))
        output = apply_fn(params, density)
        print(f"{config['network_type']} output shape:", output.shape)

    # Example usage with direct initialization
    local_network = build_local_mlp(
        n_neurons=513,
        n_layers=2,
        activation="tanh",
        n_outputs=1,
        density_normalization_factor=2.0,
        grids=grids,
    )

    global_network = build_global_mlp(
        n_neurons=513,
        n_layers=2,
        activation="tanh",
        n_outputs=1,
        density_normalization_factor=2.0,
        grids=grids,
    )

    # Test direct initialization networks
    for network, name in [(local_network, "local"), (global_network, "global")]:
        print(f"Testing {name} network")
        init_fn, apply_fn = network
        rng_key = jax.random.PRNGKey(0)
        _, params = init_fn(rng_key, input_shape=(-1, 513, 1))
        print(f"Direct {name} network params shape:", len(params), len(params[1]), len(params[2]))
        output = apply_fn(params, density)
        print(output)
        print(f"Direct {name} network output shape:", output.shape)

    # For referee report... to put instead of QNNs
    print("\nTesting Batched Global MLP")
    grids = jnp.linspace(0.0, 1.0, 3)
    density = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape: (2, 3)
    init_fn, apply_fn = build_batched_global_mlp(grids=grids)
    rng_key = jax.random.PRNGKey(0)
    _, params = init_fn(rng_key, input_shape=None)
    output = apply_fn(params, density)  # shape: (2, 1)
    print(output)
