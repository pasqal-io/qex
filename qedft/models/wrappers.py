"""Wrappers for neural network functionals.

This module provides functions to wrap neural networks with additional layers like
self-interaction correction and negativity transform. It also includes a main
function to build a neural exchange-correlation functional from a base network.
"""

from collections.abc import Callable
from typing import Union

import jax
from jax import numpy as jnp
from jax import random
from jax.example_libraries import stax
from jax_dft import neural_xc, utils
from jax_dft.neural_xc import negativity_transform
from loguru import logger

Array = Union[jnp.ndarray, float]
InitFn = Callable[[jax.random.PRNGKey], dict]
ApplyFn = Callable[[dict, Array], jnp.ndarray]
Network = tuple[InitFn, ApplyFn]


def neural_xc_functional(
    network: Network,
    grids: Array,
    network_type: str,
    encoding: str = "local",
    **kwargs,
) -> tuple[InitFn, ApplyFn]:
    """Creates initialization and application functions for neural network functionals.

    Args:
        network: Tuple of (init_fn, apply_fn) defining the neural network
        grids: Array of spatial grid points
        network_type: Type of network architecture ("mlp", "mlp_ksr", or other)
        encoding: Type of encoding to use ("local" or "amplitude")

    Returns:
        Tuple of (init_fn, xc_energy_density_fn) where xc_energy_density_fn returns jnp.ndarray
    """

    network_init_fn, network_apply_fn = network
    num_grids = grids.shape[0]

    def init_fn(rng: jax.random.PRNGKey, **kwargs) -> dict:
        """Initializes network parameters.

        Args:
            rng: JAX random number generator key
            **kwargs: Additional arguments (unused)

        Returns:
            dict: Initialized network parameters
        """
        del kwargs

        # TODO: keeping to check if the cleaned up code works
        # input_shapes = {
        #     "mlp": (-1, num_grids, 1),      # Local MLP: single density value input
        #     "mlp_ksr": (1,),                 # Global MLP: full density vector → scalar
        #     "local": (num_grids,),           # QNNs: single density value
        #     "amplitude": (-1, num_grids, 1)  # Global encodings: whole density vector
        # }
        # if network_type in ["mlp", "mlp_ksr"]:
        #     shape = input_shapes[network_type]
        # else:
        #     shape = input_shapes[encoding]

        input_shapes = {
            # For Global models
            "amplitude": (1,),  # QNNs: single density value
            # For Local models
            "local": (-1, num_grids, 1),  # Global encodings: whole density vector
        }

        shape = input_shapes[encoding]

        _, params = network_init_fn(rng, input_shape=shape)
        return params

    # May leave jitting outside, optional
    # @jax.jit
    def xc_energy_density_fn(density: Array, params: dict, **kwargs) -> jnp.ndarray:
        """Computes exchange-correlation energy density from electron density.

        Args:
            density: Input electron density array with shape (num_grids,)
            params: Dictionary of network parameters
            **kwargs: Additional keyword arguments passed to network_apply_fn

        Returns:
            jnp.ndarray: Exchange-correlation energy density. Shape depends on encoding:
                - For local encoding: Array of shape (num_grids,)
                - For amplitude encoding: Single scalar value

        Note:
            The argument order (density, params) is reversed compared to network_apply_fn
            which expects (params, density) to match the JAX convention.
        """

        # The order of the arguments is reversed compared to xc_energy_density_fn
        output = network_apply_fn(params, density, **kwargs)

        if encoding == "local":
            return jnp.asarray(output).flatten()  # Ensure jax array output
        else:  # "Amplitude" encoding
            if output.shape == (1,):
                return jnp.asarray(output[0])
            else:
                # ? Did we ever do sum here?
                return jnp.asarray(output)

    # Add Gaussian noise to the neural network

    noise_std = kwargs.get("noise_std", 0.0)
    if noise_std > 0.0:
        logger.info(f"Adding Gaussian noise to the output with std {noise_std}")
        original_fn = xc_energy_density_fn

        def noisy_neural_xc_energy_density_fn(inputs: Array, params: dict, *, rng_key):
            """Adds Gaussian noise to the network output.

            Args:
                inputs: Input electron density
                params: Network parameters
                rng_key: JAX PRNG key for generating noise

            Returns:
                The network output with added Gaussian noise
            """
            outputs = original_fn(inputs, params)
            noise = jax.random.normal(rng_key, shape=outputs.shape) * noise_std
            return outputs + noise

        # Create a wrapper that handles key creation for backward compatibility
        def noisy_wrapper(inputs: Array, params: dict):
            seed = 0
            param_values = jax.tree_util.tree_leaves(params)
            for param in param_values:
                # Large multiplier to make the seed more sensitive to parameter values to
                # ensure different seeds for different inputs
                seed += jnp.sum(jnp.abs(param)) * 1e9

            # Ensure seed is a valid integer for PRNGKey
            seed = jnp.int32(seed)
            key = random.PRNGKey(seed)

            # The old version with time (not jittable / reproducible)

            # It would trigger a recompile since it is not pure due to time.time()
            # key = random.PRNGKey(int(time.time() * 1e6))

            return noisy_neural_xc_energy_density_fn(inputs, params, rng_key=key)

        xc_energy_density_fn = noisy_wrapper

    return init_fn, xc_energy_density_fn


def wrap_network(
    network: Network,
    grids: Array,
    network_type: str,
    wrap_self_interaction: bool = False,
    wrap_with_negative_transform: bool = False,
    **kwargs,
) -> tuple[InitFn, ApplyFn]:
    """Wraps network with appropriate functional encoding and optional layers."""
    # Determine if network uses global/amplitude encoding
    is_global = network_type in [
        "gadqc",
        "gadqc_with_mlp",
        "conv_dqc",
        "conv_adqc",
        "mlp_ksr",
        "conv_dqc_tfm",
        "conv_adqc_tfm",
    ]
    # Global models output the total XC energy
    if is_global:
        encoding = "amplitude"
    else:
        # Local models output the XC energy density on all grid points
        encoding = "local"

    if wrap_self_interaction:
        logger.info("Wrapping network with self-interaction layer")
        network = neural_xc.wrap_network_with_self_interaction_layer(
            network,
            grids=grids,
            interaction_fn=utils.exponential_coulomb,
        )

    if wrap_with_negative_transform:
        logger.info("Wrapping network with negative transform")
        network = stax.serial(network, negativity_transform())

    init_fn, apply_fn = neural_xc_functional(network, grids, network_type, encoding, **kwargs)

    # For global models, ensure the output is a scalar
    if is_global:
        original_apply_fn = apply_fn
        logger.info(
            "Global model, ensuring output is scalar (wrap_self_interaction "
            "causes global models to output (num_grids,) instead of (1,))",
        )

        def global_apply_fn(density, params, **kwargs):
            # Take mean to preserve scale
            # Global should output a scalar, not a vector
            return jnp.mean(original_apply_fn(density, params, **kwargs))

        return init_fn, global_apply_fn

    return init_fn, apply_fn


def wrap_network_from_config(
    network: Network,
    grids: Array,
    config: dict,
) -> tuple[InitFn, ApplyFn]:
    """Wraps a neural network with additional layers based on configuration settings.

    This function takes a base neural network and wraps it with optional layers like
    self-interaction correction and negativity transform based on the provided config.
    The wrapped network follows JAX functional style with init/apply functions.

    Args:
        network: A tuple of (init_fn, apply_fn) functions defining the base neural network.
            init_fn: Takes PRNGKey, returns initial parameters.
            apply_fn: Takes (params, inputs), returns network outputs.
        grids: Array of spatial grid points where the network will be evaluated.
        config: Configuration dictionary with the following keys:
            network_type: str, Architecture type (e.g. 'mlp', 'mlp_ksr')
            wrap_self_interaction: bool, Add self-interaction layer (optional)
            wrap_with_negative_transform: bool, Add negativity transform (optional)

    Returns:
        A tuple of (init_fn, apply_fn) for the wrapped network where:
            init_fn: Takes PRNGKey -> params
            apply_fn: Takes (density, params) -> energy_density
                Note: apply_fn argument order is reversed from input network

    Example:
        >>> config = {'network_type': 'mlp', 'wrap_self_interaction': True}
        >>> init_fn, apply_fn = wrap_network_from_config(base_net, grids, config)
        >>> params = init_fn(random.PRNGKey(0))
        >>> energy_density = apply_fn(density, params)
    """
    # Extract network type and optional wrapper flags from config
    network_type = config["network_type"]
    wrap_self_interaction = config.get("wrap_self_interaction", False)
    wrap_with_negative_transform = config.get("wrap_with_negative_transform", False)

    return wrap_network(
        network=network,
        grids=grids,
        network_type=network_type,
        wrap_self_interaction=wrap_self_interaction,
        wrap_with_negative_transform=wrap_with_negative_transform,
    )


def build_xc_functional(
    network: Network,
    grids: Array,
    config: dict,
) -> tuple[InitFn, ApplyFn]:
    """Main function to build a neural exchange-correlation functional.

    This is the primary entry point for creating an XC functional. It takes a base neural
    network and wraps it with appropriate layers based on the config to create a complete
    functional that can be used in DFT calculations.

    Args:
        network: Base neural network as (init_fn, apply_fn) pair in JAX style.
                init_fn takes PRNGKey and returns parameters
                apply_fn takes (params, inputs) and returns outputs
        grids: Spatial grid points where functional will be evaluated
        config: Configuration dictionary with keys:
            - network_type: Architecture type ('mlp', 'mlp_ksr', etc)
            - wrap_self_interaction: Add self-interaction correction (optional)
            - wrap_with_negative_transform: Add negativity constraint (optional)

    Returns:
        (init_fn, apply_fn) pair where:
            - init_fn: Takes PRNGKey, returns initial parameters
            - apply_fn: Takes (inputs, params), returns XC energy density
                       Note reversed argument order from input network

    Example:
        >>> init_fn, apply_fn = build_xc_functional(base_net, grids, config)
        >>> params = init_fn(rng_key)
        >>> xc_energy = apply_fn(density, params)
    """
    return wrap_network_from_config(network, grids, config)


if __name__ == "__main__":
    from qedft.models.classical.classical_models import build_global_mlp, build_local_mlp
    from qedft.models.quantum.quantum_models import build_qnn

    # Set up test data
    num_points = 513
    grids = jnp.linspace(0.0, 1.0, num_points)
    prng = jax.random.PRNGKey(0)
    test_inputs = jax.random.uniform(
        prng,
        minval=-4 * jnp.pi,
        maxval=4 * jnp.pi,
        shape=(num_points,),
    )

    # Test 1: Global MLP without wrappers
    print("\nTest 1: Global MLP without wrappers")
    network = build_global_mlp(
        n_neurons=num_points,
        n_layers=2,
        activation="tanh",
        n_outputs=1,
        density_normalization_factor=2.0,
        grids=grids,
    )
    init_fn, apply_fn = network
    _, params = init_fn(prng, input_shape=(num_points,))
    result = apply_fn(params, test_inputs)
    print("Output:", result)

    # Test 2: Local MLP with self-interaction
    print("\nTest 2: Local MLP with self-interaction")
    network = build_local_mlp(
        n_neurons=num_points,
        n_layers=2,
        activation="tanh",
        n_outputs=1,
        density_normalization_factor=2.0,
        grids=grids,
    )
    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp",
        wrap_self_interaction=True,
    )
    params = init_fn(prng)

    # Note the order has changed, now it's (params, inputs)
    result = apply_fn(test_inputs, params)
    print("Output:", result)

    # Test 3: Global MLP with self-interaction
    print("\nTest 3: Global MLP with self-interaction")
    network = build_global_mlp(
        n_neurons=num_points,
        n_layers=2,
        activation="tanh",
        n_outputs=1,
        density_normalization_factor=2.0,
        grids=grids,
    )
    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp_ksr",
        wrap_self_interaction=True,
    )
    params = init_fn(prng)
    result = apply_fn(test_inputs, params)
    print("Output:", result)

    # Test 4: Global MLP with self-interaction and negativity transform
    print("\nTest 4: Global MLP with self-interaction and negativity transform")
    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp_ksr",
        wrap_self_interaction=True,
        wrap_with_negative_transform=True,
    )
    params = init_fn(prng)
    result = apply_fn(test_inputs, params)
    print("Output:", result)

    # Test 5: Global MLP with self-interaction, negativity transform, and noise
    print("\nTest 5: Global MLP with self-interaction, negativity transform, and noise")
    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp_ksr",
        wrap_self_interaction=True,
        wrap_with_negative_transform=True,
        noise_std=0.01,
    )
    params = init_fn(prng)
    result = apply_fn(test_inputs, params)
    print("Output noise:", result)

    # Same for QNNs
    from horqrux.utils.operator_utils import zero_state

    from qedft.models.quantum.hardware_ansatz import hea

    print("\nTest 6: QNN with self-interaction, negativity transform, and noise")
    n_qubits = 4
    n_layers = 2
    num_grids = 4
    n_features = 4
    qnn_type = "GlobalQNN"
    # state = random_state(n_qubits)
    state = zero_state(n_qubits)
    ansatz = hea(n_qubits, n_layers)
    grids = jnp.linspace(0, 1, num_grids)
    prng = jax.random.PRNGKey(0)
    test_inputs = jax.random.uniform(
        prng,
        minval=-4 * jnp.pi,
        maxval=4 * jnp.pi,
        shape=(num_grids,),
    )

    network = build_qnn(
        n_qubits=n_qubits,
        n_features=n_features,
        ansatz=ansatz,
        qnn_type=qnn_type,
        grids=grids,
        layer_type="ProductQNN",
        map_fn=jnp.abs,
    )
    init_fn, apply_fn = wrap_network(
        network=network,
        grids=grids,
        network_type="mlp_ksr",
        wrap_self_interaction=False,
        wrap_with_negative_transform=False,
        noise_std=0.1,
    )
    params = init_fn(prng)
    result = apply_fn(test_inputs, params)
    print("Output noise:", result)
    # JIT the apply function
    apply_fn = jax.jit(apply_fn)
    result = apply_fn(test_inputs, params)
    print("JIT Output noise:", result)
