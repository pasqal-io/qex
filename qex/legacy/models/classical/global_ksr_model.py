"""Kohn-Sham Regularizer (KSR) model for neural exchange-correlation potential.

This model here only works for the 1D case.

This module loads a neural network model for learning the exchange-correlation
functional in density functional theory using the Kohn-Sham response formalism.

The model architecture combines global and local convolutional filters with a
self-interaction correction layer to capture both short and long-range electronic
interactions.

Key Features:
  - Hybrid CNN architecture with global and local filters
  - Self-interaction correction via exponential Coulomb interactions
  - Configurable network depth and filter counts
  - Support for custom activation functions and grid spacings

See the `jax_dft` package for more details: https://github.com/google-research/google-research/tree/master/jax_dft.
See paper: https://doi.org/10.1103/PhysRevLett.126.036401
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax_dft import neural_xc, np_utils, utils

# ================================================================
# Factory function
# ================================================================


def create_ksr_model_from_config(grids: np.ndarray, config: dict = None):
    """Creates a Kohn-Sham Response model from configuration dictionary.

    Args:
        config: Dictionary containing model configuration with keys:
            - num_global_filters: Number of global convolutional filters
            - num_local_filters: Number of local convolutional filters
            - num_local_conv_layers: Number of local convolutional layers
            - activation: Activation function name (e.g. 'swish', 'relu')
            - minval: Minimum value for grid spacing
            - maxval: Maximum value for grid spacing
            - downsample_factor: Factor to downsample the grid
        grids: Grid points for the density functional calculations.

    Returns:
        network: Configured neural network model instance for XC potential calculations.
        init_fn: Initialization function for the model.
        apply_fn: Apply function for the model.
    """
    if config is None:
        # Default config from jax_dft package
        config = {
            "num_global_filters": 16,
            "num_local_filters": 16,
            "num_local_conv_layers": 2,
            "activation": "swish",
            "minval": 0.1,
            "maxval": 2.385345,
            "downsample_factor": 0,
        }

    network = neural_xc.build_global_local_conv_net(
        num_global_filters=config.get("num_global_filters", 16),
        num_local_filters=config.get("num_local_filters", 16),
        num_local_conv_layers=config.get("num_local_conv_layers", 2),
        activation=config.get("activation", "swish"),
        grids=grids,
        minval=config.get("minval", 0.1),
        maxval=config.get("maxval", 2.385345),
        downsample_factor=config.get("downsample_factor", 0),
    )

    network = neural_xc.wrap_network_with_self_interaction_layer(
        network,
        grids=grids,
        interaction_fn=utils.exponential_coulomb,
    )

    init_fn, apply_fn = neural_xc.global_functional(network, grids=grids)

    return network, init_fn, apply_fn


if __name__ == "__main__":

    # Initialize model and parameters
    grids = np.linspace(0.0, 1.0, 513)
    network, init_fn, apply_fn = create_ksr_model_from_config(grids)
    print(network)

    # Generate initial parameters
    key = random.PRNGKey(0)
    init_params = init_fn(key)

    # Flatten parameters and print count
    spec, flatten_init_params = np_utils.flatten(init_params)
    print(f"Number of parameters: {len(flatten_init_params)}")
    flatten_init_params

    # Test model on random inputs
    key = random.PRNGKey(0)
    inputs = jax.random.uniform(
        key,
        minval=-4 * jnp.pi,
        maxval=4 * jnp.pi,
        shape=(513,),
    )
    params = init_fn(key)
    print(f"Input shape: {inputs.shape}")
    print(f"Params shape: {flatten_init_params.shape}")
    print(
        "Result of applying the model to 513 inputs:",
        apply_fn(inputs, params).shape,
    )
