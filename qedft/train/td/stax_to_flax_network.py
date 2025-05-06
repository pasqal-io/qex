"""
Adapter to make STAX models into FLAX models that work with the 3D code.
"""

import jax
import jax.numpy as jnp


class StaxAdapter:
    """Adapter to make STAX models compatible with FLAX-oriented code.

    This adapter wraps a STAX model (init_fn, apply_fn) to provide the same
    interface as a FLAX model, allowing it to work with existing optimization code.
    """

    def __init__(self, init_fn, apply_fn, input_shape, rng_key=jax.random.PRNGKey(0)):
        """Initialize the STAX adapter.

        Args:
            init_fn: STAX initialization function
            apply_fn: STAX application function
            input_shape: Shape of input excluding batch dimension
            rng_key: Random key for initialization
        """
        self.init_fn = init_fn
        self.apply_fn = apply_fn

        # Initialize STAX parameters
        output_shape, self.stax_params = init_fn(rng_key, (-1,) + input_shape)

    def init(self, rng_key, inputs):
        """Mimic FLAX init method for compatibility.

        Returns parameters in FLAX-compatible structure.
        """
        return {"params": {"stax_params": self.stax_params}}

    def apply(self, params, inputs, **kwargs):
        """Apply the model using STAX apply_fn with FLAX-compatible parameters.

        Maps between FLAX nested dict structure and STAX parameter structure.
        """
        # Extract STAX parameters from FLAX parameter structure
        stax_params = params["params"]["stax_params"] if "params" in params else params
        return self.apply_fn(stax_params, inputs)


def adapt_stax_for_training(init_fn, apply_fn, input_shape, rng_key=jax.random.PRNGKey(0)):
    """Create a STAX adapter and return it with initial parameters.

    Use this function to create a drop-in replacement for FLAX models in the existing code.

    Args:
        init_fn: STAX init function
        apply_fn: STAX apply function
        input_shape: Input shape for model (without batch dimension)
        rng_key: Random key for initialization

    Returns:
        model: Adapter object with FLAX-like interface
        params: Parameters in FLAX-compatible format
    """
    adapter = StaxAdapter(init_fn, apply_fn, input_shape, rng_key)
    params = adapter.init(rng_key, jnp.ones((1,) + input_shape))
    return adapter, params


if __name__ == "__main__":

    from qedft.models.networks import GlobalMLP

    model = GlobalMLP()
    network = model.build_network(grids=jnp.linspace(0, 1, 100))
    stax_init_fn, stax_apply_fn = network  # your STAX model
    network, params = adapt_stax_for_training(stax_init_fn, stax_apply_fn, (100,))
    print(network)
    print(params)
    output = network.apply(params, jnp.ones((1, 100)))
    print("output", output)
