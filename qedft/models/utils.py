"""
Utility functions for the models.
"""

import jax
import jax.tree_util


def count_parameters(params):
    """Count the number of parameters in the model.
    Args:
        params: Model parameters from stax
    Returns:
        Number of parameters in the model
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params) if hasattr(x, "size"))


if __name__ == "__main__":
    from jax.example_libraries import stax

    network = stax.serial(
        stax.Dense(10),
        stax.Dense(10),
    )
    init_fn, apply_fn = network
    shape, params = init_fn(rng=jax.random.PRNGKey(0), input_shape=(1, 10))
    print(count_parameters(params))
