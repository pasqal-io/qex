"""
Utility functions for the Kohn-Sham DFT code.
"""

import jax.numpy as jnp


def get_discount_coefficients(num_steps: int, discount: float = 0.9) -> jnp.ndarray:
    """Gets the discount coefficients on a trajectory with num_steps steps.

    A trajectory discount factor can be applied. The last step is not discounted.
    Say the index of the last step is num_steps - 1, for the k-th step, the
    discount coefficient is discount ** (num_steps - 1 - k).
    For example, for num_steps=4 and discount=0.8, returns [0.512, 0.64, 0.8, 1.].

    Args:
      num_steps: Integer, the total number of steps in the trajectory.
      discount: Float, the discount factor over the trajectory.

    Returns:
      Float numpy array with shape (num_steps,).
    """
    return jnp.power(discount, jnp.arange(num_steps - 1, -1, -1))


if __name__ == "__main__":
    print(get_discount_coefficients(4, 0.8))
