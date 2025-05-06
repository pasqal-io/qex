"""Loss functions for optimization without jit for compatibility with other non-jit packages.
This code is based on the code from jax_dft library.
"""

import jax.numpy as jnp


def mean_square_error(target, predict):
    """Computes mean square error.

    Args:
        target: Float numpy array with shape (batch_size, *feature_shape).
        predict: Float numpy array with shape (batch_size, *feature_shape).

    Returns:
        Float.
    """
    return jnp.mean((target - predict) ** 2)


def _get_discount_coefficients(num_steps, discount):
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


def _trajectory_error(error, discount):
    """Computes trajectory error."""
    batch_size = error.shape[0]
    num_steps = error.shape[1]
    # Shape (batch_size, num_steps)
    mse = jnp.mean(error.reshape(batch_size, num_steps, -1), axis=2)
    # Shape (batch_size,)
    discounted_mse = jnp.dot(mse, _get_discount_coefficients(num_steps, discount))
    return jnp.mean(discounted_mse)


def trajectory_error(error, discount):
    """Computes trajectory error.

    A trajectory discount factor can be applied. The last step is not discounted.
    Say the index of the last step is num_steps - 1, for the k-th step, the
    discount coefficient is discount ** (num_steps - 1 - k).

    Args:
        error: Float numpy array with shape (batch_size, num_steps, *feature_dims).
        discount: Float, the discount factor over the trajectory.

    Returns:
        Float.
    """
    return _trajectory_error(error, discount)


def _trajectory_mse(target, predict, discount):
    """Computes trajectory mean square error."""
    if predict.ndim < 2:
        raise ValueError(
            "The size of the shape of predict should be "
            f"greater or equal to 2, got {predict.ndim}",
        )
    if predict.ndim - target.ndim != 1:
        raise ValueError(
            "The size of the shape of predict should be greater than "
            "the size of the shape of target by 1, "
            f"but got predict ({predict.ndim}) and target ({target.ndim})",
        )
    # Insert a dimension for num_steps on target.
    target = jnp.expand_dims(target, axis=1)
    return trajectory_error((target - predict) ** 2, discount)


def trajectory_mse(target, predict, discount):
    """Computes trajectory mean square error.

    A trajectory discount factor can be applied. The last step is not discounted.
    Say the index of the last step is num_steps - 1, for the k-th step, the
    discount coefficient is discount ** (num_steps - 1 - k).

    Args:
        target: Float numpy array with shape (batch_size, *feature_dims).
        predict: Float numpy array with shape
            (batch_size, num_steps, *feature_dims).
        discount: Float, the discount factor over the trajectory.

    Returns:
        Float.
    """
    return _trajectory_mse(target, predict, discount)
