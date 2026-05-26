"""Small standalone transforms used by quantum/classical models.

Vendored (not imported from jax_dft) so forward-going code does not depend on jax_dft.
"""

from jax import nn
from jax.example_libraries import stax


def negativity_transform():
    """Stax layer enforcing soft negativity on the network output.

    XC energy density must be negative; the layer maps x -> -swish(x), giving
    a range of approximately (-inf, 0.278].

    Returns:
        (init_fn, apply_fn) pair (stax convention).
    """

    def negative_fn(x):
        return -nn.swish(x)

    return stax.elementwise(negative_fn)
