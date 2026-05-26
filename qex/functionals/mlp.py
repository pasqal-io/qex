"""MLP network for neural XC functionals.

This is *one* network architecture, plugged into the generic XC evaluator
(`qex.functionals.xc.make_eval_xc`) to become a full functional.
"""

from collections.abc import Callable, Sequence

import flax.linen as nn
from chex import Array


class MLP(nn.Module):
    """A small dense MLP whose output is fed through `-scale * swish(...)`.

    The final negativity transform enforces a soft non-positivity prior on the
    XC energy density.
    """

    features: Sequence[int]
    act_fn: Callable = nn.gelu
    scale: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features[:-1]:
            x = self.act_fn(nn.Dense(feat)(x))
        return -self.scale * nn.swish(nn.Dense(self.features[-1])(x))
