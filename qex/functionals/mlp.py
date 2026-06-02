"""MLP network architectures for neural XC functionals.

Two flavors, matching the two encodings supported by `qex.scf.operators`:

- `LocalMLP` — outputs ε_xc(r) per grid point (per-electron energy density).
  Pair with `make_eval_xc_local` and `get_veff_local`. Classical LDA/GGA shape.

- `GlobalMLP` — outputs the scalar E_xc[ρ] (already integrated).
  Pair with `make_eval_xc_global` and `get_veff_global`.

Both apply the `-scale * swish(...)` transform on the network head as a soft
non-positivity prior on the XC energy.
"""

from collections.abc import Callable, Sequence
from typing import ClassVar

import flax.linen as nn
import jax.numpy as jnp
from chex import Array


class LocalMLP(nn.Module):
    """Per-grid MLP: ρ(r) -> ε_xc(r) (energy density per electron).

    Applied pointwise. Input shape `(n_grid,)`, output shape `(n_grid,)`.
    """

    # Local encoding evaluates ε_xc(r) pointwise from ρ alone (no extra features).
    required_features: ClassVar[tuple[str, ...]] = ()

    features: Sequence[int]
    act_fn: Callable = nn.gelu
    scale: float = 1e-2

    @nn.compact
    def __call__(self, rho: Array) -> Array:
        # `features` describes the hidden widths; the head is always a single
        # scalar per grid point (ε_xc(r)). This way callers can specify any
        # hidden architecture without accidentally producing a multi-channel
        # output that would break the local-encoding contract.
        x = rho[..., None] if rho.ndim == 1 else rho
        for feat in self.features:
            x = self.act_fn(nn.Dense(feat)(x))
        x = -self.scale * nn.swish(nn.Dense(1)(x))
        return x.squeeze(-1)


class GlobalMLP(nn.Module):
    """Whole-density MLP: ρ vector -> scalar E_xc[ρ] (already integrated).

    The input is the raw density vector on the grid; the first Dense layer's
    input dim is `n_grid`, so the model is tied to a fixed grid size.
    """

    # No geometric/extra features needed: ρ alone -> E_xc. The pipeline passes
    # an empty feature bag, so this model is immune to whatever optional features
    # the dataset happens to carry (see qex.functionals.features).
    required_features: ClassVar[tuple[str, ...]] = ()

    features: Sequence[int]
    act_fn: Callable = nn.gelu
    scale: float = 1.0

    @nn.compact
    def __call__(self, rho: Array, grid_weights: Array | None = None) -> Array:
        # `grid_weights` is part of the global-encoding call signature (used by
        # descriptor-style nets); a plain density MLP ignores it.
        del grid_weights
        h = rho
        for feat in self.features:
            h = self.act_fn(nn.Dense(feat)(h))
        out = nn.Dense(1)(h).squeeze()
        return -self.scale * nn.softplus(out)
