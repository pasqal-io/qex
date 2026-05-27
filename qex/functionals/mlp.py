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

import flax.linen as nn
import jax.numpy as jnp
from chex import Array


class LocalMLP(nn.Module):
    """Per-grid MLP: ρ(r) -> ε_xc(r) (energy density per electron).

    Applied pointwise. Input shape `(n_grid,)`, output shape `(n_grid,)`.
    """

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

    features: Sequence[int]
    act_fn: Callable = nn.gelu
    scale: float = 1.0

    @nn.compact
    def __call__(self, rho: Array) -> Array:
        h = rho
        for feat in self.features:
            h = self.act_fn(nn.Dense(feat)(h))
        out = nn.Dense(1)(h).squeeze()
        return -self.scale * nn.softplus(out)
