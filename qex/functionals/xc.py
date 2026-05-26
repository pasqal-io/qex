"""Generic XC evaluator plumbing.

Wraps any Flax network as a PySCF-style `eval_xc` callable, with `vrho` computed
by reverse-mode autodiff of the network output w.r.t. density.

Network-agnostic: the same helpers work for MLP, QNN, hybrid, etc. — anything
that exposes `(init, apply)` and returns a scalar from a density.
"""

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array


# Global encoding

def get_exc_and_vrho(
    params: dict,
    rho: Array,
    network: nn.Module,
    vxc_grad_scale: float = 1.0,
    **kwargs,
) -> tuple[Array, Array]:
    """Compute (exc, vrho) by VJP of the network output w.r.t. density."""
    to_stack = (rho,)
    for key, value in kwargs.items():
        if key == "params_grid_coords":
            x, y, z = value[:, 0], value[:, 1], value[:, 2]
            to_stack += (x, y, z)
        else:
            to_stack += (value,)

    rho_and_grad = rho
    exc, vjp_fn = jax.vjp(
        lambda x: jnp.sum(network.apply(params, x).squeeze()), rho_and_grad
    )
    (vrho,) = vjp_fn(jnp.ones_like(exc))
    return exc, vrho * vxc_grad_scale


def make_eval_xc(network: nn.Module, vxc_grad_scale: float = 1.0) -> Callable:
    """Build a PySCF-style `eval_xc` callable bound to `network` and `vxc_grad_scale`.

    Returned signature mirrors PySCF's `eval_xc`:
        eval_xc(xc_code, rho, *, params, ...)
            -> (exc, (vrho, None, None, None), None, None)
    """

    def eval_xc(
        xc_code: str,
        rho: Array,
        spin: int = 0,
        relativity: int = 0,
        deriv: int = 1,
        verbose=None,
        params: dict | None = None,
        params_grid_coords: Array | None = None,
        method: str = "density",
    ):
        if method != "density" or deriv != 1:
            raise ValueError("Only method='density' with deriv=1 is supported.")

        if params_grid_coords is not None:
            exc, vrho = get_exc_and_vrho(
                params,
                rho,
                network,
                vxc_grad_scale=vxc_grad_scale,
                params_grid_coords=params_grid_coords,
            )
        else:
            exc, vrho = get_exc_and_vrho(
                params, rho, network, vxc_grad_scale=vxc_grad_scale
            )

        return exc, (vrho, None, None, None), None, None

    return eval_xc
