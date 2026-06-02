"""Generic XC evaluator plumbing.

Wraps any Flax network as a PySCF-style `eval_xc` callable, with `vrho` computed
by reverse-mode autodiff of the network output w.r.t. density.

Two encodings, two builders:

- `make_eval_xc_local(network)` — network outputs ε_xc(r) per electron on the
  grid (shape `(n_grid,)` or `(n_grid, 1)`). `vrho(r) = δE_xc/δρ(r)` is the bare
  derivative of the network sum w.r.t. ρ.

- `make_eval_xc_global(network)` — network outputs the already-integrated
  scalar E_xc[ρ]. The discrete VJP `∂E/∂ρ_i` carries an implicit `w_i` factor
  from the quadrature; we divide by `w(r)` to recover δE/δρ(r), so callers
  can use the same `Vxc = Σ_g ao(g)·w(g)·vrho(g)·ao(g)` formula.

Network-agnostic: same helpers work for MLP, QNN, hybrid, etc.
"""

from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array


_W_FLOOR = 1e-30


def _exc_and_vrho_local(
    params: dict,
    rho: Array,
    grid_weights: Array,
    network: nn.Module,
    vxc_grad_scale: float,
) -> tuple[Array, Array]:
    """Local encoding: network output is ε_xc(r), one value per grid point.

    Returns the bare ε_xc(r) (so callers can form E_xc = Σ ε·ρ·w themselves)
    and `vrho(r) = δE_xc/δρ(r)` computed by VJP through the full integrand
    `Σ ε(ρ)·ρ·w`. This is the correct functional derivative when ε depends
    pointwise on ρ (LDA-style); the URKS reference does the same thing.
    """
    # Use `has_aux` to recover ε_xc(r) from the forward pass for free — no
    # need for a second `network.apply`. The aux output doesn't participate
    # in the VJP, so the cotangent of the integrand `Σ ε·ρ·w` w.r.t. ρ gives
    # the discrete `δE/δρ(r_i)·w_i` (URKS-style).
    def energy_fn(rho_in: Array) -> tuple[Array, Array]:
        eps = network.apply(params, rho_in).squeeze()
        return jnp.sum(eps * rho_in * grid_weights), eps

    e_tot, vjp_fn, eps_xc = jax.vjp(energy_fn, rho, has_aux=True)
    (drho,) = vjp_fn(jnp.ones_like(e_tot))
    # Undo the discrete-quadrature `w_i` factor to recover δE/δρ(r).
    vrho = drho / (grid_weights + _W_FLOOR)
    return eps_xc, vrho * vxc_grad_scale


def _exc_and_vrho_global(
    params: dict,
    rho: Array,
    grid_weights: Array,
    network: nn.Module,
    vxc_grad_scale: float,
    features: dict | None = None,
) -> tuple[Array, Array]:
    """Global encoding: network output is the scalar E_xc[ρ] already integrated.

    `features` is the named model-feature bag (see `qex.functionals.features`),
    already narrowed to this network's `required_features`. It is forwarded to
    the network **by keyword**, so the network's signature — not a positional
    ordering convention — defines the contract.
    """
    features = features or {}

    def energy_fn(rho_in: Array) -> Array:
        return network.apply(params, rho_in, grid_weights, **features)

    exc_scalar, vjp_fn = jax.vjp(energy_fn, rho)
    (drho,) = vjp_fn(jnp.ones_like(exc_scalar))
    # ∂E/∂ρ_i ≈ δE/δρ(r_i)·w_i (discrete quadrature); undo to get δE/δρ(r).
    vrho = drho / (grid_weights + _W_FLOOR)
    return exc_scalar, vrho * vxc_grad_scale


def make_eval_xc_local(network: nn.Module, vxc_grad_scale: float = 1.0) -> Callable:
    """Build a PySCF-style `eval_xc` for a LOCAL network (ε_xc(r) per electron).

    The returned callable yields `(exc(r), (vrho(r), None, None, None), None, None)`.
    Pair with `qex.scf.operators.get_veff_local` (or `get_veff(..., encoding='local')`).
    """

    def eval_xc(
        xc_code: str,
        rho: Array,
        spin: int = 0,
        relativity: int = 0,
        deriv: int = 1,
        verbose=None,
        params: dict | None = None,
        grid_weights: Array | None = None,
        **_,
    ):
        if deriv != 1:
            raise ValueError("Only deriv=1 is supported.")
        if grid_weights is None:
            raise ValueError("Local eval_xc requires `grid_weights` kwarg.")
        exc, vrho = _exc_and_vrho_local(params, rho, grid_weights, network, vxc_grad_scale)
        return exc, (vrho, None, None, None), None, None

    return eval_xc


def make_eval_xc_global(
    network: nn.Module,
    vxc_grad_scale: float = 1.0,
) -> Callable:
    """Build a PySCF-style `eval_xc` for a GLOBAL network (scalar E_xc[ρ]).

    The returned callable yields `(E_xc, (vrho(r), None, None, None), None, None)`
    where `E_xc` is a scalar. Pair with `qex.scf.operators.get_veff_global`
    (or `get_veff(..., encoding='global')`).

    `grid_weights` must be passed as a kwarg at call time so the VJP-to-δE/δρ
    correction can be applied.
    """

    def eval_xc(
        xc_code: str,
        rho: Array,
        spin: int = 0,
        relativity: int = 0,
        deriv: int = 1,
        verbose=None,
        params: dict | None = None,
        grid_weights: Array | None = None,
        features: dict | None = None,
        **_,
    ):
        if deriv != 1:
            raise ValueError("Only deriv=1 is supported.")
        if grid_weights is None:
            raise ValueError("Global eval_xc requires `grid_weights` kwarg.")
        exc, vrho = _exc_and_vrho_global(
            params, rho, grid_weights, network, vxc_grad_scale, features,
        )
        return exc, (vrho, None, None, None), None, None

    return eval_xc


# Backwards-compatible alias: the original `make_eval_xc` was local-only.
make_eval_xc = make_eval_xc_local
