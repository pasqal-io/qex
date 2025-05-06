"""
We compute the exchange-correlation energy and its derivatives with respect to
the density.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp


def get_exc_and_vrho(
    params: dict,
    rho: jnp.ndarray,
    network: Callable,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the energy density and the gradient of the energy density
    with respect to the density using JAX."""

    _, apply_fn = network
    to_stack = (rho,)
    for key, value in kwargs.items():
        if key == "params_grid_coords":
            x, y, z = value[:, 0], value[:, 1], value[:, 2]

            to_stack += (
                x,
                y,
                z,
            )
        else:
            to_stack += (value,)
    rho_and_grad = jnp.stack(to_stack, axis=1)
    exc, vjp_fn = jax.vjp(
        lambda x: apply_fn(params, x).squeeze(),
        rho_and_grad,
    )
    (vrho,) = vjp_fn(jnp.ones_like(exc))
    vrho = vrho[:, [0]].flatten()

    return exc, vrho


def exc_and_vrho_custom(network: Callable, **fixed_kwargs):
    """Creates a partial function for exc_and_vrho that can be jitted with only params and rho.
    TODO: This is what changed in comparison to the old code, I can clean it up later
    """

    @jax.jit
    def _exc_and_vrho_partial(params: dict, rho: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return get_exc_and_vrho(params, rho, network, **fixed_kwargs)

    return _exc_and_vrho_partial


def get_eval_xc(
    xc_code: str,
    rho: jnp.ndarray,
    spin: int = 0,
    relativity: int = 0,
    deriv: int = 0,
    verbose: bool | None = None,
    params: dict | None = None,
    params_grid_coords: jnp.ndarray | None = None,
    normalize_with_log: bool = False,
    normalize_with_density: bool | None = None,
    method: str = "density",
    exc_and_vrho_custom: Callable | None = None,
) -> tuple[jnp.ndarray, tuple, jnp.ndarray | None, jnp.ndarray | None]:
    """Get the exchange-correlation energy and potential."""

    if deriv != 1:
        raise ValueError("eval_xc: deriv should be set to 1.")

    rho0, _, _, _ = rho
    exc, vrho = exc_and_vrho_custom(params, rho0)

    vgamma = None
    vlapl = None
    vtau = None
    fxc = None  # 2nd order functional derivative
    kxc = None  # 3rd order functional derivative

    vxc = (vrho, vgamma, vlapl, vtau)

    return exc, vxc, fxc, kxc


def eval_xc_custom(
    xc_code: str,
    rho: jnp.ndarray,
    spin: int = 0,
    relativity: int = 0,
    deriv: int = 0,
    verbose: bool | None = None,
    params: dict | None = None,
    params_grid_coords: jnp.ndarray | None = None,
    normalize_with_log: bool = False,
    normalize_with_density: bool | None = None,
    method: str = "density",
    exc_and_vrho_custom: Callable | None = None,
) -> tuple[jnp.ndarray, tuple, jnp.ndarray | None, jnp.ndarray | None]:
    """Customizable evaluation of the XC functional."""
    return get_eval_xc(
        xc_code,
        rho,
        spin=spin,
        relativity=relativity,
        deriv=deriv,
        verbose=verbose,
        params=params,
        params_grid_coords=params_grid_coords,
        normalize_with_log=normalize_with_log,
        normalize_with_density=normalize_with_density,
        method=method,
        exc_and_vrho_custom=exc_and_vrho_custom,
    )


if __name__ == "__main__":

    from jax import random

    from qedft.models.classical.classical_models import build_local_mlp

    # Enable double precision
    jax.config.update("jax_enable_x64", True)

    # Set up model parameters
    num_grid_points = 10
    model_config = {
        "n_neurons": 513,
        "n_layers": 2,
        "activation": "tanh",
        "n_outputs": 1,
        "density_normalization_factor": 2.0,
        "grids": jnp.ones((num_grid_points,)),
    }

    # Initialize neural network model
    local_network = build_local_mlp(**model_config)
    init_fn, apply_fn = local_network

    # Create random test density
    key = random.PRNGKey(0)
    test_density = random.normal(key, shape=(num_grid_points,))

    # Initialize model parameters
    _, init_params = init_fn(key, input_shape=(num_grid_points,))

    # Method 1: Direct computation
    # Create XC functional with our neural network
    exc_and_vrho_fn = exc_and_vrho_custom(network=local_network)

    # Compute XC energy and potential
    exc, vrho = exc_and_vrho_fn(init_params, test_density)
    print("Direct computation results:")
    print(f"XC Energy: {exc}")
    print(f"XC Potential: {vrho}")

    # Method 2: Using eval_xc_custom interface
    exc2, vxc2, fxc, kxc = eval_xc_custom(
        xc_code="lda",  # Type of functional (not used for custom implementation)
        rho=(test_density, 0, 0, 0),  # Density and its derivatives
        deriv=1,  # First derivative only
        params=init_params,
        exc_and_vrho_custom=exc_and_vrho_fn,
    )

    # Verify both methods give same results
    assert jnp.allclose(exc, exc2), "XC energies do not match"
    assert jnp.allclose(vrho, vxc2[0]), "XC potentials do not match"
    print("\nBoth methods produce identical results!")
