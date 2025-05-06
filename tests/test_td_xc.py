"""Test the XC functional."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from qedft.models.classical.classical_models import build_local_mlp
from qedft.train.td.xc import eval_xc_custom, exc_and_vrho_custom


@pytest.fixture
def setup_model():
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
    init_fn, _ = local_network

    # Create random test density and initialize params
    key = random.PRNGKey(0)
    test_density = random.normal(key, shape=(num_grid_points,))
    _, init_params = init_fn(key, input_shape=(num_grid_points,))

    return local_network, test_density, init_params, num_grid_points


def test_exc_and_vrho_direct(setup_model):
    local_network, test_density, init_params, _ = setup_model

    # Create XC functional with neural network
    exc_and_vrho_fn = exc_and_vrho_custom(network=local_network)

    # Compute XC energy and potential
    exc, vrho = exc_and_vrho_fn(init_params, test_density)

    # Basic sanity checks
    assert exc.shape == (test_density.shape[0],)  # XC energy matches grid
    assert vrho.shape == test_density.shape  # Potential matches density shape
    assert not jnp.any(jnp.isnan(exc))  # Check for NaN values
    assert not jnp.any(jnp.isnan(vrho))


def test_eval_xc_custom_interface(setup_model):
    local_network, test_density, init_params, _ = setup_model

    # Create XC functional
    exc_and_vrho_fn = exc_and_vrho_custom(network=local_network)

    # Compute using both methods
    exc1, vrho1 = exc_and_vrho_fn(init_params, test_density)
    exc2, vxc2, fxc, kxc = eval_xc_custom(
        xc_code="lda",
        rho=(test_density, 0, 0, 0),
        deriv=1,
        params=init_params,
        exc_and_vrho_custom=exc_and_vrho_fn,
    )

    # Test results match between both methods
    assert jnp.allclose(exc1, exc2), "XC energies do not match"
    assert jnp.allclose(vrho1, vxc2[0]), "XC potentials do not match"

    # Test expected return values
    assert fxc is None, "fxc should be None"
    assert kxc is None, "kxc should be None"
    assert len(vxc2) == 4, "vxc tuple should have length 4"
    assert all(v is None for v in vxc2[1:]), "Higher derivatives should be None"


def test_eval_xc_invalid_deriv(setup_model):
    local_network, test_density, init_params, _ = setup_model

    exc_and_vrho_fn = exc_and_vrho_custom(network=local_network)

    # Test that invalid deriv raises ValueError
    with pytest.raises(ValueError, match="eval_xc: deriv should be set to 1"):
        eval_xc_custom(
            xc_code="lda",
            rho=(test_density, 0, 0, 0),
            deriv=2,  # Invalid value
            params=init_params,
            exc_and_vrho_custom=exc_and_vrho_fn,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
