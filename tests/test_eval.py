"""Tests for the evaluation module."""

import os
import pickle
import tempfile
from unittest.mock import patch

import jax.numpy as jnp
import pytest

from qedft.train.od import eval


class TestDataset:
    """Mock dataset for testing."""

    def __init__(self, num_samples=3):
        self.locations = [jnp.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]) for d in range(num_samples)]
        self.nuclear_charges = [jnp.array([1.0, 1.0]) for _ in range(num_samples)]


@pytest.fixture
def mock_params():
    """Create mock model parameters."""
    return {"weights": jnp.ones((10, 10)), "biases": jnp.zeros(10)}


@pytest.fixture
def mock_grids():
    """Create mock grid points."""
    return jnp.linspace(-5, 5, 100)


@pytest.fixture
def mock_initial_density():
    """Create mock initial density."""
    return [jnp.ones(100) * 0.1 for _ in range(3)]


@pytest.fixture
def mock_neural_xc_fn():
    """Create mock neural XC function."""

    def fn(density, params):
        return jnp.sum(density) * 0.1

    return fn


def test_load_model_params():
    """Test loading model parameters from a checkpoint file."""
    test_params = {"weights": jnp.ones((5, 5))}

    # Create a temporary checkpoint file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        with open(tmp.name, "wb") as f:
            pickle.dump(test_params, f)

        # Test loading the parameters
        loaded_params = eval.load_model_params(tmp.name)

    # Clean up
    os.unlink(tmp.name)

    # Check that the loaded parameters match the original
    assert isinstance(loaded_params, dict)
    assert "weights" in loaded_params
    assert loaded_params["weights"].shape == (5, 5)
    assert jnp.array_equal(loaded_params["weights"], test_params["weights"])


@patch("qedft.train.od.eval.kohn_sham")
def test_get_states(mock_kohn_sham, mock_params):
    """Test getting states for different distances."""
    # Setup
    mock_kohn_sham.return_value = {"density": jnp.ones(100), "total_energy": -1.0}

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        with open(tmp.name, "wb") as f:
            pickle.dump(mock_params, f)

        # Test
        plot_distances = [1.0, 2.0, 3.0]
        plot_set = TestDataset(num_samples=3)
        plot_initial_density = [jnp.ones(100) * 0.1 for _ in range(3)]

        states = eval.get_states(
            ckpt_path=tmp.name,
            kohn_sham_fn=lambda **kwargs: mock_kohn_sham(**kwargs),
            plot_distances=plot_distances,
            plot_set=plot_set,
            plot_initial_density=plot_initial_density,
        )

    # Clean up
    os.unlink(tmp.name)

    # Assertions
    assert mock_kohn_sham.call_count == 3
    assert "density" in states
    assert "total_energy" in states
    assert states["density"].shape == (3, 100)
    assert states["total_energy"].shape == (3,)


@patch("qedft.train.od.eval.get_states")
@patch("qedft.train.od.eval.kohn_sham")
def test_eval_trained_model(
    mock_kohn_sham,
    mock_get_states,
    mock_grids,
    mock_neural_xc_fn,
    mock_initial_density,
):
    """Test evaluating a trained model."""
    # Setup
    mock_states = {
        "density": jnp.ones((3, 100)),
        "total_energy": jnp.array([-1.0, -2.0, -3.0]),
    }
    mock_get_states.return_value = mock_states

    # Test
    plot_distances = [1.0, 2.0, 3.0]
    plot_set = TestDataset(num_samples=3)

    result = eval.eval_trained_model(
        ckpt_path="dummy_path.pkl",
        plot_distances=plot_distances,
        plot_set=plot_set,
        plot_initial_density=mock_initial_density,
        num_electrons=2,
        num_iterations=100,
        grids=mock_grids,
        neural_xc_energy_density_fn=mock_neural_xc_fn,
        use_amplitude_encoding=True,
        use_lda=False,
    )

    # Assertions
    assert mock_get_states.call_count == 1
    assert result == mock_states
    assert "density" in result
    assert "total_energy" in result


def test_kohn_sham_with_amplitude_encoding(mock_params, mock_grids, mock_neural_xc_fn):
    """Test Kohn-Sham calculation with amplitude encoding."""
    with patch("qedft.train.od.scf.kohn_sham_amplitude_encoded") as mock_ks_amp:
        mock_ks_amp.return_value = {"density": jnp.ones(100), "total_energy": -1.0}

        result = eval.kohn_sham(
            params=mock_params,
            locations=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            nuclear_charges=jnp.array([1.0, 1.0]),
            num_electrons=2,
            num_iterations=100,
            grids=mock_grids,
            neural_xc_energy_density_fn=mock_neural_xc_fn,
            use_amplitude_encoding=True,
        )

        assert mock_ks_amp.call_count == 1
        assert "density" in result
        assert "total_energy" in result


def test_kohn_sham_without_amplitude_encoding(mock_params, mock_grids, mock_neural_xc_fn):
    """Test Kohn-Sham calculation without amplitude encoding."""
    with patch("qedft.train.od.eval.scf_jax_dft.kohn_sham") as mock_ks:
        mock_ks.return_value = {"density": jnp.ones(100), "total_energy": -1.0}

        result = eval.kohn_sham(
            params=mock_params,
            locations=jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            nuclear_charges=jnp.array([1.0, 1.0]),
            num_electrons=2,
            num_iterations=100,
            grids=mock_grids,
            neural_xc_energy_density_fn=mock_neural_xc_fn,
            use_amplitude_encoding=False,
        )

        assert mock_ks.call_count == 1
        assert "density" in result
        assert "total_energy" in result


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
