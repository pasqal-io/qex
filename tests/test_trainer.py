"""Test the trainer."""

from pathlib import Path
from unittest.mock import Mock, patch

import jax
import jax.random
import numpy as np
import pytest

from qedft.train.od.trainer import KSDFTTrainer


@pytest.fixture
def mock_config():
    return {
        "molecule_name": "h2",
        "dataset1": [128, 384],
        "network_type": "mlp",
        "n_neurons": 32,
        "n_layers": 2,
        "activation": "tanh",
        "density_normalization_factor": 2.0,
        "wrap_with_negative_transform": True,
        "wrap_self_interaction": True,
        "maxfun": 10,
        "maxiter": 10,
    }


@pytest.fixture
def mock_network():
    network = Mock()
    network.get_config.return_value = {}
    # Update the mock init_fn to handle both key and shape arguments
    network.build_network.return_value = (
        lambda key, shape=None: {"params": np.zeros(10)},  # init_fn
        lambda params, x: np.zeros_like(x),  # apply_fn
    )
    return network


def test_trainer_initialization(mock_config, mock_network):
    with patch("jax.config.update"):
        trainer = KSDFTTrainer(
            config_dict=mock_config,
            network=mock_network,
            data_path=Path("/dummy/path"),
            seed=42,
        )

        assert trainer.config == mock_config
        assert trainer.network == mock_network
        # Fix: Check if seed is a DeviceArray (the actual type of PRNGKey)
        assert isinstance(trainer.seed, jax.Array)
        # Optionally, verify it's a 2-element array which is characteristic of PRNGKeys
        assert trainer.seed.shape == (2,)


@pytest.fixture
def mock_dataset():
    dataset = Mock()
    dataset.grids = np.linspace(-5, 5, 100)
    return dataset


@patch("qedft.train.od.trainer.load_molecular_datasets_from_config")
@patch("qedft.train.od.trainer.scf.get_initial_density")
def test_prepare_dataset(
    mock_get_density,
    mock_load_datasets,
    mock_config,
    mock_network,
    mock_dataset,
):
    # Setup mocks
    mock_train_set = np.array([[1.0, 2.0], [3.0, 4.0]])
    mock_initial_density = np.ones(10)

    mock_load_datasets.return_value = [(mock_dataset, mock_train_set)]
    mock_get_density.return_value = mock_initial_density

    # Create trainer and test prepare_dataset
    with patch("jax.config.update"):
        trainer = KSDFTTrainer(
            config_dict=mock_config,
            network=mock_network,
            data_path=Path("/dummy/path"),
        )

        dataset, train_set, initial_density, grids = trainer.prepare_dataset()

        assert dataset == mock_dataset
        assert np.array_equal(train_set, mock_train_set)
        assert np.array_equal(initial_density, mock_initial_density)
        assert np.array_equal(grids, mock_dataset.grids)


@patch("qedft.train.od.trainer.scipy.optimize.fmin_l_bfgs_b")
@patch("qedft.train.od.trainer.create_training_step")
@patch("qedft.train.od.trainer.build_xc_functional")
def test_train(
    mock_build_xc,
    mock_create_training_step,
    mock_fmin_l_bfgs_b,
    mock_config,
    mock_network,
):
    # Setup mocks
    mock_params = np.ones(10)
    mock_loss = 0.5
    mock_info = {"success": True}

    # Mock the build_xc_functional to return proper init and apply functions
    mock_build_xc.return_value = (
        lambda key, shape=None: {"params": np.zeros(10)},  # init_fn
        lambda params, x: np.zeros_like(x),  # apply_fn
    )

    mock_fmin_l_bfgs_b.return_value = (mock_params, mock_loss, mock_info)
    mock_create_training_step.return_value = lambda x: 0.0

    # Create trainer with mocked prepare_dataset
    with patch("jax.config.update"):
        trainer = KSDFTTrainer(
            config_dict=mock_config,
            network=mock_network,
            data_path=Path("/dummy/path"),
        )

        # Mock prepare_dataset
        trainer.prepare_dataset = Mock(
            return_value=(
                mock_dataset,
                np.ones((2, 2)),
                np.ones(10),
                np.linspace(-5, 5, 100),
            ),
        )

        # Test train method
        params, loss, info = trainer.train()

        assert np.array_equal(params, mock_params)
        assert loss == mock_loss
        assert info == mock_info


def test_main():
    with patch("qedft.train.od.trainer.KSDFTTrainer") as MockTrainer:
        # Mock the trainer's train method
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = (
            np.zeros(10),  # params
            0.1,  # loss
            {"success": True},  # info
        )
        MockTrainer.return_value = mock_trainer_instance

        # Import and run main
        from qedft.train.od.trainer import main

        main()

        # Verify trainer was created and train was called
        assert MockTrainer.called
        assert mock_trainer_instance.train.called


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
