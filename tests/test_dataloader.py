"""Tests for dataset loading functionality.

This module tests the dataset loading utilities, including:
- Basic dataset loading with various configurations
- Mock dataset implementations for testing
- Error handling for mismatched parameters
- Path handling and configuration loading

The tests verify correct loading behavior and error conditions using mock datasets.
"""

from pathlib import Path

import numpy as np
import pytest
from jax_dft import datasets

from qedft.data_io.dataset_loader import (
    load_molecular_datasets,
    load_molecular_datasets_from_config,
)


class MockDataset:
    """Mock dataset class for testing.

    Simulates a molecular dataset with basic properties needed for testing the loader
    functionality.

    Attributes:
        num_electrons: Number of electrons in the system.
        grids: 2D array representing spatial grids.
        path: Path to the mock dataset.
        molecule_locations: Array of molecular coordinates.
    """

    def __init__(self, path, num_grids):
        """Initializes mock dataset.

        Args:
            path: Path to mock dataset location.
            num_grids: Number of grid points in each dimension.
        """
        self.num_electrons = 2
        self.grids = np.zeros((num_grids, num_grids))
        self.path = path
        # Place molecule off-center for grid centering tests
        self.molecule_locations = np.array([[num_grids / 4, num_grids / 4]])

    def get_molecules(self, distances):
        """Returns mock molecule set for given distances.

        Args:
            distances: Array of molecular distances.

        Returns:
            MockMoleculeSet instance.
        """
        return MockMoleculeSet(distances)


class MockMoleculeSet:
    """Mock molecule set for testing.

    Represents a collection of molecules with fixed locations.

    Attributes:
        locations: Array of molecular coordinates.
    """

    def __init__(self, distances):
        """Initializes mock molecule set.

        Args:
            distances: Array of molecular distances (unused in mock).
        """
        self.locations = np.array([[num_grids / 2, num_grids / 2] for num_grids in [512]])


@pytest.fixture
def mock_datasets(monkeypatch):
    """Fixture that patches the dataset module to use mock implementations."""
    monkeypatch.setattr(datasets, "Dataset", MockDataset)


def test_load_molecular_datasets_basic(mock_datasets):
    """Tests basic dataset loading functionality.

    Verifies that a single dataset can be loaded with basic parameters.
    """
    molecule_names = ["h2"]
    datasets_params = {"dataset1": [174]}
    num_grids = 512

    result = load_molecular_datasets(
        molecule_names=molecule_names,
        datasets_params=datasets_params,
        num_grids=num_grids,
        check_grid_centering=False,
    )

    assert len(result) == 1, "Expected one dataset to be loaded"
    assert isinstance(
        result[0][0],
        MockDataset,
    ), "Loaded dataset should be an instance of MockDataset"


def test_load_molecular_datasets_with_dict_num_grids(mock_datasets):
    """Tests dataset loading with dictionary-specified grid sizes.

    Verifies loading multiple datasets with different grid sizes per molecule.
    """
    molecule_names = ["h2", "h2o"]
    datasets_params = {"dataset1": [174], "dataset2": [348]}
    num_grids = {"h2": 512, "h2o": 1024}

    result = load_molecular_datasets(
        molecule_names=molecule_names,
        datasets_params=datasets_params,
        num_grids=num_grids,
        check_grid_centering=False,
    )

    assert len(result) == 2, "Expected two datasets to be loaded"


def test_load_molecular_datasets_mismatch_error():
    """Tests error handling for mismatched parameters.

    Verifies that appropriate error is raised when molecule names don't match
    dataset parameters.
    """
    molecule_names = ["h2"]
    datasets_params = {"dataset1": [174], "dataset2": [348]}
    num_grids = 512

    with pytest.raises(ValueError, match="Number of molecule names must match"):
        load_molecular_datasets(
            molecule_names=molecule_names,
            datasets_params=datasets_params,
            num_grids=num_grids,
            check_grid_centering=False,
        )


def test_load_molecular_datasets_with_base_path(mock_datasets):
    """Tests dataset loading with custom base path.

    Verifies that datasets can be loaded from specified base directory.
    """
    molecule_names = ["h2"]
    datasets_params = {"dataset1": [174]}
    num_grids = 512
    base_path = Path("data/test")

    result = load_molecular_datasets(
        molecule_names=molecule_names,
        datasets_params=datasets_params,
        num_grids=num_grids,
        base_path=base_path,
        check_grid_centering=False,
    )

    assert len(result) == 1, "Expected one dataset to be loaded"
    assert str(base_path) in result[0][0].path, "Base path should be part of the dataset path"


def test_load_molecular_datasets_from_config(mock_datasets):
    """Tests loading datasets from configuration dictionary.

    Verifies that datasets can be properly loaded using a config dict.
    """
    config = {
        "molecule_names": ["h2"],
        "dataset1": [174],
        "num_grids": 512,
    }
    base_path = Path("data/test")

    result = load_molecular_datasets_from_config(
        config,
        base_path,
        check_grid_centering=False,
    )

    assert len(result) == 1, "Expected one dataset to be loaded"
    assert isinstance(
        result[0][0],
        MockDataset,
    ), "Loaded dataset should be an instance of MockDataset"


def test_load_molecular_datasets_from_config_multiple_molecules(mock_datasets):
    """Tests loading multiple molecules from configuration.

    Verifies loading of multiple datasets with different parameters from config.
    """
    config = {
        "molecule_names": ["h2", "h2o"],
        "dataset1": [174],
        "dataset2": [348],
        "num_grids": {"h2": 512, "h2o": 1024},
    }
    base_path = Path("data/test")

    result = load_molecular_datasets_from_config(
        config,
        base_path,
        check_grid_centering=False,
    )

    assert len(result) == 2, "Expected two datasets to be loaded"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
