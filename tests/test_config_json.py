"""Tests for JSON configuration functionality.

This module tests the Config class's JSON file handling capabilities, including:
- Loading configurations from JSON files
- Saving configurations to JSON files
- Round-trip JSON serialization
- Default configuration loading

The tests verify correct behavior of the JSON configuration system used
throughout the codebase.
"""

import json
import os
from pathlib import Path

import pytest

import qedft
from qedft.config.config import Config


@pytest.fixture
def test_json_path(tmp_path):
    """Creates a temporary JSON config file for testing.

    Args:
        tmp_path: Pytest fixture providing temporary directory path.

    Returns:
        Path: Path to temporary JSON configuration file.
    """
    test_config = {
        "model": {
            "quantum": {
                "n_qubits": 5,
                "depth": 3,
            },
        },
        "dft": {
            "num_grids": 100,
            "convergence_threshold": 1e-6,
        },
    }

    json_path = tmp_path / "test_config.json"
    with open(json_path, "w") as f:
        json.dump(test_config, f)

    return json_path


def test_load_from_json(test_json_path):
    """Tests loading configuration from a JSON file.

    Verifies that nested configuration values are correctly loaded from JSON.

    Args:
        test_json_path: Path to test JSON configuration file.
    """
    config = Config()
    config.load_from_json(test_json_path)

    # Test nested values
    assert config.get("model.quantum.n_qubits") == 5
    assert config.get("model.quantum.depth") == 3
    assert config.get("dft.num_grids") == 100
    assert config.get("dft.convergence_threshold") == 1e-6


def test_save_to_json(tmp_path):
    """Tests saving configuration to a JSON file.

    Verifies that configuration values are correctly serialized to JSON.

    Args:
        tmp_path: Temporary directory path for saving test files.
    """
    # Create a config with some values
    config = Config()
    config.set("model.quantum.n_qubits", 5)
    config.set("dft.num_grids", 100)

    # Save to JSON
    save_path = tmp_path / "saved_config.json"
    config.save_to_json(save_path)

    # Read the saved file and verify contents
    with open(save_path) as f:
        saved_config = json.load(f)

    assert saved_config["model"]["quantum"]["n_qubits"] == 5
    assert saved_config["dft"]["num_grids"] == 100


def test_load_save_json_roundtrip(test_json_path, tmp_path):
    """Tests round-trip JSON serialization.

    Verifies that loading and then saving a configuration preserves all values.

    Args:
        test_json_path: Path to source JSON configuration file.
        tmp_path: Temporary directory path for saving test files.
    """
    # Load original config
    config = Config()
    config.load_from_json(test_json_path)

    # Save to new file
    save_path = tmp_path / "roundtrip_config.json"
    config.save_to_json(save_path)

    # Load both files and compare
    with open(test_json_path) as f:
        original_config = json.load(f)
    with open(save_path) as f:
        saved_config = json.load(f)

    assert original_config == saved_config


def test_load_json_config():
    """Tests loading default JSON configuration file.

    Verifies that the default configuration file can be loaded and contains
    expected values.
    """
    # Get the absolute path to the config file
    # Default configuration file path relative to the project root
    project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
    config_path = project_path / "tests" / "test_files" / "test_json.json"

    # Load the JSON file
    with open(config_path) as f:
        config = json.load(f)

    # Test that it's a dictionary
    assert isinstance(config, dict)

    # Test a few key values to ensure proper loading
    assert config["experiment_name"] == "test_name"
    assert config["n_qubits"] == 6
    assert config["n_layers"] == 8
    assert isinstance(config["dataset"], list)
    assert len(config["dataset"]) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
