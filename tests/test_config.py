"""Tests for configuration management functionality.

This module tests the Config class and related utilities, including:
- Basic configuration initialization and access
- YAML file loading and saving
- Nested dictionary operations
- Command line argument parsing
- Configuration updates and validation

The tests verify correct behavior of the configuration system used throughout
the codebase.
"""

import argparse
import tempfile

import pytest
import yaml

from qedft.config.config import Config, setup_config


@pytest.fixture
def sample_config_dict():
    """Returns a sample configuration dictionary for testing.

    The dictionary contains a nested structure with model and training parameters
    that represent a typical configuration.

    Returns:
        dict: Sample configuration with quantum, classical and training parameters.
    """
    return {
        "model": {
            "quantum": {
                "n_qubits": 4,
                "depth": 2,
            },
            "classical": {
                "layers": 3,
            },
        },
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Creates a temporary YAML file with the sample configuration.

    Args:
        sample_config_dict: Fixture providing sample configuration dictionary.

    Returns:
        str: Path to temporary YAML file containing the configuration.
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
    ) as f:
        yaml.dump(sample_config_dict, f)
        return f.name


def test_config_initialization():
    """Tests basic Config class initialization."""
    config = Config()
    assert isinstance(config.config, dict)


def test_config_set_and_get():
    """Tests setting and getting configuration values.

    Verifies:
    - Setting and retrieving simple values
    - Default value handling for missing keys
    - Nested dictionary access
    """
    config = Config()

    # Test setting and getting simple values
    config.set("model.quantum.n_qubits", 5)
    assert config.get("model.quantum.n_qubits") == 5

    # Test getting non-existent value with default
    assert config.get("nonexistent.key", "default") == "default"

    # Test setting nested values
    config.set("new.nested.key", "value")
    assert config.get("new.nested.key") == "value"


def test_config_example():
    """Tests configuration usage examples.

    Demonstrates typical usage patterns including:
    - Default configuration setup
    - Setting and getting values
    - Nested dictionary structure verification
    """
    # Test default config setup
    config = setup_config()
    assert isinstance(config, Config)

    # Test setting and getting values
    config = Config()
    config.set("model.quantum.n_qubits", 42)
    config.set("dft.num_grids", 42)

    # Verify values were set correctly
    assert config.get("model.quantum.n_qubits") == 42
    assert config.get("dft.num_grids") == 42

    # Verify nested dictionary structure
    assert "model" in config.config
    assert "quantum" in config.config["model"]
    assert "dft" in config.config


def test_config_load_from_yaml(temp_config_file, sample_config_dict):
    """Tests loading configuration from YAML file.

    Args:
        temp_config_file: Path to temporary YAML config file.
        sample_config_dict: Expected configuration dictionary.
    """
    config = Config(config_path=temp_config_file)
    n_qubits = sample_config_dict["model"]["quantum"]["n_qubits"]
    assert config.get("model.quantum.n_qubits") == n_qubits
    assert config.get("training.epochs") == sample_config_dict["training"]["epochs"]


def test_config_save_to_yaml(temp_config_file):
    """Tests saving configuration to YAML file.

    Args:
        temp_config_file: Path to temporary YAML file for saving.
    """
    config = Config()
    config.set("test.key", "value")

    # Save to temporary file
    config.save_to_yaml(temp_config_file)

    # Load it back and verify
    with open(temp_config_file) as f:
        loaded_config = yaml.safe_load(f)
    assert loaded_config["test"]["key"] == "value"


def test_config_update():
    """Tests updating configuration with nested dictionaries."""
    config = Config()
    updates = {
        "model": {
            "quantum": {
                "n_qubits": 8,
            },
        },
    }
    config.update_config(updates)
    assert config.get("model.quantum.n_qubits") == 8


def test_config_dict_access():
    """Tests dictionary-style access to configuration values."""
    config = Config()
    config.set("test.key", "value")
    assert config["test"]["key"] == "value"


def test_add_arguments_to_parser():
    """Tests adding configuration options to argument parser."""
    parser = argparse.ArgumentParser()
    config = Config()
    config.add_arguments_to_parser(parser)

    # Test parsing some arguments
    args = parser.parse_args(["--n_qubits", "6"])
    assert hasattr(args, "n_qubits")
    assert args.n_qubits == 6


def test_config_from_args():
    """Tests creating configuration from parsed arguments."""
    parser = argparse.ArgumentParser()
    config = Config()
    config.add_arguments_to_parser(parser)

    # Test creating config from args
    args = parser.parse_args(["--n_qubits", "6"])
    config = config.from_args(args)
    assert config.get("n_qubits") == 6


def test_setup_config():
    """Tests default configuration setup."""
    config = setup_config()
    assert isinstance(config, Config)


def test_config_with_kwargs():
    """Tests configuration with keyword arguments."""
    config = Config()
    config.set("model.quantum.n_qubits", 10)
    config.set("training.epochs", 200)
    config.set("model.quantum.depth", 100)

    assert config.get("training.epochs") == 200
    assert config.get("model.quantum.n_qubits") == 10
    assert config.get("model.quantum.depth") == 100


def test_invalid_config_path():
    """Tests error handling for invalid configuration file paths."""
    with pytest.raises(FileNotFoundError):
        Config(config_path="nonexistent.yaml")


def test_nested_update():
    """Tests nested configuration updates.

    Verifies that nested updates preserve existing values at deeper levels
    while updating specified values.
    """
    config = Config()
    initial = {
        "model": {
            "quantum": {
                "n_qubits": 4,
                "depth": 2,
            },
        },
    }
    update = {
        "model": {
            "quantum": {
                "n_qubits": 8,
            },
        },
    }
    config.update_config(initial)
    config.update_config(update)
    assert config.get("model.quantum.n_qubits") == 8
    # Should preserve existing values
    assert config.get("model.quantum.depth") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
