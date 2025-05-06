"""Tests for leaf configuration functionality.

This module tests the Config class's leaf configuration methods, including:
- Flattening of nested configurations
- Empty configuration handling
- Key collision resolution
- Mixed data type support

The tests verify correct behavior of the leaf configuration system used
throughout the codebase.
"""

import pytest

from qedft.config.config import Config


def test_get_leaf_config():
    """Tests leaf configuration flattening with nested structures.

    Verifies that nested configuration dictionaries are correctly flattened
    into a single-level leaf configuration, preserving values at all levels.

    The test covers:
    - Basic nested structures
    - Multi-level nesting
    - Mixed data types
    - Value preservation
    """
    config = Config()

    # Test basic nested structure
    config.set("model.quantum.n_qubits", 42)
    config.set("dft.num_grids", 100)

    leaf_config = config.get_leaf_config()
    assert leaf_config["n_qubits"] == 42
    assert leaf_config["num_grids"] == 100

    # Test multi-level nesting
    config.set("model.classical.layers.hidden", 256)
    config.set("model.classical.layers.output", 10)
    config.set("optimizer.params.lr", 0.001)

    leaf_config = config.get_leaf_config()
    assert leaf_config["n_qubits"] == 42
    assert leaf_config["num_grids"] == 100
    assert leaf_config["hidden"] == 256
    assert leaf_config["output"] == 10
    assert leaf_config["lr"] == 0.001

    # Test mixed data types
    config.set("training.batch_size", 32)
    config.set("training.shuffle", True)
    config.set("model.name", "quantum_net")

    leaf_config = config.get_leaf_config()
    assert leaf_config["n_qubits"] == 42
    assert leaf_config["num_grids"] == 100
    assert leaf_config["hidden"] == 256
    assert leaf_config["output"] == 10
    assert leaf_config["lr"] == 0.001
    assert leaf_config["batch_size"] == 32
    assert leaf_config["shuffle"] is True
    assert leaf_config["name"] == "quantum_net"


def test_get_leaf_config_empty():
    """Tests leaf configuration behavior with empty config.

    Verifies that get_leaf_config handles empty configurations gracefully
    by returning an empty dictionary or default values if specified.
    """
    config = Config()
    leaf_config = config.get_leaf_config()
    assert isinstance(leaf_config, dict)


def test_get_leaf_config_overwrite():
    """Tests leaf configuration key collision handling.

    Verifies that when identical keys exist at different paths in the
    configuration tree, the most recently set value is preserved in the
    leaf configuration.
    """
    config = Config()

    config.set("model.layers", 3)
    config.set("network.layers", 5)

    leaf_config = config.get_leaf_config()
    # Most recently set value should be preserved
    assert leaf_config["layers"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
