"""A configuration management system for QeDFT that makes it easy to handle settings.

This module provides a flexible way to manage configuration settings through code, files,
and command line arguments. It supports nested configurations with intuitive dot notation
access.

Example:
  Simplest way
  >>> config = Config('path/to/config.yaml')
  >>> config_dict = config.config
  or
  >>> config = Config()
  >>> config.set('model.quantum.n_qubits', 4)
  >>> config.get('model.quantum.n_qubits')
  4
  Overwrite with settings from file
  >>> config.load_from_yaml('settings.yaml')
  >>> config['training']['batch_size']
  32
  Save to file
  >>> config.save_to_json('config.json')
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from yaml import SafeLoader

import qedft


class Config:
    """A flexible configuration manager for QeDFT settings.

    The Config class provides an intuitive interface for managing hierarchical settings.
    It supports loading from YAML/JSON files, programmatic modification, command line
    arguments, dot notation access, and saving to files.

    Attributes:
        config: The configuration dictionary containing all settings.
        default_config_path: Path to the default configuration file.

    Example:
        Basic usage:
        >>> config = Config()
        >>> config.set('model.quantum.n_qubits', 4)
        >>> config.get('model.quantum.n_qubits')
        4

        Loading from file:
        >>> config.load_from_yaml('settings.yaml')

        Command line integration:
        >>> parser = argparse.ArgumentParser()
        >>> config.add_arguments_to_parser(parser)

        Dictionary-style access:
        >>> config['training']['batch_size']
        32

        Saving to file:
        >>> config.save_to_json('config.json')

        Loading with custom paths:
        >>> config = Config(
        ...     config_path='custom_config.yaml',
        ...     default_config_path='default_config.yaml'
        ... )

        Accessing nested settings:
        >>> config.get('model.quantum.n_qubits', default=2)
        4
        >>> config.get('nonexistent.path', default='fallback')
        'fallback'
    """

    def __init__(
        self,
        config_path: str | None = None,
        default_config_path: str | None = None,
        **kwargs: Any,
    ):
        # Set default config path if not provided
        if default_config_path is None:
            project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
            self.default_config_path = project_path / "qedft" / "config" / "default_config.yaml"
        else:
            self.default_config_path = Path(default_config_path)

        if config_path is None:
            self.config = self._load_default_config()
        else:
            self.config = {}
            self.load_from_yaml(config_path)

        for key, value in kwargs.items():
            self.set(key, value)

    def _load_default_config(self) -> dict[str, Any]:
        try:
            with open(self.default_config_path) as f:
                return yaml.load(f, Loader=SafeLoader)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Couldn't find the default config file at {self.default_config_path}",
            ) from exc

    def set(self, key: str, value: Any) -> None:
        """Sets a configuration value using dot notation.

        The method automatically creates any missing intermediate dictionaries in the
        hierarchy.

        Args:
            key: Setting path using dots (e.g. 'model.quantum.n_qubits')
            value: Value to set at the specified path

        Example:
            >>> config = Config()
            >>> config.set('model.quantum.n_qubits', 4)
            >>> config.set('training.batch_size', 32)
            >>> config.set('optimizer.learning_rate', 0.001)
            >>> config.get('model.quantum.n_qubits')
            4

        Raises:
            TypeError: If key is not a string.
            ValueError: If key is empty.
        """
        if not isinstance(key, str):
            raise TypeError("The key must be a string")
        if not key:
            raise ValueError("The key can't be empty")
        keys = key.split(".")
        current = self.config

        # Navigate to the nested location
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Gets a configuration value using dot notation.

        Provides a clean way to access nested settings with fallback values.

        Args:
            key: Setting path using dots (e.g. 'model.quantum.n_qubits')
            default: Value to return if setting doesn't exist

        Returns:
            The setting value if found, otherwise the default value

        Example:
            >>> config = Config()
            >>> config.set('model.quantum.n_qubits', 4)
            >>> config.get('model.quantum.n_qubits', default=2)
            4
            >>> config.get('nonexistent.path', default='fallback')
            'fallback'
            >>> config.get('training.batch_size', default=32)
            32
            >>> config.get('optimizer.lr', default=0.001)
            0.001
        """
        try:
            current = self.config
            for k in key.split("."):
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def load_from_yaml(self, config_path: str) -> None:
        """Loads settings from a YAML file.

        Updates existing settings with values from the file. Nested settings are merged
        properly.

        Args:
            config_path: Path to YAML configuration file

        Example YAML file:
            model:
              quantum:
                n_qubits: 4
                depth: 3
            training:
              batch_size: 32

        Example:
            >>> config = Config()
            >>> config.load_from_yaml('settings.yaml')
            >>> config.get('model.quantum.n_qubits')
            4

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        config_path = os.path.abspath(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Couldn't find the config file: {config_path}")
        with open(config_path) as f:
            yaml_config = yaml.load(f, Loader=SafeLoader)
            self.update_config(yaml_config)

    def update_config(self, updates: dict[str, Any]) -> None:
        """Update settings with new values, handling nested settings properly."""

        def update_recursive(base: dict[str, Any], updates: dict[str, Any]) -> None:
            for key, value in updates.items():
                if isinstance(value, dict) and key in base:
                    update_recursive(base[key], value)
                else:
                    base[key] = value

        update_recursive(self.config, updates)

    def save_to_yaml(self, filepath: str) -> None:
        """Save your current settings to a YAML file.

        Args:
            filepath: Where to save the file

        Example:
            >>> config = Config()
            >>> config.set('model.quantum.n_qubits', 4)
            >>> config.save_to_yaml('settings.yaml')
        """
        with open(filepath, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def __getitem__(self, key: str) -> Any:
        """Gets a setting using dictionary-style access.

        This lets you use square brackets like:
        >>> config = Config()
        >>> config.set('n_qubits', 4)
        >>> config['n_qubits']
        4

        Args:
            key: The setting name

        Returns:
            The setting's value

        Raises:
            KeyError: If the key doesn't exist.
        """
        return self.config[key]

    def add_arguments_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """Now an instance method instead of class method"""

        def add_dict_arguments(d: dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                arg_name = f"{prefix}{key}" if prefix else key
                cmd_arg = f"--{arg_name.replace('.', '-')}"

                if isinstance(value, dict):
                    add_dict_arguments(value, f"{arg_name}.")
                    continue

                if isinstance(value, bool):
                    parser.add_argument(
                        cmd_arg,
                        dest=arg_name,
                        action="store_true" if not value else "store_false",
                    )
                elif isinstance(value, (list, tuple)):
                    parser.add_argument(
                        cmd_arg,
                        dest=arg_name,
                        type=type(value[0]) if value else str,
                        nargs="+",
                        default=value,
                    )
                else:
                    parser.add_argument(
                        cmd_arg,
                        dest=arg_name,
                        type=type(value) if value is not None else str,
                        default=value,
                    )

        with open(self.default_config_path) as f:  # Changed from cls.DEFAULT_CONFIG_PATH
            add_dict_arguments(yaml.load(f, Loader=SafeLoader))

    def from_args(self, args: argparse.Namespace) -> "Config":
        """Now an instance method instead of class method"""
        # Convert args to dictionary, filtering out None values
        arg_dict = {k: v for k, v in vars(args).items() if v is not None}
        return Config(**arg_dict)  # Changed from cls to Config

    def load_from_json(self, config_path: str, merge_with_defaults: bool = False) -> None:
        """Loads settings from a JSON file.

        Args:
            config_path: Where to find the JSON file
            merge_with_defaults: If True, update existing settings. If False, replace them all.

        Example:
            >>> config = Config()
            >>> config.load_from_json('settings.json', merge_with_defaults=True)
        """
        with open(config_path) as f:
            json_config = json.load(f)
            if merge_with_defaults:
                self.update_config(json_config)
            else:
                self.config = json_config

    def save_to_json(self, filepath: str) -> None:
        """Saves your current settings to a JSON file.

        Args:
            filepath: Where to save the file

        Example:
            >>> config = Config()
            >>> config.set('model.quantum.n_qubits', 4)
            >>> config.save_to_json('settings.json')
        """
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=4)

    def get_leaf_config(self) -> dict[str, Any]:
        """Gets a flattened version of your settings, with just the end values.

        Returns:
            A dictionary with just the leaf settings

        Example:
            >>> config = Config()
            >>> config.set('model.quantum.n_qubits', 42)
            >>> config.get_leaf_config()
            {'n_qubits': 42}
        """
        leaf_config = {}

        def collect_leaves(d: dict[str, Any]) -> None:
            for key, value in d.items():
                if isinstance(value, dict):
                    collect_leaves(value)
                else:
                    leaf_config[key] = value

        collect_leaves(self.config)
        return leaf_config


def setup_config(default_config_path: str | None = None) -> Config:
    """Sets up your configuration from command-line options or defaults.

    Args:
        default_config_path: Optional path to default config file

    Returns:
        A Config ready to use with your settings

    Example:
        >>> config = setup_config('default_config.yaml')
        >>> config.get('model.quantum.n_qubits')
        4
    """
    parser = argparse.ArgumentParser(description="QeDFT configuration")
    config = Config(default_config_path=default_config_path)  # Create instance first
    config.add_arguments_to_parser(parser)  # Call instance method

    # Parse args only if running from command line and not during pytest
    if not sys.argv[0].endswith("ipykernel_launcher.py") and "pytest" not in sys.modules:
        args = parser.parse_args()
        return config.from_args(args)  # Call instance method

    # For notebook usage, return default config
    return config


if __name__ == "__main__":
    # Test command line arguments
    config = setup_config()
    print("Default config:")
    print(config.config)

    # Test with command line arguments
    sys.argv = [
        "config.py",
        "--n_qubits",
        "42",
        "--n_layers",
        "42",
    ]
    config_with_args = setup_config()
    print("\nConfig with command line arguments:")
    print(config_with_args.config)
