"""Base classes for Kohn-Sham DFT trainers.

This module provides abstract base classes that define a common API for
different types of Kohn-Sham DFT trainers, ensuring consistent interfaces
across one-dimensional and three-dimensional implementations.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import jax
from loguru import logger


class BaseKSDFTTrainer(ABC):
    """Abstract base class for Kohn-Sham DFT trainers.

    This class defines the common interface that should be implemented
    by both one-dimensional (od) and time-dependent (td) trainers.
    """

    def __init__(
        self,
        config_dict: dict[str, Any],
        network: Any,
        data_path: Path | None = None,
        seed: int = 0,
    ):
        """Initialize the trainer with configuration.

        Args:
            config_dict: Configuration dictionary containing all settings
            network: Network implementation (varies by trainer type)
            data_path: Path to the data directory
            seed: Random seed for initialization
        """
        # Enable 64-bit precision
        jax.config.update("jax_enable_x64", True)

        self.config = config_dict.copy()
        self.network = network
        self.seed = seed

        # Set up data path
        if data_path is None:
            project_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            data_dir = "od" if self.__class__.__name__.startswith("KSDFT") else "td"
            self.data_path = project_path.parent / "data" / data_dir
        else:
            self.data_path = data_path

        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")

    @abstractmethod
    def prepare_dataset(self) -> Any:
        """Load and prepare dataset from config settings.

        Returns:
            Dataset objects specific to the trainer implementation
        """

    @abstractmethod
    def train(
        self,
        checkpoint_path: Path | None = None,
        checkpoint_save_dir: Path | None = None,
    ) -> tuple:
        """Train the model using config settings.

        Args:
            checkpoint_path: Optional path to load initial parameters from checkpoint
            checkpoint_save_dir: Optional path to save checkpoints

        Returns:
            Tuple containing optimized parameters and training metrics
        """

    @abstractmethod
    def evaluate(
        self,
        checkpoint_path: Path,
        **kwargs,
    ) -> dict:
        """Evaluate a trained model on specific configurations.

        Args:
            checkpoint_path: Path to the checkpoint file
            **kwargs: Implementation-specific evaluation parameters

        Returns:
            Dictionary with evaluation results
        """


class KSDFTTrainerFactory:
    """Factory class to create appropriate KSDFT trainer based on dimensionality."""

    @staticmethod
    def create_trainer(
        dimension: str,
        config_dict: dict[str, Any],
        network: Any,
        data_path: Path | None = None,
        seed: int = 0,
    ) -> BaseKSDFTTrainer:
        """Create a trainer instance based on the specified dimension.

        Args:
            dimension: 'od' for one-dimensional or 'td' for time-dependent (3D)
            config_dict: Configuration dictionary
            network: Network implementation
            data_path: Path to data directory
            seed: Random seed

        Returns:
            Appropriate trainer instance

        Raises:
            ValueError: If dimension is not supported
        """
        if dimension.lower() == "od":
            from qedft.train.od.trainer import KSDFTTrainer

            return KSDFTTrainer(config_dict, network, data_path, seed)
        elif dimension.lower() == "td":
            from qedft.train.td.trainer_legacy_no_jit import TDKSDFTTrainer

            return TDKSDFTTrainer(config_dict, network, data_path, seed)
        else:
            raise ValueError(f"Unsupported dimension: {dimension}. Use 'od' or 'td'.")


# Example usage in a unified script:
def main():
    """Example demonstrating unified API for both trainer types."""

    # Configuration for both trainer types
    od_config = {
        "molecule_name": "h2",
        "dataset1": [128, 384],
        "network_type": "mlp_ksr",
        "n_neurons": 128,
        "n_layers": 3,
        "results_dir": "results/od_test",
    }

    td_config = {
        "features": [512, 512, 1],
        "train_bond_lengths": [0.74, 0.5, 1.5],
        "val_bond_lengths": [0.6, 0.9, 1.2],
        "n_iterations": 5,
        "batch_size": 3,
        "results_dir": "results/td_test",
    }

    # Create network appropriate for the dimension
    def get_network(dimension, config):
        if dimension == "od":
            from qedft.models.networks import GlobalMLP

            network = GlobalMLP(config)
        else:  # td
            from qedft.train.td.trainer_legacy_no_jit import MLP

            network = MLP(
                features=config.get("features", [512, 512, 1]),
                act_fn=config.get("act_fn", "gelu"),
                scale=config.get("scale", 1e-2),
            )
        return network

    network = get_network("od", od_config)

    # Create trainer using the factory
    trainer_od = KSDFTTrainerFactory.create_trainer(
        dimension="od",
        config_dict=od_config,
        network=network,
    )

    network = get_network("td", td_config)

    trainer_td = KSDFTTrainerFactory.create_trainer(
        dimension="td",
        config_dict=td_config,
        network=network,
    )

    print(trainer_od)
    print(trainer_td)


if __name__ == "__main__":
    main()
