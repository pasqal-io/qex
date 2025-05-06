"""Trainer for Kohn-Sham DFT models.

This module provides a high-level trainer class that simplifies training neural network-based
exchange-correlation functionals for Kohn-Sham DFT. It handles all the training boilerplate
like data loading, optimization, checkpointing etc., allowing users to focus on just defining
their network architecture.

By default, everything is jitted.
"""

import os
from pathlib import Path

import jax
import numpy as np
import scipy.optimize
from jax_dft import np_utils, scf
from loguru import logger

import qedft
from qedft.data_io.dataset_loader import load_molecular_datasets_from_config
from qedft.models.wrappers import build_xc_functional
from qedft.train.od.train import create_kohn_sham_fn, create_loss_fn, create_training_step


class KSDFTTrainer:
    """Trainer for Kohn-Sham DFT models."""

    def __init__(
        self,
        config_dict: dict,
        network: "KohnShamNetwork",
        data_path: Path,
        seed: int = 0,
    ):
        """Initialize trainer with configuration.

        Args:
            config_dict: Configuration dictionary containing all settings
            network: Network implementation following KohnShamNetwork protocol
            data_path: Path to the data directory
            seed: Random seed for initialization
        """
        # Enable 64-bit precision
        jax.config.update("jax_enable_x64", True)

        self.network = network
        self.data_path = data_path
        self.seed = jax.random.PRNGKey(seed)

        # Start with base config and update with network config
        self.config = config_dict.copy()
        # self.config.update(network.get_config())
        self.default_checkpoint_save_dir = (
            Path(os.path.dirname(os.path.dirname(qedft.__file__))) / "tests" / "ckpts"
        )

        logger.info(f"Initialized trainer with config: {self.config}")

    def prepare_dataset(self) -> tuple:
        """Load and prepare dataset from config settings."""
        # Load dataset using config
        base_path = self.data_path
        list_datasets = load_molecular_datasets_from_config(
            self.config,
            base_path,
            check_grid_centering=True,
        )
        dataset, train_set = list_datasets[0]

        # Get initial density and grids
        initial_density = scf.get_initial_density(
            train_set,
            method="noninteracting",
        )

        return dataset, train_set, initial_density, dataset.grids

    def train(self, checkpoint_path: Path = None, checkpoint_save_dir: Path = None) -> tuple:
        """Train the model using config settings.

        Args:
            checkpoint_path: Optional path to load initial parameters from checkpoint
            checkpoint_save_dir: Optional path to save checkpoints

        Returns:
            Tuple of (optimized_params, final_loss, optimization_info)
        """
        # Prepare dataset
        dataset, train_set, initial_density, grids = self.prepare_dataset()

        # Create and wrap network
        base_network = self.network.build_network(grids)
        network = build_xc_functional(
            network=base_network,
            grids=grids,
            config=self.config,
        )

        # Initialize network
        init_fn, neural_xc_energy_density_fn = network

        # Load from checkpoint if it exists, otherwise initialize fresh
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Loading parameters from checkpoint: {checkpoint_path}")
            init_params = np.load(checkpoint_path, allow_pickle=True)
        else:
            logger.info("Initializing fresh parameters")
            init_params = init_fn(self.seed)

        # JIT the neural_xc_energy_density_fn if requested
        # IF we use a QPU then we don't need to JIT here
        if self.config.get("jit_neural_xc", True):
            logger.info("Jitting neural_xc_energy_density_fn")
            neural_xc_energy_density_fn = jax.jit(neural_xc_energy_density_fn)

        # Need to flatten the parameters for the optimizer
        # spec defines the tree structure of the parameters
        spec, flatten_init_params = np_utils.flatten(init_params)

        # Create training components
        # batch_kohn_sham is a jitted function
        _, batch_kohn_sham = create_kohn_sham_fn(
            self.config,
            dataset,
            grids,
            neural_xc_energy_density_fn,
            spec,
        )

        loss_fn = create_loss_fn(
            batch_kohn_sham,
            grids,
            dataset,
            self.config,
        )

        # JIT the loss function if requested
        if self.config.get("jit_loss", True):
            logger.info("Jitting loss_fn")
            value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
        else:
            value_and_grad_fn = jax.value_and_grad(loss_fn)

        if checkpoint_save_dir is None:
            checkpoint_save_dir = self.default_checkpoint_save_dir
            logger.info(f"Using default checkpoint save directory: {checkpoint_save_dir}")
        else:
            # Create checkpoint save directory if it doesn't exist
            checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint save directory: {checkpoint_save_dir}")

        training_step = create_training_step(
            value_and_grad_fn,
            train_set,
            initial_density,
            save_every_n=self.config.get("save_every_n", 10),
            initial_checkpoint_index=self.config.get("initial_checkpoint_index", 0),
            checkpoint_dir=checkpoint_save_dir,
            spec=spec,
            ckpt_prefix=self.config.get("ckpt_prefix", "ckpt"),
        )

        # Run optimization using config settings
        optimizer_kwargs = {
            "maxfun": self.config.get("maxfun", 1000),
            "factr": self.config.get("factr", 1),
            "m": self.config.get("m", 20),
            "pgtol": self.config.get("pgtol", 1e-14),
            "maxiter": self.config.get("maxiter", 1000),
        }

        logger.info(f"Optimizing with L-BFGS-B: {optimizer_kwargs}")
        params, loss, info = scipy.optimize.fmin_l_bfgs_b(
            training_step,
            x0=np.array(flatten_init_params),
            **optimizer_kwargs,
        )

        return params, loss, info

    def evaluate(
        self,
        checkpoint_path: Path,
        plot_distances: list,
        use_lda: bool = False,
        output_path: str = None,
    ):
        """
        Evaluate a trained model on specific configurations.

        Args:
            checkpoint_path: Path to the checkpoint file
            plot_distances: List of distances to evaluate
            plot_initial_density: Initial density for each configuration
            use_lda: Whether to use LDA instead of neural XC
            output_path: Path to save evaluation results

        Returns:
            States from the Kohn-Sham calculations
        """
        from qedft.train.od.eval import eval_trained_model

        dataset, _, _, grids = self.prepare_dataset()

        # Data for evaluation and plotting
        plot_set = dataset.get_molecules(plot_distances)
        plot_initial_density = scf.get_initial_density(
            plot_set,
            method="noninteracting",
        )

        # Get the neural XC energy density function
        base_network = self.network.build_network(grids)
        network = build_xc_functional(
            network=base_network,
            grids=grids,
            config=self.config,
        )
        _, neural_xc_energy_density_fn = network

        # Call the evaluation function
        return eval_trained_model(
            ckpt_path=str(checkpoint_path),
            plot_distances=plot_distances,
            plot_set=plot_set,
            plot_initial_density=plot_initial_density,
            num_electrons=dataset.num_electrons,
            num_iterations=self.config.get("num_iterations", 100),
            grids=grids,
            neural_xc_energy_density_fn=neural_xc_energy_density_fn,
            use_amplitude_encoding=self.config.get("use_amplitude_encoding", False),
            use_lda=use_lda,
            alpha=self.config.get("alpha", 0.5),
            alpha_decay=self.config.get("alpha_decay", 0.9),
            num_mixing_iterations=self.config.get("num_mixing_iterations", 2),
            density_mse_converge_tolerance=self.config.get("density_mse_converge_tolerance", -1.0),
            enforce_reflection_symmetry=self.config.get("enforce_reflection_symmetry", True),
            output_path=output_path,
        )


def main():

    from qedft.config.config import Config
    from qedft.models.networks import GlobalMLP, LocalMLP

    # Get project path
    project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))

    # Load base configuration
    config = Config(config_path=project_path / "qedft" / "config" / "train_config.yaml").config

    # Update with specific settings
    config.update(
        {
            "molecule_name": "h2",
            "dataset1": [128, 384],
            "rng": 0,
            # Network architecture settings
            "network_type": "mlp_ksr",  # or 'mlp' for local, 'mlp_ksr' for global
            "n_neurons": 128,
            "n_layers": 3,
            "activation": "tanh",
            "density_normalization_factor": 2.0,
            "wrap_with_negative_transform": True,
            "wrap_self_interaction": True,
            "use_amplitude_encoding": True,
            # Optimizer settings
            "maxfun": 2,
            "maxiter": 2,
            "factr": 1,
            "m": 20,
            "pgtol": 1e-14,
            "ckpt_prefix": "ckpt",
            "output_path": str(project_path / "tests" / "ksr" / "ckpts"),
        },
    )

    # Choose network type based on config
    NetworkClass = GlobalMLP if config["network_type"] == "mlp_ksr" else LocalMLP
    network = NetworkClass(config)

    # Initialize trainer
    trainer = KSDFTTrainer(
        config_dict=config,
        network=network,
        data_path=project_path / "data" / "od",
    )

    # Train model
    params, loss, info = trainer.train()
    print(f"Training completed with final loss: {loss}")
    print(f"Optimization info: {info}")
    print(f"Params: {params}")

    # Evaluate model
    states = trainer.evaluate(
        checkpoint_path=project_path / "tests" / "ksr" / "ckpts" / "ckpt-00001",
        plot_distances=[128, 384],
    )
    print(f"States: {states}")


if __name__ == "__main__":
    main()
