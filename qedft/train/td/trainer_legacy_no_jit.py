"""Trainer for Kohn-Sham DFT models.

This module provides a high-level trainer class that simplifies training neural network-based
exchange-correlation functionals for Kohn-Sham DFT. It handles the training
boilerplate like data loading, optimization, checkpointing etc.
"""

import os
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pyscf
from chex import Array
from flax import serialization
from jax import random
from jax.ad_checkpoint import checkpoint
from loguru import logger
from pyscf.dft import numint
from tqdm import tqdm

import qedft
from qedft.data_io.td.dataset_generation import DataGenerator, MoleculeConfig
from qedft.train.td import numint_legacy as numintad
from qedft.train.td import rks_legacy as dft

warnings.filterwarnings("ignore", "Function mol.dumps drops attribute.*")
warnings.filterwarnings("ignore", "Not taking derivatives wrt the leaves*")


def forward_with_checkpoint(network, params, x):
    """Forward pass through the network with checkpointing."""
    return checkpoint(
        lambda p, inp: network.apply(p, inp),
        static_argnums=(1,),
    )(params, x)


def exc_and_vrho_global(network, params, rho):
    """Compute XC energy and potential using neural network (global version)."""
    exc, vjp_fn = jax.vjp(
        lambda x: jnp.sum(forward_with_checkpoint(network, params, x)).squeeze(),
        rho,
    )
    (vrho,) = vjp_fn(jnp.ones_like(exc))
    return exc, vrho


def exc_and_vrho_local(network, params, rho):
    """Compute XC energy and potential using neural network (local version)."""
    exc, vjp_fn = jax.vjp(
        lambda x: network.apply(params, x[:, None]).squeeze(),
        rho,
    )
    (vrho,) = vjp_fn(jnp.ones_like(exc))
    return exc, vrho


# Compile functions with JIT
jit_forward = jax.jit(forward_with_checkpoint, static_argnums=(0,))
jit_exc_and_vrho_global = jax.jit(exc_and_vrho_global, static_argnums=(0,))
jit_exc_and_vrho_local = jax.jit(exc_and_vrho_local, static_argnums=(0,))
# If no jitting is needed, use the following:
# jit_forward = lambda network, params, x: forward_with_checkpoint(network, params, x)
# jit_exc_and_vrho_global = lambda network, params, rho: exc_and_vrho_global(network, params, rho)
# jit_exc_and_vrho_local = lambda network, params, rho: exc_and_vrho_local(network, params, rho)


def eval_xc(
    xc_code,
    rho,
    spin=0,
    relativity=0,
    deriv=2,
    verbose=None,
    params=None,
    network=None,
    is_global_xc=True,
):
    """Evaluate exchange-correlation energy and derivatives."""
    if is_global_xc:
        exc, vrho = jit_exc_and_vrho_global(network, params, rho)
    else:
        exc, vrho = jit_exc_and_vrho_local(network, params, rho)
    vxc = (vrho, None, None, None)  # Only first derivative implemented
    return exc, vxc, None, None


class MLP(nn.Module):
    """Multi-layer perceptron model for XC functional approximation."""

    features: Sequence[int]
    act_fn: Callable = nn.gelu
    scale: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features[:-1]:
            x = self.act_fn(nn.Dense(feat)(x))
        return -self.scale * nn.swish(nn.Dense(self.features[-1])(x))


class TDKSDFTTrainer:
    """Trainer for time-dependent Kohn-Sham DFT models."""

    def __init__(
        self,
        config_dict: dict,
        network: nn.Module | None = None,
        data_path: Path | None = None,
        seed: int = 0,
    ):
        """Initialize trainer with configuration.

        Args:
            config_dict: Configuration dictionary containing all settings
            network: Network implementation (if None, uses default MLP)
            data_path: Path to the data directory
            seed: Random seed for initialization
        """
        # Enable 64-bit precision
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_platform_name", config_dict.get("jax_platform_name", "cpu"))

        self.config = config_dict.copy()
        self.seed = random.PRNGKey(seed)

        # Set up network
        self.network = network or MLP(
            features=self.config.get("features", [512, 512, 1]),
            act_fn=nn.gelu,
            scale=self.config.get("scale", 1e-2),
        )

        # Set up data path
        if data_path is None:
            project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
            self.data_path = project_path / "data" / "td"
        else:
            self.data_path = data_path

        # Results directory
        self.results_dir = Path(self.config.get("results_dir", "results/td_dft_training"))
        os.makedirs(self.results_dir, exist_ok=True)

        # Set up pre-jitted evaluation functions
        is_global_xc = self.config.get("is_global_xc", True)
        if is_global_xc:
            self.eval_xc_func = partial(
                eval_xc,
                network=self.network,
                is_global_xc=True,
            )
        else:
            self.eval_xc_func = partial(
                eval_xc,
                network=self.network,
                is_global_xc=False,
            )

        logger.info(f"Initialized trainer with config: {self.config}")

    def prepare_dataset(self) -> tuple[list, list]:
        """Load and prepare dataset from config settings."""
        # Initialize data generator
        generator = DataGenerator(output_dir=self.data_path)

        # Get configuration options for molecules
        if "molecule_configs" in self.config:
            # Use custom molecule configurations if provided
            train_configs = self.config["molecule_configs"].get("train", [])
            val_configs = self.config["molecule_configs"].get("validation", [])
        else:
            # Default to H2 molecules with different bond lengths
            train_bond_lengths = self.config.get("train_bond_lengths", [0.74, 0.5, 1.5])
            val_bond_lengths = self.config.get("val_bond_lengths", [0.6, 0.9, 1.2])
            basis = self.config.get("basis", "631g")
            method = self.config.get("method", "CCSD")
            grid_density = self.config.get("grid_density", 0)

            # Define H2 molecule configurations
            train_configs = [
                MoleculeConfig(
                    name=f"H2_train_{bl}",
                    atom_coords=f"H 0 0 0; H 0 0 {bl}",
                    units="Ang",
                    basis=basis,
                    method=method,
                    grid_density=grid_density,
                )
                for bl in train_bond_lengths
            ]

            val_configs = [
                MoleculeConfig(
                    name=f"H2_val_{bl}",
                    atom_coords=f"H 0 0 0; H 0 0 {bl}",
                    units="Ang",
                    basis=basis,
                    method=method,
                    grid_density=grid_density,
                )
                for bl in val_bond_lengths
            ]

        # Generate training data
        training_data = []
        logger.info("Generating training data...")
        for config in train_configs:
            mol, mf, dm_ao, energy, density, coords = generator.generate_data(
                config,
                save_data=True,
            )
            density_true = jnp.concatenate([coords, density[:, None]], axis=1)
            training_data.append((energy, density_true, mol))

        # Generate validation data
        validation_data = []
        logger.info("Generating validation data...")
        for config in val_configs:
            mol, mf, dm_ao, energy, density, coords = generator.generate_data(
                config,
                save_data=False,
            )
            density_true = jnp.concatenate([coords, density[:, None]], axis=1)
            validation_data.append((energy, density_true, mol))

        return training_data, validation_data

    def _compute_loss_and_grad(self, params, batch_data, energy_weight, density_weight):
        """Compute loss and gradients for a single batch."""

        def batch_loss_fn(p):
            # Process all molecules in the batch simultaneously
            batch_energies = []
            batch_density_losses = []

            # First, compute all energy predictions in parallel
            for E_goal, density_goal, mol in batch_data:
                # Set up DFT calculation with neural network functional
                grids = pyscf.dft.gen_grid.Grids(mol)
                grids.level = 0
                grids.becke_scheme = pyscf.dft.gen_grid.stratmann
                grids.build()
                mf = dft.RKS(mol)
                mf.grids = grids

                # Set the xc functional
                mf = mf.define_xc_(
                    description=self.eval_xc_func,
                    xctype=(
                        "NN-AmplitudeEncoding" if self.config.get("is_global_xc", True) else "NN"
                    ),
                )

                mf.max_cycle = self.config.get("max_cycle", 20)

                # Calculate energy
                E_approx = mf.kernel(params=p)
                loss_E = energy_weight * (E_approx - E_goal) ** 2
                batch_energies.append(loss_E)

                # Calculate density loss
                coords, rho_true = density_goal[:, :3], density_goal[:, 3]
                dm_ao = mf.make_rdm1(a0_repr=True)
                ao_value = numint.eval_ao(mol, coords, deriv=0)
                rho = numintad.eval_rho(mol, ao_value, dm_ao, xctype="LDA")
                loss_n = density_weight * jnp.mean((rho - rho_true) ** 2)
                batch_density_losses.append(loss_n)

            # Combine all losses using vectorized operations
            energy_loss = jnp.mean(jnp.array(batch_energies))
            density_loss = jnp.mean(jnp.array(batch_density_losses))

            return energy_loss + density_loss

        return jax.value_and_grad(batch_loss_fn)(params)

    def _compute_validation_loss(
        self,
        params,
        validation_data,
        energy_weight,
        density_weight,
        batch_size,
    ):
        """Compute the total validation loss over the validation set."""
        total_val_loss = 0.0
        num_val_molecules = len(validation_data)
        if num_val_molecules == 0:
            return 0.0

        num_batches = 0
        for batch_start in range(0, num_val_molecules, batch_size):
            batch_end = min(batch_start + batch_size, num_val_molecules)
            batch_data = validation_data[batch_start:batch_end]

            # Process all molecules in the batch simultaneously
            batch_energies = []
            batch_density_losses = []

            # Use stop_gradient to prevent gradient calculation during validation
            stopped_params = jax.lax.stop_gradient(params)

            # First, compute all energy predictions in parallel
            for E_goal, density_goal, mol in batch_data:
                # Set up DFT calculation with neural network functional
                grids = pyscf.dft.gen_grid.Grids(mol)
                grids.level = 0
                grids.becke_scheme = pyscf.dft.gen_grid.stratmann
                grids.build()
                mf = dft.RKS(mol)
                mf.grids = grids

                # Set the xc functional
                mf = mf.define_xc_(
                    description=self.eval_xc_func,
                    xctype=(
                        "NN-AmplitudeEncoding" if self.config.get("is_global_xc", True) else "NN"
                    ),
                )

                mf.max_cycle = self.config.get("max_cycle", 20)

                # Calculate energy using the stopped_params
                E_approx = mf.kernel(params=stopped_params)
                loss_E = energy_weight * (E_approx - E_goal) ** 2
                batch_energies.append(loss_E)

                # Calculate density loss
                coords, rho_true = density_goal[:, :3], density_goal[:, 3]
                dm_ao = mf.make_rdm1(a0_repr=True)
                ao_value = numint.eval_ao(mol, coords, deriv=0)
                rho = numintad.eval_rho(mol, ao_value, dm_ao, xctype="LDA")
                loss_n = density_weight * jnp.mean((rho - rho_true) ** 2)
                batch_density_losses.append(loss_n)

            # Combine all losses
            energy_loss = jnp.mean(jnp.array(batch_energies))
            density_loss = jnp.mean(jnp.array(batch_density_losses))
            loss_value = energy_loss + density_loss

            total_val_loss += loss_value
            num_batches += 1

        return total_val_loss / num_batches if num_batches > 0 else 0.0

    def _plot_training_progress(self, train_losses, val_losses, val_iterations, save_dir):
        """Plot training and validation loss progress."""
        plt.figure(figsize=(10, 6))

        # Plot all training losses with their proper iteration indices
        train_iterations = list(range(len(train_losses)))
        plt.plot(train_iterations, train_losses, label="Training Loss", color="blue", linewidth=2)

        # Plot validation losses only at their specific iterations
        if val_losses:
            plt.plot(
                val_iterations,
                val_losses,
                "o-",
                label="Validation Loss",
                color="orange",
                markersize=6,
            )

        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        loss_plot_filename = os.path.join(save_dir, "training_validation_loss.png")
        plt.savefig(loss_plot_filename, dpi=300)
        plt.close()

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
            Tuple of (optimized_params, opt_state, training_losses, validation_losses)
        """
        # Prepare dataset
        training_data, validation_data = self.prepare_dataset()

        # Initialize/load network parameters
        if not hasattr(self, "initialized_params"):
            # Get grid size from first training molecule
            first_mol_grids = pyscf.dft.gen_grid.Grids(training_data[0][2])
            first_mol_grids.level = 0
            first_mol_grids.build()
            num_grid_points = first_mol_grids.coords.shape[0]

            # Initialize parameters
            self.initialized_params = self.network.init(
                self.seed,
                jnp.ones((1, num_grid_points)),
            )
            logger.info(f"Network initialized with {num_grid_points} grid points")

        # Load from checkpoint if provided
        if checkpoint_path and checkpoint_path.exists():
            logger.info(f"Loading parameters from checkpoint: {checkpoint_path}")
            trained_params_dict = jnp.load(checkpoint_path, allow_pickle=True).item()
            params = serialization.from_state_dict(self.initialized_params, trained_params_dict)
        else:
            params = self.initialized_params

        # Set up optimizer
        learning_rate = self.config.get("learning_rate", 1e-3)
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        # Training parameters
        n_iterations = self.config.get("n_iterations", 1000)
        checkpoint_interval = self.config.get("checkpoint_interval", 10)
        validation_interval = self.config.get("validation_interval", 5)
        batch_size = self.config.get("batch_size", 3)
        energy_weight = self.config.get("energy_weight", 1.0)
        density_weight = self.config.get("density_weight", 1.0)

        # Set save directory
        if checkpoint_save_dir:
            results_dir = checkpoint_save_dir
        else:
            results_dir = self.results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Training state tracking
        num_molecules = len(training_data)
        train_losses = []
        validation_losses = []
        validation_iterations = []

        # Training loop
        logger.info("Starting training...")
        for iteration in tqdm(range(n_iterations)):
            # Shuffle data at each epoch for better training
            key = jax.random.PRNGKey(iteration)
            indices = jax.random.permutation(key, num_molecules)
            shuffled_data = [training_data[i] for i in indices]

            epoch_loss = 0.0
            num_batches = 0

            # Process batches
            for batch_start in range(0, num_molecules, batch_size):
                batch_end = min(batch_start + batch_size, num_molecules)
                batch_data = shuffled_data[batch_start:batch_end]

                # Compute loss and gradients for this batch only
                loss_value, grads = self._compute_loss_and_grad(
                    params,
                    batch_data,
                    energy_weight,
                    density_weight,
                )

                # Update parameters with the batch gradients
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)

                epoch_loss += loss_value
                num_batches += 1

            # Log epoch-level loss
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            train_losses.append(avg_epoch_loss)
            logger.info(f"Iteration {iteration}, Training Loss: {avg_epoch_loss:.6f}")

            # Validation step
            if iteration % validation_interval == 0:
                avg_val_loss = self._compute_validation_loss(
                    params,
                    validation_data,
                    energy_weight,
                    density_weight,
                    batch_size,
                )
                validation_losses.append(avg_val_loss)
                validation_iterations.append(iteration)
                logger.info(f"Iteration {iteration}, Validation Loss: {avg_val_loss:.6f}")

            # Plot training and validation loss at every iteration
            self._plot_training_progress(
                train_losses,
                validation_losses,
                validation_iterations,
                results_dir,
            )

            # Save checkpoint
            if iteration % checkpoint_interval == 0:
                try:
                    dict_output = serialization.to_state_dict(params)
                    checkpoint_filename = os.path.join(
                        results_dir,
                        f"trained_model_iter_{iteration}.npy",
                    )
                    jnp.save(checkpoint_filename, dict_output)
                    logger.info(f"Checkpoint saved to {checkpoint_filename}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint: {e}")

        # Save final model
        try:
            dict_output = serialization.to_state_dict(params)
            final_model_path = os.path.join(results_dir, "trained_model_final.npy")
            jnp.save(final_model_path, dict_output)
            logger.info(f"Final model saved to {final_model_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")

        return params, opt_state, train_losses, validation_losses, validation_iterations

    def evaluate(
        self,
        checkpoint_path: Path,
        molecule_configs: list[MoleculeConfig] | None = None,
        bond_lengths: list[float] | None = None,
        use_lda: bool = False,
        output_path: str | None = None,
    ):
        """Evaluate a trained model on specific molecules."""
        # Load trained parameters
        trained_params_dict = jnp.load(checkpoint_path, allow_pickle=True).item()

        # Initialize data generator
        generator = DataGenerator(output_dir=self.data_path)

        # Prepare evaluation data
        eval_molecules = []
        logger.info("Generating evaluation data...")

        if molecule_configs:
            # Use provided custom molecular configurations
            for config in molecule_configs:
                mol, _, _, ref_energy, _, _ = generator.generate_data(config, save_data=False)
                # Use the molecule name as identifier instead of bond length
                eval_molecules.append((config.name, ref_energy, mol))
        else:
            # Default to H2 molecules with different bond lengths
            bond_lengths = bond_lengths or [
                0.4,
                0.5,
                0.6,
                0.7,
                0.74,
                0.8,
                0.9,
                1.0,
                1.2,
                1.5,
                2.0,
                2.5,
            ]
            for bond_length in bond_lengths:
                config = MoleculeConfig(
                    name=f"H2_eval_{bond_length}",
                    atom_coords=f"H 0 0 0; H 0 0 {bond_length}",
                    units="Ang",
                    basis=self.config.get("basis", "631g"),
                    method=self.config.get("method", "CCSD"),
                    grid_density=self.config.get("grid_density", 0),
                )
                mol, _, _, ref_energy, _, _ = generator.generate_data(config, save_data=False)
                eval_molecules.append((bond_length, ref_energy, mol))

        # Initialize parameters if needed
        if not hasattr(self, "initialized_params"):
            # Create a grid from the first molecule to get size
            mol = eval_molecules[0][2]
            grids = pyscf.dft.gen_grid.Grids(mol)
            grids.level = 0
            grids.build()
            num_grid_points = grids.coords.shape[0]

            self.initialized_params = self.network.init(
                self.seed,
                jnp.ones((1, num_grid_points)),
            )

        # Load parameters into model
        params = serialization.from_state_dict(self.initialized_params, trained_params_dict)

        # Calculate energies using trained model
        identifiers = []
        reference_energies = []
        nn_energies = []
        lda_energies = [] if use_lda else None

        logger.info("Evaluating trained model...")
        for identifier, ref_energy, mol in tqdm(eval_molecules):
            # Set up DFT calculation with neural network functional
            grids = pyscf.dft.gen_grid.Grids(mol)
            grids.level = 0
            grids.becke_scheme = pyscf.dft.gen_grid.stratmann
            grids.build()
            mf = dft.RKS(mol)
            mf.grids = grids

            # Set the xc functional
            mf = mf.define_xc_(
                description=self.eval_xc_func,
                xctype="NN-AmplitudeEncoding" if self.config.get("is_global_xc", True) else "NN",
            )

            mf.max_cycle = self.config.get(
                "max_cycle",
                20,
            )  # Increase for better convergence during eval

            # Calculate energy with trained model
            nn_energy = mf.kernel(params=jax.lax.stop_gradient(params))

            # If comparing with LDA, calculate LDA energy
            if use_lda:
                mf_lda = pyscf.dft.RKS(mol)
                mf_lda.xc = "lda"
                mf_lda.kernel()
                lda_energies.append(mf_lda.e_tot)

            identifiers.append(identifier)
            reference_energies.append(ref_energy)
            nn_energies.append(nn_energy)

        # Determine if we're working with bond lengths (numeric) or molecule names
        is_bond_length_plot = all(isinstance(x, (int, float)) for x in identifiers)

        # Plot results
        if is_bond_length_plot:
            self._plot_dissociation_curve(
                identifiers,
                reference_energies,
                nn_energies,
                lda_energies,
                output_path or self.results_dir,
            )
        else:
            # For custom molecules, create a bar plot instead
            self._plot_molecule_comparison(
                identifiers,
                reference_energies,
                nn_energies,
                lda_energies,
                output_path or self.results_dir,
            )

        # Calculate errors
        energy_errors = np.array(nn_energies) - np.array(reference_energies)

        # Return results dictionary
        results = {
            "identifiers": identifiers,
            "reference_energies": reference_energies,
            "nn_energies": nn_energies,
            "energy_errors": energy_errors.tolist(),
        }

        if use_lda:
            results["lda_energies"] = lda_energies

        return results

    def _plot_dissociation_curve(
        self,
        bond_lengths,
        ref_energies,
        nn_energies,
        lda_energies=None,
        output_dir="results",
    ):
        """Plot dissociation curve and errors."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Plot dissociation curve
        plt.figure(figsize=(10, 6))
        plt.plot(bond_lengths, ref_energies, "o-", label="CCSD Reference", color="blue")
        plt.plot(bond_lengths, nn_energies, "s--", label="Neural Network XC", color="red")
        if lda_energies:
            plt.plot(bond_lengths, lda_energies, "^-.", label="LDA", color="green")
        plt.xlabel("Bond Length (Å)")
        plt.ylabel("Energy (Hartree)")
        plt.title("H₂ Dissociation Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "h2_dissociation_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Plot error
        plt.figure(figsize=(10, 4))
        energy_errors = np.array(nn_energies) - np.array(ref_energies)
        plt.plot(bond_lengths, energy_errors, "o-", color="purple")
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        plt.xlabel("Bond Length (Å)")
        plt.ylabel("Energy Error (Hartree)")
        plt.title("Error in Neural Network XC Functional (NN - CCSD)")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        error_path = os.path.join(output_dir, "h2_dissociation_error.png")
        plt.savefig(error_path, dpi=300)
        plt.close()

    def _plot_molecule_comparison(
        self,
        molecule_names,
        ref_energies,
        nn_energies,
        lda_energies=None,
        output_dir="results",
    ):
        """Plot energy comparison for different molecules."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Plot absolute energies
        plt.figure(figsize=(12, 6))
        x = np.arange(len(molecule_names))
        width = 0.25

        plt.bar(x - width, ref_energies, width, label="Reference", color="blue")
        plt.bar(x, nn_energies, width, label="Neural Network XC", color="red")
        if lda_energies:
            plt.bar(x + width, lda_energies, width, label="LDA", color="green")

        plt.xlabel("Molecule")
        plt.ylabel("Energy (Hartree)")
        plt.title("Energy Comparison Across Molecules")
        plt.xticks(x, molecule_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "molecule_energy_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Plot errors
        plt.figure(figsize=(12, 6))
        energy_errors = np.array(nn_energies) - np.array(ref_energies)
        plt.bar(x, energy_errors, width=0.4, color="purple")
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        plt.xlabel("Molecule")
        plt.ylabel("Energy Error (Hartree)")
        plt.title("Error in Neural Network XC Functional (NN - Reference)")
        plt.xticks(x, molecule_names, rotation=45, ha="right")
        plt.tight_layout()
        error_path = os.path.join(output_dir, "molecule_energy_error.png")
        plt.savefig(error_path, dpi=300)
        plt.close()


def main():

    # Get project path
    project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))

    # Create H2 bonds example
    h2_config = {
        "features": [512, 512, 1],
        "train_bond_lengths": [0.74, 0.5, 1.5],
        "val_bond_lengths": [0.6, 0.9, 1.2],
        "n_iterations": 100,  # Small for testing
        "batch_size": 3,
        "is_global_xc": True,
        "results_dir": "results/h2_test_run",
    }

    # Create custom molecule example
    custom_basis = "631g"
    custom_method = "CCSD"

    h4_train1 = MoleculeConfig(
        name="H4_square",
        atom_coords="H 0 0 0; H 0 0 0.74; H 0.74 0 0; H 0 0.74 0",
        units="Ang",
        basis=custom_basis,
        method=custom_method,
    )
    h4_train2 = MoleculeConfig(
        name="H4_square2",
        atom_coords="H 0 0 0; H 0 0 0.64; H 0.64 0 0; H 0 0.64 0",
        units="Ang",
        basis=custom_basis,
        method=custom_method,
    )

    h4_val = MoleculeConfig(
        name="H4_square2_val",
        atom_coords="H 0 0 0; H 0 0 0.84; H 0.84 0 0; H 0 0.84 0",
        units="Ang",
        basis=custom_basis,
        method=custom_method,
    )

    # Configuration with custom molecules
    custom_config = {
        "features": [512, 512, 1],
        "molecule_configs": {
            "train": [h4_train1, h4_train2],
            "validation": [h4_val],
        },
        "n_iterations": 100,
        "batch_size": 2,
        "is_global_xc": True,
        "results_dir": "results/custom_test_run",
    }

    # Choose which example to run
    use_custom_molecules = False
    config = custom_config if use_custom_molecules else h2_config

    # Initialize network with STAX
    try:
        from qedft.models.networks import GlobalMLP
        from qedft.train.td.stax_to_flax_network import adapt_stax_for_training

        model = GlobalMLP()
        num_grid_points = 1192
        network_stax = model.build_network(grids=jnp.linspace(0, 1, num_grid_points))
        stax_init_fn, stax_apply_fn = network_stax  # your STAX model
        network, params = adapt_stax_for_training(stax_init_fn, stax_apply_fn, (num_grid_points,))
        logger.info(f"Network initialized with {num_grid_points} grid points")
    except Exception as e:
        print("Adjust the input shape based on the error message: ", e)
        print(
            "The model has to have the input shape as the number of grid points (same across all molecules).",
        )

    # Initialize and train
    trainer = TDKSDFTTrainer(config, network=network)
    params, opt_state, train_losses, val_losses, val_iters = trainer.train()

    # Evaluate with custom molecules
    if use_custom_molecules:
        # Create evaluation molecules
        h4_eval = MoleculeConfig(
            name="H4_square_eval",
            atom_coords="H 0 0 0; H 0 0 0.94; H 0.94 0 0; H 0 0.94 0",
            units="Ang",
            basis=custom_basis,
            method=custom_method,
        )

        h4_eval2 = MoleculeConfig(
            name="H4_square2_eval",
            atom_coords="H 0 0 0; H 0 0 1.24; H 1.24 0 0; H 0 1.24 0",
            units="Ang",
            basis=custom_basis,
            method=custom_method,
        )

        # Evaluate with custom molecules
        results = trainer.evaluate(
            checkpoint_path=Path("results/custom_test_run/trained_model_final.npy"),
            molecule_configs=[h4_train1, h4_train2, h4_val, h4_eval, h4_eval2],
            use_lda=True,
        )
    else:
        # Evaluate H2 with different bond lengths
        results = trainer.evaluate(
            checkpoint_path=Path("results/h2_test_run/trained_model_final.npy"),
            bond_lengths=[0.6, 0.8, 1.0, 1.5],
            use_lda=True,
        )

    print("Training and evaluation complete!")
    print(f"Results saved in {trainer.results_dir}")


if __name__ == "__main__":
    main()
