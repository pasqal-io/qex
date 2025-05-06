"""Neural network based exchange-correlation functional training for DFT.

This module implements training of a neural network to approximate the
exchange-correlation (XC) functional in density functional theory (DFT).
It demonstrates how parameters are passed through the SCF iterations
to parameterize the XC functional being trained.

The module supports training both global and local XC functionals using FLAX or STAX
neural network architectures. This implementation avoids JAX JIT compilation
for better debugging and flexibility, while still maintaining reasonable performance.

Key features:
- Batched training with validation
- Checkpointing and visualization of training progress
- Evaluation on molecular dissociation curves
- Support for both energy and density-based loss functions
"""

import os
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import flax.linen as nn
import jax
import jax.lax
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
from pyscfad import gto
from tqdm import tqdm

import qedft
from qedft.train.td import numint_legacy as numintad
from qedft.train.td import rks_legacy as dft

warnings.filterwarnings("ignore", "Function mol.dumps drops attribute.*")
warnings.filterwarnings("ignore", "Not taking derivatives wrt the leaves*")

# Configure JAX to use CPU and 64-bit precision for stability
# If cuda is available, use it.
# Enable double precision
jax.config.update("jax_enable_x64", True)
# Use GPU if available
jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_platform_name", "cuda")


class MLP(nn.Module):
    """Multi-layer perceptron model for XC functional approximation.

    This neural network architecture is designed to approximate the exchange-correlation
    functional in density functional theory.

    Attributes:
        features: Sequence of integers specifying width of each layer
        act_fn: Activation function to use between layers (default: nn.gelu)
        scale: Scaling factor for the output to control magnitude (default: 1e-2)
    """

    features: Sequence[int]
    act_fn: Callable = nn.gelu
    scale: float = 1e-2

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, input_dim) containing density information

        Returns:
            Network output of shape (batch_size, 1) representing XC energy contribution
        """
        # print("Input shape: ", jnp.shape(x))
        for feat in self.features[:-1]:
            x = self.act_fn(nn.Dense(feat)(x))
        # Added a minus sign to make the network output negative as the energy is negative.
        return -self.scale * nn.swish(nn.Dense(self.features[-1])(x))


# Initialize network architecture
# Using FLAX
# network = MLP([512, 512, 1], act_fn=nn.gelu, scale=1e-2)

# Our models are in STAX format.
# One needs with STAX to know the input shape before building the network.
# For that try to run the following code and adjust the input shape based on the error message.

try:
    from qedft.models.networks import GlobalMLP
    from qedft.train.td.stax_to_flax_network import adapt_stax_for_training

    model = GlobalMLP()
    num_grid_points = 1192
    network = model.build_network(grids=jnp.linspace(0, 1, num_grid_points))
    stax_init_fn, stax_apply_fn = network  # your STAX model
    network, params = adapt_stax_for_training(stax_init_fn, stax_apply_fn, (num_grid_points,))
except Exception as e:
    print("Adjust the input shape based on the error message: ", e)
    print(
        "The model has to have the input shape as the number of grid points (same across all molecules).",
    )


# Modify your neural network forward pass
def forward_with_checkpoint(params, x):
    """Forward pass through the network with memory checkpointing.

    Uses JAX's checkpoint utility to reduce memory usage during backpropagation
    by recomputing intermediate values instead of storing them.

    Args:
        params: Neural network parameters
        x: Input array containing density information

    Returns:
        Network output representing XC energy contribution
    """
    return checkpoint(lambda p, inp: network.apply(p, inp), static_argnums=(1,))(params, x)


@jax.jit
def exc_and_vrho_global(params, rho):
    """Compute XC energy and potential using neural network for global XC.

    For global XC functionals, the network processes the entire density grid at once.

    Args:
        params: Neural network parameters
        rho: Electron density array

    Returns:
        exc: XC energy
        vrho: First derivative of XC energy wrt density (XC potential)
    """

    # Use in your XC functional
    exc, vjp_fn = jax.vjp(
        lambda x: jnp.sum(forward_with_checkpoint(params, x)).squeeze(),
        rho,
    )
    (vrho,) = vjp_fn(jnp.ones_like(exc))
    return exc, vrho


@jax.jit
def exc_and_vrho_local(params, rho):
    """Compute XC energy and potential using neural network for local XC.

    For local XC functionals, the network processes each grid point independently.

    Args:
        params: Neural network parameters
        rho: Electron density array

    Returns:
        exc: XC energy
        vrho: First derivative of XC energy wrt density (XC potential)
    """
    # Added sum here to output a single scalar.
    exc, vjp_fn = jax.vjp(
        lambda x: network.apply(params, x[:, None]).squeeze(),
        rho,
    )
    (vrho,) = vjp_fn(jnp.ones_like(exc))
    return exc, vrho


def eval_xc(
    xc_code,
    rho,
    spin=0,
    relativity=0,
    deriv=2,
    verbose=None,
    params=None,
    params_grid_coords=None,
    is_global_xc=True,
):
    """Evaluate exchange-correlation energy and derivatives.

    This function mimics the interface of standard XC functionals in PySCF
    but uses our neural network model instead.

    Args:
        xc_code: XC functional identifier (unused, kept for compatibility)
        rho: Electron density
        spin: Spin polarization (default: 0, unpolarized)
        relativity: Relativistic correction level (unused)
        deriv: Maximum derivative order to calculate (unused)
        verbose: Verbosity level (unused)
        params: Neural network parameters
        params_grid_coords: Grid coordinates for parameterized functionals (unused)
        is_global_xc: Whether to use global (True) or local (False) XC evaluation

    Returns:
        Tuple of (exc, vxc, fxc, kxc) containing energy and derivatives
    """
    # print("rho shape: ", jnp.shape(rho))
    if is_global_xc:
        exc, vrho = exc_and_vrho_global(params, rho)
    else:
        exc, vrho = exc_and_vrho_local(params, rho)
    vxc = (vrho, None, None, None)  # Only first derivative implemented
    return exc, vxc, None, None


def compute_loss_and_grad(
    params,
    batch_data,
    is_global_xc,
    energy_weight,
    density_weight,
    max_cycle,
):
    """Compute loss and gradients for a single batch.

    Calculates both energy and density losses for each molecule in the batch,
    then returns the combined loss and its gradient with respect to model parameters.

    Args:
        params: Neural network parameters
        batch_data: List of (target_energy, target_density, molecule) tuples
        is_global_xc: Whether to use global XC functional
        energy_weight: Weight for energy loss term
        density_weight: Weight for density loss term
        max_cycle: Maximum SCF cycles

    Returns:
        Tuple of (loss_value, gradients)
    """

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

            if is_global_xc:
                eval_xc_global = partial(eval_xc, is_global_xc=True)
                mf = mf.define_xc_(description=eval_xc_global, xctype="NN-AmplitudeEncoding")
            else:
                eval_xc_local = partial(eval_xc, is_global_xc=False)
                mf = mf.define_xc_(description=eval_xc_local, xctype="NN")

            mf.max_cycle = max_cycle
            # Generate grid

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


def compute_loss(params, batch_data, is_global_xc, energy_weight, density_weight, max_cycle):
    """Compute loss for a single batch without computing gradients.

    Similar to compute_loss_and_grad but used for validation where gradients aren't needed.

    Args:
        params: Neural network parameters
        batch_data: List of (target_energy, target_density, molecule) tuples
        is_global_xc: Whether to use global XC functional
        energy_weight: Weight for energy loss term
        density_weight: Weight for density loss term
        max_cycle: Maximum SCF cycles

    Returns:
        Combined loss value
    """

    # Process all molecules in the batch simultaneously
    batch_energies = []
    batch_density_losses = []

    # Use jax.lax.stop_gradient on params to prevent gradient calculation during validation
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

        if is_global_xc:
            eval_xc_global = partial(eval_xc, is_global_xc=True)
            mf = mf.define_xc_(description=eval_xc_global, xctype="NN-AmplitudeEncoding")
        else:
            eval_xc_local = partial(eval_xc, is_global_xc=False)
            mf = mf.define_xc_(description=eval_xc_local, xctype="NN")

        mf.max_cycle = max_cycle

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

    # Combine all losses using vectorized operations
    energy_loss = jnp.mean(jnp.array(batch_energies))
    density_loss = jnp.mean(jnp.array(batch_density_losses))

    return energy_loss + density_loss


def compute_validation_loss(
    params,
    validation_data,
    is_global_xc,
    energy_weight,
    density_weight,
    batch_size,
    max_cycle,
):
    """Compute the total validation loss over the validation set.

    Processes the validation data in batches to avoid memory issues.

    Args:
        params: Neural network parameters
        validation_data: List of validation data tuples
        is_global_xc: Whether to use global XC functional
        energy_weight: Weight for energy loss term
        density_weight: Weight for density loss term
        batch_size: Number of molecules to process at once
        max_cycle: Maximum SCF cycles

    Returns:
        Average validation loss across all validation molecules
    """
    total_val_loss = 0.0
    num_val_molecules = len(validation_data)
    if num_val_molecules == 0:
        return 0.0

    num_batches = 0
    for batch_start in range(0, num_val_molecules, batch_size):
        batch_end = min(batch_start + batch_size, num_val_molecules)
        batch_data = validation_data[batch_start:batch_end]
        loss_value = compute_loss(
            params,
            batch_data,
            is_global_xc,
            energy_weight,
            density_weight,
            max_cycle,
        )
        total_val_loss += loss_value
        num_batches += 1

    return total_val_loss / num_batches if num_batches > 0 else 0.0


def train(
    params: dict,
    training_data: list[tuple[float, Array, gto.Mole]],
    validation_data: list[tuple[float, Array, gto.Mole]],
    is_global_xc: bool = True,
    energy_weight: float = 1.0,
    density_weight: float = 1.0,
    n_iterations: int = 20,
    checkpoint_interval: int = 10,
    validation_interval: int = 10,
    batch_size: int = 5,
    results_dir: str = "results/default_run",
    max_cycle: int = 15,
) -> tuple[dict, dict, list, list, list]:
    """Train the neural network XC functional with batched processing and validation.

    Args:
        params: Initial neural network parameters
        training_data: List of (target_energy, target_density, molecule) tuples for training
        validation_data: List of (target_energy, target_density, molecule) tuples for validation
        is_global_xc: Whether to use global (True) or local (False) XC functional
        energy_weight: Weight for energy loss term
        density_weight: Weight for density loss term
        n_iterations: Number of training iterations
        checkpoint_interval: How often to save model checkpoints
        validation_interval: How often to evaluate on validation data
        batch_size: Number of molecules to process in each batch
        results_dir: Directory to save results and checkpoints
        max_cycle: Maximum SCF cycles

    Returns:
        Tuple containing:
        - Final model parameters
        - Final optimizer state
        - Training loss history
        - Validation loss history
        - Validation iteration indices
    """

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    num_molecules = len(training_data)

    train_losses = []
    validation_losses = []
    validation_iterations = []
    train_losses_at_validation = []

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
            loss_value, grads = compute_loss_and_grad(
                params,
                batch_data,
                is_global_xc,
                energy_weight,
                density_weight,
                max_cycle,
            )

            # Update parameters with the batch gradients
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            epoch_loss += loss_value
            num_batches += 1

            # Log batch progress (optional, can be verbose)
            # logger.info(
            #     f"Iteration {iteration}, Batch {batch_start//batch_size + 1}: Loss {loss_value:.6f}",
            # )

            # Clear memory between batches (consider if necessary)
            # jax.clear_caches()

        # Log epoch-level loss
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_epoch_loss)
        logger.info(f"Iteration {iteration}, Average Training Loss: {avg_epoch_loss:.6f}")

        # Validation step
        if iteration % validation_interval == 0:
            avg_val_loss = compute_validation_loss(
                params,
                validation_data,
                is_global_xc,
                energy_weight,
                density_weight,
                batch_size,
                max_cycle,
            )
            train_losses_at_validation.append(avg_epoch_loss)
            validation_losses.append(avg_val_loss)
            validation_iterations.append(iteration)
            logger.info(f"Iteration {iteration}, Validation Loss: {avg_val_loss:.6f}")

            # Plot training and validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses_at_validation, label="Training Loss", color="blue", linewidth=2)
            plt.plot(
                validation_losses,
                "o-",
                label="Validation Loss",
                color="orange",
                markersize=6,
            )
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.title("Training and Validation Loss Over Iterations")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            loss_plot_filename = os.path.join(results_dir, "training_validation_loss.png")
            plt.savefig(loss_plot_filename, dpi=300)
            logger.info(f"Training/validation loss plot saved to {loss_plot_filename}")
            plt.close()  # Close the figure to free memory

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
                logger.error(f"Failed to save checkpoint at iteration {iteration}: {e}")

    return params, opt_state, train_losses, validation_losses, validation_iterations


if __name__ == "__main__":

    from qedft.data_io.td.dataset_generation import DataGenerator, MoleculeConfig

    # Training parameters --------------------------------------------------------

    N_ITERATIONS = 400  # Reduced for faster testing, increase as needed
    CHECKPOINT_INTERVAL = 10
    VALIDATION_INTERVAL = 5  # Validate more frequently
    BATCH_SIZE = 3  # Adjust based on memory, ensure < len(training_data)
    IS_GLOBAL_XC = True
    ENERGY_WEIGHT = 1.0
    DENSITY_WEIGHT = 1.0
    EXPERIMENT_NAME = "h2_dft_training_legacy"
    MAX_CYCLE = 15
    # Get the number of grid points
    num_grid_points = 1192
    # Define bond lengths for training data
    # training_bond_lengths = [0.74, 0.5, 1.5]
    training_bond_lengths = [0.74, 0.5, 1.5, 2.0]
    # Define bond lengths for validation data
    validation_bond_lengths = [0.6, 0.9, 1.2]  # Different from training/evaluation
    # Define a wider range of bond lengths for evaluation
    eval_bond_lengths = [0.4, 0.5, 0.6, 0.7, 0.74, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]

    # ----------------------------------------------------------------------------

    print("Testing the network -------------------------------------------------------")
    # Initialize network parameters
    params = network.init(random.PRNGKey(42), jnp.ones((1, num_grid_points)))
    # Test eval of network
    dummy_input = jax.random.normal(random.PRNGKey(42), (1, num_grid_points))
    dummy_output = network.apply(params, dummy_input)
    print(f"Network test output (should be different for each batch): {dummy_output}")

    # Define multiple molecular systems for training
    train_molecule_configs = [
        MoleculeConfig(
            name="H2_train",
            atom_coords=f"H 0 0 0; H 0 0 {bond_length}",
            units="Ang",
            basis="631g",
            method="CCSD",
            grid_density=0,
        )
        for bond_length in training_bond_lengths
    ]
    # Define multiple molecular systems for validation
    validation_molecule_configs = [
        MoleculeConfig(
            name="H2_val",
            atom_coords=f"H 0 0 0; H 0 0 {bond_length}",
            units="Ang",
            basis="631g",
            method="CCSD",
            grid_density=0,
        )
        for bond_length in validation_bond_lengths
    ]

    # Initialize data generator
    project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
    output_dir = project_path / "data" / "td"
    generator = DataGenerator(output_dir=output_dir)

    # Generate training data
    training_data = []
    logger.info("Generating training data...")
    for config in train_molecule_configs:
        mol, mf, dm_ao, energy, density, coords = generator.generate_data(config, save_data=True)
        density_true = jnp.concatenate([coords, density[:, None]], axis=1)
        training_data.append((energy, density_true, mol))

    # Generate validation data
    validation_data = []
    logger.info("Generating validation data...")
    for config in validation_molecule_configs:
        # Avoid saving validation data unless needed
        mol, mf, dm_ao, energy, density, coords = generator.generate_data(config, save_data=False)
        density_true = jnp.concatenate([coords, density[:, None]], axis=1)
        validation_data.append((energy, density_true, mol))

    # Initialize network parameters (ensure this happens after data generation if grid size depends on it)
    # Assuming grid size is consistent or determined beforehand. If not, adjust initialization.
    # Example: Use grid from the first training molecule to determine input size
    first_mol_grids = pyscf.dft.gen_grid.Grids(training_data[0][2])
    first_mol_grids.level = 0
    first_mol_grids.build()
    num_grid_points = first_mol_grids.coords.shape[0]

    params = network.init(random.PRNGKey(42), jnp.ones((1, num_grid_points)))
    logger.info(f"Network initialized with input shape for {num_grid_points} grid points.")

    # Add experiment name parameter
    RESULTS_DIR = os.path.join("results", EXPERIMENT_NAME)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info(f"Results will be saved to: {RESULTS_DIR}")

    # Train the model on all molecules ----------------------------------------

    logger.info("Starting training...")
    params, opt_state, train_loss_hist, val_loss_hist, val_iters = train(
        params,
        training_data,
        validation_data,
        IS_GLOBAL_XC,
        ENERGY_WEIGHT,
        DENSITY_WEIGHT,
        n_iterations=N_ITERATIONS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        validation_interval=VALIDATION_INTERVAL,
        batch_size=BATCH_SIZE,
        results_dir=RESULTS_DIR,  # Pass the results directory
        max_cycle=MAX_CYCLE,
    )
    logger.info("Training finished.")

    # Save final model parameters to results directory
    dict_output = serialization.to_state_dict(params)
    final_model_path = os.path.join(RESULTS_DIR, "trained_model_final.npy")
    jnp.save(final_model_path, dict_output)
    logger.info(f"Final model saved to {final_model_path}")

    # --- Plotting ------------------------------------------------------------

    # Update all plot saving to use the results directory
    # Plot Training and Validation Loss (final plot)
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(N_ITERATIONS),
        train_loss_hist,
        label="Training Loss",
        color="blue",
        linewidth=2,
    )
    plt.plot(val_iters, val_loss_hist, "o-", label="Validation Loss", color="orange", markersize=6)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Iterations")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.yscale("log")
    plt.tight_layout()
    loss_plot_filename = os.path.join(RESULTS_DIR, "final_training_validation_loss.png")
    plt.savefig(loss_plot_filename, dpi=300)
    logger.info(f"Final training/validation loss plot saved to {loss_plot_filename}")
    plt.close()

    # --- Plot dissociation curve comparing trained model with CCSD reference -

    # Load trained parameters (using the final ones here)
    ckpt = "trained_model_final.npy"  # "trained_model_final.npy" trained_model_iter_440
    trained_params_dict = jnp.load(
        os.path.join(RESULTS_DIR, ckpt),
        allow_pickle=True,
    ).item()
    params = serialization.from_state_dict(params, trained_params_dict)
    # No need to reload if we use the 'params' returned by train() directly

    # Create molecules for each bond length
    eval_molecules = []
    logger.info("Generating evaluation data...")
    for bond_length in eval_bond_lengths:
        config = MoleculeConfig(
            name="H2_eval",
            atom_coords=f"H 0 0 0; H 0 0 {bond_length}",
            units="Ang",
            basis="631g",
            method="CCSD",
            grid_density=0,
        )
        # Generate data but don't save it repeatedly
        mol, mf, dm_ao, ref_energy, density, coords = generator.generate_data(
            config,
            save_data=False,
        )
        eval_molecules.append((bond_length, ref_energy, mol))

    # Calculate energies using trained model
    bond_lengths_plot = []
    ccsd_energies_plot = []
    nn_energies_plot = []

    logger.info("Evaluating trained model on dissociation curve --------------------------------")
    for bond_length, ref_energy, mol in tqdm(eval_molecules):
        # Set up DFT calculation with trained neural network functional
        grids = pyscf.dft.gen_grid.Grids(mol)
        grids.level = 0
        grids.becke_scheme = pyscf.dft.gen_grid.stratmann
        grids.build()
        mf = dft.RKS(mol)
        mf.grids = grids
        if IS_GLOBAL_XC:
            eval_xc_global = partial(eval_xc, is_global_xc=True)
            mf = mf.define_xc_(description=eval_xc_global, xctype="NN-AmplitudeEncoding")
        else:
            eval_xc_local = partial(eval_xc, is_global_xc=False)
            mf = mf.define_xc_(description=eval_xc_local, xctype="NN")

        mf.max_cycle = MAX_CYCLE  # Increase for potentially better convergence during eval

        # Calculate energy with trained model
        # Use stop_gradient as we only need the energy value, not gradients
        nn_energy = mf.kernel(params=jax.lax.stop_gradient(params))

        bond_lengths_plot.append(bond_length)
        ccsd_energies_plot.append(ref_energy)
        nn_energies_plot.append(nn_energy)

    # Create the dissociation curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(bond_lengths_plot, ccsd_energies_plot, "o-", label="CCSD Reference", color="blue")
    plt.plot(bond_lengths_plot, nn_energies_plot, "s--", label="Neural Network XC", color="red")
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Energy (Hartree)")
    plt.title("H₂ Dissociation Curve: CCSD vs Neural Network XC Functional")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    comparison_plot_filename = os.path.join(RESULTS_DIR, "h2_dissociation_comparison.png")
    plt.savefig(comparison_plot_filename, dpi=300)
    logger.info(f"Dissociation comparison plot saved to {comparison_plot_filename}")
    plt.close()

    # Create the error plot
    plt.figure(figsize=(10, 4))
    energy_errors = np.array(nn_energies_plot) - np.array(ccsd_energies_plot)
    plt.plot(bond_lengths_plot, energy_errors, "o-", color="purple")
    plt.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Energy Error (Hartree)")
    plt.title("Error in Neural Network XC Functional (NN - CCSD)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    error_plot_filename = os.path.join(RESULTS_DIR, "h2_dissociation_error.png")
    plt.savefig(error_plot_filename, dpi=300)
    logger.info(f"Dissociation error plot saved to {error_plot_filename}")
    plt.close()

    # Save the training and validation configurations for reproducibility
    import json

    config = {
        "training_bond_lengths": training_bond_lengths,
        "validation_bond_lengths": validation_bond_lengths,
        "evaluation_bond_lengths": eval_bond_lengths,
        "n_iterations": N_ITERATIONS,
        "batch_size": BATCH_SIZE,
        "energy_weight": ENERGY_WEIGHT,
        "density_weight": DENSITY_WEIGHT,
        "is_global_xc": IS_GLOBAL_XC,
    }

    config_path = os.path.join(RESULTS_DIR, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Experiment configuration saved to {config_path}")

    logger.info(f"All results saved to {RESULTS_DIR}")
