"""Noise effect on the SOTA XC functional (the KSR model)

This script is used to evaluate the effect of Gaussian noise
on the training loss of a Kohn-Sham model.
- The noise is added to the output of the neural network.
- We check if the model is still able to converge to the correct solution.
"""

import os
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy
from jax import config, random
from jax_dft import np_utils

from qedft.config.config import Config
from qedft.data_io.dataset_loader import load_molecular_datasets_from_config
from qedft.models.classical.classical_models import build_global_mlp, build_local_mlp
from qedft.models.classical.global_ksr_model import create_ksr_model_from_config
from qedft.models.wrappers import wrap_network_from_config
from qedft.train.od.train import create_kohn_sham_fn, create_loss_fn, create_training_step

# Set the default dtype as float64
config.update("jax_enable_x64", True)
# set gpu device
config.update("jax_platform_name", "cuda")  # "cuda" or "cpu"

# Get the project path
import qedft

# Get the project path
project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
print(f"Project path: {project_path}")


def add_gaussian_noise_layer(network, noise_std=0.1):
    """
    Wraps a network with a Gaussian noise layer.

    Args:
        network: A tuple of (init_fn, apply_fn)
        noise_std: Standard deviation of the Gaussian noise

    Returns:
        A tuple of (init_fn, apply_fn) with noise added
    """
    init_fn, apply_fn = network

    def noisy_apply_fn(params, inputs, rng_key=None, **kwargs):
        outputs = apply_fn(params, inputs, **kwargs)
        if rng_key is not None and noise_std > 0:
            noise = jax.random.normal(rng_key, shape=outputs.shape) * noise_std
            return outputs + noise
        return outputs

    return init_fn, noisy_apply_fn


def evaluate_with_noise(config_dict, noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5]):
    """
    Evaluates model performance with different noise levels.

    Args:
        config_dict: Configuration dictionary
        noise_levels: List of noise standard deviations to test

    Returns:
        Dictionary of results for each noise level
    """
    # Load dataset
    base_path = project_path / "data" / "od"
    list_datasets = load_molecular_datasets_from_config(
        config_dict,
        base_path,
        check_grid_centering=True,
    )

    dataset = list_datasets[0][0]  # first dataset
    train_set = list_datasets[0][1]
    grids = dataset.grids
    initial_density = jax.jit(lambda x: x)(train_set.density)  # Force copy to device

    # Results dictionary
    results = {
        "noise_levels": noise_levels,
        "losses": [],
        "convergence": [],
    }

    # Test each noise level
    for noise_std in noise_levels:
        print(f"\n--- Testing noise level: {noise_std} ---")

        # Create model based on config
        if config_dict["network_type"] == "ksr":
            network, init_fn, neural_xc_energy_density_fn = create_ksr_model_from_config(grids)
        elif config_dict["network_type"] == "mlp_local":
            network = build_local_mlp(
                n_neurons=config_dict.get("mlp_neurons", 513),
                n_layers=config_dict.get("mlp_layers", 2),
                activation=config_dict.get("mlp_activation", "tanh"),
                n_outputs=1,
                rng_key=jax.random.PRNGKey(config_dict.get("rng", 0)),
                density_normalization_factor=config_dict.get("density_normalization_factor", 2.0),
                grids=grids,
            )
            network = wrap_network_from_config(network, grids, config_dict)
            init_fn, neural_xc_energy_density_fn = network
        elif config_dict["network_type"] == "mlp_global":
            network = build_global_mlp(
                n_neurons=config_dict.get("mlp_neurons", 513),
                n_layers=config_dict.get("mlp_layers", 2),
                activation=config_dict.get("mlp_activation", "tanh"),
                n_outputs=1,
                rng_key=jax.random.PRNGKey(config_dict.get("rng", 0)),
                density_normalization_factor=config_dict.get("density_normalization_factor", 2.0),
                grids=grids,
            )
            network = wrap_network_from_config(network, grids, config_dict)
            init_fn, neural_xc_energy_density_fn = network

        # Add noise layer
        if noise_std > 0:
            # Create a wrapper for the neural_xc_energy_density_fn that adds noise
            original_fn = neural_xc_energy_density_fn

            def noisy_neural_xc_energy_density_fn(inputs, params):
                outputs = original_fn(inputs, params)
                # Generate a new random key for each call
                rng_key = random.PRNGKey(int(time.time() * 1000) % (2**32))
                noise = jax.random.normal(rng_key, shape=outputs.shape) * noise_std
                return outputs + noise

            neural_xc_energy_density_fn = noisy_neural_xc_energy_density_fn

        # Initialize parameters
        prng = random.PRNGKey(config_dict.get("rng", 0))
        if config_dict["network_type"] == "ksr":
            init_params = init_fn(prng)
        else:
            input_shape = (
                (1,) if config_dict["network_type"] == "mlp_global" else (-1, grids.shape[0], 1)
            )
            init_params = init_fn(prng, input_shape=input_shape)

        spec, flatten_init_params = np_utils.flatten(init_params)

        # Create Kohn-Sham function and loss function
        kohn_sham_fn, batch_kohn_sham = create_kohn_sham_fn(
            config_dict,
            dataset,
            grids,
            neural_xc_energy_density_fn,
            spec,
        )

        loss_fn = create_loss_fn(batch_kohn_sham, grids, dataset, config_dict)
        value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        # Create training step
        training_step = create_training_step(
            value_and_grad_fn,
            train_set,
            initial_density,
            save_every_n=config_dict.get("save_every_n", 10),
            initial_checkpoint_index=0,
            checkpoint_dir=project_path / "scripts_ckpts",
            spec=spec,
        )

        # Train with L-BFGS-B
        try:
            x, f, info = scipy.optimize.fmin_l_bfgs_b(
                training_step,
                x0=np.array(flatten_init_params),
                maxfun=config_dict.get("maxfun", 5),
                factr=config_dict.get("factr", 1e12),
                m=config_dict.get("m", 10),
                pgtol=config_dict.get("pgtol", 1e-5),
                maxiter=config_dict.get("maxiter", 5),
            )

            results["losses"].append(f)
            results["convergence"].append(info["warnflag"] == 0)
            print(f"Final loss: {f}, Converged: {info['warnflag'] == 0}")

        except Exception as e:
            print(f"Error during training with noise {noise_std}: {e}")
            results["losses"].append(float("nan"))
            results["convergence"].append(False)

    return results


def plot_results(results):
    """
    Plots the results of noise evaluation.

    Args:
        results: Dictionary with noise levels and corresponding losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results["noise_levels"], results["losses"], "o-", linewidth=2)
    plt.xlabel("Noise Standard Deviation")
    plt.ylabel("Final Loss")
    plt.title("Effect of Gaussian Noise on Training Loss")
    plt.grid(True)
    plt.yscale("log")

    # Add convergence indicators
    for i, converged in enumerate(results["convergence"]):
        color = "green" if converged else "red"
        marker = "o" if converged else "x"
        plt.plot(
            results["noise_levels"][i],
            results["losses"][i],
            marker,
            color=color,
            markersize=10,
        )

    plt.savefig(project_path / "noise_evaluation_results.png")
    plt.show()


if __name__ == "__main__":
    # Load configuration
    config = Config(config_path=project_path / "qedft" / "config" / "train_config.yaml")
    config_dict = config.config

    # Set model type - can be changed to test different models
    config_dict.update(
        {
            "network_type": "ksr",  # Options: 'ksr', 'mlp_local', 'mlp_global'
            "maxiter": 1000,  # Reduced for faster testing
            "maxfun": 1000,
            "save_every_n": 50,
        },
    )

    # Define noise levels to test
    noise_levels = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]

    # Run evaluation
    results = evaluate_with_noise(config_dict, noise_levels)

    # Plot results
    plot_results(results)

    # Save results
    import json

    with open(project_path / "noise_evaluation_results.json", "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {
            "noise_levels": [float(x) for x in results["noise_levels"]],
            "losses": [float(x) for x in results["losses"]],
            "convergence": [bool(x) for x in results["convergence"]],
        }
        json.dump(serializable_results, f, indent=2)
