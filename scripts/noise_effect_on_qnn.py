"""Noise effect on the GlobalQNNClassicalToQuantum model

This script is used to evaluate the effect of different types of noise
on the training loss of a GlobalQNNClassicalToQuantum model.
- The noise can be applied to quantum gates (bitflip, depolarizing, etc.)
- Gaussian noise can be added to the QNN output
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
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

from qedft.config.config import Config
from qedft.data_io.dataset_loader import load_molecular_datasets_from_config
from qedft.models.networks import GlobalQNNClassicalToQuantum, GlobalQNNQuantumToClassical
from qedft.train.od.train import create_kohn_sham_fn, create_loss_fn, create_training_step
from qedft.models.wrappers import wrap_network_from_config

# Set the default dtype as float64
config.update("jax_enable_x64", True)
# set gpu device
config.update("jax_platform_name", "cpu")  # "cuda" or "cpu"

# Get the project path
import qedft

# Get the project path
project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
print(f"Project path: {project_path}")


def create_noise_config(noise_type, noise_level):
    """
    Creates noise configuration for the GlobalQNNClassicalToQuantum model.

    Args:
        noise_type: Type of noise ('bitflip', 'depolarizing', 'amplitude_damping', 'gaussian', 'none')
        noise_level: Noise level (probability for quantum noise, std for gaussian)

    Returns:
        Dictionary with noise configuration
    """
    if noise_type == 'none':
        return {
            'noise': None,
            'add_gaussian_noise_to_qnn_output': False,
            'gaussian_noise_std': 0.0
        }
    elif noise_type == 'gaussian':
        return {
            'noise': None,
            'add_gaussian_noise_to_qnn_output': True,
            'gaussian_noise_std': noise_level
        }
    else:
        # Map noise type to DigitalNoiseType and add gaussian noise to QNN output
        noise_type_map = {
            'bitflip': DigitalNoiseType.BITFLIP,
            'depolarizing': DigitalNoiseType.DEPOLARIZING,
            'amplitude_damping': DigitalNoiseType.AMPLITUDE_DAMPING,
            'phase_flip': DigitalNoiseType.PHASEFLIP,
            'phase_damping': DigitalNoiseType.PHASE_DAMPING,
        }

        if noise_type not in noise_type_map:
            raise ValueError(f"Unknown noise type: {noise_type}")

        digital_noise = DigitalNoiseInstance(noise_type_map[noise_type], noise_level)
        return {
            'noise': (digital_noise,),
            'add_gaussian_noise_to_qnn_output': True,
            'gaussian_noise_std': 0.0
        }


def evaluate_with_noise(config_dict, noise_configs):
    """
    Evaluates model performance with different noise configurations.

    Args:
        config_dict: Configuration dictionary
        noise_configs: List of dictionaries with noise configurations

    Returns:
        Dictionary of results for each noise configuration
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
        "noise_configs": noise_configs,
        "losses": [],
        "convergence": [],
        "noise_types": [],
        "noise_levels": [],
    }

    # Test each noise configuration
    for noise_config in noise_configs:
        noise_type = noise_config.get('type', 'none')
        noise_level = noise_config.get('level', 0.0)

        print(f"\n--- Testing noise: {noise_type} with level {noise_level} ---")

        # Create model based on config
        model_config = config_dict.copy()
        model_config.update(noise_config)

        # Create model
        classical_to_quantum = model_config.get("classical_to_quantum", True)
        if classical_to_quantum is False:
            print("Using GlobalQNNQuantumToClassical model")
            model = GlobalQNNQuantumToClassical(model_config)
        else:
            print("Using GlobalQNNClassicalToQuantum model")
            model = GlobalQNNClassicalToQuantum(model_config)

        # Build network with noise
        noise_params = create_noise_config(noise_type, noise_level)
        network = model.build_network(grids, noise=noise_params['noise'])

        # Ensure proper output shape through wrapper
        prng = random.PRNGKey(config_dict.get("seed", 42))
        wrapped_network = wrap_network_from_config(network, grids, model_config)  # Use model_config here

        # Test direct initialization networks
        init_fn, neural_xc_energy_density_fn = wrapped_network
        init_params = init_fn(prng, input_shape=(1,))

        # JIT the neural_xc_energy_density_fn
        neural_xc_energy_density_fn = jax.jit(neural_xc_energy_density_fn)
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
            results["noise_types"].append(noise_type)
            results["noise_levels"].append(noise_level)
            print(f"Final loss: {f}, Converged: {info['warnflag'] == 0}")

        except Exception as e:
            print(f"Error during training with noise {noise_type} level {noise_level}: {e}")
            results["losses"].append(float("nan"))
            results["convergence"].append(False)
            results["noise_types"].append(noise_type)
            results["noise_levels"].append(noise_level)

    return results


def plot_results(results):
    """
    Plots the results of noise evaluation.

    Args:
        results: Dictionary with noise configurations and corresponding losses
    """
    # Create subplots for different noise types
    noise_types = list(set(results["noise_types"]))
    n_types = len(noise_types)

    fig, axes = plt.subplots(1, n_types, figsize=(5*n_types, 6))
    if n_types == 1:
        axes = [axes]

    for i, noise_type in enumerate(noise_types):
        # Filter results for this noise type
        mask = [nt == noise_type for nt in results["noise_types"]]
        levels = [results["noise_levels"][j] for j, m in enumerate(mask) if m]
        losses = [results["losses"][j] for j, m in enumerate(mask) if m]
        convergence = [results["convergence"][j] for j, m in enumerate(mask) if m]

        ax = axes[i]
        ax.plot(levels, losses, "o-", linewidth=2, label=f"{noise_type} noise")
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Final Loss")
        ax.set_title(f"Effect of {noise_type.capitalize()} Noise on Training Loss")
        ax.grid(True)
        ax.set_yscale("log")

        # Add convergence indicators
        for j, converged in enumerate(convergence):
            color = "green" if converged else "red"
            marker = "o" if converged else "x"
            ax.plot(
                levels[j],
                losses[j],
                marker,
                color=color,
                markersize=10,
            )

        ax.legend()

    plt.tight_layout()
    plt.savefig(project_path / "global_qnn_noise_evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Load configuration
    config = Config(config_path=project_path / "qedft" / "config" / "train_config_global.yaml")
    config_dict = config.config

    # Update config with proper parameters for output shape
    config_dict.update(
        {
            "network_type": "mlp_ksr",
            "n_qubits": 6,
            "n_var_layers": 4,
            "n_features": 3,  # Match n_qubits
            "largest_kernel_width": 2**4,  # Match n_qubits
            "max_number_conv_layers": 1,
            "normalization": 1.0,
            "use_bias_mlp": False,
            "diff_mode": "ad",
            "n_shots": 0,
            "maxiter": 1000,
            "maxfun": 1000,
            "save_every_n": 50,
        },
    )

    # Define noise configurations to test
    noise_configs = [
        # No noise baseline
        {"type": "none", "level": 0.0},

        # Gaussian noise on QNN output
        {"type": "gaussian", "level": 0.001},
        {"type": "gaussian", "level": 0.01},
        {"type": "gaussian", "level": 0.1},
        {"type": "gaussian", "level": 1.0},

        # Quantum noise on gates
        {"type": "bitflip", "level": 0.001},
        {"type": "bitflip", "level": 0.01},
        {"type": "bitflip", "level": 0.1},

        {"type": "depolarizing", "level": 0.001},
        {"type": "depolarizing", "level": 0.01},
        {"type": "depolarizing", "level": 0.1},

        {"type": "amplitude_damping", "level": 0.001},
        {"type": "amplitude_damping", "level": 0.01},
        {"type": "amplitude_damping", "level": 0.1},
    ]

    # Run evaluation
    results = evaluate_with_noise(config_dict, noise_configs)

    # Plot results
    plot_results(results)

    # Save results
    import json

    with open(project_path / "global_qnn_noise_evaluation_results.json", "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {
            "noise_types": results["noise_types"],
            "noise_levels": [float(x) for x in results["noise_levels"]],
            "losses": [float(x) if not np.isnan(x) else None for x in results["losses"]],
            "convergence": [bool(x) for x in results["convergence"]],
        }
        json.dump(serializable_results, f, indent=2)

    print("\nResults saved to:")
    print(f"- {project_path / 'global_qnn_noise_evaluation_results.json'}")
    print(f"- {project_path / 'global_qnn_noise_evaluation_results.png'}")