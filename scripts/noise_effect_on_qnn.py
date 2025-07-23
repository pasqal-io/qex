"""Noise effect on the GlobalQNNClassicalToQuantum model

This script is used to evaluate the effect of different types of noise
on the training loss of a GlobalQNNClassicalToQuantum model.
- The noise can be applied to quantum gates (bitflip, depolarizing, etc.)
- Gaussian noise can be added to the QNN output
- We check if the model is still able to converge to the correct solution.
"""

import os
from pathlib import Path

import qedft
import jax
import matplotlib.pyplot as plt
import numpy as np
import scipy
from jax import config, random
from jax_dft import np_utils
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

from qedft.config.config import Config
from qedft.data_io.dataset_loader import load_molecular_datasets_from_config
from qedft.models.networks import (
    GlobalQNNClassicalToQuantum,
    GlobalQNNQuantumToClassical,
)
from qedft.train.od.train import (
    create_kohn_sham_fn,
    create_loss_fn,
    create_training_step,
)
from qedft.models.wrappers import wrap_network_from_config

# Set the default dtype as float64
config.update("jax_enable_x64", True)
# set gpu device
config.update("jax_platform_name", "cpu")  # "cuda" or "cpu"

# Get the project path
project_path = Path(os.path.dirname(os.path.dirname(qedft.__file__)))
print(f"Project path: {project_path}")


def calculate_noise_probabilities(
    t1_ns: float, t2_ns: float, gate_time_ns: float, readout_error: float
):
    """
    Calculate realistic noise probabilities from IBM device parameters.

    Args:
        t1_ns: T1 relaxation time in nanoseconds
        t2_ns: T2 dephasing time in nanoseconds
        gate_time_ns: Gate execution time in nanoseconds
        readout_error: Readout error probability

    Returns:
        Dictionary with calculated noise probabilities
    """
    # Calculate amplitude damping probability: P = 1 - exp(-t_gate/T1)
    amplitude_damping_prob = 1.0 - np.exp(-gate_time_ns / t1_ns)

    # Calculate phase damping probability: P = 1 - exp(-t_gate/T2)
    phase_damping_prob = 1.0 - np.exp(-gate_time_ns / t2_ns)

    # Print summary of noise probabilities
    print("\n===========================================")
    print("Summary of noise probabilities:")
    print(f"T1: {t1_ns} ns, T2: {t2_ns} ns, Gate time: {gate_time_ns} ns")
    print(f"Amplitude damping probability: {amplitude_damping_prob}")
    print(f"Phase damping probability: {phase_damping_prob}")
    print(f"Readout error: {readout_error}")

    return {
        "amplitude_damping": amplitude_damping_prob,
        "phase_damping": phase_damping_prob,
        "readout_error": readout_error,
    }


def create_realistic_noise_config(
    noise_scale: float = 1.0, device_type: str = "best"
):
    """
    Creates realistic noise configuration based on IBM device parameters.

    Args:
        noise_scale: Scale factor to multiply all noise probabilities
        device_type: 'best' or 'worst' to use best/worst device parameters

    Returns:
        Tuple of (noise_instances, gaussian_noise_config)
    """
    # IBM device parameters from your data
    device_params = {
        "best": {
            "t1_ns": 593.86,
            "t2_ns": 533.33,
            "gate_time_ns": 49.78,
            "readout_error": 0.212158,
        },
        "worst": {
            "t1_ns": 433.44,
            "t2_ns": 403.38,
            "gate_time_ns": 60.00,
            "readout_error": 0.504395,
        },
    }

    params = device_params[device_type]
    noise_probs = calculate_noise_probabilities(**params)

    # Scale noise probabilities
    scaled_probs = {
        key: min(prob * noise_scale, 1.0)  # Cap at 1.0
        for key, prob in noise_probs.items()
    }

    print(f"Scaled noise probabilities: {scaled_probs}")

    # Create noise instances for quantum gates
    quantum_noise = (
        DigitalNoiseInstance(
            DigitalNoiseType.AMPLITUDE_DAMPING,
            scaled_probs["amplitude_damping"]
        ),
        DigitalNoiseInstance(
            DigitalNoiseType.PHASE_DAMPING, scaled_probs["phase_damping"]
        ),
    )

    # Add measurement/sampling noise as Gaussian noise
    # Scale readout error to reasonable Gaussian noise std
    gaussian_noise_std = scaled_probs["readout_error"] * 0.1

    return {
        "noise": quantum_noise,
        "add_gaussian_noise_to_qnn_output": True,
        "gaussian_noise_std": gaussian_noise_std,
    }


def evaluate_with_noise(
    config_dict, noise_scales=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
):
    """
    Evaluates model performance with different noise scale factors.

    Args:
        config_dict: Configuration dictionary
        noise_scales: List of noise scale factors to test

    Returns:
        Dictionary of results for each noise scale
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
    # Force copy to device
    initial_density = jax.jit(lambda x: x)(train_set.density)

    # Results dictionary
    results = {
        "noise_scales": noise_scales,
        "losses_best": [],
        "losses_worst": [],
        "convergence_best": [],
        "convergence_worst": [],
    }

    # Test each noise scale with both best and worst device parameters
    for noise_scale in noise_scales:
        for device_type in ["best", "worst"]:
            print(
                f"\n--- Testing {device_type} device with "
                f"noise scale {noise_scale} ---"
            )

            # Create model based on config
            model_config = config_dict.copy()

            # Create model
            classical_to_quantum = model_config.get(
                "classical_to_quantum", True
            )
            if classical_to_quantum is False:
                print("Using GlobalQNNQuantumToClassical model")
                model = GlobalQNNQuantumToClassical(model_config)
            else:
                print("Using GlobalQNNClassicalToQuantum model")
                model = GlobalQNNClassicalToQuantum(model_config)

            # Build network with realistic noise
            if noise_scale == 0.0:
                # No noise case
                noise_config = {
                    "noise": None,
                    "add_gaussian_noise_to_qnn_output": False,
                    "gaussian_noise_std": 0.0,
                }
            else:
                noise_config = create_realistic_noise_config(
                    noise_scale, device_type
                )

            network = model.build_network(grids, noise=noise_config["noise"])

            # Ensure proper output shape through wrapper
            prng = random.PRNGKey(config_dict.get("seed", 42))
            # Use model_config here
            wrapped_network = wrap_network_from_config(
                network, grids, model_config
            )

            # Test direct initialization networks
            init_fn, neural_xc_energy_density_fn = wrapped_network
            init_params = init_fn(prng)  # type: ignore

            # Add Gaussian noise to output if specified
            if noise_config["add_gaussian_noise_to_qnn_output"]:
                original_fn = neural_xc_energy_density_fn
                gaussian_std = noise_config["gaussian_noise_std"]

                def noisy_neural_xc_energy_density_fn(inputs, params):
                    outputs = original_fn(inputs, params)
                    # Use a consistent noise key
                    noise_key = random.PRNGKey(42)
                    noise = jax.random.normal(
                        noise_key, shape=outputs.shape
                    ) * gaussian_std
                    return outputs + noise

                neural_xc_energy_density_fn = (
                    noisy_neural_xc_energy_density_fn
                )

            # JIT the neural_xc_energy_density_fn
            neural_xc_energy_density_fn = jax.jit(
                neural_xc_energy_density_fn
            )
            spec, flatten_init_params = np_utils.flatten(init_params)

            # Create Kohn-Sham function and loss function
            kohn_sham_fn, batch_kohn_sham = create_kohn_sham_fn(
                config_dict,
                dataset,
                grids,
                neural_xc_energy_density_fn,
                spec,
            )

            loss_fn = create_loss_fn(
                batch_kohn_sham, grids, dataset, config_dict
            )
            value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

            # Create checkpoint directory specific to device and noise scale
            device_noise_str = (
                f"{device_type}_scale_{noise_scale:.6f}".replace(".", "p")
            )
            checkpoint_dir = project_path / "scripts_ckpts" / device_noise_str

            # Create training step
            training_step = create_training_step(
                value_and_grad_fn,
                train_set,
                initial_density,
                save_every_n=config_dict.get("save_every_n", 10),
                initial_checkpoint_index=0,
                checkpoint_dir=str(checkpoint_dir),
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

                final_loss = f
                converged = info["warnflag"] == 0
                print(f"Final loss: {final_loss}, Converged: {converged}")

            except Exception as e:
                error_msg = (
                    f"Error during training with {device_type} device "
                    f"noise scale {noise_scale}: {e}"
                )
                print(error_msg)
                final_loss = float("nan")
                converged = False

            # Store results
            if device_type == "best":
                results["losses_best"].append(final_loss)
                results["convergence_best"].append(converged)
            else:
                results["losses_worst"].append(final_loss)
                results["convergence_worst"].append(converged)

    return results


def plot_results(results):
    """
    Plots the results of noise evaluation.

    Args:
        results: Dictionary with noise scales and corresponding losses
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    noise_scales = results["noise_scales"]

    # Plot for best device
    ax1.plot(
        noise_scales,
        results["losses_best"],
        "o-",
        linewidth=2,
        label="Best Device",
        color="blue",
    )
    ax1.set_xlabel("Noise Scale Factor")
    ax1.set_ylabel("Final Loss")
    ax1.set_title("Effect of Noise on Training Loss (Best Device)")
    ax1.grid(True)
    ax1.set_yscale("log")

    # Add convergence indicators for best device
    for i, converged in enumerate(results["convergence_best"]):
        color = "green" if converged else "red"
        marker = "o" if converged else "x"
        ax1.plot(
            noise_scales[i],
            results["losses_best"][i],
            marker,
            color=color,
            markersize=10,
        )

    # Plot for worst device
    ax2.plot(
        noise_scales,
        results["losses_worst"],
        "o-",
        linewidth=2,
        label="Worst Device",
        color="red",
    )
    ax2.set_xlabel("Noise Scale Factor")
    ax2.set_ylabel("Final Loss")
    ax2.set_title("Effect of Noise on Training Loss (Worst Device)")
    ax2.grid(True)
    ax2.set_yscale("log")

    # Add convergence indicators for worst device
    for i, converged in enumerate(results["convergence_worst"]):
        color = "green" if converged else "red"
        marker = "o" if converged else "x"
        ax2.plot(
            noise_scales[i],
            results["losses_worst"][i],
            marker,
            color=color,
            markersize=10,
        )

    plt.tight_layout()
    save_path = project_path / "global_qnn_noise_evaluation_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Load configuration
    config_path = str(
        project_path / "qedft" / "config" / "train_config_global.yaml"
    )
    config = Config(config_path=config_path)
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

    # Define noise scale factors to test (similar to noise_effect_on_ksr.py)
    noise_scales = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    # similar scaling to noise_effect_on_ksr.py
    # noise_scales = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]

    # Print realistic noise parameters for reference
    print("=== Realistic IBM Device Noise Parameters ===")
    # for device_type in ["best", "worst"]:
    for device_type in ["best"]:
        for scale in noise_scales:
            noise_config = create_realistic_noise_config(scale, device_type)
            print(f"\n{device_type.capitalize()} device, scale {scale}:")
            print(noise_config)
    # Run evaluation
    results = evaluate_with_noise(config_dict, noise_scales)

    # Plot results
    plot_results(results)

    # Save results
    import json

    save_path = project_path / "global_qnn_noise_evaluation_results.json"
    with open(save_path, "w") as f:
        # Convert numpy values to Python native types for JSON serialization
        serializable_results = {
            "noise_scales": [float(x) for x in results["noise_scales"]],
            "losses_best": [
                float(x) if not np.isnan(x) else None
                for x in results["losses_best"]
            ],
            "losses_worst": [
                float(x) if not np.isnan(x) else None
                for x in results["losses_worst"]
            ],
            "convergence_best": [
                bool(x) for x in results["convergence_best"]
            ],
            "convergence_worst": [
                bool(x) for x in results["convergence_worst"]
            ],
        }
        json.dump(serializable_results, f, indent=2)

    print("\nResults saved to:")
    print(f"- {project_path / 'global_qnn_noise_evaluation_results.json'}")
    print(f"- {project_path / 'global_qnn_noise_evaluation_results.png'}")