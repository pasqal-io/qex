"""Test to compare speed of jitted vs non-jitted LocalQNN."""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from horqrux.utils import random_state

from qedft.models.quantum.feature_maps import chebyshev_gates
from qedft.models.quantum.quantum_models import QNN
from qedft.models.quantum.variational_ansatz import hea


def measure_performance(n_qubits=2, n_layers=2, batch_size=4, n_features=2, n_runs=100):
    """Measure performance difference between jitted and non-jitted forward pass.

    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of layers in the ansatz
        batch_size: Number of samples to process at once
        n_features: Number of input features
        n_runs: Number of runs to average timing over

    Returns:
        tuple: (mean_time_standard, mean_time_jitted, speedup)
    """
    # Create initial quantum state
    state = random_state(n_qubits)

    # Create the model
    feature_map_fn = chebyshev_gates
    ansatz = hea(n_qubits, n_layers)

    qnn = QNN(
        n_qubits=n_qubits,
        feature_map_fn=feature_map_fn,
        ansatz=ansatz,
        state=state,
    )

    # JIT the forward pass
    jitted_forward = jax.jit(qnn.__call__)

    # Generate random data and parameters
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    param_values = jax.random.uniform(subkey, shape=(qnn.n_vparams,))

    key, subkey = jax.random.split(key)
    # I need shape [[n_features],[n_features]]
    print(f"Shape of x: {jax.random.uniform(subkey, shape=(batch_size, n_features)).shape}")
    x = jax.random.uniform(subkey, shape=(batch_size, n_features))

    # Warmup to compile JIT
    _ = jitted_forward(param_values, x)

    # Time non-jitted version
    start_time = time.time()
    for _ in range(n_runs):
        _ = qnn(param_values, x)
    end_time = time.time()
    time_standard = end_time - start_time

    # Time jitted version
    start_time = time.time()
    for _ in range(n_runs):
        _ = jitted_forward(param_values, x)
    end_time = time.time()
    time_jitted = end_time - start_time

    # Calculate speedup
    speedup = time_standard / time_jitted

    return time_standard / n_runs, time_jitted / n_runs, speedup


def test_jit_speedup():
    """Test the speedup from using JIT compilation."""
    # Basic test with default parameters
    time_std, time_jit, speedup = measure_performance()

    print(f"Standard execution time: {time_std:.6f} seconds per batch")
    print(f"JIT execution time:      {time_jit:.6f} seconds per batch")
    print(f"Speedup factor:          {speedup:.2f}x")

    # Verify correctness
    n_qubits = 2
    n_layers = 2
    state = random_state(n_qubits)

    qnn = QNN(
        n_qubits=n_qubits,
        feature_map_fn=chebyshev_gates,
        ansatz=hea(n_qubits, n_layers),
        state=state,
    )

    jitted_forward = jax.jit(qnn.__call__)

    key = jax.random.PRNGKey(42)
    param_values = jax.random.uniform(key, shape=(qnn.n_vparams,))
    x = jnp.array([0.5, 0.6, 0.7, 0.8])

    result_std = qnn(param_values, x)
    result_jit = jitted_forward(param_values, x)

    assert jnp.allclose(result_std, result_jit)
    print("Results match between jitted and non-jitted versions.")

    return speedup


def plot_scaling_study():
    """Run performance tests with different batch sizes and feature dimensions."""
    batch_sizes = [1, 5, 10, 20, 50, 100]
    n_features_list = [2]

    plt.figure(figsize=(12, 8))

    for n_features in n_features_list:
        speedups = []
        for batch_size in batch_sizes:
            _, _, speedup = measure_performance(
                batch_size=batch_size,
                n_features=n_features,
                n_runs=20,
            )
            speedups.append(speedup)

        plt.plot(batch_sizes, speedups, marker="o", label=f"Features = {n_features}")

    plt.xlabel("Batch Size")
    plt.ylabel("Speedup Factor (x)")
    plt.title("JIT Speedup with Different Batch Sizes and Feature Dimensions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("./results/jit_speedup_scaling.png")
    plt.close()


if __name__ == "__main__":
    speedup = test_jit_speedup()
    print("\nRunning scaling study (this may take a while)...")
    plot_scaling_study()
    print(f"Scaling study complete. Results saved to 'jit_speedup_scaling.png'")
