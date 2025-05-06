"""Test to compare speed of jitted vs non-jitted LocalQNN."""

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from horqrux.utils import random_state
from tqdm import tqdm

from qedft.models.quantum.feature_maps import chebyshev_gates
from qedft.models.quantum.quantum_models import QNN
from qedft.models.quantum.variational_ansatz import hea


def test_jit_speedup(n_runs=100):
    """Test the speedup from using JIT compilation."""
    # Initialize circuit parameters
    n_qubits = 9
    n_layers = 4

    # Create initial quantum state
    state = random_state(n_qubits)

    # Create model
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

    # Generate example input data (following the pattern from the original file)
    key = jax.random.PRNGKey(0)
    param_values = jax.random.uniform(key, shape=(qnn.n_vparams,))
    x = jnp.array([0.5, 0.6, 0.7, 0.8])  # Input vector

    # Verify correctness
    y_std = qnn(param_values, x)
    y_jit = jitted_forward(param_values, x)
    assert jnp.allclose(y_std, y_jit)
    print("Results match between jitted and non-jitted versions.")

    # Warmup JIT compilation
    _ = jitted_forward(param_values, x)

    # Time non-jitted version
    times_std = []
    for _ in tqdm(range(n_runs)):
        start_time = time.time()
        _ = qnn(param_values, x)
        end_time = time.time()
        times_std.append(end_time - start_time)

    print("Finished non-jitted version")

    # Time jitted version
    times_jit = []
    for _ in tqdm(range(n_runs)):
        start_time = time.time()
        _ = jitted_forward(param_values, x)
        end_time = time.time()
        times_jit.append(end_time - start_time)

    print("Finished jitted version")

    # Calculate mean and standard deviation
    mean_std = np.mean(times_std)
    std_dev_std = np.std(times_std)
    mean_jit = np.mean(times_jit)
    std_dev_jit = np.std(times_jit)
    speedup = mean_std / mean_jit

    print(f"Standard execution time: {mean_std*1000:.3f} ± {std_dev_std*1000:.3f} ms")
    print(f"JIT execution time:      {mean_jit*1000:.3f} ± {std_dev_jit*1000:.3f} ms")
    print(f"Speedup factor:          {speedup:.2f}x")

    # Plot timing histogram
    plt.figure(figsize=(10, 6))
    plt.hist(np.array(times_std) * 1000, alpha=0.5, label="Standard", bins=30)
    plt.hist(np.array(times_jit) * 1000, alpha=0.5, label="JIT", bins=30)
    plt.xlabel("Execution Time (ms)")
    plt.ylabel("Frequency")
    plt.title("Execution Time Distribution: Standard vs JIT")
    plt.legend()
    plt.savefig("./results/jit_speedup_histogram.png")
    plt.close()

    return speedup


if __name__ == "__main__":
    print("Testing single input speedup:")
    test_jit_speedup()
    print("\nTest complete. Results saved as PNG files.")
