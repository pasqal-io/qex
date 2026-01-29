"""
Simple function-based Quantum Neural Network implementation.

Here I wanted to understand why it is so slow to compile.

This module provides a minimal QNN implementation without classes to reduce
compilation time and complexity.

Original settings of what is in the paper:

{"experiment_name": "conv_adqc1_q6d8_mlp171_1_s3", "network_type": "conv_adqc", "molecule_name": "h2", "dataset": [80, 128, 192, 240, 384, 448, 544, 592], "rng": 3, "activation": "tanh", "n_qubits": 6, "n_neurons": 32, "n_layers": 8, "n_reupload_layers": 1, "save_every_n": 20, "use_rzz_parametrized_entanglers": false, "chebychev_reuploading": false, "add_reversed_rzz": false, "entangling_block_type": "alternate_linear", "single_qubit_rotations": ["rz", "rx", "rz"], "use_same_parameters": false, "add_negative_transform": false, "wrap_with_self_interaction_layer": false, "wrap_with_global_functional": false, "use_correlators_in_output": false, "output_operators": ["Z"], "use_bias_in_output": false, "max_train_steps": 10000, "factr": 0.01, "pgtol": 1e-14, "m": 40, "save_plot_loss": false, "num_iterations": 25, "ks_iter_to_ignore": 10, "discount": 0.9, "alpha": 0.5, "alpha_decay": 0.9, "num_mixing_iterations": 1, "density_mse_converge_tolerance": -1.0, "stop_gradient_step": -1, "enforce_reflection_symmetry": true, "use_relative_encoding": false, "num_grids": 513, "max_number_conv_layers": 1, "list_qubits_per_layer": [], "force_qubits_per_layer_is_kernel_width": false, "feature_map_type": "direct", "final_mlp_layers": [171, 1], "dont_use_parametrized_observables": false, "use_nel_as_input_with_mlp": false, "nel_exc_combination_type": "sum", "nel_mlp_layers": [1], "density_normalization_factor": 2.0}

"""

import jax
import jax.numpy as jnp
from chex import Array
from functools import partial
from horqrux import QuantumCircuit, expectation, zero_state
from horqrux.primitives.parametric import RY, RX, RZ
from horqrux.primitives.primitive import NOT, Z
from horqrux.composite import Observable
from horqrux.noise import NoiseProtocol
from horqrux.utils.operator_utils import TargetQubits
from horqrux.primitives.primitive import Primitive
import time
import numpy as np
from qedft.models.quantum.entangling_layers import entangling_ops

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cuda")  # "cuda" or "cpu"

# Add gradient checkpointing import
from jax import checkpoint


def create_feature_map_gates(x: Array, n_qubits: int, feature_type: str = "direct", noise: NoiseProtocol | None = None) -> list[Primitive]:
    """Create feature map gates for encoding classical data.

    Args:
        x: Input data array
        n_qubits: Number of qubits
        feature_type: Type of feature map ("direct", "chebyshev", "product")
        noise: Optional noise protocol to apply to gates

    Returns:
        List of gates implementing the feature map
    """
    target_idx = tuple((i,) for i in range(n_qubits))

    # Create noise-aware gate constructor
    if noise is not None:
        ry = partial(RY, noise=noise)
    else:
        ry = RY

    # Ensure we have enough data for all qubits by padding or repeating
    if x.shape[0] < n_qubits:
        # Pad with zeros if we don't have enough features
        x_padded = jnp.pad(x, (0, n_qubits - x.shape[0]))
    else:
        # Take only the first n_qubits features
        x_padded = x[:n_qubits]

    if feature_type == "direct":
        # Direct encoding: map input values directly to rotation angles
        encoding = x_padded
        return [ry(angle, idx) for angle, idx in zip(encoding, target_idx)]

    elif feature_type == "product":
        # Product encoding using arcsin, with clipping to avoid NaNs
        x_clipped = jnp.clip(x_padded, -0.99, 0.99)  # Avoid domain errors
        encoding = jnp.arcsin(x_clipped)
        return [ry(angle, idx) for angle, idx in zip(encoding, target_idx)]

    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


def create_ansatz_gates(n_qubits: int, n_layers: int, noise: NoiseProtocol | None = None) -> tuple[list[Primitive], list[str]]:
    """Create simple hardware-efficient ansatz gates with consistent parameter names.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        noise: Optional noise protocol to apply to gates

    Returns:
        Tuple of (gates, parameter_names)
    """
    gates = []
    param_names = []

    # Create noise-aware gate constructors
    if noise is not None:
        rz = partial(RZ, noise=noise)
        rx = partial(RX, noise=noise)
        not_gate = partial(NOT, noise=noise)
    else:
        rz = RZ
        rx = RX
        # not_gate = NOT

    for layer in range(n_layers):
        # Rotation gates for each qubit (RY, RX, RY pattern)
        for i in range(n_qubits):
            rz1_param = f"rz1_{layer}_{i}"
            rx_param = f"rx_{layer}_{i}"
            rz2_param = f"rz2_{layer}_{i}"

            gates.append(rz(rz1_param, (i,)))
            gates.append(rx(rx_param, (i,)))
            gates.append(rz(rz2_param, (i,)))
            param_names.extend([rz1_param, rx_param, rz2_param])

        # Alternate linear entangling gates
        gates.extend(entangling_ops(n_qubits, entangling_block_type="alternate_linear", noise=noise))

        # Entangling gates (circular CNOTs)
        # for i in range(n_qubits):
        #     gates.append(not_gate((i + 1) % n_qubits, i))

    return gates, param_names


def create_measurement_ops(n_qubits: int) -> list[Observable]:
    """Create measurement operators (Pauli-Z on each qubit) wrapped in Observable.

    Args:
        n_qubits: Number of qubits

    Returns:
        List of Observable objects containing Pauli-Z operators
    """
    return [Observable([Z(i)]) for i in range(n_qubits)]


def simple_qnn_core(params: Array, x: Array, n_qubits: int, n_layers: int, feature_type: str, noise: NoiseProtocol | None = None) -> Array:
    """Core QNN function without vmap.

    Args:
        params: Variational parameters array
        x: Input data (single sample)
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        feature_type: Type of feature map
        noise: Optional noise protocol to apply to gates

    Returns:
        Expectation value (scalar)
    """
    # Create gates
    feature_gates = create_feature_map_gates(x, n_qubits, feature_type, noise)
    ansatz_gates, param_names = create_ansatz_gates(n_qubits, n_layers, noise)
    measurement_ops = create_measurement_ops(n_qubits)

    # Combine all gates
    all_gates = feature_gates + ansatz_gates

    # Create parameter dictionary
    param_dict = {name: val for name, val in zip(param_names, params)}

    # Create circuit
    circuit = QuantumCircuit(n_qubits=n_qubits, operations=all_gates)

    # Initial state
    state = zero_state(n_qubits)

    # Compute expectation values and sum them
    expectations = expectation(state, circuit, measurement_ops, param_dict)

    return jnp.sum(expectations)


def simple_qnn_core_checkpointed(params: Array, x: Array, n_qubits: int, n_layers: int, feature_type: str, noise: NoiseProtocol | None = None) -> Array:
    """Core QNN function with gradient checkpointing for memory efficiency.

    This version applies checkpointing to individual layers rather than using scan.
    """
    # Create feature map gates and measurement ops
    feature_gates = create_feature_map_gates(x, n_qubits, feature_type, noise)
    ansatz_gates, param_names = create_ansatz_gates(n_qubits, n_layers, noise)
    measurement_ops = create_measurement_ops(n_qubits)

    # Combine all gates
    all_gates = feature_gates + ansatz_gates

    # Create parameter dictionary
    param_dict = {name: val for name, val in zip(param_names, params)}

    # Apply checkpointing to the circuit evaluation
    @checkpoint
    def compute_circuit():
        circuit = QuantumCircuit(n_qubits=n_qubits, operations=all_gates)
        state = zero_state(n_qubits)
        expectations = expectation(state, circuit, measurement_ops, param_dict)
        return jnp.sum(expectations)

    return compute_circuit()


@partial(jax.jit, static_argnums=(2, 3, 4))  # JIT with static arguments
def simple_qnn_core_optimized(params: Array, x: Array, n_qubits: int, n_layers: int, feature_type: str, noise: NoiseProtocol | None = None) -> Array:
    """Optimized core QNN function with static compilation."""
    # Pre-compile gate structure (this will be cached)
    feature_gates = create_feature_map_gates(x, n_qubits, feature_type, noise)
    ansatz_gates, param_names = create_ansatz_gates(n_qubits, n_layers, noise)
    measurement_ops = create_measurement_ops(n_qubits)

    # Combine all gates
    all_gates = feature_gates + ansatz_gates

    # Create parameter dictionary efficiently
    param_dict = dict(zip(param_names, params))

    # Create circuit once
    circuit = QuantumCircuit(n_qubits=n_qubits, operations=all_gates)

    # Initial state
    state = zero_state(n_qubits)

    # Compute expectation values
    expectations = expectation(state, circuit, measurement_ops, param_dict)

    return jnp.sum(expectations)

# Cache compiled circuits to avoid recompilation
_circuit_cache = {}

def get_cached_circuit(n_qubits: int, n_layers: int, feature_type: str, noise: NoiseProtocol | None = None):
    """Get or create a cached circuit structure."""
    cache_key = (n_qubits, n_layers, feature_type, noise)

    if cache_key not in _circuit_cache:
        # Create a dummy input to get gate structure
        dummy_x = jnp.zeros(n_qubits)
        feature_gates = create_feature_map_gates(dummy_x, n_qubits, feature_type, noise)
        ansatz_gates, param_names = create_ansatz_gates(n_qubits, n_layers, noise)
        measurement_ops = create_measurement_ops(n_qubits)

        _circuit_cache[cache_key] = {
            'param_names': param_names,
            'measurement_ops': measurement_ops,
            'ansatz_gates': ansatz_gates
        }

    return _circuit_cache[cache_key]


def linear_combination(quantum_outputs: Array, weights: Array, bias: float) -> float:
    """Combine quantum outputs with a linear layer.

    Args:
        quantum_outputs: Array of quantum expectation values from batch
        weights: Linear weights for combining outputs
        bias: Bias term

    Returns:
        Single scalar output
    """
    return jnp.dot(quantum_outputs, weights) + bias


def get_n_params(n_qubits: int, n_layers: int, num_batches: int) -> tuple[int, int, int]:
    """Get the number of parameters for the QNN and linear layer.

    Args:
        n_qubits: Number of qubits
        n_layers: Number of layers
        num_batches: Number of batches (for linear layer weights)

    Returns:
        Tuple of (quantum_params, linear_weights, total_params)
    """
    # Each layer has 3 rotation gates per qubit (RY, RX, RY)
    quantum_params = 3 * n_qubits * n_layers
    linear_weights = num_batches  # One weight per batch
    bias_params = 1  # One bias term
    total_params = quantum_params + linear_weights + bias_params

    return quantum_params, linear_weights, total_params


def add_gaussian_noise_layer(network: tuple, noise_std: float = 0.1):
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
        rng = kwargs.get('rng', jax.random.PRNGKey(0))
        if rng is not None and noise_std > 0:
            noise = jax.random.normal(rng, shape=outputs.shape) * noise_std
            return jnp.array(outputs + noise).reshape((-1,))
        # return jnp.array([outputs,])
        return jnp.array(outputs).reshape((-1,))

    return init_fn, noisy_apply_fn




def ibm_noise_config(noise_scale: float = 0.001, device_type: str = "best"):
    """Create IBM noise configuration.

    """

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

    return quantum_noise


def build_simple_qnn_with_linear(n_qubits: int, num_batches: int, num_features: int, n_layers: int = 2, feature_type: str = "direct", add_gaussian_noise_to_qnn_output: bool = False, noise_std: float = 0.001, gate_noise: NoiseProtocol | None = None):
    """Build initialization and application functions for simple QNN with linear output layer.

    Args:
        n_qubits: Number of qubits
        num_batches: Number of batches that will be processed (determines linear layer input size)
        num_features: Number of features per batch
        n_layers: Number of variational layers
        feature_type: Type of feature map
        add_gaussian_noise_to_qnn_output: Whether to add Gaussian noise to the output of the QNN
        noise_std: Standard deviation of the Gaussian noise
        gate_noise: Optional noise protocol to apply to quantum gates

    Returns:
        (init_fn, apply_fn) tuple
    """
    quantum_params, linear_weights, total_params = get_n_params(n_qubits, n_layers, num_batches)

    # Create vmapped version for quantum part
    vmapped_qnn = jax.vmap(
        simple_qnn_core,
        in_axes=(None, 0, None, None, None, None)  # Only vmap over x (axis 0)
    )

    def init_fn(rng, input_shape=None):
        """Initialize parameters for both quantum and classical parts."""
        # Split random key
        key1, key2, key3 = jax.random.split(rng, 3)

        # Initialize quantum parameters
        quantum_params_init = jax.random.uniform(
            key1, (quantum_params,), minval=-0.1, maxval=0.1
        )

        # Initialize linear weights
        linear_weights_init = jax.random.normal(key2, (linear_weights,)) * 0.1

        # Initialize bias
        bias_init = jax.random.normal(key3, ()) * 0.1

        # Combine all parameters
        all_params = jnp.concatenate([
            quantum_params_init,
            linear_weights_init,
            jnp.array([bias_init])
        ])

        return (-1, 1), all_params

    def apply_fn(params, x, rng_key=None, **kwargs):
        """Apply QNN with linear combination layer.

        Args:
            params: Model parameters
            x: Input data of shape (total_inputs,) - will be reshaped to (num_batches, num_features)
        """
        # Use provided RNG or create a new one based on current time
        if rng_key is None:
            import time
            rng_key = jax.random.PRNGKey(int(time.time() * 1000000) % (2**32))

        # Split RNG for different uses
        rng1, rng2 = jax.random.split(rng_key)

        # Reshape input from (total_inputs,) to (num_batches, num_features)
        expected_total_inputs = num_batches * num_features

        # Ensure input has the expected size
        if x.shape[0] != expected_total_inputs:
            raise ValueError(
                f"Input size {x.shape[0]} doesn't match expected size {expected_total_inputs} "
                f"(num_batches={num_batches} * num_features={num_features})"
            )

        x_batched = x.reshape(num_batches, num_features)

        # Split parameters
        quantum_params_vals = params[:quantum_params]
        linear_weights_vals = params[quantum_params:quantum_params + linear_weights]
        bias_val = params[-1]

        # Apply quantum circuit to get batch outputs
        quantum_outputs = vmapped_qnn(quantum_params_vals, x_batched, n_qubits, n_layers, feature_type, gate_noise)

        if add_gaussian_noise_to_qnn_output:
            if noise_std > 0:
                noise = jax.random.normal(rng1, shape=quantum_outputs.shape) * noise_std
                quantum_outputs = quantum_outputs + noise

        # Apply linear combination
        final_output = linear_combination(quantum_outputs, linear_weights_vals, bias_val)

        return final_output

    return init_fn, apply_fn


def build_simple_qnn(n_qubits: int, n_layers: int = 2, feature_type: str = "direct", gate_noise: NoiseProtocol | None = None):
    """Build initialization and application functions for simple QNN (original version).

    Args:
        n_qubits: Number of qubits
        n_layers: Number of variational layers
        feature_type: Type of feature map
        gate_noise: Optional noise protocol to apply to quantum gates

    Returns:
        (init_fn, apply_fn) tuple
    """
    quantum_params, _, _ = get_n_params(n_qubits, n_layers, 1)  # num_batches=1 for compatibility

    # Create vmapped version with correct in_axes
    vmapped_qnn = jax.vmap(
        simple_qnn_core,
        in_axes=(None, 0, None, None, None, None)  # Only vmap over x (axis 0)
    )

    def init_fn(rng, input_shape=None):
        """Initialize parameters."""
        return (-1, 1), jax.random.uniform(rng, (quantum_params,), minval=-0.1, maxval=0.1)

    def apply_fn(params, x, rng_key=None, **kwargs):
        """Apply QNN to input data."""
        if rng_key is None:
            raise ValueError("rng_key is required for proper randomness")
        return vmapped_qnn(params, x, n_qubits, n_layers, feature_type, gate_noise)

    return init_fn, apply_fn



# Add compilation hints for faster JIT
def set_jax_compilation_options():
    """Set JAX options for faster compilation."""
    # Enable XLA optimizations
    jax.config.update('jax_enable_x64', True)

    # Set compilation cache
    import os
    os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'

    # Enable persistent compilation cache
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)

# Call at module level
set_jax_compilation_options()

# Add progress tracking for long compilations
def create_circuit_with_progress(n_qubits: int, n_layers: int, feature_type: str = "direct"):
    """Create circuit with compilation progress tracking."""
    print(f"Compiling circuit: {n_qubits} qubits, {n_layers} layers...")

    start_time = time.time()

    # Your circuit creation code here
    quantum_params, _, _ = get_n_params(n_qubits, n_layers, 1)
    init_fn, apply_fn = build_simple_qnn_with_linear_optimized(
        n_qubits, 1, n_qubits, n_layers, feature_type, use_checkpointing=True
    )

    compile_time = time.time() - start_time
    print(f"✓ Circuit compiled in {compile_time:.2f} seconds")

    return init_fn, apply_fn


def build_simple_qnn_with_linear_optimized(n_qubits: int, num_batches: int, num_features: int, n_layers: int = 2, feature_type: str = "direct", add_gaussian_noise_to_qnn_output: bool = False, noise_std: float = 0.001, gate_noise: NoiseProtocol | None = None, use_checkpointing: bool = True):
    """Build optimized QNN with compilation optimizations.

    Args:
        use_checkpointing: Whether to use gradient checkpointing for memory efficiency
    """
    quantum_params, linear_weights, total_params = get_n_params(n_qubits, n_layers, num_batches)

    # Choose core function based on optimization settings
    if use_checkpointing:
        core_fn = simple_qnn_core_checkpointed
    else:
        core_fn = simple_qnn_core_optimized

    # Pre-compile the vmapped version with static arguments
    vmapped_qnn = jax.vmap(
        core_fn,
        in_axes=(None, 0, None, None, None, None)
    )

    # Pre-JIT the vmapped function for faster subsequent calls
    # Remove gate_noise from static_argnums since it's not hashable
    @partial(jax.jit, static_argnums=(2, 3, 4))
    def jitted_vmapped_qnn(params, x_batched, n_qubits, n_layers, feature_type, gate_noise):
        return vmapped_qnn(params, x_batched, n_qubits, n_layers, feature_type, gate_noise)

    def init_fn(rng, input_shape=None):
        """Initialize parameters for both quantum and classical parts."""
        key1, key2, key3 = jax.random.split(rng, 3)

        quantum_params_init = jax.random.uniform(
            key1, (quantum_params,), minval=-0.1, maxval=0.1
        )
        linear_weights_init = jax.random.normal(key2, (linear_weights,)) * 0.1
        bias_init = jax.random.normal(key3, ()) * 0.1

        all_params = jnp.concatenate([
            quantum_params_init,
            linear_weights_init,
            jnp.array([bias_init])
        ])

        return (-1, 1), all_params

    def apply_fn(params, x, rng_key=None, **kwargs):
        """Optimized apply function with pre-compiled components."""
        if rng_key is None:
            import time
            rng_key = jax.random.PRNGKey(int(time.time() * 1000000) % (2**32))

        rng1, rng2 = jax.random.split(rng_key)

        expected_total_inputs = num_batches * num_features

        if x.shape[0] != expected_total_inputs:
            raise ValueError(
                f"Input size {x.shape[0]} doesn't match expected size {expected_total_inputs} "
                f"(num_batches={num_batches} * num_features={num_features})"
            )

        x_batched = x.reshape(num_batches, num_features)

        quantum_params_vals = params[:quantum_params]
        linear_weights_vals = params[quantum_params:quantum_params + linear_weights]
        bias_val = params[-1]

        # Use pre-compiled function
        quantum_outputs = jitted_vmapped_qnn(
            quantum_params_vals, x_batched, n_qubits, n_layers, feature_type, gate_noise
        )

        if add_gaussian_noise_to_qnn_output and noise_std > 0:
            noise = jax.random.normal(rng1, shape=quantum_outputs.shape) * noise_std
            quantum_outputs = quantum_outputs + noise

        final_output = linear_combination(quantum_outputs, linear_weights_vals, bias_val)

        return final_output

    return init_fn, apply_fn


if __name__ == "__main__":

    from qedft.models.utils import count_parameters

    # Test the simple QNN with linear layer
    n_qubits = 6
    n_layers = 12
    num_features = 3  # Features per batch

    # Setup realistic data dimensions
    num_inputs = 513
    num_batches = num_inputs // num_features  # 171 batches

    # Input is now a flat array of shape (513,)
    x = jnp.ones((num_inputs,))

    # Initialize
    key = jax.random.PRNGKey(0)

    print("=== Testing Optimized QNN with Linear Output Layer ===")
    init_fn, apply_fn = build_simple_qnn_with_linear_optimized(n_qubits=n_qubits, num_batches=num_batches, num_features=num_features, n_layers=n_layers, feature_type="direct", add_gaussian_noise_to_qnn_output=True, noise_std=0.001, gate_noise=None, use_checkpointing=True)
    _, params = init_fn(key)
    result = apply_fn(params, x)
    # jit it
    time_start = time.time()
    jitted_apply = jax.jit(apply_fn)
    result_jit = jitted_apply(params, x)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    print(f"JIT result: {result_jit}")
    print("✓ JIT compilation works correctly")
    print(f"Result (scalar): {result}")
    print(f"JIT result: {result_jit}")
    print(f"Result shape: {result.shape}")

    print("=== Testing QNN with Linear Output Layer ===")
    init_fn, apply_fn = build_simple_qnn_with_linear(n_qubits, num_batches, num_features, n_layers, "direct")
    _, params = init_fn(key)

    init_fn_noisy, apply_fn_noisy = build_simple_qnn_with_linear(n_qubits, num_batches, num_features, n_layers, "direct", add_gaussian_noise_to_qnn_output=True, noise_std=0.001)

    quantum_params, linear_weights, total_params = get_n_params(n_qubits, n_layers, num_batches)
    print(f"Quantum parameters: {quantum_params}")
    print(f"Linear weights: {linear_weights}")
    print(f"Total parameters: {total_params}")
    print(f"Actual parameter count: {len(params)}")

    print(f"Input shape: {x.shape}")
    print(f"Will be reshaped to: ({num_batches}, {num_features})")
    print(f"Number of batches: {num_batches}")
    print(f"Features per batch: {num_features}")

    # Apply QNN with linear layer
    result = apply_fn(params, x)
    print(f"Result (scalar): {result}")
    print(f"Result shape: {result.shape}")

    # Apply QNN with linear layer and noisy output
    result_noisy = apply_fn_noisy(params, x)
    print(f"Noisy result (scalar): {result_noisy}")
    print(f"Noisy result shape: {result_noisy.shape}")

    # Use the wrap function to make neural XC functional
    print("=== Testing Neural XC Functional ===")
    from qedft.models.wrappers import wrap_network
    init_fn_neural_xc, apply_fn_neural_xc = wrap_network((init_fn, apply_fn), x, "mlp_ksr", wrap_self_interaction=False, wrap_with_negative_transform=False)
    params_neural_xc = init_fn_neural_xc(key)
    result_neural_xc = apply_fn_neural_xc(x, params_neural_xc)
    print(f"Neural XC result (scalar): {result_neural_xc}")
    print(f"Neural XC result shape: {result_neural_xc.shape}")

    # Test JIT compilation
    jitted_apply = jax.jit(apply_fn)
    result_jit = jitted_apply(params, x)
    print(f"JIT result: {result_jit}")

    # Apply QNN with linear layer and noisy output
    jitted_apply_noisy = jax.jit(apply_fn_noisy)
    result_jit_noisy = jitted_apply_noisy(params, x)
    print(f"JIT noisy result: {result_jit_noisy}")
    # Verify they match
    assert jnp.allclose(result, result_jit)
    print("✓ JIT compilation works correctly")

    # Test JIT compilation
    jitted_apply_noisy = jax.jit(apply_fn_noisy)
    result_jit_noisy = jitted_apply_noisy(params, x)
    print(f"JIT noisy result: {result_jit_noisy}")
    # Verify they match
    assert jnp.allclose(result_noisy, result_jit_noisy)
    print("✓ JIT compilation works correctly")

    print("\n=== Testing Original QNN (returns batch outputs) ===")
    # For the original QNN, we need to pass the reshaped data
    x_batched = x.reshape(num_batches, num_features)
    init_fn_orig, apply_fn_orig = build_simple_qnn(n_qubits, n_layers, "direct")
    _, params_orig = init_fn_orig(key)

    result_orig = apply_fn_orig(params_orig, x_batched)
    print(f"Original QNN input shape: {x_batched.shape}")
    print(f"Original QNN result shape: {result_orig.shape}")
    print(f"Original QNN result (first 5): {result_orig[:5]}")

    # Test different feature types with linear layer
    print("\n=== Testing Different Feature Types with Linear Layer ===")
    for feature_type in ["direct", "product"]:
        try:
            init_fn, apply_fn = build_simple_qnn_with_linear(n_qubits, num_batches, num_features, n_layers, feature_type)
            _, params = init_fn(key)
            result = apply_fn(params, x)
            print(f"{feature_type} feature map result: {result}")
        except Exception as e:
            print(f"{feature_type} feature map failed: {e}")

    # Test with quantum gate noise
    print("\n=== Testing with Quantum Gate Noise ===")
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    # Was like 90 sec to eval and to jit was 35 sec
    # gate_noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.01),)

    # More noise sources so should be slower
    gate_noise_ibm = ibm_noise_config(noise_scale=1.0, device_type="best")
    # 10 sec
    print(gate_noise_ibm)

    try:
        # This is massively slower than the noiseless version
        init_fn_noise, apply_fn_noise = build_simple_qnn_with_linear(
            n_qubits, num_batches, num_features, n_layers, "direct", gate_noise=gate_noise_ibm
        )
        _, params_noise = init_fn_noise(key)

        # Time it to compare with the noiseless version
        # 1 min 30 sec
        start_time = time.time()
        result_noise = apply_fn_noise(params_noise, x)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds for noisy eval.")
        print(f"Result with gate noise: {result_noise}")

        # JIT it to compare how fast it is
        start_time_jit = time.time()
        jitted_apply_noise = jax.jit(apply_fn_noise)
        result_jit_noise = jitted_apply_noise(params_noise, x)
        end_time_jit = time.time()
        print(f"Time taken: {end_time_jit - start_time_jit} seconds for JIT eval.")
        print(f"JIT noisy result: {result_jit_noise}")
        # Verify they match
        assert jnp.allclose(result_noise, result_jit_noise)
        print("✓ JIT compilation works correctly")

        # Compare with noiseless version
        result_clean = apply_fn(params, x)
        print(f"Result without noise: {result_clean}")
        print(f"Difference due to noise: {abs(result_noise - result_clean)}")

    except Exception as e:
        print(f"Gate noise test failed: {e}")

    print(f"\nTotal parameter count: {count_parameters(params)}")

    # Test error handling for wrong input size
    print("\n=== Testing Input Size Validation ===")
    try:
        wrong_x = jnp.ones((500,))  # Wrong size
        apply_fn(params, wrong_x)
    except ValueError as e:
        print(f"✓ Correctly caught input size error: {e}")