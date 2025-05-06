"""Quantum measurement functions.

This module provides functions for measuring quantum states, particularly focused on
magnetization measurements. The functions are designed to work with JAX for automatic
differentiation and acceleration.

Key functions:
- qubit_magnetization: Computes per-qubit magnetization
- total_magnetization: Computes total system magnetization
- total_magnetization_via_inner_product: Alternative magnetization calculation
- total_magnetization_ops: Creates measurement operators
"""

from collections.abc import Callable
from functools import reduce
from operator import add

import jax.numpy as jnp
import numpy as np
from chex import Array
from horqrux import expectation
from horqrux.apply import apply_gates as apply_gate
from horqrux.composite import Observable
from horqrux.primitives.primitive import Primitive, Z
from horqrux.utils import State, inner


def qubit_magnetization(state: State) -> Array:
    """Calculate the magnetization of each qubit in a quantum state.

    Computes the magnetization (expectation value of Pauli Z) for each qubit by
    applying Z gates and calculating expectation values. The function is JAX-compatible
    and differentiable.

    Args:
        state: Quantum state array of shape (..., 2, 2, ..., 2) with n_qubits dimensions.

    Returns:
        Array of shape (n_qubits,) containing magnetization values in [-1, 1].

    Example:
        >>> import jax.numpy as jnp
        >>> from horqrux import zero_state, X, apply_gate
        >>> state = zero_state(2)  # |00⟩ state
        >>> qubit_magnetization(state)
        Array([1., 1.], dtype=float32)  # Both qubits have +1 magnetization

        >>> # Flip first qubit
        >>> flipped = apply_gate(state, X(0))  # |10⟩ state
        >>> qubit_magnetization(flipped)
        Array([-1., 1.], dtype=float32)  # First qubit -1, second +1

        >>> # Works with gradients
        >>> import jax
        >>> jax.grad(lambda s: jnp.sum(qubit_magnetization(s)))(state)
        Array([...], dtype=float32)  # Gradient w.r.t state
    """

    def qubit_magnetization(idx):
        projection = apply_gate(state, Z(idx))
        return jnp.real(
            jnp.dot(jnp.conj(state.flatten()), projection.flatten()),
        )

    temp = [qubit_magnetization(idx) for idx in np.arange(state.ndim)]
    return jnp.stack(temp)


def total_magnetization(state: State, n_out: int = 1) -> Array:
    """Calculate the total magnetization of a quantum state.

    Computes the total magnetization by summing individual qubit magnetizations.
    The result can be split into multiple outputs by specifying n_out.

    Args:
        state: Quantum state array of shape (..., 2, 2, ..., 2).
        n_out: Number of output values to reshape the result into.
              Must evenly divide the total number of qubits.

    Returns:
        Array of shape (n_out,) containing summed magnetization values.

    Example:
        >>> from horqrux import random_state
        >>> state = random_state(4)  # 4-qubit random state
        >>> total_magnetization(state)  # Single output
        Array([0.42], dtype=float32)  # Sum of all qubit magnetizations

        >>> # Split into two outputs
        >>> total_magnetization(state, n_out=2)
        Array([0.21, 0.21], dtype=float32)  # Magnetization for qubits [0,1] and [2,3]

        >>> # Differentiable
        >>> jax.grad(lambda s: total_magnetization(s, n_out=1)[0])(state)
        Array([...], dtype=float32)  # Gradient w.r.t state
    """
    magnetization = qubit_magnetization(state)
    return jnp.sum(magnetization.reshape((n_out, -1)), axis=-1)


def total_magnetization_via_inner_product(n_qubits: int) -> Callable:
    """Create a function to compute total magnetization using inner products.

    Alternative implementation using direct inner products between states.
    This approach is exact for statevector simulations but may not reflect
    measurement statistics on real quantum hardware.

    Args:
        n_qubits: Number of qubits in the system.

    Returns:
        Function that takes (state, param_dict) and returns total magnetization.

    Example:
        >>> from horqrux import random_state
        >>> mag_fn = total_magnetization_via_inner_product(2)
        >>> state = random_state(2)
        >>> mag = mag_fn(state, {})  # Empty dict since no parameters needed
        >>> print(mag)  # Returns total magnetization as real number
        0.42

        >>> # Compatible with JAX transformations
        >>> batch_mag = jax.vmap(lambda s: mag_fn(s, {}))
        >>> states = random_state(2, batch_size=10)
        >>> batch_mag(states)  # Compute magnetization for batch of states
        Array([...], shape=(10,), dtype=float32)
    """
    paulis = [Z(i) for i in range(n_qubits)]

    def _total_magnetization(out_state: Array, values: dict[str, Array]) -> Array:
        projected_state = reduce(
            add,
            [apply_gate(out_state, pauli, values) for pauli in paulis],
        )
        return inner(out_state, projected_state).real

    return _total_magnetization


def total_magnetization_ops(n_qubits: int) -> list[Primitive]:
    """Create a list of Pauli Z operators to measure total magnetization.

    Generates Pauli Z operators for each qubit that can be used with the
    expectation() function to measure total magnetization.

    Args:
        n_qubits: Number of qubits in the system.

    Returns:
        List of Pauli Z operators, one per qubit.

    Example:
        >>> ops = total_magnetization_ops(2)
        >>> print(ops)  # Returns [Z(0), Z(1)]
        >>> from horqrux import random_state, expectation
        >>> state = random_state(2)
        >>> # Compute magnetization using expectation values
        >>> mags = [expectation(state, [], op) for op in ops]
        >>> total_mag = sum(mags)  # Total magnetization

        >>> # Can be used with JAX transformations
        >>> jax.vmap(lambda s: sum(expectation(s, [], op) for op in ops))(states)
        Array([...], shape=(batch_size,), dtype=float32)
    """
    # Old horqrux version
    # observables = [Z(i) for i in range(n_qubits)]
    return [Observable([Z(i)]) for i in range(n_qubits)]


if __name__ == "__main__":

    import jax
    from horqrux import QuantumCircuit, X, random_state, zero_state
    from horqrux.primitives.parametric import RX

    # Demonstrate qubit magnetization on a simple initial state
    n_qubits = 2
    state = zero_state(n_qubits)  # Initialize |00⟩ state
    mag_initial = qubit_magnetization(state)
    print(f"Initial qubit magnetization (|00⟩ state): {mag_initial}")
    # Should be -1 for each qubit since they're in |0⟩

    # Show how magnetization changes after applying X gate
    state = apply_gate(state, X(0))  # Transform to |10⟩ state
    mag_after_x = qubit_magnetization(state)
    print(f"Qubit magnetization after X on first qubit (|10⟩ state): {mag_after_x}")
    # First qubit should now have +1 magnetization

    # Compare different magnetization measurement methods
    n_qubits_random = 4
    random_state_vector = random_state(n_qubits_random)

    # Method 1: Direct total magnetization
    mag_direct = total_magnetization(random_state_vector, n_out=1)
    print(f"Total magnetization (direct method): {mag_direct}")

    # Method 2: Inner product approach
    n_qubits_inner = 4
    # inner_state = random_state(n_qubits_inner)
    inner_state = random_state_vector
    mag_fn = total_magnetization_via_inner_product(n_qubits_inner)
    mag_inner = mag_fn(inner_state, {})
    print(f"Total magnetization (inner product method): {mag_inner}")

    # Method 3: Using Pauli Z operators with expectation values
    z_ops = total_magnetization_ops(n_qubits_inner)
    print(f"Pauli Z operators for magnetization measurement: {z_ops}")

    # Example usage of expectation values function
    gates = [RX("x", 0), RX("y", 1)]
    values = {"x": jnp.array([0.0]), "y": jnp.array([0.0])}
    circuit = QuantumCircuit(2, gates)
    state = zero_state(4)
    mag_expectation = expectation(state, circuit, z_ops, values, key=jax.random.PRNGKey(0))
    print(f"Total magnetization (expectation values): {jnp.sum(mag_expectation)}")

    # Noisy case for zero state 4 qubits

    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    low_noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.0),)
    high_noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.01),)

    z_ops = total_magnetization_ops(4)

    gates_no_noise = [RX("x", 0), RX("y", 1)]
    gates_low_noise = [RX("x", 0, noise=low_noise), RX("y", 1, noise=low_noise)]
    gates_high_noise = [RX("x", 0, noise=high_noise), RX("y", 1, noise=high_noise)]

    values = {"x": jnp.array([0.0]), "y": jnp.array([0.0])}

    circuit_no_noise = QuantumCircuit(4, gates_no_noise)
    circuit_low_noise = QuantumCircuit(4, gates_low_noise)
    circuit_high_noise = QuantumCircuit(4, gates_high_noise)

    state = zero_state(4)

    mag_expectation_no_noise = expectation(
        state,
        circuit_no_noise,
        z_ops,
        values,
        key=jax.random.PRNGKey(0),
    )
    print(
        f"Total no noise magnetization (expectation values): {jnp.sum(mag_expectation_no_noise)}",
    )
    mag_expectation_low_noise = expectation(
        state,
        circuit_low_noise,
        z_ops,
        values,
        key=jax.random.PRNGKey(0),
    )
    print(
        f"Total low noise magnetization (expectation values): {jnp.sum(mag_expectation_low_noise)}",
    )
    mag_expectation_high_noise = expectation(
        state,
        circuit_high_noise,
        z_ops,
        values,
        key=jax.random.PRNGKey(0),
    )
    print(
        f"Total high noise magnetization (expectation values): {jnp.sum(mag_expectation_high_noise)}",
    )

    assert jnp.sum(mag_expectation_low_noise) == jnp.sum(mag_expectation)
    assert jnp.sum(mag_expectation_high_noise) != jnp.sum(mag_expectation)

    # Noisy case for random state 4 qubits

    circuit_no_noise = QuantumCircuit(4, gates_no_noise)
    circuit_low_noise = QuantumCircuit(4, gates_low_noise)
    circuit_high_noise = QuantumCircuit(4, gates_high_noise)

    state = random_state(4)

    mag_expectation_no_noise = expectation(
        state,
        circuit_no_noise,
        z_ops,
        values,
        key=jax.random.PRNGKey(0),
    )
    print(f"Total magnetization (completely noiseless): {jnp.sum(mag_expectation_no_noise)}")
    mag_expectation_noisy = expectation(
        state,
        circuit_low_noise,
        z_ops,
        values,
        key=jax.random.PRNGKey(0),
    )
    print(f"Total noisy magnetization (noise to 0): {jnp.sum(mag_expectation_noisy)}")
    mag_expectation_high_noise = expectation(
        state,
        circuit_high_noise,
        z_ops,
        values,
        key=jax.random.PRNGKey(0),
    )
    print(
        f"Total high noise magnetization (expectation values): {jnp.sum(mag_expectation_high_noise)}",
    )

    # Note different behavior for noisy case, use noise 0.0 when using noise models.
    assert jnp.sum(mag_expectation_noisy) != jnp.sum(mag_expectation_no_noise)
    assert jnp.sum(mag_expectation_high_noise) != jnp.sum(mag_expectation_no_noise)
