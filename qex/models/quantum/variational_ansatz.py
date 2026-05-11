"""Quantum variational ansatz module.

This module provides functions for building quantum variational circuits, including:
- Hardware-efficient ansatz (HEA) with rotations and entangling layers
- Single and multi-layer variational circuits
- Global entangling operations

The functions are designed to work with JAX for automatic differentiation and acceleration.

Example:
    >>> import jax.numpy as jnp
    >>> from horqrux import zero_state
    >>> # Create 2-qubit circuit with 2 layers
    >>> n_qubits = 2
    >>> ansatz = hea(n_qubits, n_layers=2)
    >>> # Initialize parameters and state
    >>> theta = jnp.zeros((2, 3, n_qubits))  # [n_layers, n_rotations, n_qubits]
    >>> state = zero_state(n_qubits)
    >>> # Apply variational layers
    >>> final_state = n_variational(theta, state, tuple(range(n_qubits)))
"""

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from horqrux import zero_state
from horqrux.apply import apply_gates as apply_gate
from horqrux.primitives.parametric import RX, RZ
from horqrux.primitives.primitive import NOT
from horqrux.utils.operator_utils import State, TargetQubits


def global_entangling(state: State) -> State:
    """Globally entangles qubits using two consecutive shifted CNOT layers.
    See also entangling_layer in entangling_layers.py.

    Applies CNOT gates between adjacent qubits in two layers:
    - First layer: CNOTs between qubits (0,1), (2,3), etc.
    - Second layer: CNOTs between qubits (1,2), (3,4), etc.

    Args:
        state: Quantum state array of shape (..., 2, 2, ..., 2) with n_qubits dimensions.

    Returns:
        Entangled quantum state.

    Example:
        >>> from horqrux import zero_state
        >>> state = zero_state(3)  # |000⟩ state
        >>> entangled = global_entangling(state)
        >>> # First layer applies CNOT(1,0)
        >>> # Second layer applies CNOT(2,1)
    """
    n_qubits = state.ndim

    # Getting indices of first CNOT layer
    even_idx = np.arange(0, n_qubits - 1, step=2, dtype=int)
    even_target = tuple(even_idx + 1)
    even_control = tuple(even_idx)

    # Getting indices of second CNOT layer
    uneven_idx = np.arange(1, n_qubits - 1, step=2, dtype=int)
    uneven_target = tuple(uneven_idx + 1)
    uneven_control = tuple(uneven_idx)

    # Applying CNOT layers
    state = apply_gate(
        state,
        [NOT(t_idx, c_idx) for t_idx, c_idx in zip(even_target, even_control)],
    )
    if len(uneven_idx) != 0:
        state = apply_gate(
            state,
            [NOT(t_idx, c_idx) for t_idx, c_idx in zip(uneven_target, uneven_control)],
        )
    return state


def variational(theta: Array, state: Array, target_idx: TargetQubits) -> Array:
    """Applies a single hardware-efficient variational layer.

    Each layer consists of:
    1. RZ rotations on each qubit
    2. RX rotations on each qubit
    3. RZ rotations on each qubit
    4. Global entangling operation

    Args:
        theta: Rotation angles array of shape (3, n_qubits) for [RZ, RX, RZ] layers.
        state: Input quantum state.
        target_idx: Indices of target qubits for rotations.

    Returns:
        Quantum state after applying the variational layer.

    Example:
        >>> import jax.numpy as jnp
        >>> from horqrux import zero_state
        >>> state = zero_state(2)
        >>> theta = jnp.zeros((3, 2))  # 3 rotation layers, 2 qubits
        >>> target = tuple(range(2))
        >>> out_state = variational(theta, state, target)
    """
    # First Rz layer
    state = apply_gate(
        state,
        [RZ(angle, idx) for angle, idx in zip(theta[0], target_idx)],
    )

    # Second Rx layer
    state = apply_gate(
        state,
        [RX(angle, idx) for angle, idx in zip(theta[1], target_idx)],
    )

    # Third Rz layer
    state = apply_gate(
        state,
        [RZ(angle, idx) for angle, idx in zip(theta[2], target_idx)],
    )

    return global_entangling(state)


def n_variational(theta: Array, state: Array, target_idx: TargetQubits) -> Array:
    """Applies multiple hardware-efficient variational layers.

    Repeatedly applies variational layers, with memory-efficient checkpointing.
    Number of layers is determined by first dimension of theta.

    Args:
        theta: Rotation angles array of shape (n_layers, 3, n_qubits).
        state: Input quantum state.
        target_idx: Indices of target qubits for rotations.

    Returns:
        Final quantum state after all variational layers.

    Example:
        >>> import jax.numpy as jnp
        >>> from horqrux import zero_state
        >>> n_qubits = 2
        >>> state = zero_state(n_qubits)
        >>> # 2 layers, 3 rotations per layer, 2 qubits
        >>> theta = jnp.zeros((2, 3, n_qubits))
        >>> target = tuple(range(n_qubits))
        >>> final = n_variational(theta, state, target)
    """

    # We checkpoint for memory efficiency
    @jax.checkpoint
    def update_fn(carry: Array, theta_layer: Array) -> tuple[Array, None]:
        return variational(theta_layer, carry, target_idx), None

    return jax.lax.scan(update_fn, state, theta)[0]


if __name__ == "__main__":

    # Create a 2-qubit, 2-layer HEA circuit
    n_qubits = 2
    n_layers = 2

    # Example of n_variational application
    # Initialize parameters for 2 layers, 3 rotations per qubit, 2 qubits
    theta = jnp.zeros((n_layers, 3, n_qubits))  # Shape: (n_layers, n_rotations, n_qubits)

    # Create initial state |00⟩
    state = zero_state(n_qubits)
    print("\nInitial state:", state)

    # Define target qubits (0,1)
    target = tuple(range(n_qubits))
    # Apply variational layers
    final_state = n_variational(theta, state, target)
    print("\nFinal state after variational layers:")
    print(final_state)

    # Example with non-zero parameters
    theta_random = jax.random.uniform(
        jax.random.PRNGKey(0),
        shape=(n_layers, 3, n_qubits),
        minval=0,
        maxval=2 * jnp.pi,
    )
    final_random = n_variational(theta_random, state, target)
    print("\nFinal state with random parameters:")
    print(final_random)

    # Jit the circuit
    from functools import partial

    jitted_circ = jax.jit(partial(n_variational, state=state, target_idx=target))

    # Run circuit and get output
    final_random_jit = jitted_circ(theta_random)
    print("\nFinal state with random parameters:")
    print(final_random_jit)
    assert jnp.allclose(final_random, final_random_jit)
