"""
Entangling layers for quantum neural networks.
"""

from collections.abc import Sequence
from functools import partial
from itertools import combinations

import jax
import numpy as np
from horqrux import zero_state
from horqrux.apply import apply_gates as apply_gate
from horqrux.primitives.primitive import NOT, Primitive
from horqrux.utils import State


def get_entangler_map(
    num_block_qubits: int,
    num_circuit_qubits: int,
    entanglement: str,
    offset: int = 0,
) -> list[Sequence[int]]:
    """Get an entangler map for an arbitrary number of qubits.

    Args:
        num_block_qubits: The number of qubits of the entangling block.
        num_circuit_qubits: The number of qubits of the circuit.
        entanglement: The entanglement strategy.
        offset: The block offset, can be used if the entanglements differ per block.
            See mode ``sca`` for instance.

    Returns:
        The entangler map using mode ``entanglement`` to scatter a block of ``num_block_qubits``
        qubits on ``num_circuit_qubits`` qubits.

    Raises:
        ValueError: If the entanglement mode ist not supported.
    """
    n, m = num_circuit_qubits, num_block_qubits
    if m > n:
        raise ValueError(
            "The number of block qubits must be smaller or equal to the number"
            " of qubits in the circuit.",
        )

    if entanglement == "pairwise" and num_block_qubits > 2:
        raise ValueError(
            "Pairwise entanglement is not defined for blocks with more than 2" " qubits.",
        )

    if entanglement == "full":
        return list(combinations(list(range(n)), m))

    elif entanglement == "alternate_linear":
        # Getting indices of first CNOT layer
        even_idx = np.arange(0, num_circuit_qubits - 1, step=2, dtype=int)
        even_target = tuple(even_idx + 1)
        even_control = tuple(even_idx)

        # Getting indices of first CNOT layer
        uneven_idx = np.arange(1, num_circuit_qubits - 1, step=2, dtype=int)
        uneven_target = tuple(uneven_idx + 1)
        uneven_control = tuple(uneven_idx)

        global_ent = []
        global_ent = [(c_idx, t_idx) for (c_idx, t_idx) in zip(even_control, even_target)]
        if len(uneven_idx) != 0:
            global_ent += [(c_idx, t_idx) for (c_idx, t_idx) in zip(uneven_control, uneven_target)]
        return global_ent

    elif entanglement == "alternate_circular":
        # Getting indices of first CNOT layer
        even_idx = np.arange(0, num_circuit_qubits - 1, step=2, dtype=int)
        even_target = tuple(even_idx + 1)
        even_control = tuple(even_idx)

        # Getting indices of first CNOT layer
        uneven_idx = np.arange(1, num_circuit_qubits - 1, step=2, dtype=int)
        uneven_target = tuple(uneven_idx + 1)
        uneven_control = tuple(uneven_idx)

        global_ent = []
        global_ent = [(c_idx, t_idx) for (c_idx, t_idx) in zip(even_control, even_target)]
        if len(uneven_idx) != 0:
            global_ent += [(c_idx, t_idx) for (c_idx, t_idx) in zip(uneven_control, uneven_target)]

        # Adding the last CNOT between the
        if n > m:
            global_ent = [tuple(range(n - m + 1, n)) + (0,)] + global_ent
        else:
            global_ent = global_ent

        return global_ent

    elif entanglement == "reverse_linear":
        # reverse linear connectivity. In the case of m=2 and the entanglement_block='cx'
        # then it's equivalent to 'full' entanglement
        reverse = [tuple(range(n - i - m, n - i)) for i in range(n - m + 1)]
        return reverse
    elif entanglement in ["linear", "circular", "sca", "pairwise"]:
        linear = [tuple(range(i, i + m)) for i in range(n - m + 1)]
        # if the number of block qubits is 1, we don't have to add the 'circular' part
        if entanglement == "linear" or m == 1:
            return linear

        if entanglement == "pairwise":
            return linear[::2] + linear[1::2]

        # circular equals linear plus top-bottom entanglement (if there's space for it)
        if n > m:
            circular = [tuple(range(n - m + 1, n)) + (0,)] + linear
        else:
            circular = linear
        if entanglement == "circular":
            return circular

        # sca is circular plus shift and reverse
        shifted = circular[-offset:] + circular[:-offset]
        if offset % 2 == 1:  # if odd, reverse the qubit indices
            sca = [ind[::-1] for ind in shifted]
        else:
            sca = shifted

        return sca

    else:
        raise ValueError(f"Unsupported entanglement type: {entanglement}")


def entangling_ops(
    n_qubits: int,
    entangling_block_type: str = "full",
    noise: tuple = None,
) -> list[Primitive]:
    """Get the entangling operations for a given entangling block type."""

    if noise is not None:
        not_gate = partial(NOT, noise=noise)
    else:
        not_gate = NOT

    if entangling_block_type not in [
        "full",
        "linear",
        "sca",
        "circular",
        "pairwise",
        "reverse_linear",
        "alternate_linear",
        "alternate_circular",
    ]:
        raise ValueError(
            f"{entangling_block_type} is not a valid single qubit rotation.",
        )

    idx_ctr_tgt = get_entangler_map(
        num_block_qubits=2,  # 2-qubit gates are only allowed.
        num_circuit_qubits=n_qubits,
        entanglement=entangling_block_type,
    )

    # Applying CNOT layers.
    return [not_gate(t_idx, c_idx) for c_idx, t_idx in idx_ctr_tgt]


def entangling_layer(
    state: State,
    entangling_block_type: str = "full",
    noise: tuple = None,
) -> State:
    """Globally entangles qubits by two consequtive shifted cnot layers, i.e.
    first layer has cnot (0, 1) and the second (1, 2), etc...

    Args:
        state (Array): State to be entangled.
        entangling_block_type (str): Type of entangling block to use.
        noise (tuple): Noise to be applied to the entangling layer.
    Returns:
        Array: Entangled state.
    """
    n_qubits = state.ndim
    ops = entangling_ops(n_qubits, entangling_block_type, noise)

    # Applying CNOT gates
    state = apply_gate(
        state,
        ops,
    )
    return state


def plot_entanglement_pattern(num_qubits: int, entanglement: str, offset: int = 0):
    """Plot the entanglement pattern using ASCII art.

    Args:
        num_qubits: Number of qubits in the circuit
        entanglement: Entanglement strategy
        offset: Offset for SCA entanglement
    """
    try:
        # Get the entangler map
        entangler_map = get_entangler_map(
            num_block_qubits=2,  # Explicitly use 2-qubit blocks for visualization
            num_circuit_qubits=num_qubits,
            entanglement=entanglement,
            offset=offset,
        )

        if not entangler_map:
            print(
                f"\nEntanglement pattern '{entanglement}' produced no gates for {num_qubits} qubits.",
            )
            return

        # Initialize the circuit representation
        circuit = []
        for i in range(num_qubits):
            circuit.append([f"q{i}: ", "─" * (4 * len(entangler_map) + 1)])

        # Add CNOT gates
        for gate_idx, connection in enumerate(entangler_map):
            # Handle both 2-qubit and multi-qubit blocks
            if len(connection) == 2:
                control, target = connection
            else:
                # For patterns that might return different formats,
                # assume first is control, second is target
                control, target = connection[0], connection[1]

            # Position in the circuit
            pos = 4 * gate_idx + 2

            # Add control point
            circuit_line = list(circuit[control][1])
            circuit_line[pos] = "●"
            circuit[control][1] = "".join(circuit_line)

            # Add target point (⊕)
            circuit_line = list(circuit[target][1])
            circuit_line[pos] = "⊕"
            circuit[target][1] = "".join(circuit_line)

            # Add vertical line between control and target
            if control != target:  # Only if control and target are different qubits
                min_idx, max_idx = min(control, target), max(control, target)
                for i in range(min_idx + 1, max_idx):
                    circuit_line = list(circuit[i][1])
                    circuit_line[pos] = "│"
                    circuit[i][1] = "".join(circuit_line)

        # Print the circuit
        print(f"\nEntanglement pattern: {entanglement} (offset={offset})")
        for line in circuit:
            print("".join(line))

    except ValueError as e:
        print(f"\nError plotting entanglement pattern '{entanglement}': {e}")


if __name__ == "__main__":

    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    # Test different entanglement patterns
    state = zero_state(2)
    entanglement_types = [
        "full",
        "linear",
        "sca",
        "circular",
        "pairwise",
        "reverse_linear",
        "alternate_linear",
        "alternate_circular",
    ]

    for entangling_type in entanglement_types:
        print("-" * 79)
        print(f"Testing {entangling_type} entanglement:")

        # Test basic entangling layer
        print("\nBasic layer:")
        print(entangling_layer(state, entangling_type))

        # Test JIT-compiled layer
        print("\nJIT-compiled layer:")
        jitted_layer = jax.jit(
            partial(entangling_layer, entangling_block_type=entangling_type),
        )
        print(jitted_layer(state))

        # Test noisy layer
        print("\nNoisy layer:")
        noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.1),)
        jitted_noisy = jax.jit(
            partial(
                entangling_layer,
                entangling_block_type=entangling_type,
                noise=noise,
            ),
        )
        print(jitted_noisy(state))

        # Visualize pattern
        print("\nCircuit visualization:")
        plot_entanglement_pattern(4, entangling_type)
