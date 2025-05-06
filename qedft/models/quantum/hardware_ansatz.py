"""Noisy HEA circuit.

This module provides a noisy hardware-efficient ansatz circuit with same noise in all gates.

Example:
    >>> from qedft.models.quantum.noisy_hea import hea_noisy
    >>> from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
    >>> noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.01),)
    >>> noisy_ansatz = hea_noisy(n_qubits, n_layers, noise=noise)
"""

from collections.abc import Callable
from functools import partial
from uuid import uuid4

import jax.numpy as jnp
from horqrux import zero_state
from horqrux.apply import apply_gates as apply_gate
from horqrux.primitives.parametric import RX, RY
from horqrux.primitives.primitive import NOT, Primitive

from qedft.models.quantum.entangling_layers import entangling_ops


def hea_legacy(
    n_qubits: int,
    n_layers: int,
    rot_fns: list[Callable] = [RX, RY, RX],
    variational_param_prefix: str = "v_",
    noise: tuple = None,
) -> list[Primitive]:
    """Creates a hardware-efficient ansatz circuit with same noise in all gates.

    Generates a sequence of parameterized rotation gates followed by entangling
    operations. Each layer applies rotations to each qubit followed by CNOTs.

    ! This is the legacy version of the HEA circuit.
    ! It is kept here for reference.

    Args:
        n_qubits: Number of qubits in circuit.
        n_layers: Number of variational layers.
        rot_fns: List of rotation gate constructors, defaults to [RX, RY, RX].
        variational_param_prefix: Prefix for parameter names, defaults to "v_".
        noise: Optional tuple of DigitalNoiseInstance objects to apply to all gates.

    Returns:
        List of quantum gates defining the ansatz circuit.

    Example:
        >>> # Create 2-qubit, 2-layer HEA with default rotations
        >>> ansatz = hea_legacy(n_qubits=2, n_layers=2)
        >>> # Custom rotations
        >>> custom = hea_legacy(2, 2, rot_fns=[RZ, RX, RZ])
        >>> # Custom parameter prefix
        >>> named = hea_legacy(2, 2, variational_param_prefix="theta_")
        >>> # With noise
        >>> from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
        >>> noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.01),)
        >>> noisy = hea_legacy(2, 2, noise=noise)
    """
    gates = []
    param_names = []

    # Create noise-aware gate constructors if noise is provided
    # For compatibility
    if noise is not None:
        rot_fns = [partial(fn, noise=noise) for fn in rot_fns]
        not_gate = partial(NOT, noise=noise)
    else:
        not_gate = NOT

    for _ in range(n_layers):
        for i in range(n_qubits):
            ops = [
                fn(variational_param_prefix + str(uuid4()), qubit)
                for fn, qubit in zip(rot_fns, [i for _ in range(len(rot_fns))])
            ]
            param_names += [op.param for op in ops]
            # Gates are places right after the rotation gates.
            # This is
            ops += [not_gate((i + 1) % n_qubits, i % n_qubits) for i in range(n_qubits)]
            gates += ops

    return gates


def hea(
    n_qubits: int,
    n_layers: int,
    rot_fns: list[Callable] = [RX, RY, RX],
    variational_param_prefix: str = "v_",
    noise: tuple = None,
) -> list[Primitive]:
    """Creates a hardware-efficient ansatz circuit with same noise in all gates.

    Generates a sequence of parameterized rotation gates followed by entangling
    operations. Each layer applies rotations to each qubit followed by CNOTs.

    Args:
        n_qubits: Number of qubits in circuit.
        n_layers: Number of variational layers.
        rot_fns: List of rotation gate constructors, defaults to [RX, RY, RX].
        variational_param_prefix: Prefix for parameter names, defaults to "v_".
        noise: Optional tuple of DigitalNoiseInstance objects to apply to all gates.

    Returns:
        List of quantum gates defining the ansatz circuit.

    Example:
        >>> # Create 2-qubit, 2-layer HEA with default rotations
        >>> ansatz = hea(n_qubits=2, n_layers=2)
        >>> # Custom rotations
        >>> custom = hea(2, 2, rot_fns=[RZ, RX, RZ])
        >>> # Custom parameter prefix
        >>> named = hea(2, 2, variational_param_prefix="theta_")
        >>> # With noise
        >>> from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
        >>> noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.01),)
        >>> noisy = hea(2, 2, noise=noise)
    """
    gates = []
    param_names = []

    # Create noise-aware gate constructors if noise is provided
    # For compatibility
    if noise is not None:
        rot_fns = [partial(fn, noise=noise) for fn in rot_fns]
        not_gate = partial(NOT, noise=noise)
    else:
        not_gate = NOT

    for _ in range(n_layers):
        ops = []
        for i in range(n_qubits):
            ops += [
                fn(variational_param_prefix + str(uuid4()), qubit)
                for fn, qubit in zip(rot_fns, [i for _ in range(len(rot_fns))])
            ]
            param_names += [op.param for op in ops]

        # We place rotation gates first, then entangling gates.
        for i in range(n_qubits):
            ops += [not_gate((i + 1) % n_qubits, i % n_qubits) for i in range(n_qubits)]
        gates += ops

    return gates


def advanced_hea(
    n_qubits: int,
    n_layers: int,
    rot_fns: list[Callable] = [RX, RY, RX],
    variational_param_prefix: str = "v_",
    noise: tuple = None,
    entangling_block_type: str = "circular",
) -> list[Primitive]:
    """Creates a hardware-efficient ansatz circuit with same noise in all gates.
    More flexible version of the HEA circuit.

    Generates a sequence of parameterized rotation gates followed by entangling
    operations. Each layer applies rotations to each qubit followed by CNOTs.

    Args:
        n_qubits: Number of qubits in circuit.
        n_layers: Number of variational layers.
        rot_fns: List of rotation gate constructors, defaults to [RX, RY, RX].
        variational_param_prefix: Prefix for parameter names, defaults to "v_".
        noise: Optional tuple of DigitalNoiseInstance objects to apply to all gates.
        entangling_block_type: Type of entangling block to use, defaults to "circular".
        other options are "full", "linear", "sca", "circular", "pairwise",
        "reverse_linear", "alternate_linear", "alternate_circular".
    Returns:
        List of quantum gates defining the ansatz circuit.

    Example:
        >>> # Create 2-qubit, 2-layer HEA with default rotations
        >>> ansatz = advanced_hea(n_qubits=2, n_layers=2)
        >>> # Custom rotations
        >>> custom = advanced_hea(2, 2, rot_fns=[RZ, RX, RZ])
        >>> # Custom parameter prefix
        >>> named = advanced_hea(2, 2, variational_param_prefix="theta_")
        >>> # With noise
        >>> from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType
        >>> noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.01),)
        >>> noisy = advanced_hea(2, 2, noise=noise)
    """
    gates = []
    param_names = []

    # Create noise-aware gate constructors if noise is provided
    if noise is not None:
        rot_fns = [partial(fn, noise=noise) for fn in rot_fns]
        ent_ops = entangling_ops(n_qubits, entangling_block_type, noise)
    else:
        not_gate = NOT
        ent_ops = entangling_ops(n_qubits, entangling_block_type)

    for _ in range(n_layers):
        ops = []
        # Apply rotation gates
        for i in range(n_qubits):
            ops += [
                fn(variational_param_prefix + str(uuid4()), qubit)
                for fn, qubit in zip(rot_fns, [i for _ in range(len(rot_fns))])
            ]
            param_names += [op.param for op in ops]

        # We place rotation gates first, then entangling gates
        ops += ent_ops
        gates += ops

    return gates


if __name__ == "__main__":

    # Create a 2-qubit, 2-layer HEA circuit
    n_qubits = 2
    n_layers = 3
    ansatz = hea(n_qubits, n_layers)
    print("HEA circuit gates:")
    print(ansatz)  # Shows sequence of parameterized rotations and CNOTs
    state = zero_state(n_qubits)
    params = {op.param: 0.01 for op in ansatz if hasattr(op, "param")}
    new_state = apply_gate(state, ansatz, params)
    print("\nPure state:")
    print(new_state)
    # Should be n_qubits x n_qubits, not a density matrix
    assert new_state.shape == (2, 2)

    # Create initial state |00⟩
    state = zero_state(n_qubits)
    print("\nInitial state:", state)

    # Test noisy circuit
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.001),)
    noisy_ansatz = hea(n_qubits, n_layers, noise=noise)
    print("\nNoisy circuit gates:")
    print(noisy_ansatz)

    # Make a dictionary of parameters
    params = {op.param: 0.01 for op in noisy_ansatz if hasattr(op, "param")}
    new_state = apply_gate(state, noisy_ansatz, params)
    print("\nNoisy state:")
    print(new_state)

    # Test noisy circuit more noisy
    noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.1),)
    noisy_ansatz = hea(n_qubits, n_layers, noise=noise)
    params = {op.param: 0.01 for op in noisy_ansatz if hasattr(op, "param")}
    new_state2 = apply_gate(state, noisy_ansatz, params)
    assert not jnp.allclose(new_state.array, new_state2.array)

    # Count the number of rotation, parametrized gates in hea_legacy
    legacy_ansatz = hea_legacy(n_qubits, n_layers)
    rotation_gates_legacy = [op for op in legacy_ansatz if hasattr(op, "param")]
    print(f"Number of rotation gates in legacy: {len(rotation_gates_legacy)}")

    # Count the number of rotation, parametrized gates in hea
    new_ansatz = hea(n_qubits, n_layers)
    rotation_gates = [op for op in new_ansatz if hasattr(op, "param")]
    print(f"Number of rotation gates in new: {len(rotation_gates)}")

    # Compare the two
    assert len(rotation_gates) == len(rotation_gates_legacy)

    # Compare the number of gates without params (should be the same)
    no_params_legacy = [op for op in legacy_ansatz if not hasattr(op, "param")]
    print(f"Number of gates without params in legacy: {len(no_params_legacy)}")
    no_params = [op for op in new_ansatz if not hasattr(op, "param")]
    # In legacy, the CNOTs are places right after the rotation gates.
    # In new, the CNOTs are places right after the BLOCK of rotation gates.
    print(f"Number of gates without params in new: {len(no_params)}")
    assert len(no_params) == len(no_params_legacy)

    # Test advanced_hea
    advanced_ansatz = advanced_hea(n_qubits, n_layers, entangling_block_type="circular")
    print("\nAdvanced HEA circuit gates:")
    print(advanced_ansatz)
    params = {op.param: 0.01 for op in advanced_ansatz if hasattr(op, "param")}
    advanced_state = apply_gate(state, advanced_ansatz, params)
    print("\nAdvanced state:")
    print(advanced_state.shape)
