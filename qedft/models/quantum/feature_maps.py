"""Quantum feature maps for encoding classical data into quantum states.

This module provides functions for encoding classical data into quantum states using
different feature map strategies:

- Chebyshev: Encodes using Chebyshev polynomials via angle encoding
- Product: Applies same angle encoding to all qubits
- Direct: Maps input values directly to rotation angles

The feature maps are implemented both as gate lists and state transformations.
All functions are JAX-compatible and support automatic differentiation.

Example:
    >>> import jax.numpy as jnp
    >>> from horqrux import zero_state
    >>> # Create 2-qubit input state
    >>> state = zero_state(2)
    >>> x = jnp.array([0.5, 0.3])
    >>> target = ((0,), (1,))
    >>> # Apply Chebyshev feature map
    >>> encoded = chebyshev(x, state, target)
    >>> # Get just the gates
    >>> gates = chebyshev_gates(x, target)
"""

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from chex import Array
from horqrux.apply import apply_gates as apply_gate
from horqrux.noise import NoiseProtocol
from horqrux.primitives.parametric import RY as Ry
from horqrux.utils import State, TargetQubits
from jax.experimental import checkify


def chebyshev_gates(
    x: Array,
    target_idx: TargetQubits,
    noise: NoiseProtocol | None = None,
) -> list:
    """Returns Chebyshev Tower feature map gates.

    Implements the Chebyshev Tower feature map by encoding classical data into
    rotation angles using the formula: f(x) = 2 * i * arccos(x), where i is the
    qubit index.

    Args:
        x: Input data array of shape (n_dims,)
        target_idx: Target qubit indices for applying gates

    Returns:
        List of Ry gates implementing the feature map

    Example:
        >>> x = jnp.array([0.5, 0.3])
        >>> target = ((0,), (1,))
        >>> gates = chebyshev_gates(x, target)
        >>> print(gates)
        [Ry(1.0472, (0,)), Ry(1.8849, (1,))]

        >>> # Works with gradients
        >>> import jax
        >>> jax.grad(lambda x: sum(g.param for g in chebyshev_gates(x, target)))(x)
        Array([-1.1547, -1.1547], dtype=float32)
    """
    if noise is not None:
        ry = partial(Ry, noise=noise)
    else:
        ry = Ry

    # Normalize between -1 and 1 to avoid NaNs
    # Use the factors to avoid to be on the edge of the domain
    x_normalized = 1.95 * (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x) + 1e-10) - 0.95

    encoding = 2 * (jnp.asarray(target_idx) + 1) * jnp.arccos(x_normalized)
    n_qubits_per_dim = encoding.shape[0] // encoding.shape[1]
    if n_qubits_per_dim < 1:
        raise ValueError(
            "Number of qubits must be >= number of input dimensions. "
            f"Got {encoding.shape[0]} qubits for {encoding.shape[1]} dims.",
        )
    encoding = encoding[:n_qubits_per_dim].flatten("F")
    return [ry(angle, idx) for angle, idx in zip(encoding, target_idx)]


def chebyshev(
    x: Array,
    state: State,
    target_idx: TargetQubits,
    noise: NoiseProtocol | None = None,
) -> State:
    """Applies Chebyshev feature map to quantum state.

    Transforms the input state by applying the Chebyshev feature map gates.

    Args:
        x: Input data array of shape (n_dims,)
        state: Input quantum state
        target_idx: Target qubit indices

    Returns:
        Transformed quantum state

    Example:
        >>> from horqrux import zero_state
        >>> state = zero_state(2)
        >>> x = jnp.array([0.5, 0.3])
        >>> target = ((0,), (1,))
        >>> encoded = chebyshev(x, state, target)
        >>> print(encoded)
        Array([[0.8776+0.j, 0.4794+0.j],
              [-0.4794+0.j, 0.8776+0.j]], dtype=complex64)
    """
    return apply_gate(state, chebyshev_gates(x, target_idx, noise))


def product_gates(
    x: Array,
    target_idx: TargetQubits,
    map_fn: Callable[[float], float] = jnp.arcsin,
    noise: NoiseProtocol | None = None,
) -> list:
    """Returns Product feature map gates.

    Creates gates that apply the same angle encoding to all qubits.

    Args:
        x: Input data array of shape (n_dims,)
        target_idx: Target qubit indices
        map_fn: Function to map inputs to angles

    Returns:
        List of Ry gates

    Example:
        >>> x = jnp.array([0.5])
        >>> target = ((0,), (1,))
        >>> gates = product_gates(x, target, jnp.arcsin)
        >>> print(gates)
        [Ry(0.5236, (0,)), Ry(0.5236, (1,))]
    """
    if noise is not None:
        ry = partial(Ry, noise=noise)
    else:
        ry = Ry

    encoding = jnp.ones(len(target_idx)) * map_fn(x)
    return [ry(angle, idx) for angle, idx in zip(encoding, target_idx)]


def product(
    x: Array,
    state: State,
    target_idx: TargetQubits,
    map_fn: Callable[[float], float] = jnp.arcsin,
    noise: NoiseProtocol | None = None,
) -> State:
    """Applies Product feature map to quantum state.

    Args:
        x: Input data array of shape (n_dims,)
        state: Input quantum state
        target_idx: Target qubit indices
        map_fn: Function to map inputs to angles, default arcsin

    Returns:
        Transformed quantum state

    Example:
        >>> from horqrux import zero_state
        >>> state = zero_state(2)
        >>> x = jnp.array([0.5])
        >>> target = ((0,), (1,))
        >>> encoded = product(x, state, target)
        >>> print(encoded)
        Array([[0.9397+0.j, 0.3420+0.j],
              [-0.3420+0.j, 0.9397+0.j]], dtype=complex64)
    """
    return apply_gate(state, product_gates(x, target_idx, map_fn, noise))


def direct_gates(
    x: Array,
    target_idx: TargetQubits,
    map_fn: Callable[[float], float] = None,
    noise: NoiseProtocol | None = None,
) -> list:
    """Returns Direct feature map gates.

    Creates gates that directly map input values to rotation angles.

    Args:
        x: Input data array of shape (n_dims,)
        target_idx: Target qubit indices
        map_fn: Optional function to transform inputs

    Returns:
        List of Ry gates

    Example:
        >>> x = jnp.array([0.5, 0.3])
        >>> target = ((0,), (1,))
        >>> gates = direct_gates(x, target)
        >>> print(gates)
        [Ry(0.5000, (0,)), Ry(0.3000, (1,))]
    """
    if noise is not None:
        ry = partial(Ry, noise=noise)
    else:
        ry = Ry

    if map_fn is None:
        encoding = jnp.ones_like(jnp.asarray(target_idx)) * x
    else:
        encoding = jnp.ones_like(jnp.asarray(target_idx)) * map_fn(x)

    n_qubits_per_dim = encoding.shape[0] // encoding.shape[1]
    encoding = encoding[:n_qubits_per_dim].flatten("F")
    return [ry(angle, idx) for angle, idx in zip(encoding, target_idx)]


def direct(
    x: Array,
    state: State,
    target_idx: TargetQubits,
    map_fn=None,
    noise: NoiseProtocol | None = None,
) -> State:
    """Applies Direct feature map to quantum state.

    Args:
        x: Input data array of shape (n_dims,)
        state: Input quantum state
        target_idx: Target qubit indices
        map_fn: Optional function to transform inputs

    Returns:
        Transformed quantum state

    Example:
        >>> from horqrux import zero_state
        >>> state = zero_state(2)
        >>> x = jnp.array([0.5, 0.3])
        >>> target = ((0,), (1,))
        >>> encoded = direct(x, state, target)
        >>> print(encoded)
        Array([[0.9689+0.j, 0.2474+0.j],
              [-0.2474+0.j, 0.9689+0.j]], dtype=complex64)
    """
    return apply_gate(state, direct_gates(x, target_idx, map_fn, noise))


if __name__ == "__main__":

    import jax
    from horqrux import zero_state

    # Create test data
    x = jnp.array([0.5, 0.3])
    state = zero_state(2)
    target = ((0,), (1,))

    # Test feature maps
    print("Direct encoding:")
    result_direct = direct(x, state, target)
    print(result_direct)

    print("\nChebyshev encoding:")
    result_chebyshev = chebyshev(x, state, target)
    print(result_chebyshev)

    print("\nProduct encoding:")
    result_product = product(x, state, target)
    print(result_product)

    # Test gradients
    print("\nGradient of Chebyshev encoding:")
    grad_fn = jax.grad(lambda x: jnp.sum(chebyshev(x, state, target)).real)
    print(grad_fn(x))

    # JIT the feature maps
    from functools import partial

    checked_chebyshev = checkify.checkify(partial(chebyshev, state=state, target_idx=target))
    err, result_chebyshev_jit = jax.jit(checked_chebyshev)(x)
    err.throw()
    print("\nJIT Chebyshev encoding:")
    print(result_chebyshev_jit)

    print("\nJIT Product encoding:")
    checked_product = checkify.checkify(partial(product, state=state, target_idx=target))
    err, result_product_jit = jax.jit(checked_product)(x)
    err.throw()
    print(result_product_jit)

    print("\nJIT Direct encoding:")
    checked_direct = checkify.checkify(partial(direct, state=state, target_idx=target))
    err, result_direct_jit = jax.jit(checked_direct)(x)
    err.throw()
    print(result_direct_jit)

    assert jnp.allclose(result_chebyshev, result_chebyshev_jit)
    assert jnp.allclose(result_product, result_product_jit)
    assert jnp.allclose(result_direct, result_direct_jit)

    # Test noisy feature maps
    from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

    noise = (DigitalNoiseInstance(DigitalNoiseType.BITFLIP, 0.5),)
    chebyshev_gates_noisy = partial(chebyshev_gates, noise=noise)
    product_gates_noisy = partial(product_gates, noise=noise)
    direct_gates_noisy = partial(direct_gates, noise=noise)

    chebyshev_noisy = partial(chebyshev, noise=noise)
    product_noisy = partial(product, noise=noise)
    direct_noisy = partial(direct, noise=noise)

    print("\nNoisy Chebyshev encoding:")
    print(chebyshev_noisy(x, state, target))
