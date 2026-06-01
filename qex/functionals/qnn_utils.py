"""Lightweight QNN primitives backing `qcnn.py`.

The previous implementation went through `horqrux.QuantumCircuit` +
`expectation`, which builds a per-block circuit object and routes parameters
through a dict keyed by Python strings. Under vmap-over-blocks inside the SCF
loop (which is Python-unrolled over `max_cycle`), the XLA graph blew up and
compilation hung.

This module replaces that path with the in-tree `qex.qnn_backend` subpackage
(originally vendored from an early, slim horqrux release). Gates are bare 2×2
matrices applied via `tensordot`; variational parameters are a single
`(n_layers, 3, n_qubits)` array (no names, no dicts); layers are folded with
`lax.scan` + `jax.checkpoint`. The whole per-block QNN becomes a small,
fully-static JAX function — exactly what jit/vmap want. Keeping the backend
inside `qex` means it ships with the package on PyPI without depending on the
heavier installed `horqrux`.

Public surface used by `qcnn.py`:

- `n_ansatz_params(n_qubits, n_var_layers)` — flat parameter count for a HEA
  (kept flat so the Flax module signature in `qcnn.py` is unchanged).
- `block_qnn_apply(x_block, var_params, *, n_qubits, n_var_layers,
  feature_map, state)` — feature-map + HEA + total-magnetization-sum,
  returns a scalar.
- `make_zero_state(n_qubits)` — `|0…0>` state of shape `(2,)*n_qubits`.
- `FEATURE_MAPS` — same string keys as before ("direct", "chebyshev").

Noise paths from the old `horqrux` aren't ported (the slim clone has none);
`qcnn.py` keeps its `noise` field but it's a no-op here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from chex import Array

from qex.qnn_backend.jax_native.gates import NOT, Rx, Ry
from qex.qnn_backend.jax_native.measurement import qubit_magnetization
from qex.qnn_backend.jax_native.ops import apply_gate


def make_zero_state(n_qubits: int) -> Array:
    """`|0…0>` as a complex128 tensor of shape `(2,)*n_qubits`."""
    shape = (2,) * n_qubits
    state = jnp.zeros(shape, dtype=jnp.complex128)
    return state.at[(0,) * n_qubits].set(1.0)


# --- feature maps -----------------------------------------------------------

def _direct_state(x: Array, state: Array, n_qubits: int) -> Array:
    """Direct angle encoding: Ry(x_i) on qubit i, broadcasting if needed.

    `x` has length `n_features <= n_qubits`. Qubits beyond `len(x)` are left
    untouched (effectively encoded by `Ry(0)`).
    """
    gates = [Ry(x[i % x.shape[0]], (i,)) for i in range(n_qubits)]
    return apply_gate(state, gates)


def _chebyshev_state(x: Array, state: Array, n_qubits: int) -> Array:
    """Chebyshev tower: Ry(2(i+1) * arccos(x_norm)) on qubit i.

    Clipped well inside (-1, 1) so the arccos derivative stays finite — under
    the SCF gradient path, hitting ±1 produced gradients of order 1e14.
    """
    span = jnp.max(x) - jnp.min(x) + 1e-10
    x_normalized = 1.8 * (x - jnp.min(x)) / span - 0.9
    x_normalized = jnp.clip(x_normalized, -0.9, 0.9)
    angles = 2.0 * jnp.arccos(x_normalized)
    gates = [Ry((i + 1) * angles[i % x.shape[0]], (i,)) for i in range(n_qubits)]
    return apply_gate(state, gates)


FEATURE_MAPS = {
    "direct": _direct_state,
    "chebyshev": _chebyshev_state,
}


# --- HEA ansatz -------------------------------------------------------------

def n_ansatz_params(n_qubits: int, n_var_layers: int) -> int:
    """Flat parameter count for the HEA (3 angles × n_qubits × layers)."""
    return 3 * n_qubits * n_var_layers


def _hea_layer(state: Array, theta_layer: Array, n_qubits: int) -> Array:
    """Single HEA layer: Rx-Ry-Rx on each qubit, then ring of CNOTs.

    `theta_layer` has shape `(3, n_qubits)`.
    """
    rx0 = [Rx(theta_layer[0, q], (q,)) for q in range(n_qubits)]
    ry = [Ry(theta_layer[1, q], (q,)) for q in range(n_qubits)]
    rx1 = [Rx(theta_layer[2, q], (q,)) for q in range(n_qubits)]
    state = apply_gate(state, rx0 + ry + rx1)

    if n_qubits > 1:
        cnots = [NOT(((i + 1) % n_qubits,), (i,)) for i in range(n_qubits)]
        state = apply_gate(state, cnots)
    return state


def _apply_hea(state: Array, theta: Array, n_qubits: int) -> Array:
    """Fold `n_var_layers` HEA layers over `state` via `lax.scan` + checkpoint.

    `theta` has shape `(n_var_layers, 3, n_qubits)`.
    """

    @jax.checkpoint
    def step(carry: Array, theta_layer: Array):
        return _hea_layer(carry, theta_layer, n_qubits), None

    out, _ = jax.lax.scan(step, state, theta)
    return out


# --- total magnetization observable -----------------------------------------

def _total_magnetization(state: Array) -> Array:
    """Sum of <Z_i> over all qubits."""
    return jnp.sum(qubit_magnetization(state))


# --- per-block QNN ----------------------------------------------------------

def block_qnn_apply(
    x_block: Array,
    var_params: Array,
    *,
    n_qubits: int,
    n_var_layers: int,
    feature_map: str,
    state: Array,
) -> Array:
    """Feature map + HEA + sum-of-Z, returning a scalar.

    `var_params` is the flat `(3 * n_qubits * n_var_layers,)` vector that
    `qcnn.QCNNLayer` allocates; it's reshaped to `(n_var_layers, 3, n_qubits)`
    here so the public Flax param shape stays unchanged.
    """
    fm = FEATURE_MAPS[feature_map]
    state = fm(x_block, state, n_qubits)
    theta = var_params.reshape(n_var_layers, 3, n_qubits)
    state = _apply_hea(state, theta, n_qubits)
    return _total_magnetization(state)
