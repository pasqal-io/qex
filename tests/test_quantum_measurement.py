"""Tests for quantum measurement functionality.

This module contains tests for quantum measurement operations including:
- Basic magnetization measurements
- Expectation value calculations with exact and shot-based methods
- JIT-compiled quantum operations

The tests verify correctness of both forward computations and gradients.
"""

import jax
import jax.numpy as jnp
import pytest
from horqrux import expectation, random_state, run
from horqrux.circuit import QuantumCircuit
from horqrux.composite import Observable
from horqrux.primitives import H, Z
from horqrux.primitives.parametric import RX
from horqrux.primitives.primitive import NOT
from horqrux.utils import density_mat
from jax.experimental import checkify

from qedft.models.quantum.measurement import qubit_magnetization, total_magnetization

# Constants used across tests
N_QUBITS = 2
SHOTS_ATOL = 0.01  # Tolerance for shot-based measurements
N_SHOTS = 100_000  # Number of measurement shots


@pytest.fixture
def quantum_setup():
    """Creates a basic quantum setup for testing.

    Returns:
        tuple: (initial state, operations, observables, parameter value)
    """
    state = random_state(N_QUBITS)
    list_ops = [RX("theta", 0)]
    ops = QuantumCircuit(N_QUBITS, list_ops)
    observables = [Observable([Z(0)]), Observable([Z(1)])]
    x = jnp.pi * 0.5
    return state, ops, observables, x


def test_qubit_magnetization():
    """Tests per-qubit magnetization calculation."""
    state = jnp.array([[0, 1], [0, 1]])
    expected = jnp.array([0, -2])
    result = qubit_magnetization(state)
    assert jnp.allclose(result, expected)


def test_total_magnetization():
    """Tests total system magnetization calculation."""
    state = jnp.array([[0, 1], [0, 1]])
    expected = jnp.array([-2])
    result = total_magnetization(state)
    assert jnp.allclose(result, expected)


class TestExpectationValues:
    """Tests for quantum expectation value calculations."""

    def test_density_matrix_consistency(self, quantum_setup):
        """Verifies consistency between state vector and density matrix representations."""
        state, ops, _, x = quantum_setup
        expected_dm = density_mat(run(ops, state, {"theta": x}))
        output_dm = run(ops, density_mat(state), {"theta": x})
        assert jnp.allclose(expected_dm.array, output_dm.array)

    def test_shots_vs_exact(self, quantum_setup):
        """Compares shot-based measurements against exact calculations."""
        state, ops, observables, x = quantum_setup

        def compute_exact(x, use_dm=False):
            """Computes exact expectation values."""
            input_state = density_mat(state) if use_dm else state
            return expectation(
                input_state,
                ops,
                observables,
                {"theta": x},
                diff_mode="ad",
            )

        def compute_shots(x, use_dm=False):
            """Computes shot-based expectation values."""
            input_state = density_mat(state) if use_dm else state
            return expectation(
                input_state,
                ops,
                observables,
                {"theta": x},
                diff_mode="gpsr",
                n_shots=N_SHOTS,
            )

        # Compare results for different computation methods
        exp_exact = compute_exact(x)
        exp_exact_dm = compute_exact(x, use_dm=True)
        exp_shots = compute_shots(x)
        exp_shots_dm = compute_shots(x, use_dm=True)

        assert jnp.allclose(exp_exact, exp_exact_dm)
        assert jnp.allclose(exp_exact, exp_shots, atol=SHOTS_ATOL)
        assert jnp.allclose(exp_exact, exp_shots_dm, atol=SHOTS_ATOL)

        # Compare gradients
        grad_exact = jax.grad(lambda x: compute_exact(x).sum())(x)
        grad_shots = jax.grad(lambda x: compute_shots(x).sum())(x)
        assert jnp.isclose(grad_exact, grad_shots, atol=SHOTS_ATOL)


class TestJittedOperations:
    """Tests for JIT-compiled quantum operations."""

    @staticmethod
    def get_circuit_ops(ops):
        """Constructs the complete circuit operations.

        Args:
            ops: List of operations to append to base circuit.

        Returns:
            List of all quantum operations including base circuit.
        """
        gates = [H(0), NOT(0, 1)]
        circuit = QuantumCircuit(N_QUBITS, ops.operations + gates)
        return circuit

    @staticmethod
    @jax.jit
    def exact_jit(x, state, ops, observables):
        """JIT-compiled exact expectation calculation."""
        all_ops = TestJittedOperations.get_circuit_ops(ops)
        return expectation(
            state,
            all_ops,
            observables,
            {"theta": x},
            diff_mode="ad",
        )

    @staticmethod
    def shots_jit(x, state, ops, observables):
        """Shot-based expectation calculation."""
        all_ops = TestJittedOperations.get_circuit_ops(ops)
        return expectation(
            state,
            all_ops,
            observables,
            {"theta": x},
            diff_mode="gpsr",
            n_shots=N_SHOTS,
        )

    def test_jitted_computations(self, quantum_setup):
        """Tests consistency between JIT-compiled exact and shot-based calculations."""
        state, ops, observables, x = quantum_setup

        # Compute results with different methods
        result_exact = self.exact_jit(x, state, ops, observables)

        checked_shots = checkify.checkify(self.shots_jit)
        err, result_shots = jax.jit(checked_shots)(x, state, ops, observables)
        err.throw()

        # Compare results and gradients
        assert jnp.allclose(result_exact, result_shots, atol=SHOTS_ATOL)

        grad_exact = jax.grad(
            lambda x: self.exact_jit(x, state, ops, observables).sum(),
        )(x)
        grad_shots = jax.grad(
            lambda x: self.shots_jit(x, state, ops, observables).sum(),
        )(x)
        assert jnp.allclose(grad_exact, grad_shots, atol=SHOTS_ATOL)


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
