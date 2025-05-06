"""
Tests for entangling layers for quantum neural networks.
"""

from functools import partial

import jax
import pytest
from horqrux import zero_state
from horqrux.noise import DigitalNoiseInstance, DigitalNoiseType

from qedft.models.quantum.entangling_layers import (
    entangling_layer,
    entangling_ops,
    get_entangler_map,
    plot_entanglement_pattern,
)


class TestGetEntanglerMap:
    """Tests for the get_entangler_map function."""

    def test_full_entanglement(self):
        """Test full entanglement strategy."""
        result = get_entangler_map(2, 3, "full")
        expected = [(0, 1), (0, 2), (1, 2)]
        assert result == expected

    def test_linear_entanglement(self):
        """Test linear entanglement strategy."""
        result = get_entangler_map(2, 4, "linear")
        expected = [(0, 1), (1, 2), (2, 3)]
        assert result == expected

    def test_circular_entanglement(self):
        """Test circular entanglement strategy."""
        result = get_entangler_map(2, 4, "circular")
        expected = [(3, 0), (0, 1), (1, 2), (2, 3)]
        assert result == expected

    def test_pairwise_entanglement(self):
        """Test pairwise entanglement strategy."""
        result = get_entangler_map(2, 4, "pairwise")
        expected = [(0, 1), (2, 3), (1, 2)]
        assert result == expected

    def test_reverse_linear_entanglement(self):
        """Test reverse linear entanglement strategy."""
        result = get_entangler_map(2, 4, "reverse_linear")
        expected = [(2, 3), (1, 2), (0, 1)]
        assert result == expected

    def test_alternate_linear_entanglement(self):
        """Test alternate linear entanglement strategy."""
        result = get_entangler_map(2, 4, "alternate_linear")
        expected = [(0, 1), (2, 3), (1, 2)]
        assert result == expected

    def test_sca_entanglement(self):
        """Test sca entanglement strategy with different offsets."""
        # With offset=0, should be same as circular
        result = get_entangler_map(2, 4, "sca", offset=0)
        expected = [(3, 0), (0, 1), (1, 2), (2, 3)]
        assert result == expected

        # With offset=1, should reverse the qubit indices
        result = get_entangler_map(2, 4, "sca", offset=1)
        expected = [(3, 2), (0, 3), (1, 0), (2, 1)]
        assert result == expected

    def test_invalid_entanglement(self):
        """Test invalid entanglement strategy."""
        with pytest.raises(ValueError):
            get_entangler_map(2, 3, "invalid_strategy")

    def test_block_qubits_greater_than_circuit_qubits(self):
        """Test when block qubits > circuit qubits."""
        with pytest.raises(ValueError):
            get_entangler_map(4, 3, "full")

    def test_pairwise_with_more_than_two_qubits(self):
        """Test pairwise with more than two qubits."""
        with pytest.raises(ValueError):
            get_entangler_map(3, 4, "pairwise")


class TestEntanglingOps:
    """Tests for the entangling_ops function."""

    def test_entangling_ops_returns_correct_number(self):
        """Test entangling_ops returns correct number of operations."""
        ops = entangling_ops(4, "linear")
        assert len(ops) == 3  # For 4 qubits with linear entanglement, expect 3 ops

    def test_entangling_ops_with_noise(self):
        """Test entangling_ops with noise parameter."""
        noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.1),)
        ops = entangling_ops(3, "full", noise)
        assert len(ops) == 3  # For 3 qubits with full entanglement, expect 3 ops

    def test_invalid_entangling_block_type(self):
        """Test invalid entangling block type."""
        with pytest.raises(ValueError):
            entangling_ops(3, "invalid_type")


class TestEntanglingLayer:
    """Tests for the entangling_layer function."""

    def test_entangling_layer_preserves_shape(self):
        """Test entangling_layer preserves the shape of the state."""
        state = zero_state(3)
        result = entangling_layer(state, "full")
        assert result.shape == state.shape

    def test_entangling_layer_with_noise(self):
        """Test entangling_layer with noise parameter."""
        state = zero_state(3)
        noise = (DigitalNoiseInstance(DigitalNoiseType.DEPOLARIZING, 0.1),)
        result = entangling_layer(state, "full", noise)
        assert result.array.shape == (2, 2) * 3

    def test_jit_compatibility(self):
        """Test that entangling_layer is compatible with JAX JIT."""
        state = zero_state(3)
        jitted_layer = jax.jit(partial(entangling_layer, entangling_block_type="full"))
        result = jitted_layer(state)
        assert result.shape == state.shape


class TestPlotEntanglementPattern:
    """Tests for the plot_entanglement_pattern function."""

    def test_plot_entanglement_pattern(self, capsys):
        """Test that plot_entanglement_pattern produces output."""
        plot_entanglement_pattern(4, "linear")
        captured = capsys.readouterr()
        assert "Entanglement pattern: linear" in captured.out
        assert "q0" in captured.out
        assert "q1" in captured.out
        assert "q2" in captured.out
        assert "q3" in captured.out

    def test_plot_entanglement_pattern_with_offset(self, capsys):
        """Test plot_entanglement_pattern with offset."""
        plot_entanglement_pattern(4, "sca", offset=1)
        captured = capsys.readouterr()
        assert "Entanglement pattern: sca (offset=1)" in captured.out

    def test_plot_invalid_entanglement(self, capsys):
        """Test plot_entanglement_pattern with invalid entanglement."""
        plot_entanglement_pattern(4, "invalid_pattern")
        captured = capsys.readouterr()
        assert "Error plotting entanglement pattern" in captured.out


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
