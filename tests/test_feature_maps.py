"""Tests for quantum feature map implementations.

This module tests the core functionality of quantum feature maps, including:
- Chebyshev polynomial encoding
- Product encoding
- Direct amplitude encoding

The tests verify that each feature map correctly processes input data
and produces valid quantum states.
"""

import jax.numpy as jnp
import pytest
from horqrux.apply import apply_gates as apply_gate
from horqrux.primitives import X
from horqrux.utils import random_state
from horqrux.utils.operator_utils import TargetQubits

from qedft.models.quantum.feature_maps import chebyshev, direct, product


class TestFeatureMaps:
    """Tests for quantum feature map implementations."""

    def setUp(self):
        """Sets up common test parameters."""
        # Example 2D input vector
        self.x = jnp.array([0.5, 0.5])
        # Target qubits for 2-qubit encoding
        self.target_idx = TargetQubits(((0,), (1,)))
        # Initial 2-qubit state with X gate applied to first qubit
        self.state = apply_gate(random_state(2), X(0))

    def test_chebyshev_encoding(self):
        """Tests Chebyshev polynomial feature encoding.

        Verifies that the Chebyshev feature map successfully encodes input data
        into quantum states using Chebyshev polynomials.
        """
        self.setUp()
        result = chebyshev(self.x, self.state, self.target_idx)
        assert result is not None, "Chebyshev encoding failed to produce output"
        assert result.shape == self.state.shape, "Output state has incorrect shape"

    def test_product_encoding(self):
        """Tests product feature encoding.

        Verifies that the product feature map successfully encodes input data
        using tensor product structure.
        """
        self.setUp()
        result = product(self.x, self.state, self.target_idx)
        assert result is not None, "Product encoding failed to produce output"
        assert result.shape == self.state.shape, "Output state has incorrect shape"

    def test_direct_encoding(self):
        """Tests direct amplitude encoding.

        Verifies that the direct feature map successfully encodes input data
        directly into quantum amplitudes.
        """
        self.setUp()
        result = direct(self.x, self.state, self.target_idx)
        assert result is not None, "Direct encoding failed to produce output"
        assert result.shape == self.state.shape, "Output state has incorrect shape"


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
