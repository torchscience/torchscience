from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class BroadcastingMixin:
    """Mixin providing broadcasting tests."""

    descriptor: "OperatorDescriptor"

    def test_broadcast_scalar_with_tensor(self):
        """Test broadcasting scalar with tensor."""
        if "test_broadcast_scalar_with_tensor" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires binary operator")

        # Create scalar and tensor inputs
        spec0 = self.descriptor.input_specs[0]
        spec1 = self.descriptor.input_specs[1]

        scalar = self._make_input_for_spec(spec0, torch.float64, "cpu", ())
        tensor = self._make_input_for_spec(spec1, torch.float64, "cpu", (5,))

        result = self.descriptor.func(scalar, tensor)
        assert result.shape == (5,)

    def test_broadcast_different_shapes(self):
        """Test broadcasting with different shapes."""
        if "test_broadcast_different_shapes" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if self.descriptor.arity < 2:
            pytest.skip("Broadcasting test requires binary operator")

        spec0 = self.descriptor.input_specs[0]
        spec1 = self.descriptor.input_specs[1]

        t1 = self._make_input_for_spec(spec0, torch.float64, "cpu", (3, 1))
        t2 = self._make_input_for_spec(spec1, torch.float64, "cpu", (1, 4))

        result = self.descriptor.func(t1, t2)
        assert result.shape == (3, 4)
