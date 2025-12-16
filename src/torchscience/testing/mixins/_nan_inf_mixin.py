from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class NanInfMixin:
    """Mixin providing NaN/Inf propagation tests."""

    descriptor: "OperatorDescriptor"

    def test_nan_propagation(self):
        """Test that NaN inputs produce NaN outputs."""
        if "test_nan_propagation" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        x = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float64)

        if self.descriptor.arity == 1:
            result = self.descriptor.func(x)
        else:
            other = self._make_input_for_spec(
                self.descriptor.input_specs[1], torch.float64, "cpu", (3,)
            )
            result = self.descriptor.func(x, other)

        assert torch.isnan(result[1])

    def test_inf_handling(self):
        """Test behavior with infinite inputs."""
        if "test_inf_handling" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        x = torch.tensor(
            [1.0, float("inf"), float("-inf")], dtype=torch.float64
        )

        if self.descriptor.arity == 1:
            result = self.descriptor.func(x)
        else:
            other = self._make_input_for_spec(
                self.descriptor.input_specs[1], torch.float64, "cpu", (3,)
            )
            result = self.descriptor.func(x, other)

        # Just verify it doesn't crash - specific behavior depends on function
        assert result.shape == (3,)
