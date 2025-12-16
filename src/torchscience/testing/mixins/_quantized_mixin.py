from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class QuantizedMixin:
    """Mixin providing quantized tensor tests."""

    descriptor: "OperatorDescriptor"

    @pytest.mark.parametrize("qtype", [torch.quint8, torch.qint8])
    def test_quantized_basic(self, qtype: torch.dtype):
        """Test basic quantized tensor support."""
        if "test_quantized_basic" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        if not self.descriptor.supports_quantized:
            pytest.skip("Operator does not support quantized tensors")

        if self.descriptor.arity != 1:
            pytest.skip("Quantized test only for unary operators")

        # Create quantized tensor
        scale = 0.1
        zero_point = 0
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        qx = torch.quantize_per_tensor(x, scale, zero_point, qtype)

        result = self.descriptor.func(qx)
        assert result.is_quantized
