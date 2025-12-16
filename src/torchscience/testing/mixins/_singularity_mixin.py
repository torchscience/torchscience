from typing import TYPE_CHECKING

import pytest
import torch

if TYPE_CHECKING:
    from torchscience.testing.descriptors import OperatorDescriptor


class SingularityMixin:
    """Mixin providing singularity behavior tests."""

    descriptor: "OperatorDescriptor"

    def test_pole_behavior(self):
        """Test behavior at poles."""
        if "test_pole_behavior" in self.descriptor.skip_tests:
            pytest.skip("Test skipped by descriptor")

        pole_specs = [
            s for s in self.descriptor.singularities if s.type == "pole"
        ]
        if not pole_specs:
            pytest.skip("No pole singularities defined")

        for spec in pole_specs:
            for loc in list(spec.locations())[:10]:  # Test first 10 poles
                x = torch.tensor([loc], dtype=torch.float64)
                if self.descriptor.arity == 1:
                    result = self.descriptor.func(x)
                else:
                    other = self._make_input_for_spec(
                        self.descriptor.input_specs[1],
                        torch.float64,
                        "cpu",
                        (1,),
                    )
                    result = self.descriptor.func(x, other)

                if spec.expected_behavior == "inf":
                    assert (
                        torch.isinf(result).all() or torch.isnan(result).all()
                    ), f"Expected inf/nan at pole {loc}, got {result}"
