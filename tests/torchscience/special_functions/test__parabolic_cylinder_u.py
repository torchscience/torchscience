import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestParabolicCylinderU(OpTestCase):
    """Tests for the parabolic_cylinder_u function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="parabolic_cylinder_u",
            func=torchscience.special_functions.parabolic_cylinder_u,
            arity=2,
            input_specs=[
                InputSpec(
                    name="a",
                    position=0,
                    default_real_range=(-3.0, 3.0),
                ),
                InputSpec(
                    name="x",
                    position=1,
                    default_real_range=(0.5, 5.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0),
                    expected=math.gamma(0.25) / (2**0.75 * math.sqrt(math.pi)),
                    rtol=1e-5,
                    atol=1e-6,
                    description="U(0, 0) = Gamma(1/4) / (2^(3/4) * sqrt(pi))",
                ),
            ],
        )
