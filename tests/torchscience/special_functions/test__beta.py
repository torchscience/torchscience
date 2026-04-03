import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestBeta(OpTestCase):
    """Tests for the beta function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="beta",
            func=torchscience.special_functions.beta,
            arity=2,
            input_specs=[
                InputSpec(name="a", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="b", position=1, default_real_range=(0.5, 5.0)),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=1.0,
                    description="B(1, 1) = 1",
                ),
                SpecialValue(
                    inputs=(0.5, 0.5),
                    expected=math.pi,
                    description="B(0.5, 0.5) = pi",
                ),
                SpecialValue(
                    inputs=(1.0, 2.0),
                    expected=0.5,
                    description="B(1, 2) = 1/2",
                ),
                SpecialValue(
                    inputs=(1.0, 3.0),
                    expected=1.0 / 3.0,
                    description="B(1, 3) = 1/3",
                ),
            ],
            skip_tests={
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
            },
        )
