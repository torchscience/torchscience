import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestErfcinv(OpTestCase):
    """Tests for the inverse complementary error function erfcinv(x)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="erfcinv",
            func=torchscience.special_functions.erfcinv,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x",
                    position=0,
                    default_real_range=(0.01, 1.99),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.0,
                    description="erfcinv(1) = 0",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=0.4769362762044699,
                    description="erfcinv(0.5) ~ 0.4769362762044699",
                ),
                SpecialValue(
                    inputs=(0.0,),
                    expected=math.inf,
                    description="erfcinv(0) = +inf",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=-math.inf,
                    description="erfcinv(2) = -inf",
                ),
            ],
        )
