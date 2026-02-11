import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestErfinv(OpTestCase):
    """Tests for the inverse error function erfinv(x)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="erfinv",
            func=torchscience.special_functions.erfinv,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x",
                    position=0,
                    default_real_range=(-0.99, 0.99),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.0,
                    description="erfinv(0) = 0",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=0.4769362762044699,
                    description="erfinv(0.5) ~ 0.4769362762044699",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=math.inf,
                    description="erfinv(1) = +inf",
                ),
                SpecialValue(
                    inputs=(-1.0,),
                    expected=-math.inf,
                    description="erfinv(-1) = -inf",
                ),
            ],
        )
