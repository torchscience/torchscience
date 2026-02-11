import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestReciprocalGamma(OpTestCase):
    """Tests for the reciprocal_gamma function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        sqrt_pi = math.sqrt(math.pi)
        return OperatorDescriptor(
            name="reciprocal_gamma",
            func=torchscience.special_functions.reciprocal_gamma,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(-5.0, 5.0),
                    excluded_values={0.0, -1.0, -2.0, -3.0, -4.0, -5.0},
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0,),
                    expected=1.0,
                    description="1/Gamma(1) = 1",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=1.0,
                    description="1/Gamma(2) = 1",
                ),
                SpecialValue(
                    inputs=(3.0,),
                    expected=0.5,
                    description="1/Gamma(3) = 1/2",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=1.0 / sqrt_pi,
                    description="1/Gamma(0.5) = 1/sqrt(pi)",
                ),
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.0,
                    description="1/Gamma(0) = 0 (pole)",
                ),
            ],
        )
