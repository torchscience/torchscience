import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestExponentialIntegralE1(OpTestCase):
    """Tests for the exponential integral E_1 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="exponential_integral_e_1",
            func=torchscience.special_functions.exponential_integral_e_1,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x",
                    position=0,
                    default_real_range=(0.5, 10.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=math.inf,
                    description="E_1(0) = +inf",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=0.5597735947761608,
                    description="E_1(0.5)",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.21938393439552062,
                    description="E_1(1.0)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=0.04890051070806112,
                    description="E_1(2.0)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=0.001148295591784439,
                    description="E_1(5.0)",
                ),
                SpecialValue(
                    inputs=(10.0,),
                    expected=4.156968929685324e-06,
                    description="E_1(10.0)",
                ),
            ],
        )
