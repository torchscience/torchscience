import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestExponentialIntegralEi(OpTestCase):
    """Tests for the exponential integral Ei function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="exponential_integral_ei",
            func=torchscience.special_functions.exponential_integral_ei,
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
                    expected=-math.inf,
                    description="Ei(0) = -inf",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=0.4542199048631736,
                    description="Ei(0.5)",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=1.8951178163559368,
                    description="Ei(1.0)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=4.954234356001891,
                    description="Ei(2.0)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=40.18527536389832,
                    description="Ei(5.0)",
                ),
                SpecialValue(
                    inputs=(10.0,),
                    expected=2492.228976241877,
                    description="Ei(10.0)",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
                "test_special_values",
            },
        )
