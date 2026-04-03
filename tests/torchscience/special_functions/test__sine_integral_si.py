import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestSineIntegralSi(OpTestCase):
    """Tests for the sine integral Si function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="sine_integral_si",
            func=torchscience.special_functions.sine_integral_si,
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
                    expected=0.0,
                    description="Si(0) = 0",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=0.4931074180430667,
                    description="Si(0.5)",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.9460830703671830,
                    description="Si(1.0)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=1.6054129768026948,
                    description="Si(2.0)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=1.5499312449446073,
                    description="Si(5.0)",
                ),
                SpecialValue(
                    inputs=(float("inf"),),
                    expected=math.pi / 2,
                    description="Si(+inf) = pi/2",
                ),
                SpecialValue(
                    inputs=(float("-inf"),),
                    expected=-math.pi / 2,
                    description="Si(-inf) = -pi/2",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
