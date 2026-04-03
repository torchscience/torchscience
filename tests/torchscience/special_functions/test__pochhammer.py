import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestPochhammer(OpTestCase):
    """Tests for the pochhammer function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="pochhammer",
            func=torchscience.special_functions.pochhammer,
            arity=2,
            input_specs=[
                InputSpec(name="z", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="n", position=1, default_real_range=(0.5, 5.0)),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=1.0,
                    description="(1)_0 = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=1.0,
                    description="(1)_1 = 1! = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 2.0),
                    expected=2.0,
                    description="(1)_2 = 2! = 2",
                ),
                SpecialValue(
                    inputs=(1.0, 3.0),
                    expected=6.0,
                    description="(1)_3 = 3! = 6",
                ),
                SpecialValue(
                    inputs=(3.0, 4.0),
                    expected=360.0,
                    description="(3)_4 = 3*4*5*6 = 360",
                ),
                SpecialValue(
                    inputs=(2.0, 3.0),
                    expected=24.0,
                    description="(2)_3 = 2*3*4 = 24",
                ),
            ],
            skip_tests={
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
