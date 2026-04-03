import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestExponentialIntegralEin(OpTestCase):
    """Tests for the complementary exponential integral Ein function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="exponential_integral_ein",
            func=torchscience.special_functions.exponential_integral_ein,
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
                    description="Ein(0) = 0",
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=0.4438420791177484,
                    description="Ein(0.5)",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.7965995992970532,
                    description="Ein(1.0)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=1.3192633561695397,
                    description="Ein(2.0)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=2.1878018729269089,
                    description="Ein(5.0)",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
