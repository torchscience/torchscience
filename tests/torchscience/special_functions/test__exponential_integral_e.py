import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestExponentialIntegralE(OpTestCase):
    """Tests for the generalized exponential integral E_n(x)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="exponential_integral_e",
            func=torchscience.special_functions.exponential_integral_e,
            arity=2,
            input_specs=[
                InputSpec(
                    name="n",
                    position=0,
                    default_real_range=(0.5, 5.0),
                    supports_grad=False,
                ),
                InputSpec(
                    name="x",
                    position=1,
                    default_real_range=(0.5, 10.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(2.0, 0.0),
                    expected=1.0,
                    description="E_2(0) = 1/(2-1) = 1",
                ),
                SpecialValue(
                    inputs=(3.0, 0.0),
                    expected=0.5,
                    description="E_3(0) = 1/(3-1) = 0.5",
                ),
                SpecialValue(
                    inputs=(4.0, 0.0),
                    expected=1.0 / 3.0,
                    description="E_4(0) = 1/(4-1) = 1/3",
                ),
            ],
        )
