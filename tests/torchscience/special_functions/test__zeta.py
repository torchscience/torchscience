import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestZeta(OpTestCase):
    """Tests for the zeta function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="zeta",
            func=torchscience.special_functions.zeta,
            arity=1,
            input_specs=[
                InputSpec(
                    name="s",
                    position=0,
                    default_real_range=(1.5, 10.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(2.0,),
                    expected=math.pi**2 / 6,
                    description="zeta(2) = pi^2/6 (Basel problem)",
                ),
                SpecialValue(
                    inputs=(4.0,),
                    expected=math.pi**4 / 90,
                    description="zeta(4) = pi^4/90",
                ),
                SpecialValue(
                    inputs=(6.0,),
                    expected=math.pi**6 / 945,
                    description="zeta(6) = pi^6/945",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
            },
        )
