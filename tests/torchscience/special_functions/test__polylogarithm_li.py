import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestPolylogarithmLi(OpTestCase):
    """Tests for the polylogarithm_li function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="polylogarithm_li",
            func=torchscience.special_functions.polylogarithm_li,
            arity=2,
            input_specs=[
                InputSpec(
                    name="s",
                    position=0,
                    default_real_range=(1.5, 5.0),
                ),
                InputSpec(
                    name="z",
                    position=1,
                    default_real_range=(-0.9, 0.9),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(2.0, 0.0),
                    expected=0.0,
                    description="Li_s(0) = 0 for any s",
                ),
                SpecialValue(
                    inputs=(2.0, 1.0),
                    expected=math.pi**2 / 6,
                    rtol=1e-4,
                    atol=1e-4,
                    description="Li_2(1) = pi^2/6 (Basel problem)",
                ),
                SpecialValue(
                    inputs=(2.0, 0.5),
                    expected=math.pi**2 / 12 - math.log(2) ** 2 / 2,
                    rtol=1e-5,
                    atol=1e-5,
                    description="Li_2(0.5) = pi^2/12 - ln(2)^2/2",
                ),
                SpecialValue(
                    inputs=(3.0, 0.5),
                    expected=0.5372131936,
                    rtol=1e-5,
                    atol=1e-5,
                    description="Li_3(0.5) ~ 0.5372131936",
                ),
                SpecialValue(
                    inputs=(2.0, -1.0),
                    expected=-(math.pi**2) / 12,
                    rtol=1e-3,
                    atol=1e-3,
                    description="Li_2(-1) = -pi^2/12",
                ),
            ],
        )
