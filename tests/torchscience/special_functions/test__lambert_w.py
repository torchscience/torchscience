import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestLambertW(OpTestCase):
    """Tests for the lambert_w function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="lambert_w",
            func=torchscience.special_functions.lambert_w,
            arity=2,
            input_specs=[
                InputSpec(
                    name="k",
                    position=0,
                    default_real_range=(0.0, 1.0),
                ),
                InputSpec(
                    name="z",
                    position=1,
                    default_real_range=(0.1, 5.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0),
                    expected=0.0,
                    description="W_0(0) = 0",
                ),
                SpecialValue(
                    inputs=(0.0, math.e),
                    expected=1.0,
                    description="W_0(e) = 1",
                ),
                SpecialValue(
                    inputs=(0.0, 1.0),
                    expected=0.5671432904097838,
                    description="W_0(1) ~ 0.5671432904097838",
                ),
                SpecialValue(
                    inputs=(0.0, -1.0 / math.e),
                    expected=-1.0,
                    rtol=1e-6,
                    atol=1e-6,
                    description="W_0(-1/e) = -1 (branch point)",
                ),
            ],
        )
