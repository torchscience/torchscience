import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestWeierstrassP(OpTestCase):
    """Tests for the weierstrass_p function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="weierstrass_p",
            func=torchscience.special_functions.weierstrass_p,
            arity=3,
            input_specs=[
                InputSpec(name="z", position=0, default_real_range=(0.5, 3.0)),
                InputSpec(
                    name="g2", position=1, default_real_range=(0.5, 5.0)
                ),
                InputSpec(
                    name="g3", position=2, default_real_range=(0.5, 5.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 1.0, 0.0),
                    expected=math.inf,
                    description="P(0; g2, g3) = inf (double pole at origin)",
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_complex",
                "test_gradcheck_real",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_dtype_preservation",
                "test_low_precision_forward",
            },
        )
