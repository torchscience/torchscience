import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestLogBeta(OpTestCase):
    """Tests for the log_beta function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="log_beta",
            func=torchscience.special_functions.log_beta,
            arity=2,
            input_specs=[
                InputSpec(name="a", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="b", position=1, default_real_range=(0.5, 5.0)),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=0.0,
                    description="log B(1, 1) = log(1) = 0",
                ),
                SpecialValue(
                    inputs=(0.5, 0.5),
                    expected=math.log(math.pi),
                    description="log B(0.5, 0.5) = log(pi)",
                ),
                SpecialValue(
                    inputs=(1.0, 2.0),
                    expected=-math.log(2.0),
                    description="log B(1, 2) = -log(2)",
                ),
                SpecialValue(
                    inputs=(2.0, 2.0),
                    expected=math.log(1.0 / 6.0),
                    description="log B(2, 2) = log(1/6)",
                ),
            ],
            skip_tests={
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
