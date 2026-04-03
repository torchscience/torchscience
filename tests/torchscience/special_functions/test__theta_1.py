import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestTheta1(OpTestCase):
    """Tests for the theta_1 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="theta_1",
            func=torchscience.special_functions.theta_1,
            arity=2,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(-3.0, 3.0)
                ),
                InputSpec(
                    name="q", position=1, default_real_range=(0.01, 0.9)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5),
                    expected=0.0,
                    description="theta_1(0, q) = 0 (odd function)",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=0.0,
                    description="theta_1(z, 0) = 0",
                ),
            ],
            skip_tests={
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
