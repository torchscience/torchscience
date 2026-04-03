import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestAngerJ(OpTestCase):
    """Tests for the anger_j function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="anger_j",
            func=torchscience.special_functions.anger_j,
            arity=2,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="z", position=1, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0),
                    expected=1.0,
                    description="J_0(0) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=0.0,
                    description="J_1(0) = 0",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0),
                    expected=0.0,
                    description="J_2(0) = 0",
                ),
            ],
            skip_tests={
                "test_complex_dtypes",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
            },
        )
