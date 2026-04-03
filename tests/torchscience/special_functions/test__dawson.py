import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestDawson(OpTestCase):
    """Tests for the Dawson function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="dawson",
            func=torchscience.special_functions.dawson,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(-5.0, 5.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.0,
                    description="D(0) = 0",
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_real",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_dtype_preservation",
                "test_low_precision_forward",
            },
        )
