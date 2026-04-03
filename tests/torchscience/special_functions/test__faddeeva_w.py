import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestFaddeevaW(OpTestCase):
    """Tests for the Faddeeva function w(z) = exp(-z^2) * erfc(-iz)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="faddeeva_w",
            func=torchscience.special_functions.faddeeva_w,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z",
                    position=0,
                    default_real_range=(-5.0, 5.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0 + 0.0j,),
                    expected=1.0 + 0.0j,
                    description="w(0) = 1",
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_dtype_preservation",
                "test_gradcheck_real",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_dtype_preservation",
                "test_low_precision_forward",
                "test_real_dtypes",
                "test_special_values",
            },
        )
