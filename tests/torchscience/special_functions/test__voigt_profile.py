import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestVoigtProfile(OpTestCase):
    """Tests for the voigt_profile function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="voigt_profile",
            func=torchscience.special_functions.voigt_profile,
            arity=3,
            input_specs=[
                InputSpec(
                    name="x",
                    position=0,
                    default_real_range=(-5.0, 5.0),
                ),
                InputSpec(
                    name="sigma",
                    position=1,
                    default_real_range=(0.1, 3.0),
                ),
                InputSpec(
                    name="gamma",
                    position=2,
                    default_real_range=(0.1, 3.0),
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 1.0, 0.0),
                    expected=1.0 / math.sqrt(2 * math.pi),
                    rtol=1e-6,
                    atol=1e-8,
                    description="V(0, 1, 0) = 1/sqrt(2*pi) (Gaussian limit)",
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_complex_dtypes",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_low_precision_dtype_preservation",
                "test_low_precision_forward",
            },
        )
