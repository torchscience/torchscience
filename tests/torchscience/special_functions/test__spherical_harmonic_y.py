import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestSphericalHarmonicY(OpTestCase):
    """Tests for the spherical_harmonic_y function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="spherical_harmonic_y",
            func=torchscience.special_functions.spherical_harmonic_y,
            arity=4,
            input_specs=[
                InputSpec(
                    name="l",
                    position=0,
                    default_real_range=(0.0, 5.0),
                    supports_grad=False,
                ),
                InputSpec(
                    name="m",
                    position=1,
                    default_real_range=(-3.0, 3.0),
                    supports_grad=False,
                ),
                InputSpec(
                    name="theta",
                    position=2,
                    default_real_range=(0.1, 3.0),
                    supports_grad=True,
                ),
                InputSpec(
                    name="phi",
                    position=3,
                    default_real_range=(0.1, 6.0),
                    supports_grad=True,
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_dtype_preservation",
                "test_low_precision_forward",
                "test_nan_propagation",
                "test_nan_propagation_all_inputs",
                "test_real_dtypes",
                "test_special_values",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0, 0.0, 0.0),
                    expected=1.0 / math.sqrt(4.0 * math.pi) + 0j,
                    description="Y_0^0 = 1/sqrt(4*pi)",
                ),
                SpecialValue(
                    inputs=(0.0, 0.0, math.pi / 2, math.pi),
                    expected=1.0 / math.sqrt(4.0 * math.pi) + 0j,
                    description="Y_0^0 is constant for all angles",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0, 0.0, 0.0),
                    expected=math.sqrt(5.0 / (4.0 * math.pi)) + 0j,
                    description="Y_2^0(0, 0) = sqrt(5/(4*pi)) * P_2(1) = sqrt(5/(4*pi))",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
