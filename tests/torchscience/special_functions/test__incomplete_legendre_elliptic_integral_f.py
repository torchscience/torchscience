import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)


class TestIncompleteLegendreEllipticIntegralF(OpTestCase):
    """Tests for the incomplete_legendre_elliptic_integral_f function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="incomplete_legendre_elliptic_integral_f",
            func=torchscience.special_functions.incomplete_legendre_elliptic_integral_f,
            arity=2,
            input_specs=[
                InputSpec(
                    name="phi", position=0, default_real_range=(0.1, 1.4)
                ),
                InputSpec(
                    name="m", position=1, default_real_range=(0.01, 0.99)
                ),
            ],
            tolerances=ToleranceConfig(
                gradgradcheck_atol=1e-3,
                gradgradcheck_rtol=1e-3,
            ),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradgradcheck_complex",
                "test_gradcheck_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.5),
                    expected=0.0,
                    description="F(0, m) = 0",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=1.0,
                    description="F(phi, 0) = phi",
                ),
                SpecialValue(
                    inputs=(math.pi / 2, 0.0),
                    expected=math.pi / 2,
                    description="F(pi/2, 0) = pi/2",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )
