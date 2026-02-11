import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestLaguerrePolynomialL(OpTestCase):
    """Tests for the laguerre_polynomial_l function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="laguerre_polynomial_l",
            func=torchscience.special_functions.laguerre_polynomial_l,
            arity=3,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="alpha", position=1, default_real_range=(0.5, 3.0)
                ),
                InputSpec(
                    name="z", position=2, default_real_range=(0.1, 10.0)
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0, 1.0),
                    expected=1.0,
                    description="L_0^0(1) = 1",
                ),
                SpecialValue(
                    inputs=(0.0, 2.0, 5.0),
                    expected=1.0,
                    description="L_0^alpha(z) = 1 for all alpha, z",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0, 1.0),
                    expected=0.0,
                    description="L_1^0(1) = 1 + 0 - 1 = 0",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0, 0.0),
                    expected=2.0,
                    description="L_1^1(0) = 1 + 1 - 0 = 2",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0, 1.0),
                    expected=-0.5,
                    description="L_2^0(1) = (1*2 - 2*2*1 + 1)/2 = -0.5",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
