import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestLegendrePolynomialP(OpTestCase):
    """Tests for the legendre_polynomial_p function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="legendre_polynomial_p",
            func=torchscience.special_functions.legendre_polynomial_p,
            arity=2,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="z", position=1, default_real_range=(-0.9, 0.9)
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
                    inputs=(0.0, 0.5),
                    expected=1.0,
                    description="P_0(z) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.5),
                    expected=0.5,
                    description="P_1(z) = z",
                ),
                SpecialValue(
                    inputs=(2.0, 0.5),
                    expected=-0.125,
                    description="P_2(0.5) = (3*0.25 - 1)/2 = -0.125",
                ),
                SpecialValue(
                    inputs=(0.0, 1.0),
                    expected=1.0,
                    description="P_n(1) = 1 for all n",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=1.0,
                    description="P_1(1) = 1",
                ),
                SpecialValue(
                    inputs=(2.0, 1.0),
                    expected=1.0,
                    description="P_2(1) = 1",
                ),
                SpecialValue(
                    inputs=(0.0, -1.0),
                    expected=1.0,
                    description="P_0(-1) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, -1.0),
                    expected=-1.0,
                    description="P_1(-1) = -1",
                ),
                SpecialValue(
                    inputs=(2.0, -1.0),
                    expected=1.0,
                    description="P_2(-1) = 1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
