import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestGegenbauerPolynomialC(OpTestCase):
    """Tests for the gegenbauer_polynomial_c function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="gegenbauer_polynomial_c",
            func=torchscience.special_functions.gegenbauer_polynomial_c,
            arity=3,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="alpha", position=1, default_real_range=(0.5, 3.0)
                ),
                InputSpec(
                    name="z", position=2, default_real_range=(-0.9, 0.9)
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_compile_smoke",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 1.5, 0.5),
                    expected=1.0,
                    description="C_0^lambda(z) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 1.5, 0.5),
                    expected=1.5,
                    description="C_1^lambda(z) = 2*lambda*z = 2*1.5*0.5 = 1.5",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
