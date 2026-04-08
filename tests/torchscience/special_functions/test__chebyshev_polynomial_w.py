import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    ToleranceConfig,
)


class TestChebyshevPolynomialW(OpTestCase):
    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="chebyshev_polynomial_w",
            func=torchscience.special_functions.chebyshev_polynomial_w,
            arity=2,
            input_specs=[
                InputSpec(
                    name="n",
                    position=0,
                    default_real_range=(0.0, 10.0),
                    can_be_integer=True,
                    supports_grad=False,
                ),
                InputSpec(
                    name="x",
                    position=1,
                    default_real_range=(-0.99, 0.99),
                ),
            ],
            tolerances=ToleranceConfig(
                float64_rtol=1e-4,
                float64_atol=1e-4,
            ),
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_complex_dtypes",
                "test_dtype_preservation",
                "test_sparse_coo_basic",
                "test_low_precision_forward",
                "test_gradcheck_real",
                "test_gradgradcheck_real",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_sympy_reference_complex",
            },
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )

    pass
