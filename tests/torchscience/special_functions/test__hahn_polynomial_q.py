import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestHahnPolynomialQ(OpTestCase):
    """Tests for the hahn_polynomial_q function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="hahn_polynomial_q",
            func=torchscience.special_functions.hahn_polynomial_q,
            arity=5,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 3.0)),
                InputSpec(name="x", position=1, default_real_range=(0.1, 3.0)),
                InputSpec(
                    name="alpha", position=2, default_real_range=(0.5, 3.0)
                ),
                InputSpec(
                    name="beta", position=3, default_real_range=(0.5, 3.0)
                ),
                InputSpec(
                    name="N", position=4, default_real_range=(3.0, 10.0)
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_complex_dtypes",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 1.0, 1.0, 1.0, 5.0),
                    expected=1.0,
                    description="Q_0(x; alpha, beta, N) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0, 1.0, 1.0, 5.0),
                    expected=1.0,
                    description="Q_1(0; alpha, beta, N) = 1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
