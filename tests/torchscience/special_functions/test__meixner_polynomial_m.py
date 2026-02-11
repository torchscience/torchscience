import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestMeixnerPolynomialM(OpTestCase):
    """Tests for the meixner_polynomial_m function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="meixner_polynomial_m",
            func=torchscience.special_functions.meixner_polynomial_m,
            arity=4,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="x", position=1, default_real_range=(0.1, 5.0)),
                InputSpec(
                    name="beta", position=2, default_real_range=(0.5, 5.0)
                ),
                InputSpec(name="c", position=3, default_real_range=(0.1, 0.9)),
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
                    inputs=(0.0, 1.0, 2.0, 0.5),
                    expected=1.0,
                    description="M_0(x; beta, c) = 1",
                ),
                SpecialValue(
                    inputs=(0.0, 3.0, 1.0, 0.8),
                    expected=1.0,
                    description="M_0(x; beta, c) = 1 for all x, beta, c",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
