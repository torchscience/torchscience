import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestCharlierPolynomialC(OpTestCase):
    """Tests for the charlier_polynomial_c function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="charlier_polynomial_c",
            func=torchscience.special_functions.charlier_polynomial_c,
            arity=3,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(name="x", position=1, default_real_range=(0.1, 5.0)),
                InputSpec(name="a", position=2, default_real_range=(0.5, 5.0)),
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
                    inputs=(0.0, 1.0, 1.0),
                    expected=1.0,
                    description="C_0(x; a) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 2.0, 1.0),
                    expected=1.0,
                    description="C_1(2; 1) = 2/1 - 1 = 1",
                ),
                SpecialValue(
                    inputs=(2.0, 2.0, 1.0),
                    expected=-1.0,
                    description="C_2(2; 1) = (4 - 3*2 + 1)/1 = -1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0, 2.0),
                    expected=-1.0,
                    description="C_1(0; 2) = 0/2 - 1 = -1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
