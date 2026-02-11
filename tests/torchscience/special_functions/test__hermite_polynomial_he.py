import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestHermitePolynomialHe(OpTestCase):
    """Tests for the hermite_polynomial_he function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="hermite_polynomial_he",
            func=torchscience.special_functions.hermite_polynomial_he,
            arity=2,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="z", position=1, default_real_range=(-3.0, 3.0)
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
                    inputs=(0.0, 0.0),
                    expected=1.0,
                    description="He_0(0) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=0.0,
                    description="He_1(0) = 0",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0),
                    expected=-1.0,
                    description="He_2(0) = -1",
                ),
                SpecialValue(
                    inputs=(4.0, 0.0),
                    expected=3.0,
                    description="He_4(0) = 3",
                ),
                SpecialValue(
                    inputs=(2.0, 2.0),
                    expected=3.0,
                    description="He_2(2) = 4 - 1 = 3",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=1.0,
                    description="He_1(1) = 1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
