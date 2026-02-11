import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestHermitePolynomialH(OpTestCase):
    """Tests for the hermite_polynomial_h function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="hermite_polynomial_h",
            func=torchscience.special_functions.hermite_polynomial_h,
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
                    description="H_0(0) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=0.0,
                    description="H_1(0) = 0",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0),
                    expected=-2.0,
                    description="H_2(0) = -2",
                ),
                SpecialValue(
                    inputs=(4.0, 0.0),
                    expected=12.0,
                    description="H_4(0) = 12",
                ),
                SpecialValue(
                    inputs=(2.0, 0.5),
                    expected=-1.0,
                    description="H_2(0.5) = 4*0.25 - 2 = -1",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=2.0,
                    description="H_1(1) = 2",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
