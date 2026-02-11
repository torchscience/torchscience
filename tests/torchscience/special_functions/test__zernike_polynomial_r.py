import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestZernikePolynomialR(OpTestCase):
    """Tests for the zernike_polynomial_r function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="zernike_polynomial_r",
            func=torchscience.special_functions.zernike_polynomial_r,
            arity=3,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(1.0, 5.0)),
                InputSpec(name="m", position=1, default_real_range=(0.5, 3.0)),
                InputSpec(
                    name="rho", position=2, default_real_range=(0.1, 0.9)
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
                "test_nan_propagation_all_inputs",
            },
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0, 0.5),
                    expected=1.0,
                    description="R_0^0(rho) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0, 0.5),
                    expected=0.5,
                    description="R_1^1(rho) = rho",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0, 0.5),
                    expected=-0.5,
                    description="R_2^0(0.5) = 2*0.25 - 1 = -0.5",
                ),
                SpecialValue(
                    inputs=(2.0, 2.0, 0.5),
                    expected=0.25,
                    description="R_2^2(0.5) = 0.25",
                ),
                SpecialValue(
                    inputs=(0.0, 0.0, 1.0),
                    expected=1.0,
                    description="R_0^0(1) = 1",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0, 1.0),
                    expected=1.0,
                    description="R_2^0(1) = 1",
                ),
                SpecialValue(
                    inputs=(4.0, 0.0, 1.0),
                    expected=1.0,
                    description="R_4^0(1) = 1",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
