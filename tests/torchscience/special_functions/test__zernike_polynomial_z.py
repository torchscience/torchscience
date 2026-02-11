import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestZernikePolynomialZ(OpTestCase):
    """Tests for the zernike_polynomial_z function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="zernike_polynomial_z",
            func=torchscience.special_functions.zernike_polynomial_z,
            arity=4,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(1.0, 5.0)),
                InputSpec(name="m", position=1, default_real_range=(0.5, 3.0)),
                InputSpec(
                    name="rho", position=2, default_real_range=(0.1, 0.9)
                ),
                InputSpec(
                    name="theta", position=3, default_real_range=(0.1, 6.0)
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
                    inputs=(0.0, 0.0, 0.5, 0.0),
                    expected=1.0,
                    description="Z_0^0(rho, theta) = 1",
                ),
                SpecialValue(
                    inputs=(2.0, 2.0, 1.0, 0.0),
                    expected=1.0,
                    description="Z_2^2(1, 0) = 1^2 * cos(0) = 1",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0, 0.5, 0.0),
                    expected=-0.5,
                    description="Z_2^0(0.5, 0) = 2*0.25 - 1 = -0.5",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
        )
