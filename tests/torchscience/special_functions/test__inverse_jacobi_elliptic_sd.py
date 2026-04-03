import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
    ToleranceConfig,
)


class TestInverseJacobiEllipticSd(OpTestCase):
    """Tests for the inverse_jacobi_elliptic_sd function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="inverse_jacobi_elliptic_sd",
            func=torchscience.special_functions.inverse_jacobi_elliptic_sd,
            arity=2,
            input_specs=[
                InputSpec(
                    name="u", position=0, default_real_range=(-5.0, 5.0)
                ),
                InputSpec(
                    name="m", position=1, default_real_range=(0.01, 0.99)
                ),
            ],
            tolerances=ToleranceConfig(),
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
                    inputs=(0.0, 0.5),
                    expected=0.0,
                    description="arcsd(0, m) = 0",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_quantized=False,
            supports_meta=True,
        )
