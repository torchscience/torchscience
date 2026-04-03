import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestInverseComplementaryRegularizedIncompleteBeta(OpTestCase):
    """Tests for the inverse_complementary_regularized_incomplete_beta function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="inverse_complementary_regularized_incomplete_beta",
            func=torchscience.special_functions.inverse_complementary_regularized_incomplete_beta,
            arity=3,
            input_specs=[
                InputSpec(
                    name="a",
                    position=0,
                    default_real_range=(0.5, 5.0),
                ),
                InputSpec(
                    name="b",
                    position=1,
                    default_real_range=(0.5, 5.0),
                ),
                InputSpec(
                    name="q",
                    position=2,
                    default_real_range=(0.01, 0.99),
                ),
            ],
            skip_tests={
                "test_complex_dtypes",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
            special_values=[
                SpecialValue(
                    inputs=(2.0, 3.0, 0.0),
                    expected=1.0,
                    description="I_c^{-1}(a, b, 0) = 1",
                ),
                SpecialValue(
                    inputs=(2.0, 3.0, 1.0),
                    expected=0.0,
                    description="I_c^{-1}(a, b, 1) = 0",
                ),
                SpecialValue(
                    inputs=(3.0, 3.0, 0.5),
                    expected=0.5,
                    rtol=1e-5,
                    atol=1e-6,
                    description="I_c^{-1}(a, a, 0.5) = 0.5 (symmetric case)",
                ),
            ],
        )
