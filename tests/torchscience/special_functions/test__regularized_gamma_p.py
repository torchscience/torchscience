import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestRegularizedGammaP(OpTestCase):
    """Tests for the regularized_gamma_p function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="regularized_gamma_p",
            func=torchscience.special_functions.regularized_gamma_p,
            arity=2,
            input_specs=[
                InputSpec(name="a", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="x", position=1, default_real_range=(0.1, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=0.0,
                    description="P(1, 0) = 0",
                ),
            ],
            skip_tests={
                "test_complex_dtypes",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
        )
