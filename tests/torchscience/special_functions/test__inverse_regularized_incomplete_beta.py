import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestInverseRegularizedIncompleteBeta(OpTestCase):
    """Tests for the inverse_regularized_incomplete_beta function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="inverse_regularized_incomplete_beta",
            func=torchscience.special_functions.inverse_regularized_incomplete_beta,
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
                    name="p",
                    position=2,
                    default_real_range=(0.01, 0.99),
                ),
            ],
            skip_tests={
                "test_gradgradcheck",
                "test_gradgradcheck_complex",
            },
            special_values=[
                SpecialValue(
                    inputs=(2.0, 3.0, 0.0),
                    expected=0.0,
                    description="I^{-1}(a, b, 0) = 0",
                ),
                SpecialValue(
                    inputs=(2.0, 3.0, 1.0),
                    expected=1.0,
                    description="I^{-1}(a, b, 1) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 1.0, 0.5),
                    expected=0.5,
                    rtol=1e-6,
                    atol=1e-6,
                    description="I^{-1}(1, 1, 0.5) = 0.5 (uniform distribution)",
                ),
                SpecialValue(
                    inputs=(3.0, 3.0, 0.5),
                    expected=0.5,
                    rtol=1e-5,
                    atol=1e-6,
                    description="I^{-1}(a, a, 0.5) = 0.5 (symmetric case)",
                ),
            ],
        )
