import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestInverseRegularizedGammaQ(OpTestCase):
    """Tests for the inverse_regularized_gamma_q function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="inverse_regularized_gamma_q",
            func=torchscience.special_functions.inverse_regularized_gamma_q,
            arity=2,
            input_specs=[
                InputSpec(name="a", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="q", position=1, default_real_range=(0.01, 0.99)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 1.0),
                    expected=0.0,
                    description="Q^{-1}(1, 1) = 0",
                ),
            ],
        )
