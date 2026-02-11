import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestInverseRegularizedGammaP(OpTestCase):
    """Tests for the inverse_regularized_gamma_p function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="inverse_regularized_gamma_p",
            func=torchscience.special_functions.inverse_regularized_gamma_p,
            arity=2,
            input_specs=[
                InputSpec(name="a", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="p", position=1, default_real_range=(0.01, 0.99)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=0.0,
                    description="P^{-1}(1, 0) = 0",
                ),
            ],
        )
