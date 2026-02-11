import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestRegularizedGammaQ(OpTestCase):
    """Tests for the regularized_gamma_q function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="regularized_gamma_q",
            func=torchscience.special_functions.regularized_gamma_q,
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
                    expected=1.0,
                    description="Q(1, 0) = 1",
                ),
            ],
        )
