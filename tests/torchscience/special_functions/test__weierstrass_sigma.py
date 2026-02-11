import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestWeierstrassSigma(OpTestCase):
    """Tests for the weierstrass_sigma function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="weierstrass_sigma",
            func=torchscience.special_functions.weierstrass_sigma,
            arity=3,
            input_specs=[
                InputSpec(name="z", position=0, default_real_range=(0.5, 3.0)),
                InputSpec(
                    name="g2", position=1, default_real_range=(0.5, 5.0)
                ),
                InputSpec(
                    name="g3", position=2, default_real_range=(0.5, 5.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 1.0, 0.5),
                    expected=0.0,
                    description="sigma(0; g2, g3) = 0",
                ),
            ],
        )
