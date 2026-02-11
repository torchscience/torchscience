import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
)


class TestWeierstrassEta(OpTestCase):
    """Tests for the weierstrass_eta function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="weierstrass_eta",
            func=torchscience.special_functions.weierstrass_eta,
            arity=2,
            input_specs=[
                InputSpec(
                    name="g2", position=0, default_real_range=(0.5, 5.0)
                ),
                InputSpec(
                    name="g3", position=1, default_real_range=(0.5, 5.0)
                ),
            ],
            special_values=[],
        )
