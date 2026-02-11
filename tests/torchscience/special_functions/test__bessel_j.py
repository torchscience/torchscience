import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestBesselJ(OpTestCase):
    """Tests for the Bessel function of the first kind J_n(z)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="bessel_j",
            func=torchscience.special_functions.bessel_j,
            arity=2,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="z", position=1, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0, 0.0),
                    expected=1.0,
                    description="J_0(0) = 1",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=0.0,
                    description="J_1(0) = 0",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0),
                    expected=0.0,
                    description="J_2(0) = 0",
                ),
            ],
        )
