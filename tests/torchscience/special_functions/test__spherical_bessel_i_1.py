import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestSphericalBesselI1(OpTestCase):
    """Tests for the spherical_bessel_i_1 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="spherical_bessel_i_1",
            func=torchscience.special_functions.spherical_bessel_i_1,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(0.1, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,), expected=0.0, description="i_1(0) = 0"
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
            },
        )
