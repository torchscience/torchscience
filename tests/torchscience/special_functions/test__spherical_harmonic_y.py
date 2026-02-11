import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
)


class TestSphericalHarmonicY(OpTestCase):
    """Tests for the spherical_harmonic_y function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="spherical_harmonic_y",
            func=torchscience.special_functions.spherical_harmonic_y,
            arity=4,
            input_specs=[
                InputSpec(
                    name="l",
                    position=0,
                    default_real_range=(1.0, 5.0),
                ),
                InputSpec(
                    name="m",
                    position=1,
                    default_real_range=(0.5, 3.0),
                ),
                InputSpec(
                    name="theta",
                    position=2,
                    default_real_range=(0.1, 3.0),
                ),
                InputSpec(
                    name="phi",
                    position=3,
                    default_real_range=(0.1, 6.0),
                ),
            ],
        )
