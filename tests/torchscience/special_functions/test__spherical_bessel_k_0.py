import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
)


class TestSphericalBesselK0(OpTestCase):
    """Tests for the spherical_bessel_k_0 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="spherical_bessel_k_0",
            func=torchscience.special_functions.spherical_bessel_k_0,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
            },
        )
