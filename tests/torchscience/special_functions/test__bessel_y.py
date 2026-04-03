import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
)


class TestBesselY(OpTestCase):
    """Tests for the Bessel function of the second kind Y_n(z)."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="bessel_y",
            func=torchscience.special_functions.bessel_y,
            arity=2,
            input_specs=[
                InputSpec(name="n", position=0, default_real_range=(0.5, 5.0)),
                InputSpec(
                    name="z", position=1, default_real_range=(0.5, 10.0)
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_low_precision_forward",
            },
        )
