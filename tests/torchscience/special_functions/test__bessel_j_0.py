import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestBesselJ0(OpTestCase):
    """Tests for the bessel_j_0 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="bessel_j_0",
            func=torchscience.special_functions.bessel_j_0,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,), expected=1.0, description="J_0(0) = 1"
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
