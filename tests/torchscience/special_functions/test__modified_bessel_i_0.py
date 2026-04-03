import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestModifiedBesselI0(OpTestCase):
    """Tests for the modified_bessel_i_0 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="modified_bessel_i_0",
            func=torchscience.special_functions.modified_bessel_i_0,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(0.1, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,), expected=1.0, description="I_0(0) = 1"
                ),
            ],
            skip_tests={
                "test_gradcheck_complex",
                "test_gradgradcheck_complex",
            },
        )
