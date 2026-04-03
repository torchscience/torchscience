import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestStruveH0(OpTestCase):
    """Tests for the struve_h_0 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="struve_h_0",
            func=torchscience.special_functions.struve_h_0,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.0,
                    description="H_0(0) = 0",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
