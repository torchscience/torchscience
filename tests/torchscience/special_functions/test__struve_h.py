import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestStruveH(OpTestCase):
    """Tests for the struve_h function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="struve_h",
            func=torchscience.special_functions.struve_h,
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
                    expected=0.0,
                    description="H_0(0) = 0",
                ),
                SpecialValue(
                    inputs=(1.0, 0.0),
                    expected=0.0,
                    description="H_1(0) = 0",
                ),
                SpecialValue(
                    inputs=(2.0, 0.0),
                    expected=0.0,
                    description="H_2(0) = 0",
                ),
            ],
        )
