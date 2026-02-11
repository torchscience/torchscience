import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestKelvinBer(OpTestCase):
    """Tests for the kelvin_ber function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="kelvin_ber",
            func=torchscience.special_functions.kelvin_ber,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=1.0,
                    description="ber(0) = 1",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.984381781213087,
                    description="ber(1)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=0.751734182713808,
                    description="ber(2)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=-6.230082478666358,
                    description="ber(5)",
                ),
            ],
        )
