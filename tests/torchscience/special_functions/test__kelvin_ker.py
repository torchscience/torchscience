import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestKelvinKer(OpTestCase):
    """Tests for the kelvin_ker function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="kelvin_ker",
            func=torchscience.special_functions.kelvin_ker,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.5,),
                    expected=0.8559058721186341,
                    description="ker(0.5)",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.2867062087283160,
                    description="ker(1)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=-0.0416645139915096,
                    description="ker(2)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=-0.0115117271994922,
                    description="ker(5)",
                ),
            ],
            skip_tests={
                "test_gradgradcheck_complex",
                "test_low_precision_forward",
            },
        )
