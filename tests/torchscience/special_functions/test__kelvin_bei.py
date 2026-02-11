import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestKelvinBei(OpTestCase):
    """Tests for the kelvin_bei function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="kelvin_bei",
            func=torchscience.special_functions.kelvin_bei,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=0.0,
                    description="bei(0) = 0",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=0.24956604003665972,
                    description="bei(1)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=0.9722916273066612,
                    description="bei(2)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=0.11603438155020042,
                    description="bei(5)",
                ),
            ],
        )
