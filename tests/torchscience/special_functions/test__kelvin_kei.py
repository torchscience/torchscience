import math

import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SpecialValue,
)


class TestKelvinKei(OpTestCase):
    """Tests for the kelvin_kei function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="kelvin_kei",
            func=torchscience.special_functions.kelvin_kei,
            arity=1,
            input_specs=[
                InputSpec(
                    name="x", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=-math.pi / 4,
                    description="kei(0) = -pi/4",
                    rtol=1e-6,
                    atol=1e-6,
                ),
                SpecialValue(
                    inputs=(0.5,),
                    expected=-0.6715816950943676,
                    description="kei(0.5)",
                ),
                SpecialValue(
                    inputs=(1.0,),
                    expected=-0.49499463651872,
                    description="kei(1)",
                ),
                SpecialValue(
                    inputs=(2.0,),
                    expected=-0.20240006776470432,
                    description="kei(2)",
                ),
                SpecialValue(
                    inputs=(5.0,),
                    expected=0.01118758650986929,
                    description="kei(5)",
                ),
            ],
        )
