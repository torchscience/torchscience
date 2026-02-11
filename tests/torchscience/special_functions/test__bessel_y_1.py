import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SingularitySpec,
    SpecialValue,
)


class TestBesselY1(OpTestCase):
    """Tests for the bessel_y_1 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="bessel_y_1",
            func=torchscience.special_functions.bessel_y_1,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=float("-inf"),
                    description="Y_1(0) = -inf (logarithmic singularity)",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: iter([0.0]),
                    expected_behavior="inf",
                    description="Logarithmic singularity at z=0",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_meta=True,
        )
