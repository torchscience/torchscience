import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
    SingularitySpec,
    SpecialValue,
)


class TestModifiedBesselK1(OpTestCase):
    """Tests for the modified_bessel_k_1 function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="modified_bessel_k_1",
            func=torchscience.special_functions.modified_bessel_k_1,
            arity=1,
            input_specs=[
                InputSpec(
                    name="z", position=0, default_real_range=(0.5, 10.0)
                ),
            ],
            special_values=[
                SpecialValue(
                    inputs=(0.0,),
                    expected=float("inf"),
                    description="K_1(0) = +inf (singularity)",
                ),
            ],
            singularities=[
                SingularitySpec(
                    type="pole",
                    locations=lambda: iter([0.0]),
                    expected_behavior="inf",
                    description="Singularity at z=0",
                ),
            ],
            supports_sparse_coo=False,
            supports_sparse_csr=False,
            supports_meta=True,
        )
