import torchscience.special_functions
from torchscience.testing import (
    InputSpec,
    OperatorDescriptor,
    OpTestCase,
)


class TestSphericalHarmonicY(OpTestCase):
    """Tests for the spherical_harmonic_y function."""

    @property
    def descriptor(self) -> OperatorDescriptor:
        return OperatorDescriptor(
            name="spherical_harmonic_y",
            func=torchscience.special_functions.spherical_harmonic_y,
            arity=4,
            input_specs=[
                InputSpec(
                    name="l",
                    position=0,
                    default_real_range=(1.0, 5.0),
                ),
                InputSpec(
                    name="m",
                    position=1,
                    default_real_range=(0.5, 3.0),
                ),
                InputSpec(
                    name="theta",
                    position=2,
                    default_real_range=(0.1, 3.0),
                ),
                InputSpec(
                    name="phi",
                    position=3,
                    default_real_range=(0.1, 6.0),
                ),
            ],
            skip_tests={
                "test_autocast_cpu_bfloat16",
                "test_broadcast_all_different_shapes",
                "test_broadcast_batch_dimensions",
                "test_broadcast_different_shapes",
                "test_broadcast_scalar_with_tensor",
                "test_broadcast_tensor_with_scalar",
                "test_compile_smoke",
                "test_complex_dtypes",
                "test_cpu_device",
                "test_dtype_preservation",
                "test_gradcheck_complex",
                "test_gradcheck_real",
                "test_gradgradcheck_complex",
                "test_gradgradcheck_real",
                "test_inf_handling",
                "test_inf_handling_all_inputs",
                "test_low_precision_dtype_preservation",
                "test_low_precision_forward",
                "test_meta_tensor_large_shape",
                "test_meta_tensor_shape_inference",
                "test_nan_propagation",
                "test_nan_propagation_all_inputs",
                "test_real_dtypes",
                "test_vmap_over_batch",
            },
        )
