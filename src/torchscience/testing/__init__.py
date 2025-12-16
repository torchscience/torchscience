"""PyTorch operator testing framework.

This module provides a reusable framework for testing PyTorch operators with
comprehensive coverage of PyTorch features and mathematical properties.

Example usage:

    from tests.torchscience.testing import (
        UnaryOpTestCase,
        OperatorDescriptor,
        InputSpec,
        ToleranceConfig,
        RecurrenceSpec,
        IdentitySpec,
        SpecialValue,
    )
    import sympy

    class TestMyOperator(UnaryOpTestCase):
        @property
        def descriptor(self):
            return OperatorDescriptor(
                name="my_operator",
                func=my_operator_func,
                arity=1,
                sympy_func=sympy.my_operator,
                input_specs=[
                    InputSpec(name="x", default_real_range=(0.1, 10.0)),
                ],
            )
"""

from .base import (
    BinaryOpTestCase,
    OpTestCase,
    UnaryOpTestCase,
)
from .descriptors import (
    IdentitySpec,
    InputSpec,
    OperatorDescriptor,
    RecurrenceSpec,
    SingularitySpec,
    SpecialValue,
    ToleranceConfig,
)
from .strategies import (
    all_complex_dtypes,
    all_dtypes,
    all_floating_dtypes,
    # Device strategies
    available_devices,
    avoiding_poles,
    avoiding_values,
    broadcastable_shapes,
    broadcastable_tensor_pairs,
    chebyshev_domain,
    complex_avoiding_real_axis,
    complex_dtypes,
    # Complex strategies
    complex_numbers,
    hyperbolic_domain,
    integers_as_floats,
    low_precision_dtypes,
    negative_reals,
    non_integer_reals,
    # Numeric strategies
    positive_reals,
    # Dtype strategies
    real_dtypes,
    reals,
    recurrence_test_values,
    reflection_test_values,
    # Tensor strategies
    shapes,
    tensors,
)
from .sympy_utils import (
    SymbolicDerivativeVerifier,
    SymPyReference,
)

__all__ = [
    # Base classes
    "OpTestCase",
    "UnaryOpTestCase",
    "BinaryOpTestCase",
    # Descriptors
    "OperatorDescriptor",
    "InputSpec",
    "ToleranceConfig",
    "RecurrenceSpec",
    "IdentitySpec",
    "SingularitySpec",
    "SpecialValue",
    # Strategies - numeric
    "positive_reals",
    "negative_reals",
    "reals",
    "non_integer_reals",
    "integers_as_floats",
    "chebyshev_domain",
    "hyperbolic_domain",
    "avoiding_values",
    "avoiding_poles",
    # Strategies - complex
    "complex_numbers",
    "complex_avoiding_real_axis",
    # Strategies - tensor
    "shapes",
    "tensors",
    "broadcastable_shapes",
    "broadcastable_tensor_pairs",
    "recurrence_test_values",
    "reflection_test_values",
    # Strategies - dtype
    "real_dtypes",
    "complex_dtypes",
    "low_precision_dtypes",
    "all_floating_dtypes",
    "all_complex_dtypes",
    "all_dtypes",
    # Strategies - device
    "available_devices",
    # SymPy utilities
    "SymPyReference",
    "SymbolicDerivativeVerifier",
]
