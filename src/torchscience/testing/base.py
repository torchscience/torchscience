from abc import ABC, abstractmethod
from typing import Tuple

import torch

from .descriptors import InputSpec, OperatorDescriptor
from .mixins import (
    AutocastMixin,
    AutogradMixin,
    BroadcastingMixin,
    DeviceMixin,
    DtypeMixin,
    IdentityMixin,
    LowPrecisionMixin,
    MetaTensorMixin,
    NanInfMixin,
    QuantizedMixin,
    RecurrenceMixin,
    SingularityMixin,
    SparseMixin,
    SpecialValueMixin,
    SymPyReferenceMixin,
    TorchCompileMixin,
    VmapMixin,
)


class OpTestCase(
    ABC,
    AutogradMixin,
    DeviceMixin,
    DtypeMixin,
    LowPrecisionMixin,
    BroadcastingMixin,
    TorchCompileMixin,
    VmapMixin,
    SparseMixin,
    QuantizedMixin,
    MetaTensorMixin,
    AutocastMixin,
    NanInfMixin,
    SymPyReferenceMixin,
    RecurrenceMixin,
    IdentityMixin,
    SpecialValueMixin,
    SingularityMixin,
):
    """Base test case for PyTorch operators."""

    @property
    @abstractmethod
    def descriptor(self) -> OperatorDescriptor:
        """Return the operator descriptor."""
        ...

    def _make_standard_inputs(
        self,
        dtype: torch.dtype = torch.float64,
        device: str = "cpu",
        shape: Tuple[int, ...] = (5,),
    ) -> Tuple[torch.Tensor, ...]:
        """Generate standard test inputs based on descriptor."""
        inputs = []
        for spec in self.descriptor.input_specs:
            tensor = self._make_input_for_spec(spec, dtype, device, shape)
            inputs.append(tensor)
        return tuple(inputs)

    def _make_input_for_spec(
        self,
        spec: InputSpec,
        dtype: torch.dtype,
        device: str,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Generate input tensor based on InputSpec."""
        low, high = spec.default_real_range

        if dtype.is_complex:
            # Generate complex tensor
            real_dtype = (
                torch.float32 if dtype == torch.complex64 else torch.float64
            )
            real = (
                torch.rand(shape, dtype=real_dtype, device=device)
                * (high - low)
                + low
            )
            imag_low, imag_high = spec.default_imag_range
            imag = (
                torch.rand(shape, dtype=real_dtype, device=device)
                * (imag_high - imag_low)
                + imag_low
            )
            tensor = torch.complex(real, imag)
        else:
            tensor = (
                torch.rand(shape, dtype=dtype, device=device) * (high - low)
                + low
            )

        # Filter out excluded values
        for excluded in spec.excluded_values:
            mask = (
                torch.abs(
                    tensor.real if dtype.is_complex else tensor - excluded
                )
                < 0.1
            )
            tensor = torch.where(mask, tensor + 0.2, tensor)

        return tensor
