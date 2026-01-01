"""Interpolants for dense output from ODE solvers."""

from typing import Optional, Union

import torch


class LinearInterpolant:
    """
    Linear interpolant for ODE dense output.

    Uses piecewise linear interpolation between grid points.
    The interpolant is differentiable and supports backpropagation.

    Parameters
    ----------
    t_points : Tensor
        Time points, shape (N,), must be monotonically increasing.
    y_points : Tensor
        State values at time points, shape (N, *state_shape).
    success : Tensor, optional
        Boolean mask indicating which batch elements succeeded.
        Shape (*batch_shape,). Only set when throw=False.
    """

    def __init__(
        self,
        t_points: torch.Tensor,
        y_points: torch.Tensor,
        success: Optional[torch.Tensor] = None,
    ):
        self.t_points = t_points
        self.y_points = y_points
        self.success = success
        self._t_min = t_points[0].item()
        self._t_max = t_points[-1].item()

    def __call__(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the interpolant at time(s) t.

        Parameters
        ----------
        t : float or Tensor
            Time(s) to query. Scalar or 1D tensor.

        Returns
        -------
        y : Tensor
            State at time(s) t.
            If t is scalar: shape (*state_shape) or (*batch_shape, *state_shape)
            If t is 1D tensor of length T: shape (T, *batch_shape, *state_shape)
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(
                t, dtype=self.t_points.dtype, device=self.t_points.device
            )

        scalar_query = t.dim() == 0
        if scalar_query:
            t = t.unsqueeze(0)

        # Validate bounds
        t_min_query = t.min().item()
        t_max_query = t.max().item()
        if (
            t_min_query < self._t_min - 1e-6
            or t_max_query > self._t_max + 1e-6
        ):
            raise ValueError(
                f"Query time(s) outside interpolant range [{self._t_min}, {self._t_max}]"
            )

        # Find interval for each query point
        indices = torch.searchsorted(self.t_points, t.contiguous())
        indices = indices.clamp(1, len(self.t_points) - 1)

        # Get interval endpoints
        t0 = self.t_points[indices - 1]
        t1 = self.t_points[indices]
        y0 = self.y_points[indices - 1]
        y1 = self.y_points[indices]

        # Linear interpolation weight
        h = t1 - t0
        alpha = (t - t0) / h

        # Expand alpha for broadcasting with state dimensions
        state_dims = y0.dim() - 1
        for _ in range(state_dims):
            alpha = alpha.unsqueeze(-1)

        # Linear interpolation
        y = (1 - alpha) * y0 + alpha * y1

        if scalar_query:
            y = y.squeeze(0)

        return y


# DP5 dense output coefficients
# Reference: SciPy RK45 implementation (scipy.integrate._ivp.rk)
# The dense output formula is:
#   y(t0 + theta*h) = y0 + h * sum_i b_i(theta) * k_i
# where b_i(theta) = P[i, 0]*theta + P[i, 1]*theta^2 + P[i, 2]*theta^3 + P[i, 3]*theta^4
#
# At theta=0: all b_i(0) = 0, so y = y0 (correct)
# At theta=1: b_i(1) = B5[i] (the 5th order weights), so y = y1 (correct)
#
# P matrix shape: (7 stages, 4 powers of theta)
# fmt: off
_DP5_P = [
    [1.0, -2.8535800730693277, 3.0717434625687095, -1.1270175686556618],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 4.023133377671971, -6.249321571474188, 2.6754244809482835],
    [0.0, -3.7324019591773413, 10.068970588235294, -5.6855269578294915],
    [0.0, 2.554803831366355, -6.399112381036449, 3.521932369440949],
    [0.0, -1.3744241095453652, 3.2726577470945877, -1.7672812582195055],
    [0.0, 1.3824689314366552, -3.7649378599018604, 2.382468931436655],
]
# fmt: on


class DP5Interpolant:
    """
    Dormand-Prince 5 dense output interpolant.

    Uses the 7 RK stages to construct a 4th-order accurate interpolant.
    This is more accurate than generic cubic Hermite interpolation.

    Parameters
    ----------
    t_segments : Tensor
        Time segment endpoints, shape (n_segments, 2).
    y_segments : Tensor
        State values at segment endpoints, shape (n_segments, 2, *state_shape).
    k_segments : Tensor
        RK stages for each segment, shape (n_segments, 7, *state_shape).
    success : Tensor, optional
        Boolean mask indicating which batch elements succeeded.
    """

    def __init__(
        self,
        t_segments: torch.Tensor,
        y_segments: torch.Tensor,
        k_segments: torch.Tensor,
        success: Optional[torch.Tensor] = None,
    ):
        self._t_segments = t_segments
        self._y_segments = y_segments
        self._k_segments = k_segments
        self.success = success
        self.n_steps = len(t_segments)

        # Cache bounds
        self._t_min = t_segments[0, 0].item()
        self._t_max = t_segments[-1, 1].item()
        # Dtype-aware tolerance
        self._tol = 100 * torch.finfo(t_segments.dtype).eps

        # Precompute dense output coefficients for this dtype/device
        self._P = torch.tensor(
            _DP5_P, dtype=t_segments.dtype, device=t_segments.device
        )

    @property
    def t_points(self) -> torch.Tensor:
        """Return all unique time points (for testing step sizes)."""
        # Concatenate start of first segment with all segment ends
        t_starts = self._t_segments[:, 0]
        t_end = self._t_segments[-1, 1].unsqueeze(0)
        return torch.cat([t_starts, t_end])

    def __call__(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate the interpolant at time(s) t.

        Parameters
        ----------
        t : float or Tensor
            Time(s) to query. Scalar or 1D tensor.

        Returns
        -------
        y : Tensor
            State at time(s) t.
        """
        if isinstance(t, (int, float)):
            t = torch.tensor(
                t, dtype=self._t_segments.dtype, device=self._t_segments.device
            )

        scalar_query = t.dim() == 0
        if scalar_query:
            t = t.unsqueeze(0)

        # Validate bounds
        t_min_query = t.min().item()
        t_max_query = t.max().item()
        if (
            t_min_query < self._t_min - self._tol
            or t_max_query > self._t_max + self._tol
        ):
            raise ValueError(
                f"Query time(s) outside interpolant range [{self._t_min}, {self._t_max}]"
            )

        # Find segment for each query point
        t_ends = self._t_segments[:, 1]
        seg_indices = torch.searchsorted(t_ends, t.contiguous())
        seg_indices = seg_indices.clamp(0, len(self._t_segments) - 1)

        # Get segment data
        t0 = self._t_segments[seg_indices, 0]
        t1 = self._t_segments[seg_indices, 1]
        y0 = self._y_segments[seg_indices, 0]
        k = self._k_segments[seg_indices]  # (n_query, 7, *state_shape)

        # Compute normalized position theta in [0, 1]
        h = t1 - t0
        theta = (t - t0) / h

        # Expand for broadcasting
        state_dims = y0.dim() - 1
        for _ in range(state_dims):
            theta = theta.unsqueeze(-1)
            h = h.unsqueeze(-1)

        # Dense output formula:
        # y(t0 + theta*h) = y0 + h * sum_i b_i(theta) * k_i
        # where b_i(theta) = P[i, 0]*theta + P[i, 1]*theta^2 + P[i, 2]*theta^3 + P[i, 3]*theta^4

        theta2 = theta * theta
        theta3 = theta2 * theta
        theta4 = theta3 * theta
        theta_powers = torch.stack([theta, theta2, theta3, theta4], dim=-1)

        # Start with y0
        y = y0.clone()

        # Add contribution from each stage
        for i in range(7):
            # b_i(theta) = P[i, 0]*theta + P[i, 1]*theta^2 + P[i, 2]*theta^3 + P[i, 3]*theta^4
            coeffs = self._P[i]  # (4,)
            b_i = (theta_powers * coeffs).sum(
                dim=-1
            )  # (*query_shape, *state_shape)
            y = y + h * b_i * k[:, i]

        if scalar_query:
            y = y.squeeze(0)

        return y
