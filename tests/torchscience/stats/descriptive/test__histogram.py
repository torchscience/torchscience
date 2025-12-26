"""Tests for torchscience.statistics.descriptive.histogram."""

import torch

import torchscience.statistics.descriptive


class TestHistogramBasic:
    """Basic functionality tests."""

    def test_1d_histogram_shape(self):
        """Test 1D histogram returns correct shapes."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        assert counts.shape == (10,)
        assert edges.shape == (11,)

    def test_1d_histogram_counts_sum(self):
        """Test histogram counts sum to input size."""
        x = torch.randn(1000)
        counts, edges = torchscience.statistics.descriptive.histogram(
            x, bins=10
        )
        assert counts.sum().item() == 1000
