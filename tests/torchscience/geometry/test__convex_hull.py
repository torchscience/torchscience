# tests/torchscience/geometry/test__convex_hull.py
import torch

import torchscience  # noqa: F401 - needed to register C++ operators


class TestConvexHullImport:
    """Tests for convex_hull import."""

    def test_convex_hull_importable(self):
        """convex_hull function is importable."""
        from torchscience.geometry import convex_hull

        assert callable(convex_hull)

    def test_convex_hull_class_importable(self):
        """ConvexHull class is importable."""
        from torchscience.geometry import ConvexHull

        assert ConvexHull is not None


class TestConvexHullSchema:
    """Tests for convex_hull operator schema."""

    def test_operator_exists(self):
        """convex_hull operator is registered."""
        assert hasattr(torch.ops.torchscience, "convex_hull")


class TestConvexHullMeta:
    """Tests for convex_hull meta tensor support."""

    def test_meta_shape_inference(self):
        """Meta tensors produce correct output shapes."""
        points = torch.randn(100, 3, device="meta")
        result = torch.ops.torchscience.convex_hull(points)

        # Unpack tuple
        (
            vertices,
            simplices,
            neighbors,
            equations,
            area,
            volume,
            n_vertices,
            n_facets,
        ) = result

        # Check shapes (max_vertices and max_facets are upper bounds)
        assert vertices.dim() == 1  # (max_vertices,)
        assert simplices.dim() == 2  # (max_facets, n_dims)
        assert simplices.shape[1] == 3  # n_dims
        assert equations.dim() == 2  # (max_facets, n_dims + 1)
        assert equations.shape[1] == 4  # n_dims + 1
        assert area.shape == ()  # scalar
        assert volume.shape == ()  # scalar

    def test_meta_batched_shape_inference(self):
        """Batched meta tensors produce correct output shapes."""
        points = torch.randn(4, 100, 3, device="meta")
        result = torch.ops.torchscience.convex_hull(points)

        (
            vertices,
            simplices,
            neighbors,
            equations,
            area,
            volume,
            n_vertices,
            n_facets,
        ) = result

        assert vertices.shape[0] == 4  # batch
        assert area.shape == (4,)
        assert volume.shape == (4,)


class TestConvexHull2D:
    """Tests for 2D convex hull."""

    def test_square_hull(self):
        """Unit square hull has 4 vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],  # interior point
            ]
        )
        result = torch.ops.torchscience.convex_hull(points)
        (
            vertices,
            simplices,
            neighbors,
            equations,
            area,
            volume,
            n_vertices,
            n_facets,
        ) = result

        assert n_vertices.item() == 4
        assert n_facets.item() == 4  # 4 edges in 2D

    def test_triangle_hull(self):
        """Triangle hull has 3 vertices."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )
        result = torch.ops.torchscience.convex_hull(points)
        _, _, _, _, _, _, n_vertices, n_facets = result

        assert n_vertices.item() == 3
        assert n_facets.item() == 3

    def test_area_square(self):
        """Unit square has area 1.0."""
        points = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        result = torch.ops.torchscience.convex_hull(points)
        _, _, _, _, _, volume, _, _ = result

        # In 2D, "volume" is area
        torch.testing.assert_close(volume, torch.tensor(1.0))
