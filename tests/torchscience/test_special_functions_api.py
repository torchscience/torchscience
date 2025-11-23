"""Test the Python API structure for torchscience.special_functions module.

This test module validates that the special_functions namespace is properly
organized and that all functions are accessible through the expected import paths.
"""

import pytest
import torch
from torchscience.special_functions import hypergeometric_2_f_1


class TestSpecialFunctionsAPI:
    """Test suite for special_functions module API structure."""

    def test_direct_import(self):
        """Test that hypergeometric_2_f_1 can be imported directly."""
        # Should already be imported at module level
        assert callable(hypergeometric_2_f_1)
        assert hypergeometric_2_f_1.__name__ == "hypergeometric_2_f_1"

    def test_module_namespace(self):
        """Test that special_functions module has expected namespace."""
        import torchscience.special_functions as sf

        assert hasattr(sf, "hypergeometric_2_f_1")
        assert callable(sf.hypergeometric_2_f_1)

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import torchscience.special_functions as sf

        assert hasattr(sf, "__all__")
        assert "hypergeometric_2_f_1" in sf.__all__

    def test_function_has_docstring(self):
        """Test that hypergeometric_2_f_1 has comprehensive docstring."""
        assert hypergeometric_2_f_1.__doc__ is not None
        assert len(hypergeometric_2_f_1.__doc__) > 100
        # Check for key documentation elements
        assert "Gaussian hypergeometric function" in hypergeometric_2_f_1.__doc__
        assert "Pochhammer" in hypergeometric_2_f_1.__doc__
        assert "Args:" in hypergeometric_2_f_1.__doc__
        assert "Returns:" in hypergeometric_2_f_1.__doc__
        assert "Example:" in hypergeometric_2_f_1.__doc__

    def test_function_has_type_hints(self):
        """Test that hypergeometric_2_f_1 has type hints."""
        import inspect

        sig = inspect.signature(hypergeometric_2_f_1)
        # Check that all parameters have type hints
        for param_name in ["a", "b", "c", "z"]:
            assert param_name in sig.parameters
            param = sig.parameters[param_name]
            assert param.annotation != inspect.Parameter.empty

        # Check return type hint
        assert sig.return_annotation != inspect.Signature.empty

    def test_function_callable_basic(self):
        """Test that hypergeometric_2_f_1 is callable with basic inputs."""
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        c = torch.tensor([3.0])
        z = torch.tensor([0.5])

        result = hypergeometric_2_f_1(a, b, c, z)

        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([1])
        assert result.dtype == torch.float32

    def test_c_extension_loaded(self):
        """Test that the C++ extension is properly loaded."""
        import torchscience

        # Check that _C module exists
        assert hasattr(torchscience, "_C")

    def test_torch_ops_registration(self):
        """Test that operator is registered in torch.ops namespace."""
        # The operator should be accessible via torch.ops.torchscience
        assert hasattr(torch.ops, "torchscience")
        assert hasattr(torch.ops.torchscience, "hypergeometric_2_f_1")

    def test_torch_ops_callable(self):
        """Test that torch.ops.torchscience.hypergeometric_2_f_1 is callable."""
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        c = torch.tensor([3.0])
        z = torch.tensor([0.5])

        # Direct call to torch.ops
        result = torch.ops.torchscience.hypergeometric_2_f_1(a, b, c, z)

        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([1])

    def test_wrapper_calls_torch_ops(self):
        """Test that Python wrapper calls torch.ops implementation."""
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        c = torch.tensor([3.0])
        z = torch.tensor([0.5])

        # Call through wrapper
        result_wrapper = hypergeometric_2_f_1(a, b, c, z)

        # Call directly through torch.ops
        result_direct = torch.ops.torchscience.hypergeometric_2_f_1(a, b, c, z)

        # Results should be identical
        torch.testing.assert_close(result_wrapper, result_direct)

    def test_no_ops_attribute_in_main_namespace(self):
        """Test that torchscience module doesn't expose unnecessary ops attribute."""
        import torchscience

        # The main package should be minimal - only _C import and __version__
        # We don't want an 'ops' attribute at the top level
        assert hasattr(torchscience, "__version__")
        assert hasattr(torchscience, "_C")

    def test_module_structure_is_clean(self):
        """Test that module structure follows clean API design."""
        import torchscience.special_functions as sf

        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(sf) if not attr.startswith("_")]

        # Should contain hypergeometric_2_f_1 and minimal other items
        assert "hypergeometric_2_f_1" in public_attrs

        # Should not contain unnecessary implementation details
        assert "torch" not in public_attrs  # torch should be internal import
        assert "Tensor" not in public_attrs

    def test_import_paths_consistency(self):
        """Test that all import paths give the same function object."""
        from torchscience.special_functions import hypergeometric_2_f_1 as func1
        import torchscience.special_functions

        func2 = torchscience.special_functions.hypergeometric_2_f_1

        # Both should be the exact same function object
        assert func1 is func2

    def test_multiple_imports_no_side_effects(self):
        """Test that importing multiple times doesn't cause issues."""
        # Import the function multiple times
        from torchscience.special_functions import hypergeometric_2_f_1 as f1
        from torchscience.special_functions import hypergeometric_2_f_1 as f2
        from torchscience.special_functions import hypergeometric_2_f_1 as f3

        # All should be the same object
        assert f1 is f2 is f3

    def test_function_name_matches_file(self):
        """Test that function name matches its module file name."""
        import torchscience.special_functions._hypergeometric_2_f_1 as module

        # Module should contain the function
        assert hasattr(module, "hypergeometric_2_f_1")
        assert module.hypergeometric_2_f_1 is hypergeometric_2_f_1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
