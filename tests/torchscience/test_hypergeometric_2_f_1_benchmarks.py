"""Performance benchmarks for hypergeometric_2_f_1 operator.

This module benchmarks the performance of the hypergeometric function
across different backends, dtypes, and batch sizes.
"""

import time
from typing import Callable, Dict, List, Tuple

import mpmath
import pytest
import torch

from torchscience.special_functions import hypergeometric_2_f_1


def time_function(func: Callable, *args, **kwargs) -> Tuple[float, any]:
    """Time a function execution.

    Args:
        func: Function to time
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Tuple of (execution_time_ms, result)
    """
    # Warmup
    for _ in range(3):
        _ = func(*args, **kwargs)

    # Benchmark
    if torch.cuda.is_available() and any(arg.is_cuda for arg in args if isinstance(arg, torch.Tensor)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
    else:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    return elapsed_ms, result


class TestHypergeometric2F1Benchmarks:
    """Benchmark tests for hypergeometric_2_f_1."""

    def test_benchmark_cpu_single_element(self):
        """Benchmark single element computation on CPU."""
        a = torch.tensor([1.0], dtype=torch.float64)
        b = torch.tensor([2.0], dtype=torch.float64)
        c = torch.tensor([3.0], dtype=torch.float64)
        z = torch.tensor([0.5], dtype=torch.float64)

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z)

        print(f"\nCPU single element: {elapsed_ms:.3f} ms")
        assert elapsed_ms < 10.0, "Single element should compute in < 10ms on CPU"

    @pytest.mark.parametrize("batch_size", [10, 100, 1000, 10000])
    def test_benchmark_cpu_batch_scaling(self, batch_size):
        """Benchmark CPU performance with different batch sizes."""
        a = torch.ones(batch_size, dtype=torch.float64)
        b = torch.ones(batch_size, dtype=torch.float64) * 2.0
        c = torch.ones(batch_size, dtype=torch.float64) * 3.0
        z = torch.ones(batch_size, dtype=torch.float64) * 0.5

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z)

        throughput = batch_size / (elapsed_ms / 1000)  # elements/second
        print(f"\nCPU batch size {batch_size}: {elapsed_ms:.3f} ms ({throughput:.0f} elem/sec)")

        # Basic sanity check - shouldn't be slower than serial computation
        assert elapsed_ms < batch_size * 1.0, f"Batch {batch_size} too slow"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_benchmark_cpu_dtypes(self, dtype):
        """Benchmark CPU performance with different dtypes."""
        batch_size = 1000
        a = torch.ones(batch_size, dtype=dtype)
        b = torch.ones(batch_size, dtype=dtype) * 2.0
        c = torch.ones(batch_size, dtype=dtype) * 3.0
        z = torch.ones(batch_size, dtype=dtype) * 0.5

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z)

        print(f"\nCPU {dtype} batch {batch_size}: {elapsed_ms:.3f} ms")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark_cuda_single_element(self):
        """Benchmark single element computation on CUDA."""
        a = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        b = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        c = torch.tensor([3.0], dtype=torch.float64, device="cuda")
        z = torch.tensor([0.5], dtype=torch.float64, device="cuda")

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z)

        print(f"\nCUDA single element: {elapsed_ms:.3f} ms")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("batch_size", [100, 1000, 10000, 100000])
    def test_benchmark_cuda_batch_scaling(self, batch_size):
        """Benchmark CUDA performance with different batch sizes."""
        a = torch.ones(batch_size, dtype=torch.float64, device="cuda")
        b = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 2.0
        c = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 3.0
        z = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 0.5

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z)

        throughput = batch_size / (elapsed_ms / 1000)  # elements/second
        print(f"\nCUDA batch size {batch_size}: {elapsed_ms:.3f} ms ({throughput:.0f} elem/sec)")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_benchmark_cuda_dtypes(self, dtype):
        """Benchmark CUDA performance with different dtypes."""
        batch_size = 10000
        a = torch.ones(batch_size, dtype=dtype, device="cuda")
        b = torch.ones(batch_size, dtype=dtype, device="cuda") * 2.0
        c = torch.ones(batch_size, dtype=dtype, device="cuda") * 3.0
        z = torch.ones(batch_size, dtype=dtype, device="cuda") * 0.5

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z)

        print(f"\nCUDA {dtype} batch {batch_size}: {elapsed_ms:.3f} ms")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark_cpu_vs_cuda(self):
        """Compare CPU vs CUDA performance."""
        batch_size = 10000

        # CPU
        a_cpu = torch.ones(batch_size, dtype=torch.float64)
        b_cpu = torch.ones(batch_size, dtype=torch.float64) * 2.0
        c_cpu = torch.ones(batch_size, dtype=torch.float64) * 3.0
        z_cpu = torch.ones(batch_size, dtype=torch.float64) * 0.5

        cpu_time, cpu_result = time_function(hypergeometric_2_f_1, a_cpu, b_cpu, c_cpu, z_cpu)

        # CUDA
        a_cuda = a_cpu.cuda()
        b_cuda = b_cpu.cuda()
        c_cuda = c_cpu.cuda()
        z_cuda = z_cpu.cuda()

        cuda_time, cuda_result = time_function(hypergeometric_2_f_1, a_cuda, b_cuda, c_cuda, z_cuda)

        speedup = cpu_time / cuda_time
        print(f"\nCPU time: {cpu_time:.3f} ms")
        print(f"CUDA time: {cuda_time:.3f} ms")
        print(f"Speedup: {speedup:.2f}x")

        # Verify results match
        torch.testing.assert_close(cuda_result.cpu(), cpu_result, rtol=1e-10, atol=1e-12)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_benchmark_mps_performance(self):
        """Benchmark MPS (Apple Silicon) performance."""
        batch_size = 1000

        # MPS
        a_mps = torch.ones(batch_size, dtype=torch.float32, device="mps")
        b_mps = torch.ones(batch_size, dtype=torch.float32, device="mps") * 2.0
        c_mps = torch.ones(batch_size, dtype=torch.float32, device="mps") * 3.0
        z_mps = torch.ones(batch_size, dtype=torch.float32, device="mps") * 0.5

        mps_time, mps_result = time_function(hypergeometric_2_f_1, a_mps, b_mps, c_mps, z_mps)

        print(f"\nMPS batch {batch_size}: {mps_time:.3f} ms")

    def test_benchmark_vs_mpmath(self):
        """Compare performance with mpmath reference implementation."""
        mpmath.mp.dps = 15  # Match float64 precision

        # Single element comparison
        a_val, b_val, c_val, z_val = 1.0, 2.0, 3.0, 0.5

        # TorchScience
        a_t = torch.tensor([a_val], dtype=torch.float64)
        b_t = torch.tensor([b_val], dtype=torch.float64)
        c_t = torch.tensor([c_val], dtype=torch.float64)
        z_t = torch.tensor([z_val], dtype=torch.float64)

        torch_time, torch_result = time_function(hypergeometric_2_f_1, a_t, b_t, c_t, z_t)

        # mpmath
        start = time.perf_counter()
        for _ in range(3):  # Warmup
            _ = mpmath.hyp2f1(a_val, b_val, c_val, z_val)

        start = time.perf_counter()
        mpmath_result = mpmath.hyp2f1(a_val, b_val, c_val, z_val)
        end = time.perf_counter()
        mpmath_time = (end - start) * 1000

        speedup = mpmath_time / torch_time
        print(f"\nTorchScience: {torch_time:.3f} ms")
        print(f"mpmath: {mpmath_time:.3f} ms")
        print(f"Speedup vs mpmath: {speedup:.2f}x")

        # Verify accuracy
        expected = float(mpmath_result)
        torch.testing.assert_close(
            torch_result, torch.tensor([expected], dtype=torch.float64), rtol=1e-10, atol=1e-12
        )

    def test_benchmark_memory_usage(self):
        """Benchmark memory usage for different batch sizes."""
        batch_sizes = [100, 1000, 10000, 100000]
        memory_usage: List[Tuple[int, float]] = []

        for batch_size in batch_sizes:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                a = torch.ones(batch_size, dtype=torch.float64, device="cuda")
                b = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 2.0
                c = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 3.0
                z = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 0.5

                result = hypergeometric_2_f_1(a, b, c, z)

                peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_usage.append((batch_size, peak_memory_mb))

                print(f"\nBatch {batch_size}: {peak_memory_mb:.2f} MB peak GPU memory")
            else:
                pytest.skip("CUDA not available for memory benchmarks")

    @pytest.mark.parametrize(
        "shape",
        [
            (100, 100),
            (1000, 100),
            (100, 1000),
            (10, 10, 10, 10),
        ],
    )
    def test_benchmark_multidimensional(self, shape):
        """Benchmark multidimensional tensor performance."""
        total_elements = torch.tensor(shape).prod().item()

        a = torch.ones(shape, dtype=torch.float64)
        b = torch.ones(shape, dtype=torch.float64) * 2.0
        c = torch.ones(shape, dtype=torch.float64) * 3.0
        z = torch.ones(shape, dtype=torch.float64) * 0.5

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z)

        throughput = total_elements / (elapsed_ms / 1000)
        print(f"\nShape {shape} ({total_elements} elem): {elapsed_ms:.3f} ms ({throughput:.0f} elem/sec)")

    def test_benchmark_gradient_computation(self):
        """Benchmark backward pass performance."""
        batch_size = 1000

        a = torch.ones(batch_size, dtype=torch.float64, requires_grad=True)
        b = torch.ones(batch_size, dtype=torch.float64, requires_grad=True) * 2.0
        c = torch.ones(batch_size, dtype=torch.float64, requires_grad=True) * 3.0
        z = torch.ones(batch_size, dtype=torch.float64, requires_grad=True) * 0.5

        # Forward pass
        forward_start = time.perf_counter()
        result = hypergeometric_2_f_1(a, b, c, z)
        forward_end = time.perf_counter()
        forward_ms = (forward_end - forward_start) * 1000

        # Backward pass
        grad_output = torch.ones_like(result)
        backward_start = time.perf_counter()
        result.backward(grad_output)
        backward_end = time.perf_counter()
        backward_ms = (backward_end - backward_start) * 1000

        total_ms = forward_ms + backward_ms

        print(f"\nForward pass: {forward_ms:.3f} ms")
        print(f"Backward pass: {backward_ms:.3f} ms")
        print(f"Total: {total_ms:.3f} ms")
        print(f"Backward/Forward ratio: {backward_ms/forward_ms:.2f}x")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark_sparse_tensor(self):
        """Benchmark sparse tensor performance."""
        nnz = 10000  # Number of non-zero elements
        size = (1000, 1000)

        # Generate random sparse tensor
        indices = torch.randint(0, size[0], (2, nnz), device="cuda")
        values = torch.rand(nnz, dtype=torch.float64, device="cuda") * 0.9
        z_sparse = torch.sparse_coo_tensor(indices, values, size, device="cuda")

        a = torch.tensor([1.0], dtype=torch.float64, device="cuda")
        b = torch.tensor([2.0], dtype=torch.float64, device="cuda")
        c = torch.tensor([3.0], dtype=torch.float64, device="cuda")

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z_sparse)

        throughput = nnz / (elapsed_ms / 1000)
        print(f"\nSparse tensor ({nnz} nnz): {elapsed_ms:.3f} ms ({throughput:.0f} elem/sec)")

    def test_benchmark_complex_dtype(self):
        """Benchmark complex dtype performance."""
        batch_size = 1000

        a = torch.ones(batch_size, dtype=torch.complex128) * (1.0 + 0.1j)
        b = torch.ones(batch_size, dtype=torch.complex128) * (2.0 + 0.2j)
        c = torch.ones(batch_size, dtype=torch.complex128) * 3.0
        z = torch.ones(batch_size, dtype=torch.complex128) * 0.5

        elapsed_ms, result = time_function(hypergeometric_2_f_1, a, b, c, z)

        print(f"\nComplex128 batch {batch_size}: {elapsed_ms:.3f} ms")

    def test_benchmark_summary(self):
        """Generate comprehensive benchmark summary."""
        print("\n" + "=" * 70)
        print("HYPERGEOMETRIC_2_F_1 PERFORMANCE SUMMARY")
        print("=" * 70)

        benchmarks: Dict[str, float] = {}

        # CPU benchmarks
        for batch_size in [100, 1000, 10000]:
            a = torch.ones(batch_size, dtype=torch.float64)
            b = torch.ones(batch_size, dtype=torch.float64) * 2.0
            c = torch.ones(batch_size, dtype=torch.float64) * 3.0
            z = torch.ones(batch_size, dtype=torch.float64) * 0.5

            elapsed_ms, _ = time_function(hypergeometric_2_f_1, a, b, c, z)
            throughput = batch_size / (elapsed_ms / 1000)
            benchmarks[f"CPU_batch_{batch_size}"] = throughput

        # CUDA benchmarks if available
        if torch.cuda.is_available():
            for batch_size in [100, 1000, 10000, 100000]:
                a = torch.ones(batch_size, dtype=torch.float64, device="cuda")
                b = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 2.0
                c = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 3.0
                z = torch.ones(batch_size, dtype=torch.float64, device="cuda") * 0.5

                elapsed_ms, _ = time_function(hypergeometric_2_f_1, a, b, c, z)
                throughput = batch_size / (elapsed_ms / 1000)
                benchmarks[f"CUDA_batch_{batch_size}"] = throughput

        # Print summary table
        print(f"\n{'Benchmark':<30} {'Throughput (elem/sec)':<25}")
        print("-" * 70)
        for name, throughput in sorted(benchmarks.items()):
            print(f"{name:<30} {throughput:>20,.0f}")

        print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
