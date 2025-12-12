#include <benchmarks/csrc/macros.h>

#include <torchscience/csrc/impl/special_functions/sin_pi.h>
#include <torchscience/csrc/impl/special_functions/cos_pi.h>

using namespace torchscience::impl::special_functions;

BENCHMARK_UNARY_SUITE(CosPi, cos_pi, cos_pi_backward, std::cos);

// Additional range variants
BENCHMARK_UNARY_REAL(CosPi_SmallRange, cos_pi, double, -0.5, 0.5);
BENCHMARK_UNARY_REAL(CosPi_LargeRange, cos_pi, double, -1000.0, 1000.0);

// Additional complex variants
BENCHMARK_UNARY_COMPLEX(CosPi, cos_pi, float);

static void BM_CosPi_ComplexDouble_SmallImag(benchmark::State& state) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> real_dist(-10.0, 10.0);
    std::uniform_real_distribution<double> imag_dist(-0.1, 0.1);

    std::vector<c10::complex<double>> inputs(BENCH_SIZE);
    for (auto& z : inputs) {
        z = c10::complex<double>(real_dist(gen), imag_dist(gen));
    }

    size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(cos_pi(inputs[i++ & BENCH_MASK]));
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CosPi_ComplexDouble_SmallImag);

static void BM_CosPi_ComplexDouble_PureImag(benchmark::State& state) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<c10::complex<double>> inputs(BENCH_SIZE);
    for (auto& z : inputs) {
        z = c10::complex<double>(0.0, dist(gen));
    }

    size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(cos_pi(inputs[i++ & BENCH_MASK]));
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_CosPi_ComplexDouble_PureImag);

static void BM_StdCos_ComplexDouble_Baseline(benchmark::State& state) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(-10.0 * M_PI, 10.0 * M_PI);

    std::vector<std::complex<double>> inputs(BENCH_SIZE);
    for (auto& z : inputs) {
        z = std::complex<double>(dist(gen), dist(gen));
    }

    size_t i = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(std::cos(inputs[i++ & BENCH_MASK]));
    }

    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_StdCos_ComplexDouble_Baseline);
