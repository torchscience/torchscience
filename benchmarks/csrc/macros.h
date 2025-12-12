#pragma once

#include <benchmark/benchmark.h>

#include <c10/util/complex.h>

#include <cmath>
#include <random>
#include <vector>

constexpr size_t BENCH_SIZE = 1024;
constexpr size_t BENCH_MASK = BENCH_SIZE - 1;

// ============================================================================
// Unary function benchmarks
// ============================================================================

#define BENCHMARK_UNARY_REAL(NAME, FUNC, TYPE, MIN, MAX)                       \
  static void BM_##NAME##_##TYPE(benchmark::State& state) {                    \
    std::mt19937 gen(42);                                                      \
    std::uniform_real_distribution<TYPE> dist(MIN, MAX);                       \
    std::vector<TYPE> inputs(BENCH_SIZE);                                      \
    for (auto& x : inputs) x = dist(gen);                                      \
    size_t i = 0;                                                              \
    for (auto _ : state) {                                                     \
      benchmark::DoNotOptimize(FUNC(inputs[i++ & BENCH_MASK]));                \
    }                                                                          \
    state.SetItemsProcessed(state.iterations());                               \
  }                                                                            \
  BENCHMARK(BM_##NAME##_##TYPE)

#define BENCHMARK_UNARY_DOUBLE(NAME, FUNC)                                     \
  BENCHMARK_UNARY_REAL(NAME, FUNC, double, -10.0, 10.0)

#define BENCHMARK_UNARY_FLOAT(NAME, FUNC)                                      \
  BENCHMARK_UNARY_REAL(NAME, FUNC, float, -10.0f, 10.0f)

#define BENCHMARK_UNARY_COMPLEX(NAME, FUNC, TYPE)                              \
  static void BM_##NAME##_Complex##TYPE(benchmark::State& state) {             \
    std::mt19937 gen(42);                                                      \
    std::uniform_real_distribution<TYPE> dist(-10.0, 10.0);                    \
    std::vector<c10::complex<TYPE>> inputs(BENCH_SIZE);                        \
    for (auto& z : inputs) z = c10::complex<TYPE>(dist(gen), dist(gen));       \
    size_t i = 0;                                                              \
    for (auto _ : state) {                                                     \
      benchmark::DoNotOptimize(FUNC(inputs[i++ & BENCH_MASK]));                \
    }                                                                          \
    state.SetItemsProcessed(state.iterations());                               \
  }                                                                            \
  BENCHMARK(BM_##NAME##_Complex##TYPE)

#define BENCHMARK_UNARY_COMPLEX_DOUBLE(NAME, FUNC)                             \
  BENCHMARK_UNARY_COMPLEX(NAME, FUNC, double)

// ============================================================================
// Standard library baseline benchmarks
// ============================================================================

#define BENCHMARK_STD_BASELINE(NAME, STD_FUNC)                                 \
  static void BM_Std##NAME##_Double_Baseline(benchmark::State& state) {        \
    std::mt19937 gen(42);                                                      \
    std::uniform_real_distribution<double> dist(-10.0 * M_PI, 10.0 * M_PI);    \
    std::vector<double> inputs(BENCH_SIZE);                                    \
    for (auto& x : inputs) x = dist(gen);                                      \
    size_t i = 0;                                                              \
    for (auto _ : state) {                                                     \
      benchmark::DoNotOptimize(STD_FUNC(inputs[i++ & BENCH_MASK]));            \
    }                                                                          \
    state.SetItemsProcessed(state.iterations());                               \
  }                                                                            \
  BENCHMARK(BM_Std##NAME##_Double_Baseline)

// ============================================================================
// Complete unary benchmark suite
// ============================================================================

#define BENCHMARK_UNARY_SUITE(NAME, FUNC, BACKWARD_FUNC, STD_FUNC)             \
  BENCHMARK_UNARY_DOUBLE(NAME, FUNC);                                          \
  BENCHMARK_UNARY_FLOAT(NAME, FUNC);                                           \
  BENCHMARK_UNARY_DOUBLE(NAME##Backward, BACKWARD_FUNC);                       \
  BENCHMARK_STD_BASELINE(NAME, STD_FUNC);                                      \
  BENCHMARK_UNARY_COMPLEX_DOUBLE(NAME, FUNC);                                  \
  BENCHMARK_UNARY_COMPLEX_DOUBLE(NAME##Backward, BACKWARD_FUNC)

// ============================================================================
// Binary function benchmarks
// ============================================================================

#define BENCHMARK_BINARY_REAL(NAME, FUNC, TYPE, MIN1, MAX1, MIN2, MAX2)        \
  static void BM_##NAME##_##TYPE(benchmark::State& state) {                    \
    std::mt19937 gen(42);                                                      \
    std::uniform_real_distribution<TYPE> dist1(MIN1, MAX1);                    \
    std::uniform_real_distribution<TYPE> dist2(MIN2, MAX2);                    \
    std::vector<TYPE> inputs1(BENCH_SIZE), inputs2(BENCH_SIZE);                \
    for (size_t j = 0; j < BENCH_SIZE; ++j) {                                  \
      inputs1[j] = dist1(gen);                                                 \
      inputs2[j] = dist2(gen);                                                 \
    }                                                                          \
    size_t i = 0;                                                              \
    for (auto _ : state) {                                                     \
      benchmark::DoNotOptimize(FUNC(inputs1[i & BENCH_MASK],                   \
                                    inputs2[i & BENCH_MASK]));                 \
      ++i;                                                                     \
    }                                                                          \
    state.SetItemsProcessed(state.iterations());                               \
  }                                                                            \
  BENCHMARK(BM_##NAME##_##TYPE)

#define BENCHMARK_BINARY_DOUBLE(NAME, FUNC, MIN1, MAX1, MIN2, MAX2)            \
  BENCHMARK_BINARY_REAL(NAME, FUNC, double, MIN1, MAX1, MIN2, MAX2)

#define BENCHMARK_BINARY_FLOAT(NAME, FUNC, MIN1, MAX1, MIN2, MAX2)             \
  BENCHMARK_BINARY_REAL(NAME, FUNC, float,                                     \
                        static_cast<float>(MIN1), static_cast<float>(MAX1),    \
                        static_cast<float>(MIN2), static_cast<float>(MAX2))

#define BENCHMARK_STD_BINARY_BASELINE(NAME, STD_FUNC, MIN1, MAX1, MIN2, MAX2)  \
  static void BM_Std##NAME##_Double_Baseline(benchmark::State& state) {        \
    std::mt19937 gen(42);                                                      \
    std::uniform_real_distribution<double> dist1(MIN1, MAX1);                  \
    std::uniform_real_distribution<double> dist2(MIN2, MAX2);                  \
    std::vector<double> inputs1(BENCH_SIZE), inputs2(BENCH_SIZE);              \
    for (size_t j = 0; j < BENCH_SIZE; ++j) {                                  \
      inputs1[j] = dist1(gen);                                                 \
      inputs2[j] = dist2(gen);                                                 \
    }                                                                          \
    size_t i = 0;                                                              \
    for (auto _ : state) {                                                     \
      benchmark::DoNotOptimize(STD_FUNC(inputs1[i & BENCH_MASK],               \
                                        inputs2[i & BENCH_MASK]));             \
      ++i;                                                                     \
    }                                                                          \
    state.SetItemsProcessed(state.iterations());                               \
  }                                                                            \
  BENCHMARK(BM_Std##NAME##_Double_Baseline)
