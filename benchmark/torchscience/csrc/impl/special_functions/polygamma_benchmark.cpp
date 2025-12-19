#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <vector>

#include "impl/special_functions/polygamma.h"

using namespace torchscience::impl::special_functions;

template <typename T>
std::vector<T> generate_random_data(size_t count, T min_val, T max_val, unsigned seed = 42) {
  std::vector<T> data(count);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<T> dist(min_val, max_val);
  for (size_t i = 0; i < count; ++i) {
    data[i] = dist(gen);
  }
  return data;
}

// Order 1 (trigamma) - dispatches to specialized implementation
static void BM_Polygamma_Order1_Float(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.1f, 10.0f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(1, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order1_Float)->Range(64, 4096);

static void BM_Polygamma_Order1_Double(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.1, 10.0);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(1, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order1_Double)->Range(64, 4096);

// Order 2 (tetragamma) - dispatches to specialized implementation
static void BM_Polygamma_Order2_Float(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.1f, 10.0f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(2, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order2_Float)->Range(64, 4096);

static void BM_Polygamma_Order2_Double(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.1, 10.0);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(2, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order2_Double)->Range(64, 4096);

// Order 3 (pentagamma) - dispatches to specialized implementation
static void BM_Polygamma_Order3_Float(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.1f, 10.0f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(3, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order3_Float)->Range(64, 4096);

static void BM_Polygamma_Order3_Double(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.1, 10.0);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(3, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order3_Double)->Range(64, 4096);

// Higher orders - general formula
static void BM_Polygamma_Order5_Float(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.5f, 10.0f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(5, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order5_Float)->Range(64, 4096);

static void BM_Polygamma_Order5_Double(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.5, 10.0);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(5, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order5_Double)->Range(64, 4096);

static void BM_Polygamma_Order10_Float(benchmark::State& state) {
  auto data = generate_random_data<float>(state.range(0), 0.5f, 10.0f);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(10, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order10_Float)->Range(64, 4096);

static void BM_Polygamma_Order10_Double(benchmark::State& state) {
  auto data = generate_random_data<double>(state.range(0), 0.5, 10.0);
  for (auto _ : state) {
    for (const auto& x : data) {
      benchmark::DoNotOptimize(polygamma(10, x));
    }
  }
  state.SetItemsProcessed(state.iterations() * data.size());
}
BENCHMARK(BM_Polygamma_Order10_Double)->Range(64, 4096);

BENCHMARK_MAIN();
