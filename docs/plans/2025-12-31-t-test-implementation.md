# T-Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement one-sample, two-sample, and paired t-test functions with C++ backends.

**Architecture:** Kernel layer provides core math (t-statistic, p-value via incomplete_beta). CPU layer handles tensor dispatch and batching along last dimension. No autograd support (hypothesis tests don't need backprop).

**Tech Stack:** C++17, PyTorch ATen, existing `incomplete_beta` kernel

---

## Task 1: Kernel - t_test_common.h (shared utilities)

**Files:**
- Create: `src/torchscience/csrc/kernel/statistics/hypothesis_test/t_test_common.h`

**Step 1: Create the kernel file with t-distribution p-value computation**

```cpp
// src/torchscience/csrc/kernel/statistics/hypothesis_test/t_test_common.h
#pragma once

#include <cmath>
#include <limits>
#include <string_view>

#include <c10/macros/Macros.h>

#include "../../special_functions/incomplete_beta.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Compute p-value from t-statistic and degrees of freedom.
 *
 * Uses the relationship between t-distribution CDF and incomplete beta:
 *   CDF(t, df) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 0.5)  for t >= 0
 *   CDF(t, df) = 0.5 * I_{df/(df+t^2)}(df/2, 0.5)      for t < 0
 *
 * @param t_stat The t-statistic
 * @param df Degrees of freedom (can be non-integer for Welch's)
 * @param alternative 0=two-sided, 1=less, 2=greater
 * @return p-value
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
T t_pvalue(T t_stat, T df, int alternative) {
  if (std::isnan(t_stat) || std::isnan(df) || df <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Compute x = df / (df + t^2) for incomplete beta
  T t_sq = t_stat * t_stat;
  T x = df / (df + t_sq);

  // One-tail p-value using incomplete beta
  // p_one_tail = 0.5 * I_x(df/2, 0.5)
  T half = T(0.5);
  T ibeta_val = special_functions::incomplete_beta(x, df * half, half);
  T p_one_tail = half * ibeta_val;

  // Handle alternative hypothesis
  if (alternative == 0) {
    // two-sided: p = 2 * min(p_one_tail, 1 - p_one_tail)
    // Simplified: p = 2 * p_one_tail (since we compute lower tail)
    return T(2) * p_one_tail;
  } else if (alternative == 1) {
    // less: P(T < t)
    if (t_stat < T(0)) {
      return p_one_tail;
    } else {
      return T(1) - p_one_tail;
    }
  } else {
    // greater: P(T > t)
    if (t_stat > T(0)) {
      return p_one_tail;
    } else {
      return T(1) - p_one_tail;
    }
  }
}

/**
 * Parse alternative hypothesis string to int.
 * @return 0=two-sided, 1=less, 2=greater, -1=invalid
 */
inline int parse_alternative(std::string_view alt) {
  if (alt == "two-sided") return 0;
  if (alt == "less") return 1;
  if (alt == "greater") return 2;
  return -1;
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
```

**Step 2: Verify file compiles**

Run: `ls src/torchscience/csrc/kernel/statistics/hypothesis_test/`
Expected: `t_test_common.h`

**Step 3: Commit**

```bash
git add src/torchscience/csrc/kernel/statistics/hypothesis_test/t_test_common.h
git commit -m "feat(kernel): add t-test common utilities for p-value computation"
```

---

## Task 2: Kernel - one_sample_t_test.h

**Files:**
- Create: `src/torchscience/csrc/kernel/statistics/hypothesis_test/one_sample_t_test.h`

**Step 1: Create the one-sample t-test kernel**

```cpp
// src/torchscience/csrc/kernel/statistics/hypothesis_test/one_sample_t_test.h
#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "t_test_common.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Compute one-sample t-test for a contiguous array.
 *
 * Tests whether sample mean differs from popmean.
 *
 * @param data Input array of n samples
 * @param n Number of samples
 * @param popmean Hypothesized population mean
 * @param alternative 0=two-sided, 1=less, 2=greater
 * @return (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> one_sample_t_test(
    const T* data,
    int64_t n,
    T popmean,
    int alternative
) {
  T nan = std::numeric_limits<T>::quiet_NaN();

  if (n < 2) {
    return std::make_tuple(nan, nan, nan);
  }

  // Compute mean
  T sum = T(0);
  for (int64_t i = 0; i < n; ++i) {
    sum += data[i];
  }
  T mean = sum / T(n);

  // Compute variance (unbiased, divide by n-1)
  T sum_sq = T(0);
  for (int64_t i = 0; i < n; ++i) {
    T d = data[i] - mean;
    sum_sq += d * d;
  }
  T var = sum_sq / T(n - 1);

  if (var <= T(0)) {
    // Zero variance - t-statistic undefined
    return std::make_tuple(nan, nan, nan);
  }

  // t = (mean - popmean) / (std / sqrt(n))
  T std_dev = std::sqrt(var);
  T se = std_dev / std::sqrt(T(n));
  T t_stat = (mean - popmean) / se;

  // Degrees of freedom
  T df = T(n - 1);

  // P-value
  T pvalue = t_pvalue(t_stat, df, alternative);

  return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/kernel/statistics/hypothesis_test/one_sample_t_test.h
git commit -m "feat(kernel): add one-sample t-test kernel"
```

---

## Task 3: Kernel - two_sample_t_test.h

**Files:**
- Create: `src/torchscience/csrc/kernel/statistics/hypothesis_test/two_sample_t_test.h`

**Step 1: Create the two-sample t-test kernel**

```cpp
// src/torchscience/csrc/kernel/statistics/hypothesis_test/two_sample_t_test.h
#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "t_test_common.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Compute two-sample t-test for contiguous arrays.
 *
 * Tests whether means of two independent samples differ.
 *
 * @param data1 First sample array
 * @param n1 Size of first sample
 * @param data2 Second sample array
 * @param n2 Size of second sample
 * @param equal_var If true, use Student's t-test (pooled variance).
 *                  If false, use Welch's t-test (separate variances).
 * @param alternative 0=two-sided, 1=less, 2=greater
 * @return (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> two_sample_t_test(
    const T* data1,
    int64_t n1,
    const T* data2,
    int64_t n2,
    bool equal_var,
    int alternative
) {
  T nan = std::numeric_limits<T>::quiet_NaN();

  if (n1 < 1 || n2 < 1) {
    return std::make_tuple(nan, nan, nan);
  }

  // Compute means
  T sum1 = T(0), sum2 = T(0);
  for (int64_t i = 0; i < n1; ++i) sum1 += data1[i];
  for (int64_t i = 0; i < n2; ++i) sum2 += data2[i];
  T mean1 = sum1 / T(n1);
  T mean2 = sum2 / T(n2);

  // Compute variances (unbiased)
  T ss1 = T(0), ss2 = T(0);
  for (int64_t i = 0; i < n1; ++i) {
    T d = data1[i] - mean1;
    ss1 += d * d;
  }
  for (int64_t i = 0; i < n2; ++i) {
    T d = data2[i] - mean2;
    ss2 += d * d;
  }

  T var1 = (n1 > 1) ? ss1 / T(n1 - 1) : T(0);
  T var2 = (n2 > 1) ? ss2 / T(n2 - 1) : T(0);

  T t_stat, df;

  if (equal_var) {
    // Student's t-test with pooled variance
    if (n1 + n2 < 3) {
      return std::make_tuple(nan, nan, nan);
    }

    // Pooled variance: s_p^2 = ((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2)
    T pooled_var = (ss1 + ss2) / T(n1 + n2 - 2);

    if (pooled_var <= T(0)) {
      return std::make_tuple(nan, nan, nan);
    }

    // SE = s_p * sqrt(1/n1 + 1/n2)
    T se = std::sqrt(pooled_var * (T(1) / T(n1) + T(1) / T(n2)));
    t_stat = (mean1 - mean2) / se;
    df = T(n1 + n2 - 2);
  } else {
    // Welch's t-test with separate variances
    if (n1 < 2 || n2 < 2) {
      return std::make_tuple(nan, nan, nan);
    }

    T v1_n1 = var1 / T(n1);
    T v2_n2 = var2 / T(n2);
    T se_sq = v1_n1 + v2_n2;

    if (se_sq <= T(0)) {
      return std::make_tuple(nan, nan, nan);
    }

    T se = std::sqrt(se_sq);
    t_stat = (mean1 - mean2) / se;

    // Welch-Satterthwaite degrees of freedom
    T num = se_sq * se_sq;
    T denom = (v1_n1 * v1_n1) / T(n1 - 1) + (v2_n2 * v2_n2) / T(n2 - 1);
    df = num / denom;
  }

  T pvalue = t_pvalue(t_stat, df, alternative);

  return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/kernel/statistics/hypothesis_test/two_sample_t_test.h
git commit -m "feat(kernel): add two-sample t-test kernel (Student's and Welch's)"
```

---

## Task 4: Kernel - paired_t_test.h

**Files:**
- Create: `src/torchscience/csrc/kernel/statistics/hypothesis_test/paired_t_test.h`

**Step 1: Create the paired t-test kernel**

```cpp
// src/torchscience/csrc/kernel/statistics/hypothesis_test/paired_t_test.h
#pragma once

#include <cmath>
#include <limits>
#include <tuple>

#include <c10/macros/Macros.h>

#include "one_sample_t_test.h"

namespace torchscience::kernel::statistics::hypothesis_test {

/**
 * Compute paired t-test for contiguous arrays.
 *
 * Tests whether means of paired samples differ.
 * Internally computes differences and runs one-sample t-test on them.
 *
 * @param data1 First sample array
 * @param data2 Second sample array (must be same size as data1)
 * @param n Number of paired samples
 * @param alternative 0=two-sided, 1=less, 2=greater
 * @param scratch Scratch buffer of size n for differences
 * @return (t_statistic, p_value, degrees_of_freedom)
 */
template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE
std::tuple<T, T, T> paired_t_test(
    const T* data1,
    const T* data2,
    int64_t n,
    int alternative,
    T* scratch
) {
  // Compute differences: d = x1 - x2
  for (int64_t i = 0; i < n; ++i) {
    scratch[i] = data1[i] - data2[i];
  }

  // Run one-sample t-test on differences with popmean=0
  return one_sample_t_test(scratch, n, T(0), alternative);
}

}  // namespace torchscience::kernel::statistics::hypothesis_test
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/kernel/statistics/hypothesis_test/paired_t_test.h
git commit -m "feat(kernel): add paired t-test kernel"
```

---

## Task 5: CPU - one_sample_t_test.h

**Files:**
- Create: `src/torchscience/csrc/cpu/statistics/hypothesis_test/one_sample_t_test.h`

**Step 1: Create the CPU implementation**

```cpp
// src/torchscience/csrc/cpu/statistics/hypothesis_test/one_sample_t_test.h
#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/statistics/hypothesis_test/one_sample_t_test.h"
#include "../../kernel/statistics/hypothesis_test/t_test_common.h"

namespace torchscience::cpu::statistics::hypothesis_test {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> one_sample_t_test(
    const at::Tensor& input,
    double popmean,
    c10::string_view alternative
) {
  int alt = kernel::statistics::hypothesis_test::parse_alternative(alternative);
  TORCH_CHECK(alt >= 0,
      "alternative must be 'two-sided', 'less', or 'greater', got '",
      std::string(alternative), "'");

  TORCH_CHECK(input.dim() >= 1,
      "one_sample_t_test: input must have at least 1 dimension");

  auto input_contig = input.contiguous();
  int64_t n_samples = input_contig.size(-1);
  int64_t batch_size = input_contig.numel() / n_samples;

  // Output shape: all dims except last
  std::vector<int64_t> output_shape;
  for (int64_t i = 0; i < input.dim() - 1; ++i) {
    output_shape.push_back(input.size(i));
  }

  auto options = input_contig.options();
  at::Tensor t_stat = output_shape.empty()
      ? at::empty({}, options)
      : at::empty(output_shape, options);
  at::Tensor pvalue = at::empty_like(t_stat);
  at::Tensor df = at::empty_like(t_stat);

  AT_DISPATCH_FLOATING_TYPES(
      input_contig.scalar_type(),
      "one_sample_t_test_cpu",
      [&]() {
        const scalar_t* data_ptr = input_contig.data_ptr<scalar_t>();
        scalar_t* t_ptr = t_stat.data_ptr<scalar_t>();
        scalar_t* p_ptr = pvalue.data_ptr<scalar_t>();
        scalar_t* df_ptr = df.data_ptr<scalar_t>();
        scalar_t pm = static_cast<scalar_t>(popmean);

        at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
          for (int64_t b = begin; b < end; ++b) {
            auto [t, p, d] = kernel::statistics::hypothesis_test::one_sample_t_test(
                data_ptr + b * n_samples,
                n_samples,
                pm,
                alt
            );
            t_ptr[b] = t;
            p_ptr[b] = p;
            df_ptr[b] = d;
          }
        });
      }
  );

  return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("one_sample_t_test", &torchscience::cpu::statistics::hypothesis_test::one_sample_t_test);
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/statistics/hypothesis_test/one_sample_t_test.h
git commit -m "feat(cpu): add one-sample t-test CPU implementation"
```

---

## Task 6: CPU - two_sample_t_test.h

**Files:**
- Create: `src/torchscience/csrc/cpu/statistics/hypothesis_test/two_sample_t_test.h`

**Step 1: Create the CPU implementation**

```cpp
// src/torchscience/csrc/cpu/statistics/hypothesis_test/two_sample_t_test.h
#pragma once

#include <tuple>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/statistics/hypothesis_test/two_sample_t_test.h"
#include "../../kernel/statistics/hypothesis_test/t_test_common.h"

namespace torchscience::cpu::statistics::hypothesis_test {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> two_sample_t_test(
    const at::Tensor& input1,
    const at::Tensor& input2,
    bool equal_var,
    c10::string_view alternative
) {
  int alt = kernel::statistics::hypothesis_test::parse_alternative(alternative);
  TORCH_CHECK(alt >= 0,
      "alternative must be 'two-sided', 'less', or 'greater', got '",
      std::string(alternative), "'");

  TORCH_CHECK(input1.dim() >= 1 && input2.dim() >= 1,
      "two_sample_t_test: inputs must have at least 1 dimension");

  // Batch dimensions must match (all dims except last)
  TORCH_CHECK(input1.dim() == input2.dim(),
      "two_sample_t_test: inputs must have same number of dimensions");

  for (int64_t i = 0; i < input1.dim() - 1; ++i) {
    TORCH_CHECK(input1.size(i) == input2.size(i),
        "two_sample_t_test: batch dimensions must match at dim ", i);
  }

  auto input1_contig = input1.contiguous();
  auto input2_contig = input2.contiguous();
  int64_t n1 = input1_contig.size(-1);
  int64_t n2 = input2_contig.size(-1);
  int64_t batch_size = input1_contig.numel() / n1;

  // Output shape: all dims except last
  std::vector<int64_t> output_shape;
  for (int64_t i = 0; i < input1.dim() - 1; ++i) {
    output_shape.push_back(input1.size(i));
  }

  auto options = input1_contig.options();
  at::Tensor t_stat = output_shape.empty()
      ? at::empty({}, options)
      : at::empty(output_shape, options);
  at::Tensor pvalue = at::empty_like(t_stat);
  at::Tensor df = at::empty_like(t_stat);

  AT_DISPATCH_FLOATING_TYPES(
      input1_contig.scalar_type(),
      "two_sample_t_test_cpu",
      [&]() {
        const scalar_t* data1_ptr = input1_contig.data_ptr<scalar_t>();
        const scalar_t* data2_ptr = input2_contig.data_ptr<scalar_t>();
        scalar_t* t_ptr = t_stat.data_ptr<scalar_t>();
        scalar_t* p_ptr = pvalue.data_ptr<scalar_t>();
        scalar_t* df_ptr = df.data_ptr<scalar_t>();

        at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
          for (int64_t b = begin; b < end; ++b) {
            auto [t, p, d] = kernel::statistics::hypothesis_test::two_sample_t_test(
                data1_ptr + b * n1,
                n1,
                data2_ptr + b * n2,
                n2,
                equal_var,
                alt
            );
            t_ptr[b] = t;
            p_ptr[b] = p;
            df_ptr[b] = d;
          }
        });
      }
  );

  return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("two_sample_t_test", &torchscience::cpu::statistics::hypothesis_test::two_sample_t_test);
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/statistics/hypothesis_test/two_sample_t_test.h
git commit -m "feat(cpu): add two-sample t-test CPU implementation"
```

---

## Task 7: CPU - paired_t_test.h

**Files:**
- Create: `src/torchscience/csrc/cpu/statistics/hypothesis_test/paired_t_test.h`

**Step 1: Create the CPU implementation**

```cpp
// src/torchscience/csrc/cpu/statistics/hypothesis_test/paired_t_test.h
#pragma once

#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/statistics/hypothesis_test/paired_t_test.h"
#include "../../kernel/statistics/hypothesis_test/t_test_common.h"

namespace torchscience::cpu::statistics::hypothesis_test {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> paired_t_test(
    const at::Tensor& input1,
    const at::Tensor& input2,
    c10::string_view alternative
) {
  int alt = kernel::statistics::hypothesis_test::parse_alternative(alternative);
  TORCH_CHECK(alt >= 0,
      "alternative must be 'two-sided', 'less', or 'greater', got '",
      std::string(alternative), "'");

  TORCH_CHECK(input1.dim() >= 1 && input2.dim() >= 1,
      "paired_t_test: inputs must have at least 1 dimension");

  // Shapes must match exactly for paired test
  TORCH_CHECK(input1.sizes() == input2.sizes(),
      "paired_t_test: input shapes must match exactly");

  auto input1_contig = input1.contiguous();
  auto input2_contig = input2.contiguous();
  int64_t n_samples = input1_contig.size(-1);
  int64_t batch_size = input1_contig.numel() / n_samples;

  // Output shape: all dims except last
  std::vector<int64_t> output_shape;
  for (int64_t i = 0; i < input1.dim() - 1; ++i) {
    output_shape.push_back(input1.size(i));
  }

  auto options = input1_contig.options();
  at::Tensor t_stat = output_shape.empty()
      ? at::empty({}, options)
      : at::empty(output_shape, options);
  at::Tensor pvalue = at::empty_like(t_stat);
  at::Tensor df = at::empty_like(t_stat);

  AT_DISPATCH_FLOATING_TYPES(
      input1_contig.scalar_type(),
      "paired_t_test_cpu",
      [&]() {
        const scalar_t* data1_ptr = input1_contig.data_ptr<scalar_t>();
        const scalar_t* data2_ptr = input2_contig.data_ptr<scalar_t>();
        scalar_t* t_ptr = t_stat.data_ptr<scalar_t>();
        scalar_t* p_ptr = pvalue.data_ptr<scalar_t>();
        scalar_t* df_ptr = df.data_ptr<scalar_t>();

        // Thread-local scratch buffers
        at::parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
          std::vector<scalar_t> scratch(n_samples);
          for (int64_t b = begin; b < end; ++b) {
            auto [t, p, d] = kernel::statistics::hypothesis_test::paired_t_test(
                data1_ptr + b * n_samples,
                data2_ptr + b * n_samples,
                n_samples,
                alt,
                scratch.data()
            );
            t_ptr[b] = t;
            p_ptr[b] = p;
            df_ptr[b] = d;
          }
        });
      }
  );

  return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::cpu::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("paired_t_test", &torchscience::cpu::statistics::hypothesis_test::paired_t_test);
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/cpu/statistics/hypothesis_test/paired_t_test.h
git commit -m "feat(cpu): add paired t-test CPU implementation"
```

---

## Task 8: Meta - t-test shape inference

**Files:**
- Create: `src/torchscience/csrc/meta/statistics/hypothesis_test/one_sample_t_test.h`
- Create: `src/torchscience/csrc/meta/statistics/hypothesis_test/two_sample_t_test.h`
- Create: `src/torchscience/csrc/meta/statistics/hypothesis_test/paired_t_test.h`

**Step 1: Create meta implementations**

```cpp
// src/torchscience/csrc/meta/statistics/hypothesis_test/one_sample_t_test.h
#pragma once

#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> one_sample_t_test(
    const at::Tensor& input,
    [[maybe_unused]] double popmean,
    [[maybe_unused]] c10::string_view alternative
) {
  TORCH_CHECK(input.dim() >= 1,
      "one_sample_t_test: input must have at least 1 dimension");

  std::vector<int64_t> output_shape;
  for (int64_t i = 0; i < input.dim() - 1; ++i) {
    output_shape.push_back(input.size(i));
  }

  auto options = input.options();
  at::Tensor t_stat = output_shape.empty()
      ? at::empty({}, options)
      : at::empty(output_shape, options);
  at::Tensor pvalue = at::empty_like(t_stat);
  at::Tensor df = at::empty_like(t_stat);

  return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("one_sample_t_test", &torchscience::meta::statistics::hypothesis_test::one_sample_t_test);
}
```

```cpp
// src/torchscience/csrc/meta/statistics/hypothesis_test/two_sample_t_test.h
#pragma once

#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> two_sample_t_test(
    const at::Tensor& input1,
    [[maybe_unused]] const at::Tensor& input2,
    [[maybe_unused]] bool equal_var,
    [[maybe_unused]] c10::string_view alternative
) {
  TORCH_CHECK(input1.dim() >= 1,
      "two_sample_t_test: input must have at least 1 dimension");

  std::vector<int64_t> output_shape;
  for (int64_t i = 0; i < input1.dim() - 1; ++i) {
    output_shape.push_back(input1.size(i));
  }

  auto options = input1.options();
  at::Tensor t_stat = output_shape.empty()
      ? at::empty({}, options)
      : at::empty(output_shape, options);
  at::Tensor pvalue = at::empty_like(t_stat);
  at::Tensor df = at::empty_like(t_stat);

  return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("two_sample_t_test", &torchscience::meta::statistics::hypothesis_test::two_sample_t_test);
}
```

```cpp
// src/torchscience/csrc/meta/statistics/hypothesis_test/paired_t_test.h
#pragma once

#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::statistics::hypothesis_test {

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> paired_t_test(
    const at::Tensor& input1,
    [[maybe_unused]] const at::Tensor& input2,
    [[maybe_unused]] c10::string_view alternative
) {
  TORCH_CHECK(input1.dim() >= 1,
      "paired_t_test: input must have at least 1 dimension");

  std::vector<int64_t> output_shape;
  for (int64_t i = 0; i < input1.dim() - 1; ++i) {
    output_shape.push_back(input1.size(i));
  }

  auto options = input1.options();
  at::Tensor t_stat = output_shape.empty()
      ? at::empty({}, options)
      : at::empty(output_shape, options);
  at::Tensor pvalue = at::empty_like(t_stat);
  at::Tensor df = at::empty_like(t_stat);

  return std::make_tuple(t_stat, pvalue, df);
}

}  // namespace torchscience::meta::statistics::hypothesis_test

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("paired_t_test", &torchscience::meta::statistics::hypothesis_test::paired_t_test);
}
```

**Step 2: Commit**

```bash
git add src/torchscience/csrc/meta/statistics/hypothesis_test/
git commit -m "feat(meta): add t-test meta implementations for shape inference"
```

---

## Task 9: Schema registration in torchscience.cpp

**Files:**
- Modify: `src/torchscience/csrc/torchscience.cpp`

**Step 1: Add includes at appropriate location (after existing statistics includes)**

Find: `#include "cpu/statistics/descriptive/histogram.h"`
Add after:
```cpp
#include "cpu/statistics/hypothesis_test/one_sample_t_test.h"
#include "cpu/statistics/hypothesis_test/two_sample_t_test.h"
#include "cpu/statistics/hypothesis_test/paired_t_test.h"
```

Find: `#include "meta/statistics/descriptive/histogram.h"`
Add after:
```cpp
#include "meta/statistics/hypothesis_test/one_sample_t_test.h"
#include "meta/statistics/hypothesis_test/two_sample_t_test.h"
#include "meta/statistics/hypothesis_test/paired_t_test.h"
```

**Step 2: Add schema definitions (after existing statistics.descriptive schemas)**

Find: `module.def("histogram`
Add after the histogram schemas:
```cpp
  // statistics.hypothesis_test
  module.def("one_sample_t_test(Tensor input, float popmean, str alternative) -> (Tensor, Tensor, Tensor)");
  module.def("two_sample_t_test(Tensor input1, Tensor input2, bool equal_var, str alternative) -> (Tensor, Tensor, Tensor)");
  module.def("paired_t_test(Tensor input1, Tensor input2, str alternative) -> (Tensor, Tensor, Tensor)");
```

**Step 3: Commit**

```bash
git add src/torchscience/csrc/torchscience.cpp
git commit -m "feat: register t-test operator schemas"
```

---

## Task 10: Python API - one_sample_t_test.py

**Files:**
- Create: `src/torchscience/statistics/hypothesis_test/_one_sample_t_test.py`

**Step 1: Create the Python wrapper**

```python
"""One-sample t-test implementation."""

from torch import Tensor

import torchscience._csrc  # noqa: F401


def one_sample_t_test(
    input: Tensor,
    popmean: float = 0.0,
    *,
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Perform a one-sample t-test.

    Tests whether the mean of a sample differs from a known population mean.

    Mathematical Definition
    -----------------------
    The t-statistic is computed as:

    .. math::

        t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}

    where :math:`\bar{x}` is the sample mean, :math:`\mu_0` is the hypothesized
    population mean, :math:`s` is the sample standard deviation, and :math:`n`
    is the sample size.

    The degrees of freedom is :math:`df = n - 1`.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape ``(*batch, n_samples)``. The last dimension
        contains the samples; all preceding dimensions are batch dimensions.
    popmean : float, optional
        Hypothesized population mean. Default: 0.0.
    alternative : str, optional
        The alternative hypothesis. One of:

        - ``"two-sided"``: mean differs from popmean (default)
        - ``"less"``: mean is less than popmean
        - ``"greater"``: mean is greater than popmean

    Returns
    -------
    statistic : Tensor
        The t-statistic. Shape is ``(*batch,)``.
    pvalue : Tensor
        The p-value for the test. Shape is ``(*batch,)``.
    df : Tensor
        Degrees of freedom. Shape is ``(*batch,)``.

    Examples
    --------
    Basic usage:

    >>> import torch
    >>> import torchscience.statistics.hypothesis_test as ht
    >>> x = torch.randn(100) + 0.5  # Sample with true mean ~0.5
    >>> stat, pval, df = ht.one_sample_t_test(x, popmean=0.0)
    >>> print(f"t={stat.item():.3f}, p={pval.item():.3f}, df={df.item():.0f}")

    Batched computation:

    >>> x = torch.randn(5, 100)  # 5 batches of 100 samples each
    >>> stat, pval, df = ht.one_sample_t_test(x)
    >>> print(stat.shape)  # torch.Size([5])

    One-sided test:

    >>> x = torch.randn(100) + 1.0
    >>> stat, pval, df = ht.one_sample_t_test(x, alternative="greater")

    Notes
    -----
    - Returns NaN for batches with fewer than 2 samples.
    - Returns NaN for batches with zero variance.
    - Requires at least 1 dimension in the input tensor.

    See Also
    --------
    scipy.stats.ttest_1samp : SciPy's one-sample t-test.
    two_sample_t_test : Test for difference between two independent samples.
    paired_t_test : Test for difference between paired samples.
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(
            f"alternative must be 'two-sided', 'less', or 'greater', "
            f"got '{alternative}'"
        )

    import torch

    return torch.ops.torchscience.one_sample_t_test(input, popmean, alternative)
```

**Step 2: Commit**

```bash
git add src/torchscience/statistics/hypothesis_test/_one_sample_t_test.py
git commit -m "feat(python): add one_sample_t_test Python API"
```

---

## Task 11: Python API - two_sample_t_test.py

**Files:**
- Create: `src/torchscience/statistics/hypothesis_test/_two_sample_t_test.py`

**Step 1: Create the Python wrapper**

```python
"""Two-sample t-test implementation."""

from torch import Tensor

import torchscience._csrc  # noqa: F401


def two_sample_t_test(
    input1: Tensor,
    input2: Tensor,
    *,
    equal_var: bool = False,
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Perform a two-sample t-test for independent samples.

    Tests whether the means of two independent samples differ.

    Mathematical Definition
    -----------------------
    **Welch's t-test** (``equal_var=False``, default):

    .. math::

        t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}

    with Welch-Satterthwaite degrees of freedom:

    .. math::

        df = \frac{(s_1^2/n_1 + s_2^2/n_2)^2}
                  {(s_1^2/n_1)^2/(n_1-1) + (s_2^2/n_2)^2/(n_2-1)}

    **Student's t-test** (``equal_var=True``):

    .. math::

        t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{1/n_1 + 1/n_2}}

    where the pooled standard deviation is:

    .. math::

        s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}

    and :math:`df = n_1 + n_2 - 2`.

    Parameters
    ----------
    input1 : Tensor
        First sample tensor of shape ``(*batch, n1_samples)``.
    input2 : Tensor
        Second sample tensor of shape ``(*batch, n2_samples)``.
        Batch dimensions must match input1, but sample sizes can differ.
    equal_var : bool, optional
        If True, use Student's t-test assuming equal population variances.
        If False (default), use Welch's t-test which does not assume equal
        variances. Welch's test is generally more robust.
    alternative : str, optional
        The alternative hypothesis. One of:

        - ``"two-sided"``: means differ (default)
        - ``"less"``: mean of input1 is less than mean of input2
        - ``"greater"``: mean of input1 is greater than mean of input2

    Returns
    -------
    statistic : Tensor
        The t-statistic. Shape is ``(*batch,)``.
    pvalue : Tensor
        The p-value for the test. Shape is ``(*batch,)``.
    df : Tensor
        Degrees of freedom. Shape is ``(*batch,)``. For Welch's test, this
        may be non-integer.

    Examples
    --------
    Basic usage:

    >>> import torch
    >>> import torchscience.statistics.hypothesis_test as ht
    >>> x = torch.randn(100)
    >>> y = torch.randn(100) + 0.5  # Different mean
    >>> stat, pval, df = ht.two_sample_t_test(x, y)
    >>> print(f"t={stat.item():.3f}, p={pval.item():.3f}")

    Different sample sizes:

    >>> x = torch.randn(50)
    >>> y = torch.randn(100)
    >>> stat, pval, df = ht.two_sample_t_test(x, y)

    Student's t-test (equal variances assumed):

    >>> stat, pval, df = ht.two_sample_t_test(x, y, equal_var=True)

    Batched computation:

    >>> x = torch.randn(5, 100)  # 5 batches
    >>> y = torch.randn(5, 80)   # Same batch size, different n_samples
    >>> stat, pval, df = ht.two_sample_t_test(x, y)
    >>> print(stat.shape)  # torch.Size([5])

    Notes
    -----
    - Welch's t-test (default) is preferred when equal variance cannot be
      assumed. It provides more accurate p-values when variances differ.
    - Returns NaN when sample sizes are insufficient:
      - Welch's: requires n1 >= 2 and n2 >= 2
      - Student's: requires n1 + n2 >= 3
    - Returns NaN for batches with zero variance in both samples.

    See Also
    --------
    scipy.stats.ttest_ind : SciPy's two-sample t-test.
    one_sample_t_test : Test sample mean against known value.
    paired_t_test : Test for difference between paired samples.
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(
            f"alternative must be 'two-sided', 'less', or 'greater', "
            f"got '{alternative}'"
        )

    import torch

    return torch.ops.torchscience.two_sample_t_test(
        input1, input2, equal_var, alternative
    )
```

**Step 2: Commit**

```bash
git add src/torchscience/statistics/hypothesis_test/_two_sample_t_test.py
git commit -m "feat(python): add two_sample_t_test Python API"
```

---

## Task 12: Python API - paired_t_test.py

**Files:**
- Create: `src/torchscience/statistics/hypothesis_test/_paired_t_test.py`

**Step 1: Create the Python wrapper**

```python
"""Paired t-test implementation."""

from torch import Tensor

import torchscience._csrc  # noqa: F401


def paired_t_test(
    input1: Tensor,
    input2: Tensor,
    *,
    alternative: str = "two-sided",
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Perform a paired t-test for related samples.

    Tests whether the means of paired/matched samples differ. This is
    equivalent to a one-sample t-test on the differences.

    Mathematical Definition
    -----------------------
    Given paired observations :math:`(x_{1i}, x_{2i})`, compute differences
    :math:`d_i = x_{1i} - x_{2i}`, then:

    .. math::

        t = \frac{\bar{d}}{s_d / \sqrt{n}}

    where :math:`\bar{d}` is the mean of differences, :math:`s_d` is the
    standard deviation of differences, and :math:`n` is the number of pairs.

    The degrees of freedom is :math:`df = n - 1`.

    Parameters
    ----------
    input1 : Tensor
        First sample tensor of shape ``(*batch, n_samples)``.
    input2 : Tensor
        Second sample tensor. Must have the same shape as input1.
    alternative : str, optional
        The alternative hypothesis. One of:

        - ``"two-sided"``: mean difference differs from zero (default)
        - ``"less"``: mean of input1 is less than mean of input2
        - ``"greater"``: mean of input1 is greater than mean of input2

    Returns
    -------
    statistic : Tensor
        The t-statistic. Shape is ``(*batch,)``.
    pvalue : Tensor
        The p-value for the test. Shape is ``(*batch,)``.
    df : Tensor
        Degrees of freedom. Shape is ``(*batch,)``.

    Examples
    --------
    Basic usage (before/after measurement):

    >>> import torch
    >>> import torchscience.statistics.hypothesis_test as ht
    >>> before = torch.randn(20)
    >>> after = before + torch.randn(20) * 0.5 + 0.3  # Treatment effect
    >>> stat, pval, df = ht.paired_t_test(before, after)
    >>> print(f"t={stat.item():.3f}, p={pval.item():.3f}")

    One-sided test (expecting improvement):

    >>> stat, pval, df = ht.paired_t_test(before, after, alternative="less")

    Batched computation:

    >>> before = torch.randn(5, 30)  # 5 experiments, 30 subjects each
    >>> after = before + 0.5
    >>> stat, pval, df = ht.paired_t_test(before, after)
    >>> print(stat.shape)  # torch.Size([5])

    Notes
    -----
    - Input shapes must match exactly (paired observations).
    - Returns NaN for batches with fewer than 2 pairs.
    - Returns NaN if all differences are identical (zero variance).
    - Use this test when observations are naturally paired (e.g., before/after,
      matched subjects, repeated measures).

    See Also
    --------
    scipy.stats.ttest_rel : SciPy's paired t-test.
    one_sample_t_test : Test sample mean against known value.
    two_sample_t_test : Test for difference between independent samples.
    """
    if alternative not in ("two-sided", "less", "greater"):
        raise ValueError(
            f"alternative must be 'two-sided', 'less', or 'greater', "
            f"got '{alternative}'"
        )

    import torch

    return torch.ops.torchscience.paired_t_test(input1, input2, alternative)
```

**Step 2: Commit**

```bash
git add src/torchscience/statistics/hypothesis_test/_paired_t_test.py
git commit -m "feat(python): add paired_t_test Python API"
```

---

## Task 13: Update hypothesis_test __init__.py

**Files:**
- Modify: `src/torchscience/statistics/hypothesis_test/__init__.py`

**Step 1: Add exports**

```python
from ._one_sample_t_test import one_sample_t_test
from ._paired_t_test import paired_t_test
from ._two_sample_t_test import two_sample_t_test

__all__ = [
    "one_sample_t_test",
    "paired_t_test",
    "two_sample_t_test",
]
```

**Step 2: Commit**

```bash
git add src/torchscience/statistics/hypothesis_test/__init__.py
git commit -m "feat: export t-test functions from hypothesis_test module"
```

---

## Task 14: Update statistics __init__.py

**Files:**
- Modify: `src/torchscience/statistics/__init__.py`

**Step 1: Add hypothesis_test export**

Replace contents with:
```python
from . import descriptive
from . import hypothesis_test

__all__ = [
    "descriptive",
    "hypothesis_test",
]
```

**Step 2: Commit**

```bash
git add src/torchscience/statistics/__init__.py
git commit -m "feat: export hypothesis_test submodule from statistics"
```

---

## Task 15: Tests - basic correctness

**Files:**
- Create: `tests/torchscience/statistics/hypothesis_test/__init__.py`
- Create: `tests/torchscience/statistics/hypothesis_test/test__t_test.py`

**Step 1: Create test __init__.py**

```python
# tests/torchscience/statistics/hypothesis_test/__init__.py
```

**Step 2: Create comprehensive test file**

```python
"""Tests for t-test functions."""

import pytest
import torch
from scipy import stats

import torchscience.statistics.hypothesis_test as ht


class TestOneSampleTTest:
    """Tests for one_sample_t_test."""

    def test_basic_correctness(self):
        """Test against scipy.stats.ttest_1samp."""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64)

        stat, pval, df = ht.one_sample_t_test(x, popmean=0.0)
        scipy_result = stats.ttest_1samp(x.numpy(), 0.0)

        assert torch.allclose(stat, torch.tensor(scipy_result.statistic), rtol=1e-5)
        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)
        assert df.item() == 99

    def test_nonzero_popmean(self):
        """Test with non-zero population mean."""
        torch.manual_seed(42)
        x = torch.randn(50, dtype=torch.float64) + 2.0

        stat, pval, df = ht.one_sample_t_test(x, popmean=2.0)
        scipy_result = stats.ttest_1samp(x.numpy(), 2.0)

        assert torch.allclose(stat, torch.tensor(scipy_result.statistic), rtol=1e-5)
        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)

    def test_alternative_less(self):
        """Test alternative='less'."""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64) - 1.0  # Mean < 0

        stat, pval, df = ht.one_sample_t_test(x, popmean=0.0, alternative="less")
        scipy_result = stats.ttest_1samp(x.numpy(), 0.0, alternative="less")

        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)

    def test_alternative_greater(self):
        """Test alternative='greater'."""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64) + 1.0  # Mean > 0

        stat, pval, df = ht.one_sample_t_test(x, popmean=0.0, alternative="greater")
        scipy_result = stats.ttest_1samp(x.numpy(), 0.0, alternative="greater")

        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)

    def test_batched(self):
        """Test batched computation."""
        torch.manual_seed(42)
        x = torch.randn(5, 100, dtype=torch.float64)

        stat, pval, df = ht.one_sample_t_test(x)

        assert stat.shape == (5,)
        assert pval.shape == (5,)
        assert df.shape == (5,)

        # Verify each batch matches unbatched result
        for i in range(5):
            stat_i, pval_i, df_i = ht.one_sample_t_test(x[i])
            assert torch.allclose(stat[i], stat_i)
            assert torch.allclose(pval[i], pval_i)

    def test_multidim_batch(self):
        """Test with multiple batch dimensions."""
        torch.manual_seed(42)
        x = torch.randn(2, 3, 50, dtype=torch.float64)

        stat, pval, df = ht.one_sample_t_test(x)

        assert stat.shape == (2, 3)
        assert pval.shape == (2, 3)
        assert df.shape == (2, 3)

    def test_insufficient_samples(self):
        """Test NaN for n < 2."""
        x = torch.tensor([1.0])

        stat, pval, df = ht.one_sample_t_test(x)

        assert torch.isnan(stat)
        assert torch.isnan(pval)
        assert torch.isnan(df)

    def test_zero_variance(self):
        """Test NaN for zero variance."""
        x = torch.ones(10)

        stat, pval, df = ht.one_sample_t_test(x, popmean=0.5)

        assert torch.isnan(stat)
        assert torch.isnan(pval)

    def test_invalid_alternative(self):
        """Test error for invalid alternative."""
        x = torch.randn(10)

        with pytest.raises(ValueError, match="alternative must be"):
            ht.one_sample_t_test(x, alternative="invalid")


class TestTwoSampleTTest:
    """Tests for two_sample_t_test."""

    def test_welch_correctness(self):
        """Test Welch's t-test against scipy."""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64)
        y = torch.randn(100, dtype=torch.float64) + 0.5

        stat, pval, df = ht.two_sample_t_test(x, y, equal_var=False)
        scipy_result = stats.ttest_ind(x.numpy(), y.numpy(), equal_var=False)

        assert torch.allclose(stat, torch.tensor(scipy_result.statistic), rtol=1e-5)
        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)

    def test_student_correctness(self):
        """Test Student's t-test against scipy."""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64)
        y = torch.randn(100, dtype=torch.float64) + 0.5

        stat, pval, df = ht.two_sample_t_test(x, y, equal_var=True)
        scipy_result = stats.ttest_ind(x.numpy(), y.numpy(), equal_var=True)

        assert torch.allclose(stat, torch.tensor(scipy_result.statistic), rtol=1e-5)
        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)
        assert df.item() == 198  # n1 + n2 - 2

    def test_different_sample_sizes(self):
        """Test with different sample sizes."""
        torch.manual_seed(42)
        x = torch.randn(50, dtype=torch.float64)
        y = torch.randn(100, dtype=torch.float64)

        stat, pval, df = ht.two_sample_t_test(x, y)
        scipy_result = stats.ttest_ind(x.numpy(), y.numpy(), equal_var=False)

        assert torch.allclose(stat, torch.tensor(scipy_result.statistic), rtol=1e-5)
        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)

    def test_alternative_less(self):
        """Test alternative='less'."""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=torch.float64)
        y = torch.randn(100, dtype=torch.float64) + 1.0

        stat, pval, df = ht.two_sample_t_test(x, y, alternative="less")
        scipy_result = stats.ttest_ind(
            x.numpy(), y.numpy(), equal_var=False, alternative="less"
        )

        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)

    def test_batched(self):
        """Test batched computation."""
        torch.manual_seed(42)
        x = torch.randn(5, 100, dtype=torch.float64)
        y = torch.randn(5, 80, dtype=torch.float64)

        stat, pval, df = ht.two_sample_t_test(x, y)

        assert stat.shape == (5,)
        assert pval.shape == (5,)
        assert df.shape == (5,)

    def test_equal_var_differs(self):
        """Test that equal_var=True/False give different results."""
        torch.manual_seed(42)
        x = torch.randn(50, dtype=torch.float64)
        y = torch.randn(50, dtype=torch.float64) * 2  # Different variance

        _, pval_welch, df_welch = ht.two_sample_t_test(x, y, equal_var=False)
        _, pval_student, df_student = ht.two_sample_t_test(x, y, equal_var=True)

        # Welch's df is non-integer, Student's is n1+n2-2
        assert df_student.item() == 98
        assert not torch.allclose(df_welch, df_student)

    def test_insufficient_samples_welch(self):
        """Test NaN for n < 2 in Welch's test."""
        x = torch.tensor([1.0])
        y = torch.randn(10)

        stat, pval, df = ht.two_sample_t_test(x, y, equal_var=False)

        assert torch.isnan(stat)
        assert torch.isnan(pval)


class TestPairedTTest:
    """Tests for paired_t_test."""

    def test_basic_correctness(self):
        """Test against scipy.stats.ttest_rel."""
        torch.manual_seed(42)
        x = torch.randn(50, dtype=torch.float64)
        y = x + torch.randn(50, dtype=torch.float64) * 0.5 + 0.3

        stat, pval, df = ht.paired_t_test(x, y)
        scipy_result = stats.ttest_rel(x.numpy(), y.numpy())

        assert torch.allclose(stat, torch.tensor(scipy_result.statistic), rtol=1e-5)
        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)
        assert df.item() == 49

    def test_alternative_less(self):
        """Test alternative='less'."""
        torch.manual_seed(42)
        before = torch.randn(30, dtype=torch.float64)
        after = before + 0.5  # Improvement

        stat, pval, df = ht.paired_t_test(before, after, alternative="less")
        scipy_result = stats.ttest_rel(
            before.numpy(), after.numpy(), alternative="less"
        )

        assert torch.allclose(pval, torch.tensor(scipy_result.pvalue), rtol=1e-5)

    def test_batched(self):
        """Test batched computation."""
        torch.manual_seed(42)
        x = torch.randn(5, 30, dtype=torch.float64)
        y = x + 0.3

        stat, pval, df = ht.paired_t_test(x, y)

        assert stat.shape == (5,)
        assert pval.shape == (5,)
        assert df.shape == (5,)

    def test_insufficient_samples(self):
        """Test NaN for n < 2."""
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])

        stat, pval, df = ht.paired_t_test(x, y)

        assert torch.isnan(stat)
        assert torch.isnan(pval)

    def test_shape_mismatch_error(self):
        """Test error when shapes don't match."""
        x = torch.randn(10)
        y = torch.randn(20)

        with pytest.raises(RuntimeError, match="shapes must match"):
            ht.paired_t_test(x, y)


class TestDtypes:
    """Tests for dtype handling."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_one_sample_dtypes(self, dtype):
        """Test one_sample_t_test with different dtypes."""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=dtype)

        stat, pval, df = ht.one_sample_t_test(x)

        assert stat.dtype == dtype
        assert pval.dtype == dtype
        assert df.dtype == dtype

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_two_sample_dtypes(self, dtype):
        """Test two_sample_t_test with different dtypes."""
        torch.manual_seed(42)
        x = torch.randn(100, dtype=dtype)
        y = torch.randn(100, dtype=dtype)

        stat, pval, df = ht.two_sample_t_test(x, y)

        assert stat.dtype == dtype
        assert pval.dtype == dtype
        assert df.dtype == dtype
```

**Step 3: Run tests to verify they fail (C++ not compiled yet)**

Run: `uv run pytest tests/torchscience/statistics/hypothesis_test/ -v --tb=short 2>&1 | head -50`
Expected: ImportError or AttributeError (operators not registered yet)

**Step 4: Commit tests**

```bash
git add tests/torchscience/statistics/hypothesis_test/
git commit -m "test: add comprehensive t-test tests"
```

---

## Task 16: Build and run tests

**Step 1: Rebuild the extension**

Run: `uv run pip install -e . --no-build-isolation`
Expected: Successful build

**Step 2: Run the tests**

Run: `uv run pytest tests/torchscience/statistics/hypothesis_test/ -v`
Expected: All tests pass

**Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: address any build or test issues"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Kernel: t_test_common.h | 1 new |
| 2 | Kernel: one_sample_t_test.h | 1 new |
| 3 | Kernel: two_sample_t_test.h | 1 new |
| 4 | Kernel: paired_t_test.h | 1 new |
| 5 | CPU: one_sample_t_test.h | 1 new |
| 6 | CPU: two_sample_t_test.h | 1 new |
| 7 | CPU: paired_t_test.h | 1 new |
| 8 | Meta: all three t-tests | 3 new |
| 9 | Schema registration | 1 modified |
| 10 | Python: one_sample_t_test.py | 1 new |
| 11 | Python: two_sample_t_test.py | 1 new |
| 12 | Python: paired_t_test.py | 1 new |
| 13 | hypothesis_test __init__.py | 1 modified |
| 14 | statistics __init__.py | 1 modified |
| 15 | Tests | 2 new |
| 16 | Build and verify | - |

**Total: 14 new files, 3 modified files, ~16 commits**
