#pragma once

#include <vector>
#include <ATen/core/Tensor.h>

namespace torchscience::impl {

// ReductionTraits interface requirements:
//
// struct ExampleReductionTraits {
//     // Reduce a contiguous 1D array to a single value
//     template<typename T>
//     static T reduce(const T* data, int64_t n, /* operator-specific params */);
//
//     // Backward: compute gradient w.r.t. each input element
//     template<typename T>
//     static void backward(
//         T grad_output,
//         const T* input,
//         int64_t n,
//         T* grad_input,
//         /* operator-specific params */
//     );
//
//     // Double backward (optional - default returns zeros)
//     template<typename T>
//     static void backward_backward(
//         const T* grad_grad_input,
//         T grad_output,
//         const T* input,
//         int64_t n,
//         T& grad_grad_output,
//         T* new_grad_input,
//         /* operator-specific params */
//     );
// };

}  // namespace torchscience::impl
