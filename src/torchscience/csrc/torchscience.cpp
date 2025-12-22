#include <torch/extension.h>

#include "cpu/special_functions.h"
#include "autograd/special_functions.h"
#include "autocast/special_functions.h"
#include "meta/special_functions.h"
#include "sparse/coo/cpu/special_functions.h"
#include "sparse/coo/cpu/optimization/test_functions.h"
#include "sparse/csr/cpu/special_functions.h"
#include "sparse/csr/cpu/optimization/test_functions.h"
#include "quantized/cpu/special_functions.h"
#include "quantized/cpu/optimization/test_functions.h"

#include "composite/signal_processing/window_functions.h"
#include "composite/signal_processing/waveform.h"
#include "composite/optimization/test_functions.h"
// Note: batching/optimization/test_functions.h disabled - functorch batch rules
// require different registration for custom ops (not TORCH_LIBRARY_IMPL)

#include "cpu/signal_processing/filter.h"
#include "cpu/optimization/test_functions.h"
#include "cpu/statistics/descriptive/kurtosis.h"
#include "cpu/integral_transform/hilbert_transform.h"
#include "cpu/integral_transform/inverse_hilbert_transform.h"
#include "autograd/signal_processing/filter.h"
#include "autograd/optimization/test_functions.h"
#include "autograd/statistics/descriptive/kurtosis.h"
#include "autograd/integral_transform/hilbert_transform.h"
#include "autograd/integral_transform/inverse_hilbert_transform.h"
#include "meta/signal_processing/filter.h"
#include "meta/optimization/test_functions.h"
#include "meta/statistics/descriptive/kurtosis.h"
#include "meta/integral_transform/hilbert_transform.h"
#include "meta/integral_transform/inverse_hilbert_transform.h"
#include "autocast/signal_processing/filter.h"
#include "autocast/statistics/descriptive/kurtosis.h"
#include "autocast/integral_transform/hilbert_transform.h"
#include "autocast/integral_transform/inverse_hilbert_transform.h"

#include "sparse/coo/cpu/integral_transform/hilbert_transform.h"
#include "sparse/coo/cpu/integral_transform/inverse_hilbert_transform.h"
#include "sparse/csr/cpu/integral_transform/hilbert_transform.h"
#include "sparse/csr/cpu/integral_transform/inverse_hilbert_transform.h"
#include "quantized/cpu/integral_transform/hilbert_transform.h"
#include "quantized/cpu/integral_transform/inverse_hilbert_transform.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/optimization/test_functions.cu"
#include "cuda/statistics/descriptive/kurtosis.cu"
#include "cuda/integral_transform/hilbert_transform.cu"
#include "cuda/integral_transform/inverse_hilbert_transform.cu"
#include "sparse/coo/cuda/special_functions.h"
#include "sparse/coo/cuda/optimization/test_functions.h"
#include "sparse/coo/cuda/integral_transform/hilbert_transform.h"
#include "sparse/coo/cuda/integral_transform/inverse_hilbert_transform.h"
#include "sparse/csr/cuda/special_functions.h"
#include "sparse/csr/cuda/optimization/test_functions.h"
#include "sparse/csr/cuda/integral_transform/hilbert_transform.h"
#include "sparse/csr/cuda/integral_transform/inverse_hilbert_transform.h"
#include "quantized/cuda/special_functions.h"
#include "quantized/cuda/optimization/test_functions.h"
#include "quantized/cuda/integral_transform/hilbert_transform.h"
#include "quantized/cuda/integral_transform/inverse_hilbert_transform.h"
#endif

extern "C" {
  PyObject* PyInit__csrc(void) {
    static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_csrc",
      nullptr,
      -1,
      nullptr,
    };

    return PyModule_Create(&module_def);
  }
}

TORCH_LIBRARY(torchscience, module) {
  // `torchscience.optimization.test_functions`
  module.def("rosenbrock(Tensor x, Tensor a, Tensor b) -> Tensor");
  module.def("rosenbrock_backward(Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("rosenbrock_backward_backward(Tensor grad_grad_x, Tensor grad_grad_a, Tensor grad_grad_b, Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  // `torchscience.signal_processing.filter`
  module.def("butterworth_analog_bandpass_filter(int n, Tensor omega_p1, Tensor omega_p2) -> Tensor");
  module.def("butterworth_analog_bandpass_filter_backward(Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor)");
  module.def("butterworth_analog_bandpass_filter_backward_backward(Tensor grad_grad_omega_p1, Tensor grad_grad_omega_p2, Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor, Tensor)");

  // `torchscience.signal_processing.waveform`
  module.def("sine_wave(int n, float frequency=1.0, float sample_rate=1.0, float amplitude=1.0, float phase=0.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False) -> Tensor");

  // `torchscience.signal_processing.window_function`
  module.def("rectangular_window(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False) -> Tensor");

  // `torchscience.special_functions`
  module.def("chebyshev_polynomial_t(Tensor v, Tensor z) -> Tensor");
  module.def("chebyshev_polynomial_t_backward(Tensor gradient_output, Tensor v, Tensor z) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_t_backward_backward(Tensor gradient_gradient_v, Tensor gradient_gradient_z, Tensor gradient_output, Tensor v, Tensor z) -> (Tensor, Tensor, Tensor)");

  module.def("gamma(Tensor z) -> Tensor");
  module.def("gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("gamma_backward_backward(Tensor gradient_gradient_z, Tensor gradient_output, Tensor z) -> (Tensor, Tensor)");

  module.def("hypergeometric_2_f_1(Tensor a, Tensor b, Tensor c, Tensor z) -> Tensor");
  module.def("hypergeometric_2_f_1_backward(Tensor gradient_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("hypergeometric_2_f_1_backward_backward(Tensor gradient_gradient_a, Tensor gradient_gradient_b, Tensor gradient_gradient_c, Tensor gradient_gradient_z, Tensor gradient_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  module.def("incomplete_beta(Tensor z, Tensor a, Tensor b) -> Tensor");
  module.def("incomplete_beta_backward(Tensor gradient_output, Tensor z, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("incomplete_beta_backward_backward(Tensor gradient_gradient_z, Tensor gradient_gradient_a, Tensor gradient_gradient_b, Tensor gradient_output, Tensor z, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  // `torchscience.statistics.descriptive`
  module.def("kurtosis(Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  module.def("kurtosis_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  module.def("kurtosis_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> (Tensor, Tensor)");

  // `torchscience.integral_transform`
  // n=-1 means use input size along dim (no padding/truncation)
  // padding_mode: 0=constant, 1=reflect, 2=replicate, 3=circular
  module.def("hilbert_transform(Tensor input, int n=-1, int dim=-1, int padding_mode=0, float padding_value=0.0, Tensor? window=None) -> Tensor");
  module.def("hilbert_transform_backward(Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("hilbert_transform_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");

  module.def("inverse_hilbert_transform(Tensor input, int n=-1, int dim=-1, int padding_mode=0, float padding_value=0.0, Tensor? window=None) -> Tensor");
  module.def("inverse_hilbert_transform_backward(Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("inverse_hilbert_transform_backward_backward(Tensor grad_grad_input, Tensor grad_output, Tensor input, int n, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");
}
