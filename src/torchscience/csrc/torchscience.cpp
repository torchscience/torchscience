#include <torch/extension.h>

// special_functions
#include "cpu/special_functions.h"
#include "meta/special_functions.h"
#include "autograd/special_functions.h"
#include "autocast/special_functions.h"

// combinatorics
#include "cpu/combinatorics.h"
#include "meta/combinatorics.h"
#include "autograd/combinatorics.h"
#include "autocast/combinatorics.h"
#include "sparse/coo/cpu/special_functions.h"
#include "sparse/csr/cpu/special_functions.h"
#include "quantized/cpu/special_functions.h"

// other operators - Phase 2
#include "composite/signal_processing/window_functions.h"
#include "composite/signal_processing/waveform.h"
#include "composite/signal_processing/noise.h"
#include "composite/optimization/test_functions.h"

#include "cpu/distance/minkowski_distance.h"
#include "cpu/graphics/shading/cook_torrance.h"
#include "cpu/graphics/color/srgb_to_hsv.h"
#include "cpu/graphics/color/hsv_to_srgb.h"
#include "cpu/signal_processing/filter.h"
#include "cpu/optimization/test_functions.h"
#include "cpu/statistics/descriptive/kurtosis.h"
#include "cpu/statistics/descriptive/histogram.h"
#include "cpu/statistics/hypothesis_test/one_sample_t_test.h"
#include "cpu/statistics/hypothesis_test/two_sample_t_test.h"
#include "cpu/statistics/hypothesis_test/paired_t_test.h"
#include "cpu/integral_transform/hilbert_transform.h"
#include "cpu/integral_transform/inverse_hilbert_transform.h"
#include "cpu/test/sum_squares.h"
#include "cpu/graph_theory/floyd_warshall.h"

#include "autograd/distance/minkowski_distance.h"
#include "autograd/graphics/shading/cook_torrance.h"
#include "autograd/graphics/color/srgb_to_hsv.h"
#include "autograd/graphics/color/hsv_to_srgb.h"
#include "autograd/signal_processing/filter.h"
#include "autograd/optimization/test_functions.h"
#include "autograd/statistics/descriptive/kurtosis.h"
#include "autograd/integral_transform/hilbert_transform.h"
#include "autograd/integral_transform/inverse_hilbert_transform.h"
#include "autograd/test/sum_squares.h"

#include "meta/distance/minkowski_distance.h"
#include "meta/graphics/shading/cook_torrance.h"
#include "meta/graphics/color/srgb_to_hsv.h"
#include "meta/graphics/color/hsv_to_srgb.h"
#include "meta/signal_processing/filter.h"
#include "meta/optimization/test_functions.h"
#include "meta/statistics/descriptive/kurtosis.h"
#include "meta/statistics/descriptive/histogram.h"
#include "meta/statistics/hypothesis_test/one_sample_t_test.h"
#include "meta/statistics/hypothesis_test/two_sample_t_test.h"
#include "meta/statistics/hypothesis_test/paired_t_test.h"
#include "meta/integral_transform/hilbert_transform.h"
#include "meta/integral_transform/inverse_hilbert_transform.h"
#include "meta/test/sum_squares.h"
#include "meta/graph_theory/floyd_warshall.h"

#include "autocast/signal_processing/filter.h"
#include "autocast/statistics/descriptive/kurtosis.h"
#include "autocast/integral_transform/hilbert_transform.h"
#include "autocast/integral_transform/inverse_hilbert_transform.h"
#include "autocast/test/sum_squares.h"

#include "sparse/coo/cpu/optimization/test_functions.h"
#include "sparse/coo/cpu/integral_transform/hilbert_transform.h"
#include "sparse/coo/cpu/integral_transform/inverse_hilbert_transform.h"
#include "sparse/csr/cpu/optimization/test_functions.h"
#include "sparse/csr/cpu/integral_transform/hilbert_transform.h"
#include "sparse/csr/cpu/integral_transform/inverse_hilbert_transform.h"
#include "quantized/cpu/optimization/test_functions.h"
#include "quantized/cpu/integral_transform/hilbert_transform.h"
#include "quantized/cpu/integral_transform/inverse_hilbert_transform.h"

#ifdef TORCHSCIENCE_CUDA
#include "cuda/graphics/shading/cook_torrance.cu"
#include "cuda/optimization/test_functions.cu"
#include "cuda/statistics/descriptive/kurtosis.cu"
#include "cuda/statistics/descriptive/histogram.cu"
#include "cuda/integral_transform/hilbert_transform.cu"
#include "cuda/integral_transform/inverse_hilbert_transform.cu"
#include "cuda/graph_theory/floyd_warshall.cu"
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
  // special_functions
  module.def("gamma(Tensor z) -> Tensor");
  module.def("gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("gamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  module.def("digamma(Tensor z) -> Tensor");
  module.def("digamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("digamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  module.def("trigamma(Tensor z) -> Tensor");
  module.def("trigamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("trigamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  module.def("beta(Tensor a, Tensor b) -> Tensor");
  module.def("beta_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  module.def("chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor");
  module.def("chebyshev_polynomial_t_backward(Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor)");
  module.def("chebyshev_polynomial_t_backward_backward(Tensor gg_x, Tensor gg_n, Tensor grad_output, Tensor x, Tensor n) -> (Tensor, Tensor, Tensor)");

  module.def("incomplete_beta(Tensor x, Tensor a, Tensor b) -> Tensor");
  module.def("incomplete_beta_backward(Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("incomplete_beta_backward_backward(Tensor gg_x, Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  module.def("hypergeometric_2_f_1(Tensor a, Tensor b, Tensor c, Tensor z) -> Tensor");
  module.def("hypergeometric_2_f_1_backward(Tensor grad_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("hypergeometric_2_f_1_backward_backward(Tensor gg_a, Tensor gg_b, Tensor gg_c, Tensor gg_z, Tensor grad_output, Tensor a, Tensor b, Tensor c, Tensor z) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");

  module.def("polygamma(Tensor n, Tensor z) -> Tensor");
  module.def("polygamma_backward(Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor)");
  module.def("polygamma_backward_backward(Tensor gg_n, Tensor gg_z, Tensor grad_output, Tensor n, Tensor z) -> (Tensor, Tensor, Tensor)");

  module.def("log_beta(Tensor a, Tensor b) -> Tensor");
  module.def("log_beta_backward(Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor)");
  module.def("log_beta_backward_backward(Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");

  module.def("log_gamma(Tensor z) -> Tensor");
  module.def("log_gamma_backward(Tensor grad_output, Tensor z) -> Tensor");
  module.def("log_gamma_backward_backward(Tensor gg_z, Tensor grad_output, Tensor z) -> (Tensor, Tensor)");

  // distance
  module.def("minkowski_distance(Tensor x, Tensor y, float p, Tensor? weight) -> Tensor");
  module.def("minkowski_distance_backward(Tensor grad_output, Tensor x, Tensor y, float p, Tensor? weight, Tensor dist_output) -> (Tensor, Tensor, Tensor)");

  // graphics.shading
  module.def("cook_torrance(Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> Tensor");
  module.def("cook_torrance_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def("cook_torrance_backward_backward(Tensor gg_normal, Tensor gg_view, Tensor gg_light, Tensor gg_roughness, Tensor gg_f0, Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // graphics.color
  module.def("srgb_to_hsv(Tensor input) -> Tensor");
  module.def("srgb_to_hsv_backward(Tensor grad_output, Tensor input) -> Tensor");

  module.def("hsv_to_srgb(Tensor input) -> Tensor");
  module.def("hsv_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  // optimization.test_functions
  module.def("rosenbrock(Tensor x, Tensor a, Tensor b) -> Tensor");
  module.def("rosenbrock_backward(Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("rosenbrock_backward_backward(Tensor gg_x, Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  // signal_processing.filter
  module.def("butterworth_analog_bandpass_filter(int n, Tensor omega_p1, Tensor omega_p2) -> Tensor");
  module.def("butterworth_analog_bandpass_filter_backward(Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor)");
  module.def("butterworth_analog_bandpass_filter_backward_backward(Tensor gg_omega_p1, Tensor gg_omega_p2, Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor, Tensor)");

  // signal_processing.waveform
  module.def("sine_wave(int n, float frequency, float sample_rate, float amplitude, float phase, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");

  // signal_processing.window_function
  module.def("rectangular_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");

  // statistics.descriptive
  module.def("kurtosis(Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  module.def("kurtosis_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> Tensor");
  module.def("kurtosis_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim, bool fisher, bool bias) -> (Tensor, Tensor)");

  module.def("histogram(Tensor input, int bins, float[]? range, Tensor? weight, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");
  module.def("histogram_edges(Tensor input, Tensor bins, Tensor? weight, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");

  // statistics.hypothesis_test
  module.def("one_sample_t_test(Tensor input, float popmean, str alternative) -> (Tensor, Tensor, Tensor)");
  module.def("two_sample_t_test(Tensor input1, Tensor input2, bool equal_var, str alternative) -> (Tensor, Tensor, Tensor)");
  module.def("paired_t_test(Tensor input1, Tensor input2, str alternative) -> (Tensor, Tensor, Tensor)");

  // integral_transform
  module.def("hilbert_transform(Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("hilbert_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("hilbert_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");

  module.def("inverse_hilbert_transform(Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("inverse_hilbert_transform_backward(Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> Tensor");
  module.def("inverse_hilbert_transform_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int n_param, int dim, int padding_mode, float padding_value, Tensor? window) -> (Tensor, Tensor)");

  // test (for validating reduction macros)
  module.def("sum_squares(Tensor input, int[]? dim, bool keepdim) -> Tensor");
  module.def("sum_squares_backward(Tensor grad_output, Tensor input, int[]? dim, bool keepdim) -> Tensor");
  module.def("sum_squares_backward_backward(Tensor gg_input, Tensor grad_output, Tensor input, int[]? dim, bool keepdim) -> (Tensor, Tensor)");

  // graph_theory
  module.def("floyd_warshall(Tensor input, bool directed) -> (Tensor, Tensor, bool)");

  // combinatorics
  module.def("binomial_coefficient(Tensor n, Tensor k) -> Tensor");
  module.def("binomial_coefficient_backward(Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor)");
  module.def("binomial_coefficient_backward_backward(Tensor gg_n, Tensor gg_k, Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor, Tensor)");

  // signal_processing.noise
  module.def("pink_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
}
