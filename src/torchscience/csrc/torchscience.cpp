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
#include "cpu/signal_processing/window_functions.h"
#include "meta/signal_processing/window_functions.h"
#include "autograd/signal_processing/window_functions.h"
#include "cpu/signal_processing/waveform/sine_wave.h"
#include "cpu/signal_processing/waveform/sine_wave_backward.h"
#include "cpu/signal_processing/waveform/square_wave.h"
#include "meta/signal_processing/waveform/sine_wave.h"
#include "autograd/signal_processing/waveform/sine_wave.h"
// noise - CompositeExplicitAutograd
#include "cpu/signal_processing/noise/white_noise.h"
#include "cpu/signal_processing/noise/pink_noise.h"
#include "cpu/signal_processing/noise/brown_noise.h"
#include "cpu/signal_processing/noise/blue_noise.h"
#include "cpu/signal_processing/noise/violet_noise.h"
#include "cpu/signal_processing/noise/poisson_noise.h"
#include "cpu/signal_processing/noise/shot_noise.h"
#include "cpu/signal_processing/noise/impulse_noise.h"
#include "composite/optimization/test_functions.h"

#include "cpu/distance/minkowski_distance.h"
#include "cpu/graphics/shading/cook_torrance.h"
#include "cpu/graphics/shading/phong.h"
#include "cpu/graphics/shading/schlick_reflectance.h"
#include "cpu/graphics/lighting/spotlight.h"
#include "cpu/graphics/tone_mapping/reinhard.h"
#include "cpu/graphics/texture_mapping/cube_mapping.h"
#include "cpu/graphics/projection/perspective_projection.h"
#include "cpu/graphics/color/srgb_to_hsv.h"
#include "cpu/graphics/color/hsv_to_srgb.h"
#include "cpu/graphics/color/srgb_to_srgb_linear.h"
#include "cpu/graphics/color/srgb_linear_to_srgb.h"
#include "cpu/signal_processing/filter.h"
#include "cpu/optimization/test_functions.h"
#include "cpu/optimization/combinatorial.h"
#include "cpu/statistics/descriptive/kurtosis.h"
#include "cpu/statistics/descriptive/histogram.h"
#include "cpu/statistics/hypothesis_test/one_sample_t_test.h"
#include "cpu/statistics/hypothesis_test/two_sample_t_test.h"
#include "cpu/statistics/hypothesis_test/paired_t_test.h"
#include "cpu/integral_transform/hilbert_transform.h"
#include "cpu/integral_transform/inverse_hilbert_transform.h"
#include "cpu/test/sum_squares.h"
#include "cpu/graph_theory/floyd_warshall.h"
#include "cpu/graph_theory/connected_components.h"
#include "cpu/graph_theory/dijkstra.h"
#include "cpu/graph_theory/bellman_ford.h"
#include "cpu/graph_theory/minimum_spanning_tree.h"
#include "cpu/graph_theory/maximum_bipartite_matching.h"
#include "cpu/information_theory/kullback_leibler_divergence.h"
#include "cpu/information_theory/jensen_shannon_divergence.h"
#include "cpu/space_partitioning/kd_tree.h"
#include "cpu/space_partitioning/k_nearest_neighbors.h"
#include "cpu/space_partitioning/range_search.h"
#include "cpu/space_partitioning/bvh.h"
#include "cpu/geometry/ray_intersect.h"
#include "cpu/geometry/closest_point.h"
#include "cpu/geometry/ray_occluded.h"
#include "cpu/geometry/transform/reflect.h"
#include "cpu/geometry/transform/refract.h"
#include "cpu/geometry/transform/quaternion_multiply.h"
#include "cpu/geometry/transform/quaternion_inverse.h"
#include "cpu/geometry/transform/quaternion_normalize.h"
#include "cpu/geometry/transform/quaternion_apply.h"
#include "cpu/geometry/transform/quaternion_to_matrix.h"
#include "cpu/geometry/transform/matrix_to_quaternion.h"
#include "cpu/geometry/transform/quaternion_slerp.h"
#include "cpu/geometry/convex_hull.h"
#include "cpu/encryption/chacha20.h"
#include "cpu/encryption/sha256.h"
#include "meta/encryption/chacha20.h"
#include "meta/encryption/sha256.h"
#include "cpu/privacy/gaussian_mechanism.h"
#include "cpu/privacy/laplace_mechanism.h"
#include "meta/privacy/gaussian_mechanism.h"
#include "meta/privacy/laplace_mechanism.h"
#include "autograd/privacy/gaussian_mechanism.h"
#include "autograd/privacy/laplace_mechanism.h"

// probability
#include "cpu/probability/normal.h"
#include "meta/probability/normal.h"
#include "autograd/probability/normal.h"

#include "autograd/distance/minkowski_distance.h"
#include "autograd/graphics/shading/cook_torrance.h"
#include "autograd/graphics/shading/phong.h"
#include "autograd/graphics/shading/schlick_reflectance.h"
#include "autograd/graphics/lighting/spotlight.h"
#include "autograd/graphics/tone_mapping/reinhard.h"
#include "autograd/graphics/projection/perspective_projection.h"
#include "autograd/graphics/color/srgb_to_hsv.h"
#include "autograd/graphics/color/hsv_to_srgb.h"
#include "autograd/graphics/color/srgb_to_srgb_linear.h"
#include "autograd/graphics/color/srgb_linear_to_srgb.h"
#include "autograd/signal_processing/filter.h"
#include "autograd/optimization/test_functions.h"
#include "autograd/optimization/combinatorial.h"
#include "autograd/statistics/descriptive/kurtosis.h"
#include "autograd/integral_transform/hilbert_transform.h"
#include "autograd/integral_transform/inverse_hilbert_transform.h"
#include "autograd/test/sum_squares.h"
#include "autograd/information_theory/kullback_leibler_divergence.h"
#include "autograd/information_theory/jensen_shannon_divergence.h"
#include "autograd/geometry/transform/reflect.h"
#include "autograd/geometry/transform/refract.h"
#include "autograd/geometry/transform/quaternion_multiply.h"
#include "autograd/geometry/transform/quaternion_inverse.h"
#include "autograd/geometry/transform/quaternion_normalize.h"
#include "autograd/geometry/transform/quaternion_apply.h"
#include "autograd/geometry/transform/quaternion_to_matrix.h"
#include "autograd/geometry/transform/matrix_to_quaternion.h"
#include "autograd/geometry/transform/quaternion_slerp.h"

#include "meta/distance/minkowski_distance.h"
#include "meta/graphics/shading/cook_torrance.h"
#include "meta/graphics/shading/phong.h"
#include "meta/graphics/shading/schlick_reflectance.h"
#include "meta/graphics/lighting/spotlight.h"
#include "meta/graphics/tone_mapping/reinhard.h"
#include "meta/graphics/texture_mapping/cube_mapping.h"
#include "meta/graphics/projection/perspective_projection.h"
#include "meta/graphics/color/srgb_to_hsv.h"
#include "meta/graphics/color/hsv_to_srgb.h"
#include "meta/graphics/color/srgb_to_srgb_linear.h"
#include "meta/graphics/color/srgb_linear_to_srgb.h"
#include "meta/signal_processing/filter.h"
#include "meta/optimization/test_functions.h"
#include "meta/optimization/combinatorial.h"
#include "meta/statistics/descriptive/kurtosis.h"
#include "meta/statistics/descriptive/histogram.h"
#include "meta/statistics/hypothesis_test/one_sample_t_test.h"
#include "meta/statistics/hypothesis_test/two_sample_t_test.h"
#include "meta/statistics/hypothesis_test/paired_t_test.h"
#include "meta/integral_transform/hilbert_transform.h"
#include "meta/integral_transform/inverse_hilbert_transform.h"
#include "meta/test/sum_squares.h"
#include "meta/graph_theory/floyd_warshall.h"
#include "meta/graph_theory/connected_components.h"
#include "meta/graph_theory/dijkstra.h"
#include "meta/graph_theory/bellman_ford.h"
#include "meta/graph_theory/minimum_spanning_tree.h"
#include "meta/graph_theory/maximum_bipartite_matching.h"
#include "meta/information_theory/kullback_leibler_divergence.h"
#include "meta/information_theory/jensen_shannon_divergence.h"
#include "meta/space_partitioning/kd_tree.h"
#include "meta/space_partitioning/k_nearest_neighbors.h"
#include "meta/space_partitioning/range_search.h"
#include "meta/geometry/transform/reflect.h"
#include "meta/geometry/transform/refract.h"
#include "meta/geometry/transform/quaternion_multiply.h"
#include "meta/geometry/transform/quaternion_inverse.h"
#include "meta/geometry/transform/quaternion_normalize.h"
#include "meta/geometry/transform/quaternion_apply.h"
#include "meta/geometry/transform/quaternion_to_matrix.h"
#include "meta/geometry/transform/matrix_to_quaternion.h"
#include "meta/geometry/transform/quaternion_slerp.h"
#include "meta/geometry/convex_hull.h"
#include "autograd/space_partitioning/k_nearest_neighbors.h"
#include "autograd/space_partitioning/range_search.h"

#include "autocast/signal_processing/filter.h"
#include "autocast/statistics/descriptive/kurtosis.h"
#include "autocast/integral_transform/hilbert_transform.h"
#include "autocast/integral_transform/inverse_hilbert_transform.h"
#include "autocast/test/sum_squares.h"
#include "autocast/space_partitioning/kd_tree.h"
#include "autocast/space_partitioning/k_nearest_neighbors.h"
#include "autocast/space_partitioning/range_search.h"

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
#include "cuda/space_partitioning/kd_tree.cuh"
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

  module.def("regularized_gamma_p(Tensor a, Tensor x) -> Tensor");
  module.def("regularized_gamma_p_backward(Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor)");
  module.def("regularized_gamma_p_backward_backward(Tensor grad_grad_a, Tensor grad_grad_x, Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor, Tensor)");

  module.def("regularized_gamma_q(Tensor a, Tensor x) -> Tensor");
  module.def("regularized_gamma_q_backward(Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor)");
  module.def("regularized_gamma_q_backward_backward(Tensor grad_grad_a, Tensor grad_grad_x, Tensor grad, Tensor a, Tensor x) -> (Tensor, Tensor, Tensor)");

  // distance
  module.def("minkowski_distance(Tensor x, Tensor y, float p, Tensor? weight) -> Tensor");
  module.def("minkowski_distance_backward(Tensor grad_output, Tensor x, Tensor y, float p, Tensor? weight, Tensor dist_output) -> (Tensor, Tensor, Tensor)");

  // graphics.shading
  module.def("cook_torrance(Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> Tensor");
  module.def("cook_torrance_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def("cook_torrance_backward_backward(Tensor gg_normal, Tensor gg_view, Tensor gg_light, Tensor gg_roughness, Tensor gg_f0, Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // Phong shading
  module.def("phong(Tensor normal, Tensor view, Tensor light, Tensor shininess) -> Tensor");
  module.def("phong_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor shininess) -> (Tensor, Tensor, Tensor, Tensor)");

  // Schlick reflectance (Fresnel approximation)
  module.def("schlick_reflectance(Tensor cosine, Tensor r0) -> Tensor");
  module.def("schlick_reflectance_backward(Tensor grad_output, Tensor cosine, Tensor r0) -> Tensor");

  // graphics.lighting
  module.def("spotlight(Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor)");
  module.def("spotlight_backward(Tensor grad_irradiance, Tensor light_pos, Tensor surface_pos, Tensor spot_direction, Tensor intensity, Tensor inner_angle, Tensor outer_angle) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // graphics.tone_mapping
  module.def("reinhard(Tensor input, Tensor? white_point) -> Tensor");
  module.def("reinhard_backward(Tensor grad_output, Tensor input, Tensor? white_point) -> (Tensor, Tensor)");

  // graphics.color
  module.def("srgb_to_hsv(Tensor input) -> Tensor");
  module.def("srgb_to_hsv_backward(Tensor grad_output, Tensor input) -> Tensor");

  module.def("hsv_to_srgb(Tensor input) -> Tensor");
  module.def("hsv_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  module.def("srgb_to_srgb_linear(Tensor input) -> Tensor");
  module.def("srgb_to_srgb_linear_backward(Tensor grad_output, Tensor input) -> Tensor");

  module.def("srgb_linear_to_srgb(Tensor input) -> Tensor");
  module.def("srgb_linear_to_srgb_backward(Tensor grad_output, Tensor input) -> Tensor");

  // graphics.texture_mapping
  module.def("cube_mapping(Tensor direction) -> (Tensor, Tensor, Tensor)");

  // graphics.projection
  module.def("perspective_projection(Tensor fov, Tensor aspect, Tensor near, Tensor far) -> Tensor");
  module.def("perspective_projection_backward(Tensor grad_output, Tensor fov, Tensor aspect, Tensor near, Tensor far) -> (Tensor, Tensor, Tensor, Tensor)");

  // optimization.test_functions
  module.def("rosenbrock(Tensor x, Tensor a, Tensor b) -> Tensor");
  module.def("rosenbrock_backward(Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor)");
  module.def("rosenbrock_backward_backward(Tensor gg_x, Tensor gg_a, Tensor gg_b, Tensor grad_output, Tensor x, Tensor a, Tensor b) -> (Tensor, Tensor, Tensor, Tensor)");

  // optimization.combinatorial
  module.def("sinkhorn(Tensor C, Tensor a, Tensor b, float epsilon, int maxiter, float tol) -> Tensor");
  module.def("sinkhorn_backward(Tensor grad_output, Tensor P, Tensor C, float epsilon) -> Tensor");

  // signal_processing.filter
  module.def("butterworth_analog_bandpass_filter(int n, Tensor omega_p1, Tensor omega_p2) -> Tensor");
  module.def("butterworth_analog_bandpass_filter_backward(Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor)");
  module.def("butterworth_analog_bandpass_filter_backward_backward(Tensor gg_omega_p1, Tensor gg_omega_p2, Tensor grad_output, int n, Tensor omega_p1, Tensor omega_p2) -> (Tensor, Tensor, Tensor)");

  // signal_processing.waveform
  module.def("sine_wave(int? n=None, Tensor? t=None, *, "
             "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");
  module.def("sine_wave_backward(Tensor grad_output, int? n, Tensor? t, "
             "Tensor frequency, float sample_rate, Tensor amplitude, Tensor phase) -> "
             "(Tensor, Tensor, Tensor, Tensor)");

  module.def("square_wave(int? n=None, Tensor? t=None, *, "
             "Tensor frequency, float sample_rate=1.0, Tensor amplitude, Tensor phase, Tensor duty, "
             "ScalarType? dtype=None, Layout? layout=None, Device? device=None) -> Tensor");

  // signal_processing.window_function
  module.def("rectangular_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");

  // Parameterless windows
  module.def("hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_hann_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("hamming_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_hamming_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("blackman_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_blackman_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("bartlett_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_bartlett_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("cosine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_cosine_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("nuttall_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");
  module.def("periodic_nuttall_window(int n, ScalarType? dtype, Layout? layout, Device? device, bool requires_grad) -> Tensor");

  // Parameterized windows: Gaussian
  module.def("gaussian_window(int n, Tensor std, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_gaussian_window(int n, Tensor std, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("gaussian_window_backward(Tensor grad_output, Tensor output, int n, Tensor std) -> Tensor");
  module.def("periodic_gaussian_window_backward(Tensor grad_output, Tensor output, int n, Tensor std) -> Tensor");

  // Parameterized windows: General Hamming
  module.def("general_hamming_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_general_hamming_window(int n, Tensor alpha, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("general_hamming_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");
  module.def("periodic_general_hamming_window_backward(Tensor grad_output, Tensor output, int n, Tensor alpha) -> Tensor");

  // Parameterized windows: General Cosine
  module.def("general_cosine_window(int n, Tensor coeffs, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("periodic_general_cosine_window(int n, Tensor coeffs, ScalarType? dtype, Layout? layout, Device? device) -> Tensor");
  module.def("general_cosine_window_backward(Tensor grad_output, Tensor output, int n, Tensor coeffs) -> Tensor");
  module.def("periodic_general_cosine_window_backward(Tensor grad_output, Tensor output, int n, Tensor coeffs) -> Tensor");

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
  module.def("connected_components(Tensor adjacency, bool directed, str connection) -> (int, Tensor)");
  module.def("dijkstra(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor)");
  module.def("bellman_ford(Tensor adjacency, int source, bool directed) -> (Tensor, Tensor, bool)");
  module.def("minimum_spanning_tree(Tensor adjacency) -> (Tensor, Tensor)");
  module.def("maximum_bipartite_matching(Tensor biadjacency) -> (Tensor, Tensor, Tensor)");

  // combinatorics
  module.def("binomial_coefficient(Tensor n, Tensor k) -> Tensor");
  module.def("binomial_coefficient_backward(Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor)");
  module.def("binomial_coefficient_backward_backward(Tensor gg_n, Tensor gg_k, Tensor grad_output, Tensor n, Tensor k) -> (Tensor, Tensor, Tensor)");

  // signal_processing.noise
  module.def("white_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("pink_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("brown_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("blue_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("violet_noise(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("poisson_noise(int[] size, Tensor rate, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, Generator? generator=None) -> Tensor");
  module.def("shot_noise(int[] size, Tensor rate, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool requires_grad=False, Generator? generator=None) -> Tensor");
  module.def("impulse_noise(int[] size, Tensor p_salt, Tensor p_pepper, float salt_value, float pepper_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, Generator? generator=None) -> Tensor");

  // information_theory
  module.def("kullback_leibler_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> Tensor");
  module.def("kullback_leibler_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor)");
  module.def("kullback_leibler_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, bool pairwise) -> (Tensor, Tensor, Tensor)");

  module.def("jensen_shannon_divergence(Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> Tensor");
  module.def("jensen_shannon_divergence_backward(Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> (Tensor, Tensor)");
  module.def("jensen_shannon_divergence_backward_backward(Tensor gg_p, Tensor gg_q, Tensor grad_output, Tensor p, Tensor q, int dim, str input_type, str reduction, float? base, bool pairwise) -> (Tensor, Tensor, Tensor)");

  // space_partitioning
  // Batched tree build - always use this, even for single trees (pass B=1)
  // Input: points (B, n, d), leaf_size
  // Returns: tuple of pre-padded (B, max_*) tensors for efficient consumption
  // (points, split_dim, split_val, left, right, indices, leaf_starts, leaf_counts)
  module.def("kd_tree_build_batched(Tensor points, int leaf_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // k-nearest neighbors query
  module.def("k_nearest_neighbors(Tensor points, Tensor split_dim, Tensor split_val, Tensor left, Tensor right, Tensor indices, Tensor leaf_starts, Tensor leaf_counts, Tensor queries, int k, float p) -> (Tensor, Tensor)");

  // range search query (returns nested tensors)
  module.def("range_search(Tensor points, Tensor split_dim, Tensor split_val, Tensor left, Tensor right, Tensor indices, Tensor leaf_starts, Tensor leaf_counts, Tensor queries, float radius, float p) -> (Tensor, Tensor)");

  // space_partitioning.bvh
  module.def("bvh_build(Tensor vertices, Tensor faces) -> Tensor");
  module.def("bvh_destroy(int scene_handle) -> ()");

  // geometry.ray_intersect
  module.def("bvh_ray_intersect(int scene_handle, Tensor origins, Tensor directions) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // geometry.closest_point
  module.def("bvh_closest_point(int scene_handle, Tensor query_points) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  // geometry.ray_occluded
  module.def("bvh_ray_occluded(int scene_handle, Tensor origins, Tensor directions) -> Tensor");

  // geometry.transform
  module.def("reflect(Tensor direction, Tensor normal) -> Tensor");
  module.def("reflect_backward(Tensor grad_output, Tensor direction, Tensor normal) -> (Tensor, Tensor)");

  module.def("refract(Tensor direction, Tensor normal, Tensor eta) -> Tensor");
  module.def("refract_backward(Tensor grad_output, Tensor direction, Tensor normal, Tensor eta) -> (Tensor, Tensor, Tensor)");

  // Quaternion operations
  module.def("quaternion_multiply(Tensor q1, Tensor q2) -> Tensor");
  module.def("quaternion_multiply_backward(Tensor grad_output, Tensor q1, Tensor q2) -> (Tensor, Tensor)");

  module.def("quaternion_inverse(Tensor q) -> Tensor");
  module.def("quaternion_inverse_backward(Tensor grad_output, Tensor q) -> Tensor");

  module.def("quaternion_normalize(Tensor q) -> Tensor");
  module.def("quaternion_normalize_backward(Tensor grad_output, Tensor q) -> Tensor");

  module.def("quaternion_apply(Tensor q, Tensor point) -> Tensor");
  module.def("quaternion_apply_backward(Tensor grad_output, Tensor q, Tensor point) -> (Tensor, Tensor)");

  module.def("quaternion_to_matrix(Tensor q) -> Tensor");
  module.def("quaternion_to_matrix_backward(Tensor grad_output, Tensor q) -> Tensor");

  module.def("matrix_to_quaternion(Tensor matrix) -> Tensor");
  module.def("matrix_to_quaternion_backward(Tensor grad_output, Tensor matrix) -> Tensor");

  module.def("quaternion_slerp(Tensor q1, Tensor q2, Tensor t) -> Tensor");
  module.def("quaternion_slerp_backward(Tensor grad_output, Tensor q1, Tensor q2, Tensor t) -> (Tensor, Tensor, Tensor)");

  // geometry.convex_hull
  module.def("convex_hull(Tensor points) -> "
             "(Tensor vertices, Tensor simplices, Tensor neighbors, "
             "Tensor equations, Tensor area, Tensor volume, "
             "Tensor n_vertices, Tensor n_facets)");

  // encryption
  module.def("chacha20(Tensor key, Tensor nonce, int num_bytes, int counter=0) -> Tensor");
  module.def("sha256(Tensor data) -> Tensor");

  // Privacy operators
  module.def("gaussian_mechanism(Tensor x, Tensor noise, float sigma) -> Tensor");
  module.def("gaussian_mechanism_backward(Tensor grad_output) -> Tensor");

  module.def("laplace_mechanism(Tensor x, Tensor noise, float b) -> Tensor");
  module.def("laplace_mechanism_backward(Tensor grad_output) -> Tensor");

  // Probability - Normal distribution
  module.def("normal_cdf(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_cdf_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("normal_cdf_backward_backward(Tensor grad_grad_x, Tensor grad_grad_loc, Tensor grad_grad_scale, Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor, Tensor)");
  module.def("normal_pdf(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_pdf_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("normal_ppf(Tensor p, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_ppf_backward(Tensor grad, Tensor p, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("normal_sf(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_sf_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
  module.def("normal_logpdf(Tensor x, Tensor loc, Tensor scale) -> Tensor");
  module.def("normal_logpdf_backward(Tensor grad, Tensor x, Tensor loc, Tensor scale) -> (Tensor, Tensor, Tensor)");
}
