#include <torch/extension.h>

#include "core/schema_generation.h"
#include "operators/special_functions.def"
#include "core/reduction_schema.h"
#include "operators/reductions.def"
#include "core/transform_schema.h"
#include "operators/transforms.def"

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

#include "cpu/distance/minkowski_distance.h"
#include "cpu/graphics/shading/cook_torrance.h"
#include "cpu/signal_processing/filter.h"
#include "cpu/optimization/test_functions.h"
#include "cpu/statistics/descriptive/kurtosis.h"
#include "cpu/statistics/descriptive/histogram.h"
#include "cpu/integral_transform/hilbert_transform.h"
#include "cpu/integral_transform/inverse_hilbert_transform.h"
#include "autograd/distance/minkowski_distance.h"
#include "autograd/graphics/shading/cook_torrance.h"
#include "autograd/signal_processing/filter.h"
#include "autograd/optimization/test_functions.h"
#include "autograd/statistics/descriptive/kurtosis.h"
#include "autograd/integral_transform/hilbert_transform.h"
#include "autograd/integral_transform/inverse_hilbert_transform.h"
#include "meta/distance/minkowski_distance.h"
#include "meta/graphics/shading/cook_torrance.h"
#include "meta/signal_processing/filter.h"
#include "meta/optimization/test_functions.h"
#include "meta/statistics/descriptive/kurtosis.h"
#include "meta/statistics/descriptive/histogram.h"
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
#include "cuda/graphics/shading/cook_torrance.cu"
#include "cuda/optimization/test_functions.cu"
#include "cuda/statistics/descriptive/kurtosis.cu"
#include "cuda/statistics/descriptive/histogram.cu"
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
  // `torchscience.distance`
  module.def("minkowski_distance(Tensor x, Tensor y, float p, Tensor? weight) -> Tensor");
  module.def("minkowski_distance_backward(Tensor grad_output, Tensor x, Tensor y, float p, Tensor? weight, Tensor dist_output) -> (Tensor, Tensor, Tensor)");

  // `torchscience.graphics.shading`
  module.def("cook_torrance(Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> Tensor");
  module.def("cook_torrance_backward(Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  module.def("cook_torrance_backward_backward(Tensor gg_normal, Tensor gg_view, Tensor gg_light, Tensor gg_roughness, Tensor gg_f0, Tensor grad_output, Tensor normal, Tensor view, Tensor light, Tensor roughness, Tensor f0) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

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

  // `torchscience.special_functions` - auto-generated from X-macro
  #define DEFINE_OP(name, arity, impl) DEFINE_POINTWISE_SCHEMA(module, name, arity);
  TORCHSCIENCE_SPECIAL_FUNCTIONS(DEFINE_OP)
  #undef DEFINE_OP

  // `torchscience.statistics.descriptive` - auto-generated from X-macro
  #define DEFINE_REDUCTION(name, extra_args, extra_count, impl) \
      DEFINE_REDUCTION_SCHEMA(module, name, extra_args, extra_count, impl);
  TORCHSCIENCE_REDUCTIONS(DEFINE_REDUCTION)
  #undef DEFINE_REDUCTION

  // `torchscience.statistics.descriptive` - histogram (non-differentiable)
  module.def("histogram(Tensor input, int bins, float[]? range, Tensor? weight, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");
  module.def("histogram_edges(Tensor input, Tensor bins, Tensor? weight, bool density, str closed, str out_of_bounds) -> (Tensor, Tensor)");

  // `torchscience.integral_transform` - auto-generated from X-macro
  // n=-1 means use input size along dim (no padding/truncation)
  // padding_mode: 0=constant, 1=reflect, 2=replicate, 3=circular
  #define DEFINE_TRANSFORM(name, extra_args, impl) \
      DEFINE_TRANSFORM_SCHEMA(module, name, extra_args, impl);
  TORCHSCIENCE_TRANSFORMS(DEFINE_TRANSFORM)
  #undef DEFINE_TRANSFORM
}
